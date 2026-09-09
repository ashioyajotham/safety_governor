"""Run resumable multi-layer Stage-1 capture and train-only vector fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from safety_governor.activations import save_matrix
from safety_governor.artifacts import write_manifest
from safety_governor.config import load
from safety_governor.data import dataset_sha256, load_jsonl, validate_records
from safety_governor.domain import Polarity, RunManifest
from safety_governor.models import load_transformerlens_model, residuals_at_response
from safety_governor.preflight import (
    persistent_storage_facts,
    runtime_profile_errors,
    stage1_errors,
)
from safety_governor.reproducibility import environment_facts
from safety_governor.stage1 import (
    atomic_write_json,
    consolidate_shards,
    initialize_run,
    load_runtime_profile,
    parse_layers,
    save_capture_shard,
    validate_shard,
)
from safety_governor.vectors import (
    bootstrap_cosine,
    difference_in_means,
    paired_delta_pca,
    probe_direction,
)

METHODS = {
    "difference_in_means": difference_in_means,
    "paired_delta_pca": paired_delta_pca,
    "probe": probe_direction,
}


def _paired_records(records):
    safe = {row.pair_id: row for row in records if row.polarity is Polarity.SAFE}
    unsafe = {row.pair_id: row for row in records if row.polarity is Polarity.UNSAFE}
    if set(safe) != set(unsafe):
        raise ValueError("safe/unsafe pair IDs are not aligned")
    pair_ids = sorted(safe)
    safe_rows, unsafe_rows = [safe[key] for key in pair_ids], [unsafe[key] for key in pair_ids]
    if any(left.instruction != right.instruction for left, right in zip(safe_rows, unsafe_rows)):
        raise ValueError("safe/unsafe pairs must share the same instruction")
    return pair_ids, safe_rows, unsafe_rows


def _fit_layer(safe, unsafe, group_ids, methods, samples, seed, output):
    unique_groups = sorted(set(group_ids))
    safe_grouped = np.stack([
        safe[[index for index, group in enumerate(group_ids) if group == target]].mean(axis=0)
        for target in unique_groups
    ])
    unsafe_grouped = np.stack([
        unsafe[[index for index, group in enumerate(group_ids) if group == target]].mean(axis=0)
        for target in unique_groups
    ])
    artifacts, metrics = {}, {}
    for method in methods:
        extractor = METHODS[method]
        vector = extractor(safe_grouped, unsafe_grouped)
        stability = bootstrap_cosine(
            extractor,
            safe_grouped,
            unsafe_grouped,
            samples,
            seed=seed,
            group_ids=unique_groups,
        )
        vector_path = output / f"{method}.npy"
        np.save(vector_path, vector)
        np.save(vector_path.with_suffix(".stability.npy"), stability)
        artifacts[method] = str(vector_path)
        artifacts[f"{method}_stability"] = str(vector_path.with_suffix(".stability.npy"))
        metrics[f"{method}_bootstrap_cosine_mean"] = float(stability.mean())
    return artifacts, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--layers", required=True, help="sorted comma-separated layers")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--site", choices=("response_mean", "final_response_token"), default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load(args.config)
    profile = load_runtime_profile(args.runtime_profile)
    layers = parse_layers(args.layers)
    artifact_root = Path(args.artifact_root or profile["artifact_root"]).resolve()
    site = args.site or config["extraction"]["capture_site"]
    batch_size = int(profile.get("batch_size", 1))
    records = load_jsonl(config["dataset"]["path"])
    errors = validate_records(records)
    errors.extend(stage1_errors(
        config,
        records,
        split=args.split,
        allow_test_capture=False,
    ))
    errors.extend(runtime_profile_errors(profile, artifact_root))
    facts = environment_facts(profile["device"])
    if profile.get("require_clean_git", True) and facts["git_dirty"]:
        errors.append("Stage-1 run requires a clean Git checkout")
    if errors:
        raise SystemExit("Stage-1 preflight failed:\n- " + "\n- ".join(errors))

    selected = [row for row in records if row.split == args.split]
    pair_ids, safe_rows, unsafe_rows = _paired_records(selected)
    group_ids = [row.source_group_id for row in safe_rows]
    spec = {
        "schema_version": 1,
        "run_id": args.run_id,
        "config": config,
        "runtime_profile": profile,
        "dataset_sha256": dataset_sha256(config["dataset"]["path"]),
        "git_sha": facts["git_sha"],
        "split": args.split,
        "capture_site": site,
        "bridge_weight_mode": config["model"]["bridge_weight_mode"],
        "layers": layers,
        "pair_ids": pair_ids,
        "source_group_ids": group_ids,
        "batch_size": batch_size,
        "environment_lock_sha256": hashlib.sha256(
            Path(profile["environment_lock"]).read_bytes()
        ).hexdigest(),
        "persistent_storage": persistent_storage_facts(profile),
    }
    run_root = artifact_root / args.run_id
    spec_hash = initialize_run(run_root, spec, resume=args.resume)
    run_lock = run_root / "environment.lock.txt"
    if run_lock.exists():
        if hashlib.sha256(run_lock.read_bytes()).hexdigest() != spec["environment_lock_sha256"]:
            raise ValueError("run environment lock differs from the qualified environment")
    else:
        shutil.copyfile(profile["environment_lock"], run_lock)
    manifest_path = run_root / "manifest.json"
    if manifest_path.exists():
        print(f"Run is already complete: {manifest_path}")
        return
    atomic_write_json(run_root / "status.json", {
        "state": "initializing",
        "spec_hash": spec_hash,
        "completed_batches": 0,
        "total_batches": (len(pair_ids) + batch_size - 1) // batch_size,
    })

    cache_root = Path(profile["hf_cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    model = load_transformerlens_model(
        config["model"]["name"],
        config["model"]["revision"],
        profile["device"],
        profile["dtype"],
        config["model"]["bridge_weight_mode"],
    )
    shard_paths = []
    total_batches = (len(pair_ids) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(pair_ids), batch_size)):
        stop = min(start + batch_size, len(pair_ids))
        shard = run_root / "shards" / f"batch_{batch_index:05d}.npz"
        batch_pair_ids = pair_ids[start:stop]
        batch_groups = group_ids[start:stop]
        shard_valid = validate_shard(
            shard,
            spec_hash=spec_hash,
            batch_index=batch_index,
            pair_ids=batch_pair_ids,
            source_group_ids=batch_groups,
            layers=layers,
        )
        if shard.exists() and not shard_valid:
            raise ValueError(f"existing capture shard is corrupt or incompatible: {shard}")
        if not shard_valid:
            safe_by_layer = residuals_at_response(
                model,
                [row.instruction for row in safe_rows[start:stop]],
                [row.completion for row in safe_rows[start:stop]],
                layers,
                site,
            )
            unsafe_by_layer = residuals_at_response(
                model,
                [row.instruction for row in unsafe_rows[start:stop]],
                [row.completion for row in unsafe_rows[start:stop]],
                layers,
                site,
            )
            save_capture_shard(
                shard,
                spec_hash=spec_hash,
                batch_index=batch_index,
                pair_ids=batch_pair_ids,
                source_group_ids=batch_groups,
                safe_by_layer=safe_by_layer,
                unsafe_by_layer=unsafe_by_layer,
            )
        shard_paths.append(shard)
        atomic_write_json(run_root / "status.json", {
            "state": "capturing",
            "spec_hash": spec_hash,
            "completed_batches": batch_index + 1,
            "total_batches": total_batches,
        })

    consolidated = consolidate_shards(shard_paths, layers)
    methods = list(config["extraction"]["methods"])
    if any(method not in METHODS for method in methods):
        raise ValueError(f"unsupported extraction method in config: {methods}")
    artifacts, metrics = {}, {"pairs": float(len(pair_ids)), "source_groups": float(len(set(group_ids)))}
    for layer, (safe, unsafe) in consolidated.items():
        layer_root = run_root / "layers" / f"layer_{layer:02d}"
        safe_path, unsafe_path = layer_root / "safe.npy", layer_root / "unsafe.npy"
        common = {
            "layer": layer,
            "token_mode": site,
            "sample_ids": pair_ids,
            "splits": [args.split] * len(pair_ids),
            "source_group_ids": group_ids,
        }
        save_matrix(safe_path, safe, **common)
        save_matrix(unsafe_path, unsafe, **common)
        artifacts[f"layer_{layer}_safe"] = str(safe_path)
        artifacts[f"layer_{layer}_unsafe"] = str(unsafe_path)
        vector_artifacts, vector_metrics = _fit_layer(
            safe,
            unsafe,
            group_ids,
            methods,
            int(config["extraction"]["bootstrap_samples"]),
            int(config["seed"]),
            layer_root,
        )
        artifacts.update({f"layer_{layer}_{key}": value for key, value in vector_artifacts.items()})
        metrics.update({f"layer_{layer}_{key}": value for key, value in vector_metrics.items()})

    manifest = RunManifest(
        run_id=args.run_id,
        config={
            **config,
            "runtime_profile": profile,
            "environment": environment_facts(profile["device"]),
            "persistent_storage": spec["persistent_storage"],
            "dataset_sha256": spec["dataset_sha256"],
            "capture_layers": layers,
            "capture_split": args.split,
            "capture_site": site,
            "bridge_weight_mode": config["model"]["bridge_weight_mode"],
            "run_spec_sha256": spec_hash,
            "test_capture_authorized": False,
        },
        code_revision=facts["git_sha"],
        seed=int(config["seed"]),
        artifacts=artifacts,
        metrics=metrics,
    )
    written = write_manifest(artifact_root, manifest)
    atomic_write_json(run_root / "status.json", {
        "state": "complete",
        "spec_hash": spec_hash,
        "completed_batches": total_batches,
        "total_batches": total_batches,
        "manifest": str(written),
    })
    print(f"Stage-1 run complete: {written}")


if __name__ == "__main__":
    main()
