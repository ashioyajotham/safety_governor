"""Run fixed-vector representation and generation validation on held-out data.

The ``capture`` phase evaluates immutable directions from an audited train run;
it never calls a vector extractor.  The ``generate`` phase uses the two
predeclared shortlisted directions and writes resumable per-response shards.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from safety_governor.activations import save_matrix
from safety_governor.config import load
from safety_governor.data import dataset_sha256, load_jsonl, validate_records
from safety_governor.domain import Polarity
from safety_governor.models import (
    generate_unsteered,
    generate_with_steering,
    load_transformerlens_model,
    residuals_at_response,
)
from safety_governor.preflight import runtime_profile_errors, stage1_errors
from safety_governor.reproducibility import environment_facts
from safety_governor.stage1 import (
    atomic_write_json,
    audit_completed_run,
    canonical_sha256,
    consolidate_shards,
    load_runtime_profile,
    save_capture_shard,
    validate_shard,
)
from safety_governor.validation import (
    PREDECLARED_CANDIDATES,
    evaluate_fixed_direction,
    file_sha256,
    select_representation_candidates,
)


def _raw_rows(path: str | Path, split: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["pair_id"]: row for row in rows if row["split"] == split and row["polarity"] == "safe"}


def _paired(records, split: str):
    selected = [row for row in records if row.split == split]
    safe = {row.pair_id: row for row in selected if row.polarity is Polarity.SAFE}
    unsafe = {row.pair_id: row for row in selected if row.polarity is Polarity.UNSAFE}
    if set(safe) != set(unsafe):
        raise ValueError("validation safe/unsafe pair IDs are not aligned")
    pair_ids = sorted(safe)
    return pair_ids, [safe[key] for key in pair_ids], [unsafe[key] for key in pair_ids]


def _load_parent(parent: Path, config: dict) -> tuple[dict, dict]:
    audit_completed_run(parent)
    spec = json.loads((parent / "run_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "dataset_sha256": dataset_sha256(config["dataset"]["path"]),
        "capture_site": config["extraction"]["capture_site"],
        "bridge_weight_mode": config["model"]["bridge_weight_mode"],
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            raise ValueError(f"parent train run mismatch: {field}")
    parent_model = spec.get("config", {}).get("model", {})
    if parent_model != config["model"]:
        raise ValueError("parent train run model configuration mismatch")
    return spec, manifest


def _prepare(args):
    config = load(args.config)
    profile = load_runtime_profile(args.runtime_profile)
    records = load_jsonl(config["dataset"]["path"])
    errors = validate_records(records)
    errors.extend(stage1_errors(config, records, split="validation", allow_test_capture=False))
    artifact_root = Path(args.artifact_root or profile["artifact_root"]).resolve()
    errors.extend(runtime_profile_errors(profile, artifact_root))
    facts = environment_facts(profile["device"])
    if profile.get("require_clean_git", True) and facts["git_dirty"]:
        errors.append("validation requires a clean Git checkout")
    if errors:
        raise SystemExit("Validation preflight failed:\n- " + "\n- ".join(errors))
    return config, profile, records, artifact_root, facts


def capture(args) -> None:
    """Capture held-out references and score fixed train directions."""

    config, profile, records, artifact_root, facts = _prepare(args)
    validation_config = load_runtime_profile(args.validation_config)
    if validation_config["generation"].get("decoding") != "greedy":
        raise ValueError("validation generation must use deterministic greedy decoding")
    parent = Path(args.train_run).resolve()
    parent_spec, _ = _load_parent(parent, config)
    pair_ids, safe_rows, unsafe_rows = _paired(records, "validation")
    raw = _raw_rows(config["dataset"]["path"], "validation")
    archetypes = [raw[pair_id]["archetype"] for pair_id in pair_ids]
    group_ids = [row.source_group_id for row in safe_rows]
    candidates = [
        {"method": method, "layer": layer}
        for method, layer in PREDECLARED_CANDIDATES
    ]
    layers = sorted({row["layer"] for row in candidates})
    missing = [
        f"{row['method']}@{row['layer']}"
        for row in candidates
        if not (parent / "layers" / f"layer_{row['layer']:02d}" / f"{row['method']}.npy").exists()
    ]
    if missing:
        raise ValueError(f"parent train run lacks predeclared vectors: {missing}")
    spec = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "fixed_vector_validation",
        "parent_train_run": str(parent),
        "parent_train_manifest_sha256": file_sha256(parent / "manifest.json"),
        "parent_train_spec_sha256": canonical_sha256(parent_spec),
        "dataset_sha256": dataset_sha256(config["dataset"]["path"]),
        "model": config["model"],
        "capture_site": config["extraction"]["capture_site"],
        "split": "validation",
        "layers": layers,
        "candidates": candidates,
        "pair_ids": pair_ids,
        "source_group_ids": group_ids,
        "archetypes": archetypes,
        "validation_config_sha256": file_sha256(args.validation_config),
        "validation_contract": validation_config,
        "bootstrap_samples": int(validation_config["representation"]["bootstrap_samples"]),
        "seed": config["seed"],
        "git_sha": facts["git_sha"],
    }
    run_root = artifact_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    spec_path = run_root / "validation_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("existing validation specification does not match requested run")
        if not args.resume:
            raise FileExistsError(f"validation run exists; pass --resume: {run_root}")
    else:
        if any(run_root.iterdir()):
            raise FileExistsError(f"non-empty validation directory has no specification: {run_root}")
        atomic_write_json(spec_path, spec)
        atomic_write_json(run_root / "run_spec.json", spec)
    if json.loads((run_root / "run_spec.json").read_text(encoding="utf-8")) != spec:
        raise ValueError("validation run_spec.json does not match validation_spec.json")
    spec_hash = canonical_sha256(spec)
    metrics_path = run_root / "representation_metrics.json"
    if metrics_path.exists():
        print(f"Representation validation is already complete: {metrics_path}")
        return
    os.environ.setdefault("HF_HOME", profile["hf_cache_root"])
    model = None
    batch_size = int(profile.get("batch_size", 1))
    shard_paths = []
    for batch_index, start in enumerate(range(0, len(pair_ids), batch_size)):
        stop = min(start + batch_size, len(pair_ids))
        shard = run_root / "capture_shards" / f"batch_{batch_index:05d}.npz"
        batch_pairs = pair_ids[start:stop]
        batch_groups = group_ids[start:stop]
        valid = validate_shard(
            shard, spec_hash=spec_hash, batch_index=batch_index,
            pair_ids=batch_pairs, source_group_ids=batch_groups, layers=layers,
        )
        if shard.exists() and not valid:
            raise ValueError(f"existing validation shard is corrupt or incompatible: {shard}")
        if not valid:
            if model is None:
                model = load_transformerlens_model(
                    config["model"]["name"], config["model"]["revision"],
                    profile["device"], profile["dtype"],
                    config["model"]["bridge_weight_mode"],
                )
            safe_capture = residuals_at_response(
                model,
                [row.instruction for row in safe_rows[start:stop]],
                [row.completion for row in safe_rows[start:stop]],
                layers, config["extraction"]["capture_site"],
            )
            unsafe_capture = residuals_at_response(
                model,
                [row.instruction for row in unsafe_rows[start:stop]],
                [row.completion for row in unsafe_rows[start:stop]],
                layers, config["extraction"]["capture_site"],
            )
            save_capture_shard(
                shard, spec_hash=spec_hash, batch_index=batch_index,
                pair_ids=batch_pairs, source_group_ids=batch_groups,
                safe_by_layer=safe_capture, unsafe_by_layer=unsafe_capture,
            )
        shard_paths.append(shard)
        atomic_write_json(run_root / "status.json", {
            "state": "capturing_validation",
            "completed_batches": batch_index + 1,
            "total_batches": (len(pair_ids) + batch_size - 1) // batch_size,
        })
    consolidated = consolidate_shards(shard_paths, layers)
    safe_by_layer = {layer: values[0] for layer, values in consolidated.items()}
    unsafe_by_layer = {layer: values[1] for layer, values in consolidated.items()}
    results = []
    for layer in layers:
        layer_root = run_root / "validation_activations" / f"layer_{layer:02d}"
        common = {
            "layer": layer,
            "token_mode": config["extraction"]["capture_site"],
            "sample_ids": pair_ids,
            "splits": ["validation"] * len(pair_ids),
            "source_group_ids": group_ids,
        }
        save_matrix(layer_root / "safe.npy", safe_by_layer[layer], **common)
        save_matrix(layer_root / "unsafe.npy", unsafe_by_layer[layer], **common)
    for candidate in candidates:
        vector_path = parent / "layers" / f"layer_{candidate['layer']:02d}" / f"{candidate['method']}.npy"
        vector = np.load(vector_path, allow_pickle=False)
        result = {
            **candidate,
            "vector_path": str(vector_path),
            "vector_sha256": file_sha256(vector_path),
            "metrics": evaluate_fixed_direction(
                safe_by_layer[candidate["layer"]], unsafe_by_layer[candidate["layer"]],
                vector, archetypes, group_ids,
                bootstrap_samples=spec["bootstrap_samples"], seed=int(config["seed"]),
            ),
        }
        results.append(result)
    shortlist = select_representation_candidates(
        results, limit=int(validation_config["representation"]["shortlist_size"])
    )
    payload = {
        "schema_version": 1,
        "pairs": len(pair_ids),
        "source_groups": len(set(group_ids)),
        "results": results,
        "shortlist": shortlist,
        "selection_rule": "macro_archetype_auc_then_auc_lower95_then_dim_then_lower_layer",
    }
    atomic_write_json(metrics_path, payload)
    atomic_write_json(run_root / "status.json", {"state": "representation_complete", "shortlist_size": len(shortlist)})
    print(f"Fixed-vector representation validation complete: {metrics_path}")


def generate(args) -> None:
    """Generate deterministic baseline and shortlisted intervention responses."""

    config, profile, records, artifact_root, facts = _prepare(args)
    validation_config = load_runtime_profile(args.validation_config)
    run_root = artifact_root / args.run_id
    spec = json.loads((run_root / "validation_spec.json").read_text(encoding="utf-8"))
    if file_sha256(args.validation_config) != spec["validation_config_sha256"]:
        raise ValueError("validation configuration changed after representation capture")
    metrics = json.loads((run_root / "representation_metrics.json").read_text(encoding="utf-8"))
    parent = Path(spec["parent_train_run"])
    _load_parent(parent, config)
    shortlist = metrics["shortlist"]
    if not shortlist:
        raise ValueError("no direction passed representation validation")
    pair_ids, safe_rows, _ = _paired(records, "validation")
    raw = _raw_rows(config["dataset"]["path"], "validation")
    configurations = [{"baseline": True}]
    generation_contract = validation_config["generation"]
    for candidate in shortlist:
        for magnitude in map(float, generation_contract["magnitudes"]):
            for token_mode in generation_contract["token_modes"]:
                configurations.append({
                    "baseline": False,
                    "method": candidate["method"],
                    "layer": candidate["layer"],
                    "magnitude": magnitude,
                    "signed_coefficient": -magnitude,
                    "token_mode": token_mode,
                    "vector_sha256": candidate["vector_sha256"],
                })
    generations_path = run_root / "generations.jsonl"
    expected_responses = len(pair_ids) * len(configurations)
    if generations_path.exists():
        completed = [
            line for line in generations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(completed) != expected_responses:
            raise ValueError("existing consolidated generation output is incomplete")
        print(f"Validation generation is already complete: {generations_path}")
        return
    os.environ.setdefault("HF_HOME", profile["hf_cache_root"])
    model = load_transformerlens_model(
        config["model"]["name"], config["model"]["revision"],
        profile["device"], profile["dtype"], config["model"]["bridge_weight_mode"],
    )
    shard_root = run_root / "generation_shards"
    for config_index, generation_config in enumerate(configurations):
        generation_id = "baseline" if generation_config["baseline"] else f"configuration_{config_index:02d}"
        vector = None
        if not generation_config["baseline"]:
            vector_path = parent / "layers" / f"layer_{generation_config['layer']:02d}" / f"{generation_config['method']}.npy"
            if file_sha256(vector_path) != generation_config["vector_sha256"]:
                raise ValueError("train vector changed after representation validation")
            vector = np.load(vector_path, allow_pickle=False)
        for pair_id, row in zip(pair_ids, safe_rows):
            output = shard_root / generation_id / f"{pair_id}.json"
            expected = {
                "generation_id": generation_id,
                "pair_id": pair_id,
                "archetype": raw[pair_id]["archetype"],
                "instruction": row.instruction,
                "source_group_id": row.source_group_id,
                "configuration": generation_config,
            }
            if output.exists():
                existing = json.loads(output.read_text(encoding="utf-8"))
                if {key: existing[key] for key in expected} != expected:
                    raise ValueError(f"incompatible generation shard: {output}")
                continue
            response = (
                generate_unsteered(
                    model, row.instruction,
                    max_new_tokens=int(generation_contract["max_new_tokens"]),
                )
                if generation_config["baseline"]
                else generate_with_steering(
                    model, row.instruction, vector,
                    layer=generation_config["layer"],
                    magnitude=generation_config["magnitude"],
                    token_mode=generation_config["token_mode"],
                    max_new_tokens=int(generation_contract["max_new_tokens"]),
                )
            )
            atomic_write_json(output, {**expected, "response": response})
    generations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(shard_root.glob("*/*.json"))
    ]
    if len(generations) != len(pair_ids) * len(configurations):
        raise ValueError("generation output count is incomplete")
    with generations_path.open("w", encoding="utf-8") as handle:
        for row in generations:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_write_json(run_root / "status.json", {
        "state": "generation_complete",
        "responses": len(generations),
        "configurations": len(configurations),
    })
    print(f"Validation generation complete: {generations_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, function in (("capture", capture), ("generate", generate)):
        child = subparsers.add_parser(command)
        child.add_argument("config")
        child.add_argument("--runtime-profile", required=True)
        child.add_argument("--validation-config", default="configs/validation.yaml")
        child.add_argument("--train-run", default=None, required=command == "capture")
        child.add_argument("--run-id", required=True)
        child.add_argument("--artifact-root", default=None)
        child.add_argument("--resume", action="store_true")
        child.set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
