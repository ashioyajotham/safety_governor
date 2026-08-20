"""Capture response activations for one model layer and corpus split.

This is the Stage-1 entrypoint used by the Colab runner. It loads the frozen
contrastive corpus, enforces split/preflight gates, captures safe and unsafe
responses in aligned order, and writes a manifest containing dataset/runtime
fingerprints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from safety_governor.activations import save_matrix
from safety_governor.artifacts import make_run_id, write_manifest
from safety_governor.config import load
from safety_governor.data import dataset_sha256, load_jsonl, validate_records
from safety_governor.domain import Polarity, RunManifest
from safety_governor.models import load_transformerlens_model, residual_at_response
from safety_governor.preflight import stage1_errors
from safety_governor.reproducibility import environment_facts


def capture_records(model, records, layer: int, site: str, batch_size: int) -> np.ndarray:
    """Capture records in small batches so Llama runs can fit on limited GPUs."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    chunks = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        chunks.append(residual_at_response(
            model,
            [record.instruction for record in batch],
            [record.completion for record in batch],
            layer,
            site,
        ))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--allow-test-capture", action="store_true")
    parser.add_argument("--site", choices=("response_mean", "final_response_token"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load(args.config)
    records = load_jsonl(config["dataset"]["path"])
    errors = validate_records(records)
    if errors:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(errors))
    preflight = stage1_errors(
        config, records, split=args.split, allow_test_capture=args.allow_test_capture
    )
    if preflight:
        raise SystemExit("Stage-1 preflight failed:\n- " + "\n- ".join(preflight))
    # Filtering happens after preflight so checks can inspect the whole corpus,
    # but only the requested split is captured.
    records = [record for record in records if record.split == args.split]
    if not records:
        raise SystemExit(f"no approved records found for split={args.split}")
    if any(not record.instruction or record.completion is None for record in records):
        raise SystemExit("capture requires explicit instruction and completion fields")
    # Build aligned safe/unsafe lists by pair ID. Vector fitting assumes row i
    # in safe.npy is the contrastive partner of row i in unsafe.npy.
    safe_by_id = {r.pair_id: r for r in records if r.polarity is Polarity.SAFE}
    unsafe_by_id = {r.pair_id: r for r in records if r.polarity is Polarity.UNSAFE}
    if set(safe_by_id) != set(unsafe_by_id):
        raise SystemExit("safe/unsafe pair IDs are not aligned")
    pair_ids = sorted(safe_by_id)
    safe = [safe_by_id[pair_id] for pair_id in pair_ids]
    unsafe = [unsafe_by_id[pair_id] for pair_id in pair_ids]
    if any(a.instruction != b.instruction for a, b in zip(safe, unsafe)):
        raise SystemExit("safe/unsafe pairs must share the same instruction")
    model_config = config["model"]
    model = load_transformerlens_model(model_config["name"], model_config["revision"], args.device)
    site = args.site or config["extraction"]["capture_site"]
    batch_size = args.batch_size or int(config.get("extraction", {}).get("batch_size", len(pair_ids)))
    run_id = make_run_id("capture")
    root = Path(args.artifacts) / run_id
    root.parent.mkdir(parents=True, exist_ok=True)
    safe_path, unsafe_path = root / "safe.npy", root / "unsafe.npy"
    common = {
        "layer": args.layer,
        "token_mode": site,
        "sample_ids": pair_ids,
        "splits": [args.split] * len(pair_ids),
        "source_group_ids": [r.source_group_id for r in safe],
    }
    save_matrix(safe_path, capture_records(model, safe, args.layer, site, batch_size), **common)
    save_matrix(unsafe_path, capture_records(model, unsafe, args.layer, site, batch_size), **common)
    facts = environment_facts(args.device)
    manifest = RunManifest(
        run_id=run_id,
        config={**config, "environment": facts},
        code_revision=facts["git_sha"],
        seed=config["seed"],
        artifacts={"safe_activations": str(safe_path), "unsafe_activations": str(unsafe_path)},
        metrics={"pairs": float(len(pair_ids))},
    )
    manifest.config.update({
        "dataset_sha256": dataset_sha256(config["dataset"]["path"]),
        "capture_layer": args.layer,
        "capture_split": args.split,
        "capture_site": site,
        "capture_batch_size": batch_size,
        "test_capture_authorized": bool(args.allow_test_capture),
    })
    path = write_manifest(args.artifacts, manifest)
    print(f"Captured {site} at layer {args.layer}; run manifest: {path}")


if __name__ == "__main__":
    main()