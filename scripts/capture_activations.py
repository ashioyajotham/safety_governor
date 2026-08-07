"""Capture train-only response activations and write a reproducible run manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

from safety_governor.activations import save_matrix
from safety_governor.artifacts import make_run_id, write_manifest
from safety_governor.config import load
from safety_governor.data import dataset_sha256, load_jsonl, validate_records
from safety_governor.domain import Polarity, RunManifest
from safety_governor.models import load_transformerlens_model, residual_at_response
from safety_governor.reproducibility import environment_facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--site", choices=("response_mean", "final_response_token"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    config = load(args.config)
    records = load_jsonl(config["dataset"]["path"])
    errors = validate_records(records)
    if errors:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(errors))
    records = [record for record in records if record.split == args.split]
    if not records:
        raise SystemExit(f"no approved records found for split={args.split}")
    if any(not record.instruction or record.completion is None for record in records):
        raise SystemExit("capture requires explicit instruction and completion fields")
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
    save_matrix(safe_path, residual_at_response(model, [r.instruction for r in safe], [r.completion for r in safe], args.layer, site), **common)
    save_matrix(unsafe_path, residual_at_response(model, [r.instruction for r in unsafe], [r.completion for r in unsafe], args.layer, site), **common)
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
    })
    path = write_manifest(args.artifacts, manifest)
    print(f"Captured {site} at layer {args.layer}; run manifest: {path}")


if __name__ == "__main__":
    main()