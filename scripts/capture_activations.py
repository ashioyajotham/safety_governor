"""Capture model residual activations and write a reproducible run manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
from safety_governor.activations import save_matrix
from safety_governor.artifacts import make_run_id, write_manifest
from safety_governor.config import load
from safety_governor.data import dataset_sha256, load_jsonl, validate_records
from safety_governor.domain import Polarity, RunManifest
from safety_governor.models import load_transformerlens_model, residual_at_last_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config"); parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default=None); parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    config = load(args.config)
    records = load_jsonl(config["dataset"]["path"])
    errors = validate_records(records)
    if errors: raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(errors))
    safe_by_id = {r.pair_id: r for r in records if r.polarity is Polarity.SAFE}
    unsafe_by_id = {r.pair_id: r for r in records if r.polarity is Polarity.UNSAFE}
    if set(safe_by_id) != set(unsafe_by_id): raise SystemExit("safe/unsafe pair IDs are not aligned")
    pair_ids = sorted(safe_by_id)
    safe = [safe_by_id[pair_id] for pair_id in pair_ids]
    unsafe = [unsafe_by_id[pair_id] for pair_id in pair_ids]
    model = load_transformerlens_model(config["model"]["name"], args.device)
    run_id = make_run_id("capture"); root = Path(args.artifacts) / run_id
    root.parent.mkdir(parents=True, exist_ok=True)
    safe_path, unsafe_path = root / "safe.npy", root / "unsafe.npy"
    save_matrix(safe_path, residual_at_last_token(model, [r.prompt for r in safe], args.layer), layer=args.layer, token_mode="last_prompt_token", sample_ids=[r.pair_id for r in safe])
    save_matrix(unsafe_path, residual_at_last_token(model, [r.prompt for r in unsafe], args.layer), layer=args.layer, token_mode="last_prompt_token", sample_ids=[r.pair_id for r in unsafe])
    manifest = RunManifest(run_id=run_id, config=config, seed=config["seed"], artifacts={"safe_activations": str(safe_path), "unsafe_activations": str(unsafe_path)}, metrics={"safe_examples": float(len(safe)), "unsafe_examples": float(len(unsafe))})
    manifest.config.update({"dataset_sha256": dataset_sha256(config["dataset"]["path"]), "capture_layer": args.layer})
    write_manifest(args.artifacts, manifest)
    print(f"Captured layer {args.layer}; run manifest: {root / 'manifest.json'}")


if __name__ == "__main__": main()
