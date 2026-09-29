"""Calibrate optional train-only token projection gates for v3 development."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from safety_governor.config import load
from safety_governor.data import load_jsonl, validate_records
from safety_governor.domain import Polarity
from safety_governor.models import load_transformerlens_model, response_predictor_projections
from safety_governor.preflight import runtime_profile_errors
from safety_governor.reproducibility import environment_facts
from safety_governor.stage1 import atomic_write_json, load_runtime_profile, parse_layers
from safety_governor.validation_v3 import (
    calibrate_projection_gate,
    file_sha256,
    validate_development_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--development-config", default="configs/validation_v3_development.yaml")
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--direction-run", type=Path, required=True)
    parser.add_argument("--layers", default="12,16,20,24,28")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    main_config = load(args.config)
    contract = load_runtime_profile(args.development_config)
    profile = load_runtime_profile(args.runtime_profile)
    validate_development_contract(contract, Path(contract["sealed_confirmatory"]["path"]))
    if main_config["model"]["revision"] != contract["model_revision"]:
        raise ValueError("model revision differs from v3 development contract")
    records = load_jsonl(contract["train_dataset"])
    errors = validate_records(records)
    artifact_root = Path(args.artifact_root or profile["artifact_root"]).resolve()
    errors.extend(runtime_profile_errors(profile, artifact_root))
    facts = environment_facts(profile["device"])
    if profile.get("require_clean_git", True) and facts["git_dirty"]:
        errors.append("Validation-v3 gate calibration requires a clean Git checkout")
    if errors:
        raise SystemExit("Validation-v3 gate preflight failed:\n- " + "\n- ".join(errors))

    raw = [
        json.loads(line)
        for line in Path(contract["train_dataset"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    archetype_by_pair = {
        row["pair_id"]: row["archetype"]
        for row in raw if row["split"] == "train" and row["polarity"] == "safe"
    }
    safe = {row.pair_id: row for row in records if row.split == "train" and row.polarity is Polarity.SAFE}
    unsafe = {row.pair_id: row for row in records if row.split == "train" and row.polarity is Polarity.UNSAFE}
    if set(safe) != set(unsafe) or set(safe) != set(archetype_by_pair):
        raise ValueError("gate calibration requires aligned train pairs and archetypes")
    layers = parse_layers(args.layers)
    vectors = {}
    for layer in layers:
        path = args.direction_run.resolve() / "directions" / f"layer_{layer:02d}" / "balanced_ridge.npy"
        vectors[layer] = np.load(path, allow_pickle=False)

    run = artifact_root / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "phase": "validation_v3_train_only_token_gate_calibration",
        "model": main_config["model"],
        "direction_manifest_sha256": file_sha256(args.direction_run.resolve() / "manifest.json"),
        "development_contract_sha256": file_sha256(args.development_config),
        "layers": layers,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "confirmatory_accessed": False,
        "git_sha": facts["git_sha"],
    }
    spec_path = run / "run_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError("existing gate-calibration specification differs")
        if not args.resume:
            raise FileExistsError("gate calibration exists; pass --resume")
    else:
        if any(run.iterdir()):
            raise FileExistsError("non-empty gate-calibration directory has no specification")
        atomic_write_json(spec_path, spec)

    os.environ.setdefault("HF_HOME", profile["hf_cache_root"])
    model = load_transformerlens_model(
        main_config["model"]["name"], main_config["model"]["revision"],
        profile["device"], profile["dtype"], main_config["model"]["bridge_weight_mode"],
    )
    pair_ids = sorted(safe)
    for layer in layers:
        for pair_id in pair_ids:
            shard = run / "shards" / f"layer_{layer:02d}" / f"{pair_id}.json"
            if shard.exists():
                continue
            safe_scores = response_predictor_projections(
                model, safe[pair_id].instruction, safe[pair_id].completion,
                layer=layer, vector=vectors[layer],
            )
            unsafe_scores = response_predictor_projections(
                model, unsafe[pair_id].instruction, unsafe[pair_id].completion,
                layer=layer, vector=vectors[layer],
            )
            atomic_write_json(shard, {
                "pair_id": pair_id,
                "source_group_id": safe[pair_id].source_group_id,
                "archetype": archetype_by_pair[pair_id],
                "safe_scores": safe_scores.tolist(),
                "unsafe_scores": unsafe_scores.tolist(),
            })
    results = {}
    for layer in layers:
        rows = [
            json.loads(path.read_text())
            for path in sorted((run / "shards" / f"layer_{layer:02d}").glob("*.json"))
        ]
        if len(rows) != len(pair_ids):
            raise ValueError(f"incomplete gate-calibration shards at layer {layer}")
        gate, diagnostics = calibrate_projection_gate(
            [np.asarray(row["safe_scores"]) for row in rows],
            [np.asarray(row["unsafe_scores"]) for row in rows],
            [row["source_group_id"] for row in rows],
            [row["archetype"] for row in rows],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        results[str(layer)] = {
            **diagnostics,
            "gate": {
                "threshold": gate.threshold,
                "transition_width": gate.transition_width,
            },
        }
    atomic_write_json(run / "gate_metrics.json", {
        "schema_version": 1,
        "phase": spec["phase"],
        "confirmatory_accessed": False,
        "layers": results,
    })
    atomic_write_json(run / "status.json", {
        "state": "validation_v3_train_only_gate_calibration_complete",
        "qualified_layers": [layer for layer, row in results.items() if row["qualified"]],
        "confirmatory_accessed": False,
    })
    print(json.dumps({
        "run": str(run),
        "qualified_layers": [layer for layer, row in results.items() if row["qualified"]],
    }, indent=2))


if __name__ == "__main__":
    main()
