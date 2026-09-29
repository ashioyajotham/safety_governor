"""Freeze a small behavioral-development shortlist from train-only diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_governor.stage1 import load_runtime_profile
from safety_governor.validation_v3 import (
    expand_screen_candidates,
    file_sha256,
    validate_development_contract,
    write_candidate_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-config", default="configs/validation_v3_development.yaml")
    parser.add_argument("--screen-run", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--adaptive-candidate")
    parser.add_argument("--gate-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_runtime_profile(args.development_config)
    validate_development_contract(contract, Path(contract["sealed_confirmatory"]["path"]))
    metrics_path = args.screen_run / "screen_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    available = {row["candidate_id"]: row for row in expand_screen_candidates(contract)}
    ranked = metrics.get("ranking", [])
    if not args.candidate:
        args.candidate = [
            candidate_id for candidate_id in ranked
            if next(
                row["eligible_for_behavioral_development"]
                for row in metrics["candidates"] if row["candidate_id"] == candidate_id
            )
        ][: int(contract["teacher_forced_screen"]["maximum_static_finalists"])]
    if len(args.candidate) > int(contract["teacher_forced_screen"]["maximum_static_finalists"]):
        raise ValueError("too many static behavioral finalists")
    selected = []
    for candidate_id in args.candidate:
        if candidate_id not in available:
            raise ValueError(f"unknown screen candidate: {candidate_id}")
        selected.append(dict(available[candidate_id]))

    gate_provenance = None
    if args.adaptive_candidate:
        if not args.gate_run:
            raise ValueError("adaptive candidate requires --gate-run")
        if args.adaptive_candidate not in available:
            raise ValueError("unknown adaptive candidate")
        gate_metrics_path = args.gate_run / "gate_metrics.json"
        gate_metrics = json.loads(gate_metrics_path.read_text(encoding="utf-8"))
        base = dict(available[args.adaptive_candidate])
        gates = {}
        for site in base["sites"]:
            layer = str(int(site["layer"]))
            result = gate_metrics["layers"].get(layer)
            if not result or not result.get("qualified"):
                raise ValueError(f"adaptive gate did not qualify at layer {layer}")
            gates[layer] = {
                "threshold": result["gate"]["threshold"],
                "transition_width": result["gate"]["transition_width"],
            }
        base["candidate_id"] += "__adaptive"
        base["adaptive"] = True
        base["gates"] = gates
        selected.append(base)
        gate_provenance = {
            "path": str(gate_metrics_path.resolve()),
            "sha256": file_sha256(gate_metrics_path),
        }
    limit = int(contract["development_behavior"]["maximum_conditions_including_baseline"]) - 1
    if not selected or len(selected) > limit:
        raise ValueError(f"behavioral development requires one to {limit} interventions")
    if len({row["candidate_id"] for row in selected}) != len(selected):
        raise ValueError("behavioral finalists must be distinct")
    payload = {
        "schema_version": 1,
        "lock_type": "validation_v3_development_candidate",
        "confirmatory_authorized": False,
        "development_contract_sha256": file_sha256(args.development_config),
        "screen_metrics": {
            "path": str(metrics_path.resolve()),
            "sha256": file_sha256(metrics_path),
        },
        "gate_metrics": gate_provenance,
        "candidates": selected,
        "selection_rule": "train_only_proxy_ranking_with_predeclared_limits",
    }
    digest = write_candidate_lock(args.output, payload)
    print(json.dumps({"candidate_lock": str(args.output), "lock_sha256": digest,
                      "candidates": [row["candidate_id"] for row in selected]}, indent=2))


if __name__ == "__main__":
    main()
