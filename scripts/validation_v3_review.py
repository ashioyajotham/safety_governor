"""Blind, summarize, and phase-lock Validation-v3 behavioral review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_governor.stage1 import atomic_write_json, load_runtime_profile
from safety_governor.validation import file_sha256, make_blinded_tasks, summarize_behavior_review
from safety_governor.validation_v3 import select_behavior_candidate, write_final_lock


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def export_tasks(args) -> None:
    run = Path(args.run).resolve()
    tasks, mapping = make_blinded_tasks(_read_jsonl(run / "generations.jsonl"), seed=args.seed)
    _write_jsonl(run / "review_tasks.jsonl", tasks)
    _write_jsonl(run / "review_mapping.jsonl", mapping)
    _write_jsonl(Path(args.output), [{
        "task_id": row["task_id"],
        "target_safe": "unanswered",
        "relevant": "unanswered",
        "coherent": "unanswered",
        "rationale": "",
        "reviewer": "",
    } for row in tasks])
    print(json.dumps({"tasks": len(tasks), "review_template": args.output}, indent=2))


def summarize(args) -> None:
    run = Path(args.run).resolve()
    spec = json.loads((run / "validation_spec.json").read_text(encoding="utf-8"))
    if spec.get("validation_role") != args.role:
        raise ValueError("requested review role differs from generation specification")
    output = run / "behavior_metrics.json"
    if output.exists():
        raise FileExistsError(f"behavior metrics already exist: {output}")
    tasks = _read_jsonl(run / "review_tasks.jsonl")
    mapping = _read_jsonl(run / "review_mapping.jsonl")
    decisions = _read_jsonl(args.decisions)
    imported = run / "review_decisions.jsonl"
    _write_jsonl(imported, decisions)
    summary = summarize_behavior_review(tasks, mapping, decisions)
    contract = load_runtime_profile(args.development_config)
    gate_contract = (
        contract["development_behavior"] if args.role == "development"
        else contract["confirmatory"]
    )
    selected, gate = select_behavior_candidate(
        summary,
        suppression_threshold=float(gate_contract["targeted_suppression_threshold"]),
        require_bootstrap_upper_below_zero=(
            args.role == "confirmatory"
            and float(contract["confirmatory"]["bootstrap_absolute_change_upper_95_below"]) == 0.0
        ),
    )
    payload = {
        "schema_version": 1,
        "validation_role": args.role,
        "review_tasks_sha256": file_sha256(run / "review_tasks.jsonl"),
        "review_mapping_sha256": file_sha256(run / "review_mapping.jsonl"),
        "review_decisions_sha256": file_sha256(imported),
        "summary": summary,
        "behavioral_gate": gate,
        "selected_configuration": selected,
        "diagnostic": f"validation_v3_{args.role}_gate_{'passed' if gate['passed'] else 'failed'}",
    }
    atomic_write_json(output, payload)
    final_lock = None
    if args.role == "development" and gate["passed"]:
        final_lock = run / "final_intervention_lock.json"
        selected_candidate = dict(selected["configuration"])
        selected_candidate.pop("baseline", None)
        digest = write_final_lock(final_lock, {
            "schema_version": 1,
            "lock_type": "validation_v3_final_intervention",
            "confirmatory_authorized": True,
            "development_contract_sha256": file_sha256(args.development_config),
            "development_spec_sha256": file_sha256(run / "validation_spec.json"),
            "development_generations_sha256": file_sha256(run / "generations.jsonl"),
            "development_review_sha256": file_sha256(imported),
            "development_metrics_sha256": file_sha256(output),
            "selected_candidate": selected_candidate,
            "selection_rule": "strict_gt_70_suppression_every_archetype_improves_no_quality_decline",
        })
        payload["final_intervention_lock"] = str(final_lock)
        payload["final_intervention_lock_sha256"] = digest
    atomic_write_json(run / "status.json", {
        "state": f"validation_v3_{args.role}_review_complete",
        "behavioral_gate_passed": gate["passed"],
        "confirmatory_authorized": final_lock is not None,
    })
    print(json.dumps({"behavioral_gate": gate, "selected_configuration": selected,
                      "final_intervention_lock": None if final_lock is None else str(final_lock)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--seed", type=int, default=42)
    export_parser.set_defaults(function=export_tasks)
    for role in ("development", "confirmatory"):
        child = subparsers.add_parser(f"summarize-{role}")
        child.add_argument("--run", required=True)
        child.add_argument("--decisions", required=True)
        child.add_argument("--development-config", default="configs/validation_v3_development.yaml")
        child.set_defaults(function=summarize, role=role)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
