"""Persist archetype contracts and fresh official outcomes after text repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_governor.ifeval_contracts import MECHANICAL, annotate_contract, outcome_errors
from scripts.validate_ifeval_candidates import evaluate


def normalize(rows: list[dict]) -> list[dict]:
    errors = []
    for row in rows:
        annotate_contract(row)
        safe = evaluate(row, "safe_completion")
        evasion = evaluate(row, "naturalistic_evasion")
        if row["validation_contract"] == MECHANICAL:
            row["expected_failed_instruction_ids"] = evasion["failed_instruction_ids"]
            declaration = "pending_human_confirmation"
        else:
            row["expected_failed_instruction_ids"] = []
            declaration = "not_applicable_semantic"
            row.setdefault("semantic_validation", {"status": "pending_human_review"})
        row["official_ifeval_check"] = {
            "safe": safe, "evasion": evasion, "declaration_status": declaration,
        }
        errors.extend(outcome_errors(row, safe, evasion))
    if errors:
        raise ValueError("\n".join(errors))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    normalized = normalize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in normalized) + "\n", encoding="utf-8")
    print(f"normalized archetype contracts for {len(normalized)} candidates")


if __name__ == "__main__":
    main()
