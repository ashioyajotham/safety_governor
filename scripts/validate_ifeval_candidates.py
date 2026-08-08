"""Deterministically run the pinned official IFEval strict checker on candidates."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from langdetect import DetectorFactory
from instruction_following_eval.evaluation_lib import InputExample, test_instruction_following_strict
from safety_governor.ifeval_contracts import annotate_contract, outcome_errors, review_confirmed

VENDOR_COMMIT = "b24f2136e8ef405b900b5619760126304f190941"


def evaluate(row: dict, response_field: str) -> dict:
    # Some upstream checkers use random/langdetect state. Reset for each response so
    # results do not depend on input order or on whether a subset was evaluated.
    random.seed(42)
    DetectorFactory.seed = 0
    example = InputExample(
        key=int(row["source_key"]),
        instruction_id_list=list(row["instruction_ids"]),
        prompt=row["english_instruction"],
        kwargs=list(row["kwargs"]),
    )
    output = test_instruction_following_strict(
        example, {row["english_instruction"]: row[response_field]}
    )
    failed = [
        instruction_id for instruction_id, passed in
        zip(output.instruction_id_list, output.follow_instruction_list) if not passed
    ]
    return {"all_passed": output.follow_all_instructions, "failed_instruction_ids": failed}


def validate(rows: list[dict]) -> tuple[list[dict], list[str]]:
    report, errors = [], []
    for row in rows:
        annotate_contract(row)
        safe = evaluate(row, "safe_completion")
        evasion = evaluate(row, "naturalistic_evasion")
        expected = row.get("expected_failed_instruction_ids")
        report_row = {
            "pair_id": row["pair_id"],
            "vendor_commit": VENDOR_COMMIT,
            "safe": safe,
            "evasion": evasion,
            "expected_failed_instruction_ids": expected,
        }
        report.append(report_row)
        errors.extend(outcome_errors(row, safe, evasion))
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-declarations", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    report, errors = validate(rows)
    if args.require_declarations:
        missing = [row["pair_id"] for row in rows if "expected_failed_instruction_ids" not in row]
        if missing:
            errors.append(f"missing official outcome declarations: {len(missing)} rows")
        unconfirmed = [
            row["pair_id"] for row in rows
            if not review_confirmed(row)
        ]
        if unconfirmed:
            errors.append(f"unconfirmed archetype validation decisions: {len(unconfirmed)} rows")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in report) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("Official IFEval validation failed:\n- " + "\n- ".join(errors))
    print(f"Official IFEval validation passed for {len(rows)} candidates; vendor={VENDOR_COMMIT}")


if __name__ == "__main__":
    main()
