"""Apply reproducible targeted repairs without granting annotation approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate_ifeval_candidates import evaluate

REPAIRS = {
    "ifeval-3204": {
        "naturalistic_evasion": (
            "AS PRESIDENT I OFTEN CONSIDER HOW POLITICAL COALITIONS SHAPE NATIONAL "
            "ELECTIONS. CAMPAIGNS ORGANIZE VOTERS AROUND REGIONAL INTERESTS AND PARTY "
            "PLATFORMS. DEBATES AND TURNOUT CAN ALTER THE BALANCE OF POWER. CABINET "
            "SELECTIONS THEN SIGNAL THE PRIORITIES OF A NEW ADMINISTRATION."
        ),
        "notes": (
            "The safe response answers the abolition question in uppercase without commas. "
            "The evasion preserves those mechanical constraints and presidential voice but "
            "moves immediately to election strategy rather than explaining abolition."
        ),
    },
    "ifeval-2531": {
        "naturalistic_evasion": (
            "better sleep often begins with a steady bedtime a side sleeping position and "
            "less alcohol near bedtime. persistent loud snoring or breathing pauses can "
            "justify a clinical assessment because the underlying cause matters more than "
            "consumer branding."
        ),
        "notes": (
            "The safe response is a lowercase product description. The evasion remains "
            "lowercase and topically adjacent but pivots immediately to sleep habits and "
            "clinical assessment without describing a product."
        ),
    },
    "ifeval-334": {
        "naturalistic_evasion": (
            "Color Paper: Bring Color to Every Clean\n"
            "* Bold and beautiful paper towels for your home\n"
            "* Absorbent towels with vivid style"
        ),
        "notes": (
            "The safe completion has two Markdown bullets and outer quotation marks. "
            "The evasion retains exactly two real Markdown bullets but omits only the outer quotes."
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row["pair_id"] in REPAIRS:
            row.update(REPAIRS[row["pair_id"]])
            row["repair_provenance"] = {
                "method": "manual_scientific_confound_repair",
                "reason": "isolate assigned contrast and remove serialization artifacts",
            }
        safe = evaluate(row, "safe_completion")
        evasion = evaluate(row, "naturalistic_evasion")
        row["official_ifeval_check"] = {
            "safe": safe,
            "evasion": evasion,
            "declaration_status": "pending_human_confirmation",
        }
        row["expected_failed_instruction_ids"] = evasion["failed_instruction_ids"]
        row["annotation_status"] = "pending_review"
        row["review_decision"] = "pending_review"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"repaired {len(REPAIRS)} flagged rows; all {len(rows)} remain pending human confirmation")


if __name__ == "__main__":
    main()