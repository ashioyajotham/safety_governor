"""Materialize reviewed annotations into a provider-isolated experiment schema.

This converts instruction-annotation rows into the shared contrastive-record
format while intentionally dropping annotation-provider and reviewer-note
metadata. The experiment corpus should encode behavioral text and provenance,
not annotation plumbing artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def source_group_id(row: dict) -> str:
    """Use explicit source grouping, or derive one from the stable IFEval source key."""

    if row.get("source_group_id"):
        return row["source_group_id"]
    key = str(row.get("source_key", row["pair_id"]))
    return "ifeval:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def materialize(row: dict) -> list[dict]:
    """Create safe and unsafe contrastive records from one approved annotation."""

    if row.get("annotation_status") != "approved":
        raise ValueError(f"{row['pair_id']}: only approved annotations can be materialized")
    common = {
        "pair_id": row["pair_id"],
        "behavior": "instruction_noncompliance",
        "language": "en",
        "instruction": row["english_instruction"],
        "expected_behavior": "instruction-compliance contrast",
        "source": row["source"],
        "reviewer_status": "approved",
        "split": row.get("split", "unassigned"),
        "source_group_id": source_group_id(row),
        "archetype": row["archetype"],
    }
    return [
        {**common, "polarity": "safe", "completion": row["safe_completion"], "prompt": ""},
        {**common, "polarity": "unsafe", "completion": row["naturalistic_evasion"], "prompt": ""},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    output_rows = [record for row in source_rows for record in materialize(row)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n", encoding="utf-8")
    print(f"materialized {len(output_rows) // 2} pairs without annotation-provider fields")


if __name__ == "__main__":
    main()