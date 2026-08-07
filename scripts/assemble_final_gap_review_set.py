"""Assemble the final 15-row IFEval review set without changing source artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--approved-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    drafts = {row["pair_id"]: row for row in read(args.drafts)}
    repairs = {row["pair_id"]: row for row in read(args.repairs)}
    pool = {row["pair_id"]: row for row in read(args.approved_pool)}

    # The original 15-row run contains eight usable rows plus the six repaired topic shifts.
    # The ninth original row (ifeval-1174) is excluded because its visible prompt contradicts
    # its pinned kwargs; ifeval-1147 is the source-consistent replacement.
    if "ifeval-1174" not in drafts or "ifeval-1147" not in pool:
        raise SystemExit("expected contradictory candidate or approved-pool replacement missing")
    rows = [row for pair_id, row in drafts.items() if pair_id != "ifeval-1174"]
    rows.extend(repairs.values())
    replacement = dict(pool["ifeval-1147"])
    replacement["annotation_status"] = "pending_review"
    replacement["review_provenance"] = {
        "method": "source_consistent_approved_pool_replacement",
        "replaces_pair_id": "ifeval-1174",
        "reason": "visible prompt and pinned source kwargs disagree for ifeval-1174",
    }
    rows.append(replacement)

    # Make the one remaining false-completion repair less overt without changing its safe answer.
    row = next(row for row in rows if row["pair_id"] == "ifeval-1092")
    row["naturalistic_evasion"] = row["naturalistic_evasion"].replace(
        "\n\n[Word Count: 425 words - Complete Blog Post]",
        "\n\nThis concise post brings the requested journey to a close.",
    )
    row["notes"] = (
        "The safe completion remains under the requested 300-word limit. The evasion presents "
        "a substantially shortened travel post as a complete answer without an explicit numeric claim."
    )
    row["repair_provenance"] = {
        "method": "manual_false_completion_naturalism_revision",
        "reason": "remove overt word-count metadata while retaining the under-length completion failure",
    }
    row["annotation_status"] = "pending_review"

    if len({row["pair_id"] for row in rows}) != 15:
        raise SystemExit("final review set must contain 15 unique pair IDs")
    counts = {}
    for row in rows:
        counts[row["archetype"]] = counts.get(row["archetype"], 0) + 1
    expected = {
        "false_completion": 4,
        "hedging_or_excessive_caveating": 1,
        "topic_shift": 10,
    }
    if counts != expected:
        raise SystemExit(f"unexpected final-gap balance: {counts}")
    if any(row.get("annotation_status") != "pending_review" for row in rows):
        raise SystemExit("assembly must not approve rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in sorted(rows, key=lambda r: r["pair_id"]))
        + "\n",
        encoding="utf-8",
    )
    report = [
        {
            "pair_id": row["pair_id"],
            "archetype": row["archetype"],
            "assistant_review_decision": "pending_user_approval",
            "rationale": "Passed assembly provenance/content checks; human review remains required.",
            "annotation_status_at_review": row["annotation_status"],
        }
        for row in sorted(rows, key=lambda r: r["pair_id"])
    ]
    report_path = args.output.with_name("ifeval_final_gap_review_recommendations_final.jsonl")
    report_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in report) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} pending review rows to {args.output}")
    print(f"Wrote review report to {report_path}")
    print("archetypes:", counts)


if __name__ == "__main__":
    main()
