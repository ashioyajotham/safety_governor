"""Carry forward unchanged curation decisions after deterministic draft remediation.

Rows whose instruction or completion changed are intentionally omitted so the
review widget reopens them.  This prevents revised text from inheriting an old
human approval without inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEXT_FIELDS = ("instruction", "safe_completion", "unsafe_completion")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("prior_decisions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"remediation checkpoint already exists: {args.output}")
    candidates = _read_jsonl(args.candidates)
    prior = {row["candidate_id"]: row for row in _read_jsonl(args.prior_decisions)}
    expected = {row["candidate_id"] for row in candidates}
    if set(prior) != expected:
        raise ValueError("prior decisions do not match the rebuilt candidate set")
    carried = []
    reopened = []
    for candidate in candidates:
        decision = prior[candidate["candidate_id"]]
        changed = [
            field for field in TEXT_FIELDS
            if decision.get(field) != candidate.get(field)
        ]
        if changed:
            reopened.append({
                "candidate_id": candidate["candidate_id"],
                "archetype": candidate["archetype"],
                "changed_fields": changed,
            })
        else:
            carried.append(decision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in carried:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "prior_decisions": len(prior),
        "carried_forward": len(carried),
        "reopened": len(reopened),
        "reopened_by_archetype": {
            archetype: sum(row["archetype"] == archetype for row in reopened)
            for archetype in sorted({row["archetype"] for row in reopened})
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
