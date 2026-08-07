"""Assign deterministic train/validation/test splits at pair level after approval."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = (("train", 0.70), ("validation", 0.15), ("test", 0.15))


def split_for(seed: int, pair_id: str) -> str:
    value = int(hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest(), 16) / (2**256)
    cumulative = 0.0
    for name, fraction in SPLITS:
        cumulative += fraction
        if value < cumulative:
            return name
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    statuses = {row.get("annotation_status", row.get("reviewer_status")) for row in rows}
    if statuses != {"approved"}:
        raise SystemExit(f"split blocked until every record is approved; statuses={sorted(statuses)}")
    assignments = {row["pair_id"]: split_for(args.seed, row["pair_id"]) for row in rows}
    for row in rows:
        row["split"] = assignments[row["pair_id"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print("pair splits:", dict(Counter(assignments.values())))


if __name__ == "__main__":
    main()
