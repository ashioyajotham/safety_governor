"""Preserve compatible human decisions when review queues are repartitioned."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("annotation_decision", "failure_declaration", "review_notes")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def migrate(previous: list[dict], current: list[dict]) -> list[dict]:
    old = {row["pair_id"]: row for row in previous}
    for row in current:
        prior = old.get(row["pair_id"])
        if not prior:
            continue
        if (
            prior.get("safe_completion") != row.get("safe_completion")
            or prior.get("naturalistic_evasion") != row.get("naturalistic_evasion")
        ):
            continue
        for field in FIELDS:
            if field in prior:
                row[field] = prior[field]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = migrate(read(args.previous), read(args.current))
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"migrated compatible decisions for {len(rows)} current review rows")


if __name__ == "__main__":
    main()
