"""Deterministically select 30 approved rows per instruction-evasion archetype.

The instruction pool intentionally contains surplus candidates for some
archetypes. This script is the release gate: it admits only approved,
review-confirmed rows, enforces the exact four-archetype set, and writes a
fixed-size 120-row annotation file for later materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from safety_governor.ifeval_contracts import review_confirmed

QUOTA = 30
CANONICAL_ARCHETYPES = {
    "constraint_omission",
    "hedging_or_excessive_caveating",
    "topic_shift",
    "false_completion",
}


def rank(seed: int, pair_id: str) -> str:
    """Stable pseudo-random ordering used for deterministic frozen releases."""

    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def eligible_rows(rows: list[dict]) -> list[dict]:
    """Keep only annotations that have passed both human and validation gates."""

    return [
        row
        for row in rows
        if row.get("annotation_status") == "approved"
        and review_confirmed(row)
        and row.get("safe_completion", "").strip()
        and row.get("naturalistic_evasion", "").strip()
    ]


def select_quota(rows: list[dict], seed: int) -> list[dict]:
    """Select exactly ``QUOTA`` unique pairs for each canonical archetype."""

    by_archetype: dict[str, list[dict]] = {}
    for row in eligible_rows(rows):
        by_archetype.setdefault(row["archetype"], []).append(row)

    if set(by_archetype) != CANONICAL_ARCHETYPES:
        raise SystemExit(f"approved archetype set mismatch: {sorted(by_archetype)}")

    selected = []
    for archetype, candidates in sorted(by_archetype.items()):
        unique = {row["pair_id"]: row for row in candidates}
        if len(unique) < QUOTA:
            raise SystemExit(
                f"{archetype}: {len(unique)} approved unique records; need {QUOTA}"
            )
        selected.extend(
            sorted(unique.values(), key=lambda row: rank(seed, row["pair_id"]))[:QUOTA]
        )

    if len(selected) != QUOTA * len(CANONICAL_ARCHETYPES):
        raise SystemExit(f"freeze must contain exactly 120 rows; found {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for input_path in args.inputs:
        rows.extend(
            json.loads(line)
            for line in Path(input_path).read_text(encoding="utf-8").splitlines()
            if line
        )

    selected = select_quota(rows, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n",
        encoding="utf-8",
    )
    print(
        "selected:",
        dict(Counter(row["archetype"] for row in selected)),
        "total:",
        len(selected),
    )


if __name__ == "__main__":
    main()
