"""Assign deterministic archetype-stratified splits at source-group level.

The split unit is ``source_group_id``, not ``pair_id``. This prevents related
variants of the same underlying question/argument from leaking across
train/validation/test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")


def split_for(seed: int, source_group_id: str) -> str:
    """Stable hash fallback retained for small utilities and compatibility."""
    value = int(hashlib.sha256(f"{seed}:{source_group_id}".encode()).hexdigest(), 16) / (2**256)
    if value < .70:
        return "train"
    if value < .85:
        return "validation"
    return "test"


def _counts(number_of_groups: int) -> tuple[int, int, int]:
    """Return approximate 70/15/15 counts while keeping small sets usable."""

    if number_of_groups < 3:
        return number_of_groups, 0, 0
    validation = max(1, round(number_of_groups * .15))
    test = max(1, round(number_of_groups * .15))
    train = number_of_groups - validation - test
    return train, validation, test


def assign(rows: list[dict], seed: int) -> dict[str, str]:
    """Map each pair ID to a split, preserving source-group and archetype structure."""

    pair_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pair_rows[row["pair_id"]].append(row)
    pair_groups, group_archetypes = {}, {}
    for pair_id, group in pair_rows.items():
        group_ids = {row.get("source_group_id") for row in group}
        archetypes = {row.get("archetype", "unspecified") for row in group}
        if None in group_ids or len(group_ids) != 1:
            raise ValueError(f"{pair_id}: pair requires one explicit source_group_id")
        if len(archetypes) != 1:
            raise ValueError(f"{pair_id}: pair requires one archetype")
        group_id = next(iter(group_ids))
        archetype = next(iter(archetypes))
        previous = group_archetypes.setdefault(group_id, archetype)
        if previous != archetype:
            raise ValueError(f"{group_id}: source group spans archetypes")
        pair_groups[pair_id] = group_id

    by_archetype: dict[str, list[str]] = defaultdict(list)
    for group_id, archetype in group_archetypes.items():
        by_archetype[archetype].append(group_id)
    group_splits = {}
    for archetype, group_ids in by_archetype.items():
        # Stratify within each archetype so every subtype appears in each split
        # whenever there are enough source groups.
        ordered = sorted(
            group_ids,
            key=lambda group_id: hashlib.sha256(f"{seed}:{archetype}:{group_id}".encode()).hexdigest(),
        )
        train_n, validation_n, _ = _counts(len(ordered))
        for index, group_id in enumerate(ordered):
            if index < train_n:
                group_splits[group_id] = "train"
            elif index < train_n + validation_n:
                group_splits[group_id] = "validation"
            else:
                group_splits[group_id] = "test"
    return {pair_id: group_splits[group_id] for pair_id, group_id in pair_groups.items()}


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
    try:
        assignments = assign(rows, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for row in rows:
        row["split"] = assignments[row["pair_id"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print("source-group pair splits:", dict(Counter(assignments.values())))


if __name__ == "__main__":
    main()