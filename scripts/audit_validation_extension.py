"""Audit a source-backed validation-v2 extension before any model evaluation.

The extension is deliberately separate from the frozen Stage-1 training corpus.
Rows must be approved contrastive pairs assigned to either a calibration pool
or an untouched confirmatory pool. Source groups may not cross those roles or
overlap the original train/validation/test corpus.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from safety_governor.data import validate_records
from safety_governor.data import load_jsonl as load_typed_jsonl

ARCHETYPES = {
    "arithmetic_reasoning_error",
    "factual_confabulation",
    "false_premise_agreement",
    "motivated_reasoning",
}
ROLES = {"calibration", "confirmatory"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_extension(
    extension_path: str | Path,
    base_path: str | Path,
    *,
    minimum_pairs_per_archetype_role: int = 8,
) -> dict:
    """Return an audit report or raise when extension isolation is violated."""

    extension_path, base_path = Path(extension_path), Path(base_path)
    rows = _read_jsonl(extension_path)
    errors = validate_records(load_typed_jsonl(extension_path))
    base_rows = _read_jsonl(base_path)
    base_pairs = {row["pair_id"] for row in base_rows}
    base_groups = {row["source_group_id"] for row in base_rows}
    pair_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pair_rows[row.get("pair_id", "")].append(row)
        missing = [
            field for field in (
                "validation_role", "source_record_id", "source_revision", "source_license"
            )
            if not str(row.get(field, "")).strip()
        ]
        if missing:
            errors.append(f"{row.get('pair_id')}: missing extension provenance fields {missing}")
        if row.get("validation_role") not in ROLES:
            errors.append(f"{row.get('pair_id')}: invalid validation_role")
        if row.get("archetype") not in ARCHETYPES:
            errors.append(f"{row.get('pair_id')}: invalid archetype")
        if row.get("split") != "validation":
            errors.append(f"{row.get('pair_id')}: extension rows must use split=validation")
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    archetypes_by_group: dict[str, set[str]] = defaultdict(set)
    pairs_by_source_record: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str]] = Counter()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair_id, group in pair_rows.items():
        if pair_id in base_pairs:
            errors.append(f"{pair_id}: pair overlaps the frozen corpus")
        role_values = {row.get("validation_role") for row in group}
        archetypes = {row.get("archetype") for row in group}
        source_groups = {row.get("source_group_id") for row in group}
        if len(role_values) != 1 or len(archetypes) != 1 or len(source_groups) != 1:
            errors.append(f"{pair_id}: pair metadata is not aligned")
            continue
        role, archetype, source_group = next(iter(role_values)), next(iter(archetypes)), next(iter(source_groups))
        if source_group in base_groups:
            errors.append(f"{pair_id}: source group overlaps the frozen corpus")
        roles_by_group[source_group].add(role)
        archetypes_by_group[source_group].add(archetype)
        source_identity = (str(group[0].get("source_revision")), str(group[0].get("source_record_id")))
        pairs_by_source_record[source_identity].add(pair_id)
        if role in ROLES and archetype in ARCHETYPES:
            counts[(role, archetype)] += 1
            groups[(role, archetype)].add(source_group)
    for source_group, roles in roles_by_group.items():
        if len(roles) != 1:
            errors.append(f"{source_group}: source group crosses validation roles")
        if len(archetypes_by_group[source_group]) != 1:
            errors.append(f"{source_group}: source group crosses archetypes")
    for source_identity, pair_ids in pairs_by_source_record.items():
        if len(pair_ids) != 1:
            errors.append(f"{source_identity}: source record reused by multiple pairs")
    for role in sorted(ROLES):
        for archetype in sorted(ARCHETYPES):
            count = counts[(role, archetype)]
            group_count = len(groups[(role, archetype)])
            if count < minimum_pairs_per_archetype_role:
                errors.append(
                    f"{role}/{archetype}: {count} pairs; "
                    f"requires at least {minimum_pairs_per_archetype_role}"
                )
            if group_count < minimum_pairs_per_archetype_role:
                errors.append(
                    f"{role}/{archetype}: {group_count} source groups; "
                    f"requires at least {minimum_pairs_per_archetype_role}"
                )
    if errors:
        raise ValueError("Validation-extension audit failed:\n- " + "\n- ".join(sorted(set(errors))))
    return {
        "pairs": len(pair_rows),
        "rows": len(rows),
        "minimum_pairs_per_archetype_role": minimum_pairs_per_archetype_role,
        "counts": {
            role: {archetype: counts[(role, archetype)] for archetype in sorted(ARCHETYPES)}
            for role in sorted(ROLES)
        },
        "source_groups": {
            role: {archetype: len(groups[(role, archetype)]) for archetype in sorted(ARCHETYPES)}
            for role in sorted(ROLES)
        },
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extension")
    parser.add_argument("--base", default="datasets/frozen/english_contrastive.jsonl")
    parser.add_argument("--minimum-pairs-per-archetype-role", type=int, default=8)
    args = parser.parse_args()
    report = audit_extension(
        args.extension,
        args.base,
        minimum_pairs_per_archetype_role=args.minimum_pairs_per_archetype_role,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
