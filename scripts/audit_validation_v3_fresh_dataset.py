"""Read-only local checks for a proposed fresh v3 behavioural corpus.

This does not open or certify the sealed confirmatory split. A separate
source-isolation attestation is required before creating an execution contract.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from safety_governor.data import load_jsonl, validate_records
from safety_governor.validation_v3 import file_sha256

ARCHETYPES = (
    "arithmetic_reasoning_error",
    "factual_confabulation",
    "false_premise_agreement",
    "motivated_reasoning",
)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _instructions(rows: list[dict]) -> set[str]:
    return {" ".join(str(row.get("instruction") or "").split()).casefold()
            for row in rows}


def _source_records(rows: list[dict]) -> set[tuple[str, str, str]]:
    return {(str(row.get("source")), str(row.get("source_record_id")),
             str(row.get("source_revision"))) for row in rows}


def audit(candidate: Path, train: Path, spent: Path, expected_sha256: str) -> dict:
    """Check schema, quotas and known-split overlaps without exposing text."""

    candidate, train, spent = (path.resolve() for path in (candidate, train, spent))
    if len({candidate, train, spent}) != 3:
        raise ValueError("candidate, train and spent paths must be distinct")
    candidate_rows, train_rows, spent_rows = (_rows(path) for path in
                                               (candidate, train, spent))
    digest = file_sha256(candidate)
    errors = validate_records(load_jsonl(candidate))
    if digest != expected_sha256:
        errors.append("candidate SHA-256 does not match the declared hash")
    if digest in {file_sha256(train), file_sha256(spent)}:
        errors.append("candidate file is identical to a used split")

    safe = [row for row in candidate_rows if row.get("polarity") == "safe"]
    counts = Counter(row.get("archetype") for row in safe)
    if len(safe) != 48 or len(candidate_rows) != 96:
        errors.append("fresh development requires exactly 48 complete pairs")
    if any(counts.get(name) != 12 for name in ARCHETYPES) or set(counts) != set(ARCHETYPES):
        errors.append("fresh development requires 12 pairs in each of four archetypes")
    if any(row.get("split") != "validation" or row.get("language") != "en"
           for row in candidate_rows):
        errors.append("all candidate records must be English validation rows")
    if any(row.get("validation_role") != "fresh_development" for row in candidate_rows):
        errors.append("all candidate records require fresh_development role")
    if any(not row.get(key) for row in candidate_rows
           for key in ("source_group_id", "source_record_id", "source_revision")):
        errors.append("source group, record ID and revision are required")
    by_pair: dict[str, list[dict]] = {}
    for row in candidate_rows:
        by_pair.setdefault(str(row.get("pair_id")), []).append(row)
    for pair in by_pair.values():
        if len(pair) == 2 and any(pair[0].get(key) != pair[1].get(key)
                                  for key in ("instruction", "archetype", "source",
                                              "source_record_id", "source_revision")):
            errors.append("safe and unsafe rows in a pair have inconsistent provenance")
            break
    for label, prior in (("train", train_rows), ("spent_calibration", spent_rows)):
        for key in ("pair_id", "source_group_id"):
            if {row.get(key) for row in candidate_rows} & {row.get(key) for row in prior}:
                errors.append(f"{key} overlaps {label}")
        if _instructions(candidate_rows) & _instructions(prior):
            errors.append(f"normalized instruction overlaps {label}")
        if _source_records(candidate_rows) & _source_records(prior):
            errors.append(f"source record overlaps {label}")

    return {
        "candidate_sha256": digest,
        "candidate_pairs": len(safe),
        "pairs_by_archetype": {name: counts.get(name, 0) for name in ARCHETYPES},
        "local_checks_pass": not errors,
        "errors": errors,
        "sealed_confirmatory_contents_accessed": False,
        "holdout_source_isolation": "requires_independent_attestation",
        "ready_for_generation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--train", type=Path,
                        default=Path("datasets/frozen/english_contrastive.jsonl"))
    parser.add_argument("--spent", type=Path,
                        default=Path("datasets/validation_v2/calibration.jsonl"))
    args = parser.parse_args()
    result = audit(args.candidate, args.train, args.spent, args.expected_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
