"""Custodian-only, content-silent comparison with the sealed v2 holdout.

This tool does not authorize generation or reveal holdout examples. The split
custodian must run it and retain the resulting attestation independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_instruction(value: str) -> str:
    """Casefold, remove punctuation distinctions, and collapse whitespace."""
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(" " if unicodedata.category(char).startswith("P")
                            else char for char in folded).split())


def _identities(path: Path, *, role: str, expected_pairs: int) -> dict[str, set]:
    axes: dict[str, set] = {
        "pair_id": set(), "source_group_id": set(), "source_record_id": set(),
        "normalized_instruction": set(),
    }
    pairs: dict[str, set[str]] = {}
    pair_identities: dict[str, tuple] = {}
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            required = ("pair_id", "source_group_id", "source", "source_revision",
                        "source_record_id", "instruction", "polarity")
            if any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in required):
                raise ValueError(f"{role}: missing or invalid identity field")
            if row.get("polarity") not in {"safe", "unsafe"}:
                raise ValueError(f"{role}: invalid polarity")
            if row.get("validation_role") != role:
                raise ValueError(f"{role}: unexpected validation role")
            pair = row["pair_id"]
            if row["polarity"] in pairs.setdefault(pair, set()):
                raise ValueError(f"{role}: duplicate pair polarity")
            pairs[pair].add(row["polarity"])
            identity = tuple(row[key] for key in required if key != "polarity")
            if pair in pair_identities and pair_identities[pair] != identity:
                raise ValueError(f"{role}: inconsistent pair identity")
            pair_identities[pair] = identity
            axes["pair_id"].add(pair)
            axes["source_group_id"].add(row["source_group_id"])
            axes["source_record_id"].add(row["source_record_id"])
            instruction = normalized_instruction(row["instruction"])
            if not instruction:
                raise ValueError(f"{role}: empty normalized instruction")
            axes["normalized_instruction"].add(instruction)
    if row_count != expected_pairs * 2 or len(pairs) != expected_pairs or any(
        polarities != {"safe", "unsafe"} for polarities in pairs.values()
    ):
        raise ValueError(f"{role}: wrong pair count or incomplete pairs")
    return axes


def attest(candidate: Path, holdout: Path, *, candidate_sha256: str,
           holdout_sha256: str) -> dict:
    if candidate.resolve() == holdout.resolve():
        raise ValueError("candidate and holdout must be distinct files")
    for label, value in (("candidate", candidate_sha256), ("holdout", holdout_sha256)):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{label}: expected SHA-256 must be lowercase hex")
    actual_candidate = file_sha256(candidate)
    actual_holdout = file_sha256(holdout)
    if actual_candidate != candidate_sha256 or actual_holdout != holdout_sha256:
        raise ValueError("declared corpus SHA-256 mismatch")
    fresh = _identities(candidate, role="fresh_development", expected_pairs=48)
    sealed = _identities(holdout, role="confirmatory", expected_pairs=64)
    overlaps = {axis: len(fresh[axis] & sealed[axis]) for axis in fresh}
    return {
        "schema_version": 1,
        "candidate_sha256": actual_candidate,
        "holdout_sha256": actual_holdout,
        "candidate_pairs": 48,
        "holdout_pairs": 64,
        "overlap_counts": overlaps,
        "isolated": all(count == 0 for count in overlaps.values()),
        "comparison_only": True,
        "confirmatory_generation_or_scoring_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-holdout-sha256", required=True)
    args = parser.parse_args()
    try:
        result = attest(args.candidate, args.holdout,
                        candidate_sha256=args.expected_candidate_sha256,
                        holdout_sha256=args.expected_holdout_sha256)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Isolation check blocked: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["isolated"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
