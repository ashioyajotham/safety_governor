"""Materialize fresh v3 development corpus from reviewed decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_governor.data import load_jsonl, validate_records
from safety_governor.validation_v2 import validate_review_decisions
from safety_governor.validation_v3 import file_sha256

ARCHETYPES = (
    "arithmetic_reasoning_error",
    "factual_confabulation",
    "false_premise_agreement",
    "motivated_reasoning",
)


def materialize(
    candidates_path: Path,
    decisions_path: Path,
    output_path: Path,
    *,
    target_per_archetype: int = 12,
) -> dict:
    candidates = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = [
        json.loads(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    validate_review_decisions(candidates, decisions)

    decisions_by_id = {d["candidate_id"]: d for d in decisions}
    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    approved_pairs: dict[str, list[dict]] = {name: [] for name in ARCHETYPES}
    for cid, d in decisions_by_id.items():
        if d["decision"] == "approved":
            cand = candidates_by_id[cid]
            arch = cand["archetype"]
            approved_pairs[arch].append((cand, d))

    for arch in ARCHETYPES:
        if len(approved_pairs[arch]) != target_per_archetype:
            raise ValueError(
                f"{arch}: expected {target_per_archetype} approved pairs, got {len(approved_pairs[arch])}"
            )
        groups = {c["source_group_id"] for c, _ in approved_pairs[arch]}
        if len(groups) != target_per_archetype:
            raise ValueError(
                f"{arch}: approved pairs must come from {target_per_archetype} distinct source groups"
            )

    rows: list[dict] = []
    for arch in ARCHETYPES:
        for cand, dec in approved_pairs[arch]:
            pair_id = cand["pair_id"]
            source = (
                f"{cand['source_dataset']}@{cand['source_revision']}:"
                f"{cand['source_record_id']}; {cand['source_license']}"
            )

            safe_row = {
                "archetype": arch,
                "behavior": "deceptive_reasoning",
                "completion": dec["safe_completion"].strip(),
                "expected_behavior": "Independent, evidence-grounded reasoning",
                "instruction": dec["instruction"].strip(),
                "language": "en",
                "pair_id": pair_id,
                "polarity": "safe",
                "prompt": "",
                "reviewer_status": "approved",
                "source": source,
                "source_group_id": cand["source_group_id"],
                "source_license": cand["source_license"],
                "source_record_id": cand["source_record_id"],
                "source_revision": cand["source_revision"],
                "split": "validation",
                "validation_role": "fresh_development",
            }

            unsafe_row = {
                "archetype": arch,
                "behavior": "deceptive_reasoning",
                "completion": dec["unsafe_completion"].strip(),
                "expected_behavior": "Deceptive, false, or user-pleasing reasoning",
                "instruction": dec["instruction"].strip(),
                "language": "en",
                "pair_id": pair_id,
                "polarity": "unsafe",
                "prompt": "",
                "reviewer_status": "approved",
                "source": source,
                "source_group_id": cand["source_group_id"],
                "source_license": cand["source_license"],
                "source_record_id": cand["source_record_id"],
                "source_revision": cand["source_revision"],
                "split": "validation",
                "validation_role": "fresh_development",
            }

            rows.extend([safe_row, unsafe_row])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )

    records = load_jsonl(output_path)
    errors = validate_records(records)
    if errors:
        raise ValueError(f"Materialized records failed validation: {errors}")

    digest = file_sha256(output_path)
    return {
        "status": "materialized",
        "path": str(output_path),
        "sha256": digest,
        "pairs": len(rows) // 2,
        "rows": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/working/validation_v3_fresh/review_candidates.jsonl"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/working/validation_v3_fresh/review_decisions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/validation_v3/fresh_development.jsonl"),
    )
    args = parser.parse_args()

    result = materialize(args.candidates, args.decisions, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
