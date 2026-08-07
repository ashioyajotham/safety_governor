"""Assemble one provider-neutral IFEval candidate pool for human review."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CANONICAL_ARCHETYPES = {
    "constraint_omission",
    "hedging_or_excessive_caveating",
    "topic_shift",
    "false_completion",
}
EXPECTED_COUNTS = {
    "constraint_omission": 60,
    "hedging_or_excessive_caveating": 30,
    "topic_shift": 30,
    "false_completion": 30,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_provenance(row: dict) -> dict:
    normalized = dict(row)
    annotator = normalized.pop("annotator", "")
    if not normalized.get("reviewer"):
        normalized.pop("reviewer", None)

    generation = normalized.get("generation_metadata", {})
    model = generation.get("model", "")
    method = "model_assisted" if model or "assisted" in annotator else "manual"
    provenance = dict(normalized.get("annotation_provenance", {}))
    provenance["method"] = method
    if model:
        provenance["provider"] = "google" if model.lower().startswith("gemini") else "unspecified"
        provenance["model"] = model
    normalized["annotation_provenance"] = provenance
    return normalized


def assemble(primary: list[dict], gap: list[dict]) -> list[dict]:
    rows = [normalize_provenance(row) for row in primary + gap]
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        duplicates = sorted(pair_id for pair_id, count in Counter(pair_ids).items() if count > 1)
        raise ValueError(f"duplicate pair IDs across candidate inputs: {duplicates}")
    counts = Counter(row.get("archetype") for row in rows)
    if set(counts) != CANONICAL_ARCHETYPES or dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"unexpected candidate balance: {dict(counts)}")
    if any(row.get("annotation_status") != "pending_review" for row in rows):
        raise ValueError("candidate assembly must not advance review state")
    return sorted(rows, key=lambda row: row["pair_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path, help="Remediated primary candidate pool.")
    parser.add_argument("gap", type=Path, help="Disjoint final-gap review set.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = assemble(read_jsonl(args.primary), read_jsonl(args.gap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} pending-review candidates to {args.output}")
    print("archetypes:", dict(sorted(Counter(row["archetype"] for row in rows).items())))


if __name__ == "__main__":
    main()
