"""Freeze a stratified English subset for bilingual English-to-Swahili review."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from safety_governor.data import load_jsonl, validate_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deceptive", default="datasets/curation/deceptive_reasoning_approved.jsonl")
    parser.add_argument("--instruction", default="datasets/curation/instruction_noncompliance_approved.jsonl")
    parser.add_argument("--harmful", default="data/raw/jailbreakbench_harmful_draft.jsonl")
    parser.add_argument("--output", default="data/raw/sw_translation_manifest_100.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    for path in (args.deceptive, args.instruction, args.harmful):
        records.extend(load_jsonl(path))
    errors = validate_records(records)
    if errors:
        raise SystemExit("Approved English corpus failed validation:\n- " + "\n- ".join(errors))

    pairs: dict[str, list] = defaultdict(list)
    for record in records:
        pairs[record.pair_id].append(record)
    by_behavior: dict[str, list[str]] = defaultdict(list)
    for pair_id, pair_records in pairs.items():
        by_behavior[pair_records[0].behavior.value].append(pair_id)
    quotas = {"deceptive_reasoning": 34, "instruction_noncompliance": 33, "harmful_compliance": 33}
    rng = random.Random(args.seed)
    selected: set[str] = set()
    for behavior, quota in quotas.items():
        candidate_ids = sorted(by_behavior[behavior])
        rng.shuffle(candidate_ids)
        if len(candidate_ids) < quota:
            raise SystemExit(f"Not enough {behavior} pairs for quota {quota}")
        selected.update(candidate_ids[:quota])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for pair_id in sorted(selected):
            for record in sorted(pairs[pair_id], key=lambda item: item.polarity.value):
                entry = {
                    "pair_id": record.pair_id,
                    "behavior": record.behavior.value,
                    "polarity": record.polarity.value,
                    "english_prompt": record.prompt,
                    "swahili_prompt": "",
                    "source": record.source,
                    "translation_status": "pending_bilingual_review",
                    "translator": "",
                    "reviewer": "",
                    "translation_notes": "",
                    "selection_seed": args.seed,
                }
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} pairs ({len(selected) * 2} records) to {output}")


if __name__ == "__main__":
    main()
