"""Import JailbreakBench as a quarantined reconstruction task manifest.

JBB Target is an affirmative target prefix, not a complete harmful response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def record(pair_id: str, polarity: str, prompt: str, source: str, category: str) -> dict:
    return {
        "pair_id": pair_id, "behavior": "harmful_compliance", "polarity": polarity,
        "language": "en", "prompt": prompt,
        "expected_behavior": "requires reconstruction and review",
        "source": source, "reviewer_status": "draft", "split": "unassigned",
        "category": category,
        "quarantine_reason": "JBB Target is not a full contrastive completion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/working/harmful_compliance/rebuild_tasks.jsonl")
    args = parser.parse_args()
    from datasets import load_dataset
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for item in dataset:
            task = {
                "task_id": f"jbb-{int(item['Index']):03d}",
                "behavior": "harmful_compliance",
                "goal": item["Goal"],
                "target_prefix": item["Target"],
                "category": item["Category"],
                "source": f"JailbreakBench/JBB-Behaviors (behavior {item['Index']}; source={item['Source']})",
                "status": "quarantined_rebuild_required",
                "requirements": [
                    "write a unique natural safe completion",
                    "obtain a full unsafe completion under restricted review",
                    "remove refusal/acquiescence prefix shortcuts",
                    "human review before approval",
                ],
            }
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} quarantined rebuild tasks to {output}")


if __name__ == "__main__":
    main()