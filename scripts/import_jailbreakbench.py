"""Create a restricted, review-required harmful-compliance draft from JailbreakBench."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

REFUSAL = "I’m sorry, but I can’t help with that request."


def record(pair_id: str, polarity: str, prompt: str, source: str, category: str) -> dict:
    return {
        "pair_id": pair_id, "behavior": "harmful_compliance", "polarity": polarity,
        "language": "en", "prompt": prompt,
        "expected_behavior": "Refusal of harmful request" if polarity == "safe" else "Harmful compliance target",
        "source": source, "reviewer_status": "draft", "split": "train",
        "category": category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/jailbreakbench_harmful_draft.jsonl")
    args = parser.parse_args()
    from datasets import load_dataset
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for item in dataset:
            pair_id = f"jbb-{int(item['Index']):03d}"
            source = f"JailbreakBench/JBB-Behaviors (behavior {item['Index']}; source={item['Source']})"
            goal = item["Goal"]
            handle.write(json.dumps(record(pair_id, "safe", f"User: {goal}\nAssistant: {REFUSAL}", source, item["Category"]), ensure_ascii=False) + "\n")
            handle.write(json.dumps(record(pair_id, "unsafe", f"User: {goal}\nAssistant: {item['Target']}", source, item["Category"]), ensure_ascii=False) + "\n")
    print(f"Imported {len(dataset)} restricted draft harmful-compliance pairs to {output}")


if __name__ == "__main__": main()
