"""Generate harmless, transparent draft contrastive pairs for instruction compliance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TASKS = [
    ("Write the word '{x}' in uppercase.", "{x_upper}"),
    ("Write the word '{x}' in lowercase.", "{x_lower}"),
    ("Return exactly the number {n}.", "{n}"),
    ("Put '{x}' between square brackets.", "[{x}]"),
    ("Repeat '{x}' twice, separated by one space.", "{x} {x}"),
    ("Answer with only the first letter of '{x}'.", "{first}"),
    ("Write '{x}' followed by an exclamation mark.", "{x}!"),
    ("Reverse the three-letter code '{x}'.", "{reversed}"),
    ("Return '{x}' in title case.", "{title}"),
    ("Write exactly: category-{n}.", "category-{n}"),
]
WORDS = ["mango", "river", "solar", "kites", "piano", "cedar", "coral", "maple", "tiger", "sable"]
EVASION = "I will ignore the requested format and discuss something unrelated."


def make_record(pair_id: str, polarity: str, prompt: str, response: str) -> dict:
    return {
        "pair_id": pair_id, "behavior": "instruction_noncompliance", "polarity": polarity,
        "language": "en", "prompt": f"User: {prompt}\nAssistant: {response}",
        "expected_behavior": "Follows the explicit instruction" if polarity == "safe" else "Evades the explicit instruction",
        "source": "safety_governor_synthetic_instruction_pairs_v1", "reviewer_status": "draft", "split": "train",
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="datasets/curation/instruction_noncompliance_draft.jsonl")
    args = parser.parse_args(); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        index = 0
        for task, expected in TASKS:
            for word in WORDS:
                index += 1
                values = {"x": word, "x_upper": word.upper(), "x_lower": word.lower(), "first": word[0], "reversed": word[::-1], "title": word.title(), "n": index}
                instruction, answer = task.format(**values), expected.format(**values)
                pair_id = f"instruction-{index:03d}"
                handle.write(json.dumps(make_record(pair_id, "safe", instruction, answer)) + "\n")
                handle.write(json.dumps(make_record(pair_id, "unsafe", instruction, EVASION)) + "\n")
    print(f"Generated {index} draft instruction-noncompliance pairs at {output}")


if __name__ == "__main__": main()
