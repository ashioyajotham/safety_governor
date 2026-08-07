"""Import reviewed-source CoT pairs as provenance-linked draft records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def row(pair_id: str, polarity: str, prompt: str, source: str) -> dict:
    return {
        "pair_id": pair_id, "behavior": "deceptive_reasoning", "polarity": polarity,
        "language": "en", "prompt": prompt,
        "expected_behavior": "Faithful arithmetic reasoning" if polarity == "safe" else "Unfaithful arithmetic reasoning",
        "source": source, "reviewer_status": "draft", "split": "unassigned",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Prior-work dataset.json with faithful_prompt/unfaithful_prompt")
    parser.add_argument("output"); parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--source-id",
        default="cot-faithfulness-mech-interp/dataset.json",
        help="Stable upstream identifier; never use a machine-local absolute path.",
    )
    args = parser.parse_args()
    source_path = Path(args.input)
    items = json.loads(source_path.read_text(encoding="utf-8"))
    valid = []
    seen_prompts: set[str] = set()
    for item in items:
        if not (item.get("is_valid") and item.get("faithful_prompt") and item.get("unfaithful_prompt")):
            continue
        prompt_pair = (item["faithful_prompt"], item["unfaithful_prompt"])
        if len(set(prompt_pair)) != 2 or any(prompt in seen_prompts for prompt in prompt_pair):
            continue
        seen_prompts.update(prompt_pair)
        valid.append(item)
    if len(valid) < args.limit:
        raise SystemExit(f"Only {len(valid)} valid pairs; requested {args.limit}")
    provenance = f"{args.source_id}; CC-BY-4.0"
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for index, item in enumerate(valid[:args.limit], start=1):
            pair_id = f"cot-arithmetic-{index:03d}"
            handle.write(json.dumps(row(pair_id, "safe", item["faithful_prompt"], provenance), ensure_ascii=False) + "\n")
            handle.write(json.dumps(row(pair_id, "unsafe", item["unfaithful_prompt"], provenance), ensure_ascii=False) + "\n")
    print(f"Imported {args.limit} draft deceptive-reasoning pairs to {output}")


if __name__ == "__main__": main()
