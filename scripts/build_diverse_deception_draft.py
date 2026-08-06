"""Build a source-linked, four-archetype deceptive-reasoning draft for review."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_pair(handle, pair_id, archetype, safe, unsafe, source):
    for polarity, prompt, expected in (("safe", safe, "Independent, evidence-grounded reasoning"), ("unsafe", unsafe, "Deceptive, false, or user-pleasing reasoning")):
        handle.write(json.dumps({"pair_id": pair_id, "behavior": "deceptive_reasoning", "archetype": archetype, "polarity": polarity, "language": "en", "prompt": prompt, "expected_behavior": expected, "source": source, "reviewer_status": "draft", "split": "train"}, ensure_ascii=False) + "\n")


def dialogue(messages):
    return "\n".join(("User" if m["type"] == "human" else "Assistant") + ": " + m["content"] for m in messages)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="data/raw/research_corpus/deceptive_diverse_draft.jsonl")
    parser.add_argument("--sources", default="C:/tmp/safety-governor-sources"); args = parser.parse_args()
    root = Path(args.sources); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    arithmetic = [json.loads(x) for x in Path("datasets/pilot/deceptive_reasoning_arithmetic_pilot.jsonl").read_text(encoding="utf-8").splitlines() if x]
    pairs = {r["pair_id"]: [] for r in arithmetic}
    for r in arithmetic: pairs[r["pair_id"]].append(r)
    truthful = list(csv.DictReader((root / "truthfulqa/TruthfulQA.csv").open(encoding="utf-8")))
    answer = [json.loads(x) for x in (root / "sycophancy-eval/datasets/answer.jsonl").read_text(encoding="utf-8").splitlines() if x]
    feedback = [json.loads(x) for x in (root / "sycophancy-eval/datasets/feedback.jsonl").read_text(encoding="utf-8").splitlines() if x]
    with output.open("x", encoding="utf-8") as h:
        for i, records in enumerate(list(pairs.values())[:30], 1):
            safe = next(r["prompt"] for r in records if r["polarity"] == "safe"); unsafe = next(r["prompt"] for r in records if r["polarity"] == "unsafe")
            write_pair(h, f"diverse-arithmetic-{i:03d}", "arithmetic_reasoning_error", safe, unsafe, records[0]["source"])
        for i, row in enumerate(truthful[:30], 1):
            q = row["Question"]; write_pair(h, f"diverse-factual-{i:03d}", "factual_confabulation", f"User: {q}\nAssistant: {row['Best Answer']}", f"User: {q}\nAssistant: {row['Best Incorrect Answer']}", f"TruthfulQA/TruthfulQA.csv; category={row['Category']}; source={row['Source']}")
        for i, item in enumerate(answer[:30], 1):
            base = item["base"]; prompt = dialogue(item["prompt"])
            write_pair(h, f"diverse-sycophancy-{i:03d}", "false_premise_agreement", f"{prompt}\nAssistant: {base['correct_answer']}", f"{prompt}\nAssistant: {base['incorrect_answer']}", "meg-tong/sycophancy-eval/datasets/answer.jsonl")
        for i, item in enumerate(feedback[:30], 1):
            prompt = dialogue(item["prompt"]); error = item["base"]["logical_error"]
            write_pair(h, f"diverse-motivated-{i:03d}", "motivated_reasoning", f"{prompt}\nAssistant: The argument should be assessed independently; a relevant concern is {error}.", f"{prompt}\nAssistant: I agree with the user's preferred framing and endorse the argument without independent assessment.", "meg-tong/sycophancy-eval/datasets/feedback.jsonl")
    print(f"Wrote 120 draft pairs to {output}")


if __name__ == "__main__": main()
