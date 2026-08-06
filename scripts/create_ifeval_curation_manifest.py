"""Select IFEval prompts for human-authored naturalistic evasion annotations."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

ARCHETYPES = ("constraint_omission", "hedging_or_excessive_caveating", "topic_shift", "false_completion")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", default="C:/tmp/safety-governor-sources/ifeval_input_data.jsonl")
    parser.add_argument("--output", default="data/raw/research_corpus/ifeval_naturalistic_annotation_manifest.jsonl"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x]
    random.Random(args.seed).shuffle(rows); selected = rows[:120]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as h:
        for i, item in enumerate(selected):
            archetype = ARCHETYPES[i % len(ARCHETYPES)]
            h.write(json.dumps({"pair_id": f"ifeval-{item['key']}", "behavior": "instruction_noncompliance", "archetype": archetype, "english_instruction": item["prompt"], "instruction_ids": item["instruction_id_list"], "safe_completion": "", "naturalistic_evasion": "", "source": "google-research/instruction_following_eval/data/input_data.jsonl", "annotation_status": "pending_human_annotation", "annotator": "", "reviewer": "", "notes": "", "selection_seed": args.seed}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} annotation tasks to {output}")

if __name__ == "__main__": main()
