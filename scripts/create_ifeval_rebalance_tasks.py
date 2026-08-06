"""Select disjoint IFEval prompts to fill annotation-archetype shortages."""
from __future__ import annotations
import argparse, json
from pathlib import Path

RULES = {
    "hedging_or_excessive_caveating": ({"keywords:existence", "keywords:frequency", "keywords:letter_frequency", "keywords:forbidden_words"}, 13),
    "topic_shift": ({"language:response_language", "change_case:english_lowercase", "change_case:english_capital", "change_case:capital_word_frequency"}, 14),
    "false_completion": ({"length_constraints:number_words", "length_constraints:number_sentences", "length_constraints:number_paragraphs", "length_constraints:nth_paragraph_first_word"}, 21),
}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--source", default="data/raw/sources/ifeval_input_data.jsonl"); parser.add_argument("--existing", default="data/raw/research_corpus/ifeval_naturalistic_annotation_draft.jsonl"); parser.add_argument("--output", default="data/raw/research_corpus/ifeval_rebalance_annotation_tasks.jsonl")
    args = parser.parse_args()
    existing = {json.loads(x)["pair_id"] for x in Path(args.existing).read_text(encoding="utf-8").splitlines() if x}
    rows = [json.loads(x) for x in Path(args.source).read_text(encoding="utf-8").splitlines() if x]
    rows = list({row["key"]: row for row in rows}.values())
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    used = set(existing)
    for archetype, (ids, quota) in RULES.items():
        candidates = [r for r in rows if f"ifeval-{r['key']}" not in used and ids.intersection(r["instruction_id_list"])]
        if len(candidates) < quota: raise SystemExit(f"Need {quota} {archetype} prompts; found {len(candidates)}")
        for item in candidates[:quota]:
            selected.append({"pair_id": f"ifeval-{item['key']}", "behavior": "instruction_noncompliance", "archetype": archetype, "english_instruction": item["prompt"], "instruction_ids": item["instruction_id_list"], "safe_completion": "", "naturalistic_evasion": "", "source": "google-research/instruction_following_eval/data/input_data.jsonl", "annotation_status": "pending_human_annotation", "annotator": "", "reviewer": "", "notes": "Targeted rebalance task; write naturalistic completion pair.", "selection_method": "instruction_id_semantic_rule"})
    with output.open("x", encoding="utf-8") as h:
        for row in selected: h.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} targeted annotation tasks to {output}")

if __name__ == "__main__": main()
