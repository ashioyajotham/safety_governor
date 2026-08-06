"""Build the smallest review/retry queue needed to fill frozen corpus archetype gaps."""
from __future__ import annotations
import argparse, json
from pathlib import Path

NEEDED_FROM_FAILURES = {"hedging_or_excessive_caveating": 1, "topic_shift": 9}
NEEDED_FROM_TASKS = {"topic_shift": 1, "false_completion": 4}

def read(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--failures", default="C:/Users/HomePC/Downloads/ifeval_gemini_failures.jsonl"); parser.add_argument("--tasks", default="data/raw/research_corpus/ifeval_rebalance_annotation_tasks.jsonl"); parser.add_argument("--output", default="data/raw/research_corpus/ifeval_final_gap_queue.jsonl")
    args = parser.parse_args(); queue = []
    failures = read(args.failures)
    for archetype, quota in NEEDED_FROM_FAILURES.items():
        candidates = sorted((r for r in failures if r.get("archetype") == archetype and r.get("safe_completion") and r.get("naturalistic_evasion")), key=lambda r: r["pair_id"])
        if len(candidates) < quota: raise SystemExit(f"insufficient failed {archetype} candidates")
        for row in candidates[:quota]:
            row["queue_action"] = "manual_rewrite_or_targeted_regeneration"
            row["annotation_status"] = "pending_review"
            queue.append(row)
    used = {r["pair_id"] for r in queue}
    for archetype, quota in NEEDED_FROM_TASKS.items():
        candidates = sorted((r for r in read(args.tasks) if r["archetype"] == archetype and r["pair_id"] not in used), key=lambda r: r["pair_id"])
        if len(candidates) < quota: raise SystemExit(f"insufficient targeted {archetype} candidates")
        for row in candidates[:quota]:
            row["queue_action"] = "human_annotation_or_targeted_generation"
            queue.append(row)
    if len({r["pair_id"] for r in queue}) != 15: raise SystemExit("gap queue must have 15 unique pairs")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in queue) + "\n", encoding="utf-8")
    print(f"Wrote {len(queue)} unique gap tasks to {output}")

if __name__ == "__main__": main()
