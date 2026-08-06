"""Deterministically select 30 approved records per instruction-evasion archetype."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

QUOTA = 30

def rank(seed: int, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("inputs", nargs="+"); parser.add_argument("--output", required=True); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); rows = []
    for input_path in args.inputs:
        rows.extend(json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line)
    eligible = [r for r in rows if r.get("annotation_status") == "approved" and r.get("safe_completion", "").strip() and r.get("naturalistic_evasion", "").strip()]
    by_archetype = {}
    for row in eligible: by_archetype.setdefault(row["archetype"], []).append(row)
    selected = []
    for archetype, candidates in sorted(by_archetype.items()):
        unique = {row["pair_id"]: row for row in candidates}
        if len(unique) < QUOTA: raise SystemExit(f"{archetype}: {len(unique)} approved unique records; need {QUOTA}")
        selected.extend(sorted(unique.values(), key=lambda r: rank(args.seed, r["pair_id"]))[:QUOTA])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n", encoding="utf-8")
    print("selected:", dict(Counter(r["archetype"] for r in selected)), "total:", len(selected))

if __name__ == "__main__": main()
