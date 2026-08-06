"""Verify the lightweight per-pair approval metadata for restricted JBB records."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); args = parser.parse_args()
    pairs = defaultdict(list)
    for line in Path(args.dataset).read_text(encoding="utf-8").splitlines():
        if line: pairs[json.loads(line)["pair_id"]].append(json.loads(line))
    failures = []
    for pair_id, rows in pairs.items():
        if len(rows) != 2: failures.append(f"{pair_id}: expected safe/unsafe pair"); continue
        values = {(r.get("reviewer_status"), r.get("reviewer_id", ""), r.get("reviewed_at", "")) for r in rows}
        if len(values) != 1: failures.append(f"{pair_id}: inconsistent review metadata"); continue
        status, reviewer, reviewed_at = values.pop()
        if status != "approved" or not reviewer or not reviewed_at: failures.append(f"{pair_id}: missing completed review")
    if failures: raise SystemExit("Harmful-compliance review gate failed:\n- " + "\n- ".join(failures))
    print(f"Harmful-compliance review gate passed for {len(pairs)} pairs")

if __name__ == "__main__": main()
