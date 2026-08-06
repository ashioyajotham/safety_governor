"""Fail closed on unreviewed or archetype-imbalanced research corpus drafts."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("--minimum-per-archetype", type=int, default=30); args = parser.parse_args()
    rows = [json.loads(x) for x in Path(args.dataset).read_text(encoding="utf-8").splitlines() if x]
    counts = Counter(row.get("archetype") for row in rows)
    placeholder = re.compile(r"^\s*\[\s*(?:compliant|safe|evasive|naturalistic|insert|todo|tbd)\b", re.I)
    bad_content = [row.get("pair_id", "unknown") for row in rows if not row.get("safe_completion", "").strip() or not row.get("naturalistic_evasion", "").strip() or not row.get("notes", "").strip() or placeholder.search(row.get("safe_completion", "")) or placeholder.search(row.get("naturalistic_evasion", ""))]
    statuses = Counter(row.get("annotation_status", row.get("reviewer_status")) for row in rows)
    print("archetypes:", dict(sorted(counts.items()))); print("statuses:", dict(sorted(statuses.items())))
    problems = [f"{a}: {n} < {args.minimum_per_archetype}" for a, n in counts.items() if n < args.minimum_per_archetype]
    if any(status != "approved" for status in statuses): problems.append("contains non-approved records")
    if bad_content: problems.append(f"contains {len(bad_content)} blank or template-like completion/rationale fields")
    if problems: raise SystemExit("Research-corpus gate failed: " + "; ".join(problems))
    print("Research-corpus gate passed")

if __name__ == "__main__": main()
