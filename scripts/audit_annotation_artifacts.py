"""Audit archetype-specific lexical cues that can confound vector extraction."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PHRASES = {
    "hedging_or_excessive_caveating": ("please note", "note that", "i must", "caveat", "as an ai language model"),
    "false_completion": ("requested", "complete", "compliant", "exactly", "fully"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--max-concentration", type=float, default=0.35)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.datasets for line in path.read_text(encoding="utf-8").splitlines() if line]
    pair_ids = [row.get("pair_id") for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise SystemExit("Annotation-artifact audit failed: duplicate pair IDs across inputs")
    failures = []
    for archetype, phrases in PHRASES.items():
        subset = [row for row in rows if row.get("archetype") == archetype]
        if not subset:
            failures.append(f"missing archetype: {archetype}")
            continue
        counts = Counter()
        for phrase in phrases:
            counts[phrase] = sum(phrase in row.get("naturalistic_evasion", "").lower() for row in subset)
        print(archetype, "rows=", len(subset), "phrases=", dict(counts))
        for phrase, count in counts.items():
            concentration = count / len(subset)
            if concentration > args.max_concentration:
                failures.append(f"{archetype}:{phrase}={concentration:.3f} > {args.max_concentration:.3f}")
        any_cue = sum(any(phrase in row.get("naturalistic_evasion", "").lower() for phrase in phrases) for row in subset)
        print(archetype, "any_cue=", any_cue)
        if any_cue / len(subset) > args.max_concentration:
            failures.append(f"{archetype}:combined_cues={any_cue / len(subset):.3f} > {args.max_concentration:.3f}")
    if failures:
        raise SystemExit("Annotation-artifact audit failed:\n- " + "\n- ".join(failures))
    print("Annotation-artifact audit passed")


if __name__ == "__main__":
    main()
