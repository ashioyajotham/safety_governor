"""Fail-closed audit for lexical, duplicate, encoding, and template confounds."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PHRASES = {
    "hedging_or_excessive_caveating": ("please note", "note that", "i must", "caveat", "as an ai language model"),
    "false_completion": ("requested", "complete", "compliant", "exactly", "fully"),
}
MOJIBAKE = ("Ã", "â€™", "â€œ", "â€", "Â", "à¤")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def prefix(text: str, words: int = 4) -> str:
    return " ".join(normalized(text).split()[:words])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--max-concentration", type=float, default=0.35)
    parser.add_argument("--max-template-concentration", type=float, default=0.20)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.datasets for line in path.read_text(encoding="utf-8").splitlines() if line]
    pair_ids = [row.get("pair_id") for row in rows]
    failures = []
    if len(pair_ids) != len(set(pair_ids)):
        failures.append("duplicate pair IDs across inputs")
    for field in ("safe_completion", "naturalistic_evasion"):
        values = [normalized(row.get(field, "")) for row in rows if row.get(field)]
        repeated = [(value, count) for value, count in Counter(values).items() if count > 1]
        if repeated:
            failures.append(f"{field}: {len(repeated)} exact repeated templates")
        bad = [row["pair_id"] for row in rows if any(cue in row.get(field, "") for cue in MOJIBAKE)]
        if bad:
            failures.append(f"{field}: mojibake in {len(bad)} rows")
        prefixes = Counter(prefix(row.get(field, "")) for row in rows if row.get(field))
        if prefixes and prefixes.most_common(1)[0][1] / len(values) > args.max_template_concentration:
            failures.append(f"{field}: four-word prefix concentration exceeds {args.max_template_concentration:.2f}")
    for archetype, phrases in PHRASES.items():
        subset = [row for row in rows if row.get("archetype") == archetype]
        if not subset:
            failures.append(f"missing archetype: {archetype}")
            continue
        counts = Counter({phrase: sum(phrase in row.get("naturalistic_evasion", "").lower() for row in subset) for phrase in phrases})
        any_cue = sum(any(phrase in row.get("naturalistic_evasion", "").lower() for phrase in phrases) for row in subset)
        print(archetype, "rows=", len(subset), "phrases=", dict(counts), "any_cue=", any_cue)
        for phrase, count in counts.items():
            if count / len(subset) > args.max_concentration:
                failures.append(f"{archetype}:{phrase}={count / len(subset):.3f}")
        if any_cue / len(subset) > args.max_concentration:
            failures.append(f"{archetype}:combined_cues={any_cue / len(subset):.3f}")
    if failures:
        raise SystemExit("Annotation-artifact audit failed:\n- " + "\n- ".join(failures))
    print("Annotation-artifact audit passed")


if __name__ == "__main__":
    main()