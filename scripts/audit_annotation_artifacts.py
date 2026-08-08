"""Fail closed on lexical, encoding, duplicate, and template confounds."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

PHRASES = {
    "hedging_or_excessive_caveating": (
        "please note", "note that", "i must", "caveat", "as an ai language model"
    ),
    "false_completion": ("requested", "complete", "compliant", "exactly", "fully"),
}
MOJIBAKE = ("Ã", "â€™", "â€œ", "â€", "Â", "à¤")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", normalized(text))


def completion_views(rows: list[dict]) -> list[dict]:
    """Normalize annotation-pair and polarity-row schemas into one audit view."""
    views = []
    for row in rows:
        common = {"pair_id": row.get("pair_id"), "archetype": row.get("archetype")}
        if "safe_completion" in row or "naturalistic_evasion" in row:
            views.extend((
                {**common, "polarity": "safe", "text": row.get("safe_completion", "")},
                {**common, "polarity": "unsafe", "text": row.get("naturalistic_evasion", "")},
            ))
        elif "completion" in row:
            views.append({**common, "polarity": row.get("polarity"), "text": row["completion"]})
    return views


def concentration_failures(
    views: list[dict], *, max_edge: float, max_ngram: float, max_near_duplicate: float,
    strict_archetypes: set[str] | None = None,
) -> list[str]:
    failures = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for view in views:
        groups.setdefault((view["archetype"], view["polarity"]), []).append(view)
    for (archetype, polarity), subset in sorted(groups.items()):
        strict = strict_archetypes is None or archetype in strict_archetypes
        texts = [normalized(view["text"]) for view in subset if view["text"].strip()]
        if not texts:
            failures.append(f"{archetype}/{polarity}: no completions")
            continue
        exact = Counter(texts)
        if strict and max(exact.values()) > 1:
            failures.append(f"{archetype}/{polarity}: exact duplicate completion")
        tokenized = [tokens(text) for text in texts]
        prefixes = Counter(" ".join(value[:4]) for value in tokenized if len(value) >= 4)
        suffixes = Counter(" ".join(value[-6:]) for value in tokenized if len(value) >= 6)
        for label, counts in (("four-word prefix", prefixes), ("six-word suffix", suffixes)):
            if counts:
                phrase, count = counts.most_common(1)[0]
                if strict and count / len(texts) > max_edge:
                    failures.append(
                        f"{archetype}/{polarity}: {label} {phrase!r}="
                        f"{count}/{len(texts)} > {max_edge:.2f}"
                    )
        document_ngrams = Counter()
        for value in tokenized:
            document_ngrams.update(set(zip(*(value[offset:] for offset in range(5)))))
        if document_ngrams:
            ngram, count = document_ngrams.most_common(1)[0]
            if strict and count / len(texts) > max_ngram:
                failures.append(
                    f"{archetype}/{polarity}: repeated five-gram {' '.join(ngram)!r}="
                    f"{count}/{len(texts)} > {max_ngram:.2f}"
                )
        similarities = sorted(
            SequenceMatcher(None, texts[left], texts[right]).ratio()
            for left in range(len(texts)) for right in range(left + 1, len(texts))
        )
        if similarities:
            p95 = similarities[min(len(similarities) - 1, int(.95 * len(similarities)))]
            print(f"{archetype}/{polarity}: rows={len(texts)} near_duplicate_p95={p95:.3f}")
            if strict and p95 > max_near_duplicate:
                failures.append(
                    f"{archetype}/{polarity}: near-duplicate p95={p95:.3f} > "
                    f"{max_near_duplicate:.3f}"
                )
    return failures


def audit(
    rows: list[dict], *, max_cue: float = .35, max_edge: float = .10,
    max_ngram: float = .20, max_near_duplicate: float = .85,
    strict_archetypes: set[str] | None = None,
) -> list[str]:
    views = completion_views(rows)
    failures = concentration_failures(
        views, max_edge=max_edge, max_ngram=max_ngram,
        max_near_duplicate=max_near_duplicate, strict_archetypes=strict_archetypes,
    )
    bad_encoding = [
        view["pair_id"] for view in views if any(cue in view["text"] for cue in MOJIBAKE)
    ]
    if bad_encoding:
        failures.append(f"mojibake in {len(set(bad_encoding))} pairs")
    for archetype, phrases in PHRASES.items():
        subset = [
            view for view in views
            if view["archetype"] == archetype and view["polarity"] == "unsafe"
        ]
        if not subset:
            continue
        counts = Counter({
            phrase: sum(phrase in view["text"].lower() for view in subset)
            for phrase in phrases
        })
        any_cue = sum(
            any(phrase in view["text"].lower() for phrase in phrases) for view in subset
        )
        print(archetype, "rows=", len(subset), "phrases=", dict(counts), "any_cue=", any_cue)
        if any_cue / len(subset) > max_cue:
            failures.append(
                f"{archetype}:combined_cues={any_cue / len(subset):.3f} > {max_cue:.3f}"
            )
        for phrase, count in counts.items():
            if count / len(subset) > max_cue:
                failures.append(
                    f"{archetype}:{phrase}={count / len(subset):.3f} > {max_cue:.3f}"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--max-cue-concentration", type=float, default=.35)
    parser.add_argument("--max-edge-concentration", type=float, default=.10)
    parser.add_argument("--max-ngram-concentration", type=float, default=.20)
    parser.add_argument("--max-near-duplicate", type=float, default=.85)
    parser.add_argument("--strict-archetype", action="append", default=None)
    args = parser.parse_args()
    rows = [
        json.loads(line) for path in args.datasets
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    failures = audit(
        rows, max_cue=args.max_cue_concentration,
        max_edge=args.max_edge_concentration,
        max_ngram=args.max_ngram_concentration,
        max_near_duplicate=args.max_near_duplicate,
        strict_archetypes=set(args.strict_archetype) if args.strict_archetype else None,
    )
    if failures:
        raise SystemExit("Annotation-artifact audit failed:\n- " + "\n- ".join(failures))
    print("Annotation-artifact audit passed")


if __name__ == "__main__":
    main()