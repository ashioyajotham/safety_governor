"""Report curation progress without exposing prompt text."""
from __future__ import annotations
import argparse
from collections import Counter
from safety_governor.data import load_jsonl, validate_records


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); args = parser.parse_args()
    records = load_jsonl(args.dataset); errors = validate_records(records, require_approved=False)
    print(f"records: {len(records)}; unique pair IDs: {len({r.pair_id for r in records})}")
    for key, count in sorted(Counter((r.behavior.value, r.language, r.reviewer_status) for r in records).items()): print("\t".join(key), count, sep="\t")
    if errors: raise SystemExit("Validation findings:\n- " + "\n- ".join(errors))


if __name__ == "__main__": main()
