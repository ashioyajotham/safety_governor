"""Validate a JSONL contrastive dataset without executing model code."""
from __future__ import annotations

import argparse
from safety_governor.data import load_jsonl, validate_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    errors = validate_records(load_jsonl(args.dataset))
    if errors:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(errors))
    print("Dataset validation passed")


if __name__ == "__main__":
    main()
