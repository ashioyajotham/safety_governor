"""Repair common UTF-8-as-Western-text mojibake recursively in JSONL strings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKERS = ("Ã", "â€", "â€™", "â€œ", "â€", "Â", "à¤")


def fix_text(text: str) -> str:
    value = text
    for _ in range(2):
        if not any(marker in value for marker in MARKERS):
            break
        repaired = None
        for encoding in ("cp1252", "latin1"):
            try:
                repaired = value.encode(encoding).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if repaired is None or repaired == value:
            break
        value = repaired
    return value


def fix(value):
    if isinstance(value, str):
        return fix_text(value)
    if isinstance(value, list):
        return [fix(item) for item in value]
    if isinstance(value, dict):
        return {key: fix(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    repaired = [fix(row) for row in rows]
    changed = sum(before != after for before, after in zip(rows, repaired))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in repaired) + "\n", encoding="utf-8")
    print(f"repaired mojibake in {changed} of {len(rows)} rows; review status unchanged")


if __name__ == "__main__":
    main()