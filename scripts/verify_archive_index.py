"""Verify locally retained files against the tracked archive index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index",
        type=Path,
        nargs="?",
        default=Path("datasets/manifests/archive_index.json"),
    )
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    failures = []
    checked = 0
    for entry in index["entries"]:
        destination = entry.get("destination")
        if not destination:
            continue
        path = Path(destination)
        if not path.is_file():
            failures.append(f"missing retained file: {destination}")
            continue
        expected = entry.get("destination_sha256", entry["sha256"])
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {destination}")
        checked += 1
    if failures:
        raise SystemExit("Archive verification failed:\n- " + "\n- ".join(failures))
    print(f"Archive verification passed for {checked} retained files")


if __name__ == "__main__":
    main()
