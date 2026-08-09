"""Import a completed workbench export into a new, non-overwriting local session."""
from __future__ import annotations

import argparse
from pathlib import Path

from safety_governor.review_workbench import import_review_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/working/instruction_noncompliance/review_imports"),
    )
    args = parser.parse_args()
    print(import_review_export(args.export, args.destination))


if __name__ == "__main__":
    main()
