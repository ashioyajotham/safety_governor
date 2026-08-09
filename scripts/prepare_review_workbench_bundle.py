"""Build the immutable local/Colab IFEval human-review bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

from safety_governor.review_workbench import prepare_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_bundle(args.repo_root, args.output)
    print(f"created review bundle with {sum(manifest['queue_counts'].values())} rows at {args.output}")


if __name__ == "__main__":
    main()
