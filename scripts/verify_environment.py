"""Verify exact research runtime versions and immutable model revisions."""
from __future__ import annotations

import argparse

from safety_governor.config import load
from safety_governor.preflight import runtime_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    errors = runtime_errors(load(args.config))
    if errors:
        raise SystemExit("Environment preflight failed:\n- " + "\n- ".join(errors))
    print("Environment preflight passed")


if __name__ == "__main__":
    main()