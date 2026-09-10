"""Audit a locked fixed-vector validation run before export or test access."""
from __future__ import annotations

import argparse
import json

from safety_governor.validation import audit_validation_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    print(json.dumps(audit_validation_run(args.run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
