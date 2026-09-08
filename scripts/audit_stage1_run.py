"""Audit a completed Stage-1 run before interpretation or export."""
from __future__ import annotations

import argparse
import json

from safety_governor.stage1 import audit_completed_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    args = parser.parse_args()
    print(json.dumps(audit_completed_run(args.run_directory), indent=2))


if __name__ == "__main__":
    main()
