"""Export a completed Stage-1 run as a checksummed portable archive."""
from __future__ import annotations

import argparse
import json

from safety_governor.stage1 import export_run, verify_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--include-shards", action="store_true")
    parser.add_argument("--verify", metavar="ARCHIVE")
    parser.add_argument("--checksum", default=None)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_export(args.verify, args.checksum), indent=2))
        return
    if not args.run_directory or not args.output:
        parser.error("run_directory and output are required unless --verify is used")
    print(json.dumps(export_run(
        args.run_directory,
        args.output,
        include_shards=args.include_shards,
    ), indent=2))


if __name__ == "__main__":
    main()
