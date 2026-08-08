"""Verify exact hashes for ignored local working-state artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("datasets/manifests/working_state.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            failures.append(f"hash mismatch: {path}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            records = 1
        else:
            records = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
        if records != artifact["records"]:
            failures.append(f"record count mismatch: {path}")
    if failures:
        raise SystemExit("Working-state verification failed:\n- " + "\n- ".join(failures))
    print(f"Working-state verification passed for {len(manifest['artifacts'])} artifacts")


if __name__ == "__main__":
    main()