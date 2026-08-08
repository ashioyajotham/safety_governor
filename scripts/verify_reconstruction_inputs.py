"""Verify that a clean checkout plus the local research bundle is complete."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    reconstruction = json.loads(Path("datasets/manifests/reconstruction.json").read_text(encoding="utf-8"))
    sources = json.loads(Path(reconstruction["public_sources"]).read_text(encoding="utf-8"))
    working = json.loads(Path(reconstruction["restricted_bundle"]["manifest"]).read_text(encoding="utf-8"))
    expected_working = {item["path"]: item for item in working["artifacts"]}
    failures = []
    for name, spec in sources.items():
        path = Path("data/raw/sources") / name
        if not path.is_file() or sha256(path) != spec["sha256"]:
            failures.append(f"public source missing or mismatched: {path}")
    for value in reconstruction["restricted_bundle"]["required_paths"]:
        path = Path(value)
        expected = expected_working.get(value)
        if expected is None:
            failures.append(f"restricted path absent from working manifest: {value}")
        elif not path.is_file() or sha256(path) != expected["sha256"]:
            failures.append(f"restricted input missing or mismatched: {value}")
    if failures:
        raise SystemExit("Reconstruction verification failed:\n- " + "\n- ".join(failures))
    print("Reconstruction inputs verified")


if __name__ == "__main__":
    main()
