"""Fetch immutable public corpus sources and reject unexpected bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

SOURCES = {
    "truthfulqa.csv": {
        "commit": "d71c110897f5d31c5d7f309e7bc316c152f6f031",
        "path": "TruthfulQA.csv",
        "sha256": "b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c",
        "repository": "sylinrl/TruthfulQA",
    },
    "sycophancy_answer.jsonl": {
        "commit": "9a1694221e3639887138f61deae344335eca6752",
        "path": "datasets/answer.jsonl",
        "sha256": "3da2c2bbf685cf2c6cbfc6bf67449caad6fe1d56b67924087b3792ffba47bcc1",
        "repository": "meg-tong/sycophancy-eval",
    },
    "sycophancy_feedback.jsonl": {
        "commit": "9a1694221e3639887138f61deae344335eca6752",
        "path": "datasets/feedback.jsonl",
        "sha256": "3687c5c335b41adf13bb6004fe9e3eff4067fb0457b4b225e88f55e76e18399f",
        "repository": "meg-tong/sycophancy-eval",
    },
    "ifeval_input_data.jsonl": {
        "commit": "b24f2136e8ef405b900b5619760126304f190941",
        "path": "instruction_following_eval/data/input_data.jsonl",
        "sha256": "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
        "repository": "google-research/google-research",
    },
}


def source_url(spec: dict) -> str:
    return (
        f"https://raw.githubusercontent.com/{spec['repository']}/"
        f"{spec['commit']}/{spec['path']}"
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_one(root: Path, name: str, spec: dict) -> dict:
    destination = root / name
    if destination.exists():
        actual = digest(destination)
        if actual != spec["sha256"]:
            raise ValueError(f"existing {name} hash {actual}; expected {spec['sha256']}")
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=root)
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            urlretrieve(source_url(spec), temporary_path)
            actual = digest(temporary_path)
            if actual != spec["sha256"]:
                raise ValueError(f"downloaded {name} hash {actual}; expected {spec['sha256']}")
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return {**spec, "url": source_url(spec)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/sources")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {name: fetch_one(root, name, spec) for name, spec in SOURCES.items()}
    (root / "sources_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(SOURCES)} immutable source files in {root}")


if __name__ == "__main__":
    main()