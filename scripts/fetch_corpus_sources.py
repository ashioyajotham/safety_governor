"""Fetch pinned public source files needed to reproduce restricted corpus drafts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from urllib.request import urlretrieve

SOURCES = {
    "truthfulqa.csv": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
    "sycophancy_answer.jsonl": "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/answer.jsonl",
    "sycophancy_feedback.jsonl": "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/feedback.jsonl",
    "ifeval_input_data.jsonl": "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="data/raw/sources"); args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True); manifest = {}
    for name, url in SOURCES.items():
        path = root / name; urlretrieve(url, path)
        manifest[name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "sources_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Fetched {len(SOURCES)} source files to {root}")

if __name__ == "__main__": main()
