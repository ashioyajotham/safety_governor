"""Build deterministic, source-backed Validation-v2 curation candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_governor.validation_v2 import build_candidates


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", type=Path, default=Path("data/raw/sources/gsm8k_train.jsonl"))
    parser.add_argument("--truthfulqa", type=Path, default=Path("data/raw/sources/truthfulqa.csv"))
    parser.add_argument(
        "--bigbench", type=Path,
        default=Path("data/raw/sources/bigbench_formal_fallacies.json"),
    )
    parser.add_argument("--sources", type=Path, default=Path("datasets/manifests/source_corpora.json"))
    parser.add_argument("--base", type=Path, default=Path("datasets/frozen/english_contrastive.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"candidate artifact already exists: {args.output}")
    specs = json.loads(args.sources.read_text(encoding="utf-8"))
    rows = build_candidates(
        gsm8k_path=args.gsm8k, truthfulqa_path=args.truthfulqa,
        bigbench_path=args.bigbench, source_specs=specs,
        base_rows=_read_jsonl(args.base), seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts = {}
    for row in rows:
        counts.setdefault(row["validation_role"], {}).setdefault(row["archetype"], 0)
        counts[row["validation_role"]][row["archetype"]] += 1
    print(json.dumps({"candidates": len(rows), "counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
