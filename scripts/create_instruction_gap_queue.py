"""Create a provider-neutral retry/review queue from annotation failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failed_ids = {row["pair_id"] for row in read_jsonl(args.failures)}
    tasks = {row["pair_id"]: row for row in read_jsonl(args.tasks)}
    missing = sorted(failed_ids - set(tasks))
    if missing:
        raise SystemExit(f"Missing failed IDs in task source: {missing}")

    rows = []
    for pair_id in sorted(failed_ids):
        row = dict(tasks[pair_id])
        row["queue_reason"] = "generation_or_programmatic_audit_failure"
        row["queue_action"] = "manual_rewrite_or_targeted_regeneration"
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} review-queue rows to {args.output}")


if __name__ == "__main__":
    main()