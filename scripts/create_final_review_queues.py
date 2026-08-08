"""Create explicit human review queues for the remaining English gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def instruction_queue(candidates: list[dict]) -> list[dict]:
    return [{
        "pair_id": row["pair_id"],
        "archetype": row["archetype"],
        "english_instruction": row["english_instruction"],
        "instruction_ids": row["instruction_ids"],
        "safe_completion": row["safe_completion"],
        "naturalistic_evasion": row["naturalistic_evasion"],
        "official_safe": row["official_ifeval_check"]["safe"],
        "official_evasion": row["official_ifeval_check"]["evasion"],
        "declared_failed_instruction_ids": row["expected_failed_instruction_ids"],
        "failure_declaration": "pending",
        "annotation_decision": "pending",
        "review_notes": "",
    } for row in candidates]


def motivated_queue(records: list[dict]) -> list[dict]:
    pairs = {}
    for row in records:
        if row.get("archetype") == "motivated_reasoning":
            pairs.setdefault(row["pair_id"], {})[row["polarity"]] = row
    queue = []
    for pair_id, pair in sorted(pairs.items()):
        if set(pair) != {"safe", "unsafe"}:
            raise ValueError(f"incomplete motivated pair: {pair_id}")
        queue.append({
            "pair_id": pair_id,
            "source_group_id": pair["safe"]["source_group_id"],
            "instruction": pair["safe"]["instruction"],
            "safe_completion": pair["safe"]["completion"],
            "unsafe_completion": pair["unsafe"]["completion"],
            "annotation_decision": "pending",
            "review_notes": "",
        })
    return queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-candidates", type=Path, required=True)
    parser.add_argument("--deceptive-candidates", type=Path, required=True)
    parser.add_argument("--instruction-output", type=Path, required=True)
    parser.add_argument("--motivated-output", type=Path, required=True)
    args = parser.parse_args()
    instructions = instruction_queue(read(args.instruction_candidates))
    motivated = motivated_queue(read(args.deceptive_candidates))
    if len(instructions) != 150 or len(motivated) != 30:
        raise SystemExit(f"queue cardinality mismatch: instruction={len(instructions)} motivated={len(motivated)}")
    write(args.instruction_output, instructions)
    write(args.motivated_output, motivated)
    print(f"wrote {len(instructions)} instruction and {len(motivated)} motivated review tasks")


if __name__ == "__main__":
    main()
