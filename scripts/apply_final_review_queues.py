"""Apply explicit human queue decisions without inferring approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def apply_instruction(candidates: list[dict], queue: list[dict]) -> list[dict]:
    decisions = {row["pair_id"]: row for row in queue}
    for row in candidates:
        decision = decisions[row["pair_id"]]
        declaration = decision.get("failure_declaration")
        annotation = decision.get("annotation_decision")
        note = decision.get("review_notes", "").strip()
        if declaration not in {"confirmed", "revision_required"}:
            raise ValueError(f"{row['pair_id']}: unresolved failure declaration")
        if annotation not in {"approved", "rejected"}:
            raise ValueError(f"{row['pair_id']}: unresolved annotation decision")
        if not note:
            raise ValueError(f"{row['pair_id']}: review note required")
        row["official_ifeval_check"]["declaration_status"] = (
            "human_confirmed" if declaration == "confirmed" else "revision_required"
        )
        row["annotation_status"] = annotation if declaration == "confirmed" else "pending_review"
        row["review_decision"] = row["annotation_status"]
        row["review_notes"] = note
    return candidates


def apply_motivated(records: list[dict], queue: list[dict]) -> list[dict]:
    decisions = {row["pair_id"]: row for row in queue}
    for row in records:
        if row.get("archetype") != "motivated_reasoning":
            continue
        decision = decisions[row["pair_id"]]
        status = decision.get("annotation_decision")
        note = decision.get("review_notes", "").strip()
        if status not in {"approved", "rejected"} or not note:
            raise ValueError(f"{row['pair_id']}: resolved decision and note required")
        row["reviewer_status"] = status
        row["review_notes"] = note
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-candidates", type=Path, required=True)
    parser.add_argument("--instruction-queue", type=Path, required=True)
    parser.add_argument("--instruction-output", type=Path, required=True)
    parser.add_argument("--deceptive-candidates", type=Path, required=True)
    parser.add_argument("--motivated-queue", type=Path, required=True)
    parser.add_argument("--deceptive-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        instruction = apply_instruction(read(args.instruction_candidates), read(args.instruction_queue))
        deceptive = apply_motivated(read(args.deceptive_candidates), read(args.motivated_queue))
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Review application blocked: {exc}") from exc
    write(args.instruction_output, instruction)
    write(args.deceptive_output, deceptive)
    print("applied fully resolved human review queues")


if __name__ == "__main__":
    main()
