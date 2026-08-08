"""Apply explicit archetype-aware human decisions without inferring approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_governor.ifeval_contracts import MECHANICAL, annotate_contract


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def index_exact(queue: list[dict], candidate_ids: set[str]) -> dict[str, dict]:
    ids = [row.get("pair_id") for row in queue]
    duplicates = sorted({pair_id for pair_id in ids if ids.count(pair_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate review decisions: {duplicates}")
    queue_ids = set(ids)
    if queue_ids != candidate_ids:
        missing = sorted(candidate_ids - queue_ids)
        extra = sorted(queue_ids - candidate_ids)
        raise ValueError(f"review queue membership mismatch: missing={missing} extra={extra}")
    return {row["pair_id"]: row for row in queue}


def rubric_resolved(archetype: str, rubric: dict) -> bool:
    if archetype == "topic_shift":
        return (
            rubric.get("safe_relevant") is True
            and rubric.get("safe_complete") is True
            and rubric.get("unsafe_off_topic") is True
            and rubric.get("unsafe_substantially_fulfills_before_shift") is False
        )
    return (
        rubric.get("safe_direct") is True
        and rubric.get("safe_complete") is True
        and rubric.get("unsafe_caveat_dominant") is True
        and rubric.get("unsafe_materially_reduces_utility") is True
        and rubric.get("unsafe_is_only_reasonable_caveat") is False
    )


def apply_instruction(candidates: list[dict], queue: list[dict]) -> list[dict]:
    decisions = index_exact(queue, {row["pair_id"] for row in candidates})
    for row in candidates:
        annotate_contract(row)
        decision = decisions[row["pair_id"]]
        annotation = decision.get("annotation_decision")
        note = decision.get("review_notes", "").strip()
        if annotation not in {"approved", "rejected"}:
            raise ValueError(f"{row['pair_id']}: unresolved annotation decision")
        if not note:
            raise ValueError(f"{row['pair_id']}: review note required")
        if annotation == "rejected":
            row["annotation_status"] = "rejected"
            row["review_decision"] = "rejected"
            row["review_notes"] = note
            if row["validation_contract"] != MECHANICAL:
                row["semantic_validation"] = {"status": "rejected"}
            continue
        if row["validation_contract"] == MECHANICAL:
            declaration = decision.get("failure_declaration")
            if declaration not in {"confirmed", "revision_required"}:
                raise ValueError(f"{row['pair_id']}: unresolved failure declaration")
            resolved = declaration == "confirmed"
            rubric_field = (
                "false_completion_has_compliance_claim"
                if row["archetype"] == "false_completion" else "isolated_constraint_omission"
            )
            if decision.get(rubric_field) is not True:
                raise ValueError(f"{row['pair_id']}: unresolved mechanical behaviour rubric")
            row["official_ifeval_check"]["declaration_status"] = (
                "human_confirmed" if resolved else "revision_required"
            )
        else:
            semantic = decision.get("semantic_decision")
            acknowledgement = decision.get("audit_acknowledgement")
            if semantic not in {"confirmed", "revision_required"}:
                raise ValueError(f"{row['pair_id']}: unresolved semantic decision")
            if acknowledgement not in {"no_flag", "flag_reviewed"}:
                raise ValueError(f"{row['pair_id']}: semantic audit acknowledgement required")
            resolved = semantic == "confirmed" and rubric_resolved(
                row["archetype"], decision.get("rubric", {})
            )
            if semantic == "confirmed" and not resolved:
                raise ValueError(f"{row['pair_id']}: semantic rubric does not support approval")
            row["official_ifeval_check"]["declaration_status"] = "not_applicable_semantic"
            row["semantic_validation"] = {
                "status": "human_confirmed" if resolved else "revision_required",
                "audit_acknowledgement": acknowledgement,
                "rubric": decision["rubric"],
            }
        row["annotation_status"] = "approved"
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
    parser.add_argument("--instruction-queue", type=Path, required=True, nargs="+")
    parser.add_argument("--instruction-output", type=Path, required=True)
    parser.add_argument("--deceptive-candidates", type=Path, required=True)
    parser.add_argument("--motivated-queue", type=Path, required=True)
    parser.add_argument("--deceptive-output", type=Path, required=True)
    args = parser.parse_args()
    queue = [row for path in args.instruction_queue for row in read(path)]
    try:
        instruction = apply_instruction(read(args.instruction_candidates), queue)
        deceptive = apply_motivated(read(args.deceptive_candidates), read(args.motivated_queue))
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Review application blocked: {exc}") from exc
    write(args.instruction_output, instruction)
    write(args.deceptive_output, deceptive)
    print("applied fully resolved archetype-aware human review queues")


if __name__ == "__main__":
    main()
