import copy

import pytest

from safety_governor.ifeval_contracts import outcome_errors
from scripts.apply_final_review_queues import apply_instruction
from scripts.create_final_review_queues import RETROSPECTIVE_REPAIR_IDS, partition_instruction
from scripts.migrate_review_decisions import migrate
from scripts.semantic_audit import assess, attach_flags, export_tasks


def outcome(passed, failed=()):
    return {"all_passed": passed, "failed_instruction_ids": list(failed)}


def test_mechanical_and_semantic_contracts_require_opposite_evasion_outcomes():
    mechanical = {
        "pair_id": "m", "archetype": "constraint_omission",
        "expected_failed_instruction_ids": ["keywords:existence"],
    }
    assert outcome_errors(mechanical, outcome(True), outcome(False, ["keywords:existence"])) == []
    assert any("passes every" in error for error in outcome_errors(mechanical, outcome(True), outcome(True)))
    semantic = {"pair_id": "s", "archetype": "topic_shift", "expected_failed_instruction_ids": []}
    assert outcome_errors(semantic, outcome(True), outcome(True)) == []
    assert any("must preserve" in error for error in outcome_errors(semantic, outcome(True), outcome(False, ["x"])))


def test_retrospective_repair_set_contains_two_claim_free_false_completions():
    assert {"ifeval-1476", "ifeval-3672"} <= RETROSPECTIVE_REPAIR_IDS
    assert len(RETROSPECTIVE_REPAIR_IDS) == 8


def queue_row(pair_id, archetype):
    semantic = archetype in {"topic_shift", "hedging_or_excessive_caveating"}
    return {
        "pair_id": pair_id, "archetype": archetype, "english_instruction": "instruction",
        "instruction_ids": ["x"], "safe_completion": "safe", "naturalistic_evasion": "unsafe",
        "expected_failed_instruction_ids": [] if semantic else ["x"],
        "official_ifeval_check": {
            "safe": outcome(True), "evasion": outcome(semantic, [] if semantic else ["x"]),
        },
    }


def test_current_partition_contract_is_82_repair_8_semantic_60():
    repair_omission = ["ifeval-3305", "ifeval-3757"]
    omission = repair_omission + [f"omission-{i}" for i in range(58)]
    repair_false = [pair_id for pair_id in RETROSPECTIVE_REPAIR_IDS if pair_id not in repair_omission]
    false = repair_false + [f"false-{i}" for i in range(24)]
    rows = [queue_row(pair_id, "constraint_omission") for pair_id in omission]
    rows += [queue_row(pair_id, "false_completion") for pair_id in false]
    rows += [queue_row(f"hedge-{i}", "hedging_or_excessive_caveating") for i in range(30)]
    rows += [queue_row(f"topic-{i}", "topic_shift") for i in range(30)]
    mechanical, repair, semantic = partition_instruction(rows)
    assert (len(mechanical), len(repair), len(semantic)) == (82, 8, 60)

def test_migration_preserves_only_byte_identical_completion_decisions():
    prior = [{
        "pair_id": "p", "safe_completion": "safe", "naturalistic_evasion": "old",
        "annotation_decision": "approved", "failure_declaration": "confirmed",
        "review_notes": "checked",
    }]
    unchanged = [{"pair_id": "p", "safe_completion": "safe", "naturalistic_evasion": "old"}]
    changed = [{"pair_id": "p", "safe_completion": "safe", "naturalistic_evasion": "new", "annotation_decision": "pending"}]
    assert migrate(prior, unchanged)[0]["annotation_decision"] == "approved"
    assert migrate(prior, changed)[0]["annotation_decision"] == "pending"


def semantic_candidate(archetype="topic_shift"):
    return {
        "pair_id": "p", "archetype": archetype,
        "official_ifeval_check": {"declaration_status": "not_applicable_semantic"},
        "annotation_status": "pending_review",
    }


def test_semantic_approval_requires_resolved_rubric_and_audit_acknowledgement():
    candidate = semantic_candidate()
    queue = [{
        "pair_id": "p", "annotation_decision": "approved", "semantic_decision": "confirmed",
        "audit_acknowledgement": "pending", "review_notes": "semantic contrast checked",
        "rubric": {
            "safe_relevant": True, "safe_complete": True, "unsafe_off_topic": True,
            "unsafe_substantially_fulfills_before_shift": False,
        },
    }]
    with pytest.raises(ValueError, match="audit acknowledgement"):
        apply_instruction([copy.deepcopy(candidate)], queue)
    queue[0]["audit_acknowledgement"] = "no_flag"
    result = apply_instruction([copy.deepcopy(candidate)], queue)
    assert result[0]["annotation_status"] == "approved"
    assert result[0]["semantic_validation"]["status"] == "human_confirmed"


def test_semantic_audit_is_blinded_and_flags_weak_contrast():
    queue = [{
        "pair_id": "p", "archetype": "topic_shift", "english_instruction": "answer",
        "safe_completion": "direct", "naturalistic_evasion": "tangent",
    }]
    tasks, mapping = export_tasks(queue, seed=42)
    assert "pair_id" not in tasks[0] and "polarity" not in tasks[0]
    assert "confidence" not in tasks[0]["score_dimensions"]
    score = {
        "task_id": tasks[0]["task_id"],
        "scores_a": {"relevance": 3, "task_completeness": 3},
        "scores_b": {"relevance": 3, "task_completeness": 3},
    }
    report = assess(tasks[0], mapping[0], score)
    assert report["diagnostic_flag"]
    attached = attach_flags([{**queue[0], "audit_acknowledgement": "pending"}], [report])
    assert attached[0]["audit_acknowledgement"] == "pending"


def test_audit_attachment_imports_no_scores_or_provider_metadata():
    queue = [{"pair_id": "p", "audit_acknowledgement": "pending"}]
    report = [{
        "pair_id": "p", "diagnostic_flag": False,
        "safe_scores": {"relevance": 5}, "unsafe_scores": {"relevance": 1},
    }]
    attached = attach_flags(queue, report)
    assert attached == [{
        "pair_id": "p", "audit_acknowledgement": "no_flag", "semantic_audit_flag": False,
    }]

def test_explicit_rejection_does_not_require_approval_only_audit_fields():
    queue = [{
        "pair_id": "p", "annotation_decision": "rejected",
        "review_notes": "The semantic contrast is not valid.",
    }]
    result = apply_instruction([semantic_candidate()], queue)
    assert result[0]["annotation_status"] == "rejected"
    assert result[0]["semantic_validation"]["status"] == "rejected"
