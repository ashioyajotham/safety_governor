import json
import sys

import pytest

from scripts import (
    apply_annotation_review,
    assemble_instruction_candidate_pool,
    assign_pair_splits,
    audit_repository_hygiene,
    audit_research_corpus,
    freeze_instruction_corpus,
)
from scripts.build_diverse_deception_draft import preference_signal
from scripts.import_cot_contrastive import row as cot_record
from scripts.import_jailbreakbench import record as harmful_record


ARCHETYPES = (
    "constraint_omission",
    "hedging_or_excessive_caveating",
    "topic_shift",
    "false_completion",
)


def row(pair_id, archetype, status="approved"):
    return {
        "pair_id": pair_id,
        "archetype": archetype,
        "annotation_status": status,
        "safe_completion": "safe text",
        "naturalistic_evasion": "evasive text",
        "notes": "specific contrastive rationale",
    }


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")


def test_import_records_explicit_decision_without_reviewer_identity(tmp_path, monkeypatch):
    source = tmp_path / "draft.jsonl"
    output = tmp_path / "reviewed.jsonl"
    write_jsonl(source, [row("ifeval-1", "topic_shift", "pending_review")])
    monkeypatch.setattr(sys, "argv", ["import", str(source), "--output", str(output), "--decision", "approved", "--review-note", "Reviewed completion pair."])
    apply_annotation_review.main()
    reviewed = json.loads(output.read_text(encoding="utf-8"))
    assert reviewed["annotation_status"] == "approved"
    assert reviewed["review_decision"] == "approved"
    assert reviewed["review_notes"] == "Reviewed completion pair."


def test_freeze_requires_exact_archetype_set_and_writes_120_rows(tmp_path, monkeypatch):
    source = tmp_path / "approved.jsonl"
    output = tmp_path / "frozen.jsonl"
    rows = [row(f"{archetype}-{index}", archetype) for archetype in ARCHETYPES for index in range(30)]
    for item in rows:
        if item["archetype"] in {"topic_shift", "hedging_or_excessive_caveating"}:
            item["semantic_validation"] = {"status": "human_confirmed"}
        else:
            item["official_ifeval_check"] = {"declaration_status": "human_confirmed"}
    write_jsonl(source, rows)
    monkeypatch.setattr(sys, "argv", ["freeze", str(source), "--output", str(output)])
    freeze_instruction_corpus.main()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 120


def test_audit_rejects_missing_canonical_archetype(tmp_path, monkeypatch):
    source = tmp_path / "incomplete.jsonl"
    write_jsonl(source, [row(f"topic-{index}", "topic_shift") for index in range(120)])
    monkeypatch.setattr(sys, "argv", ["audit", str(source)])
    with pytest.raises(SystemExit, match="archetype set mismatch"):
        audit_research_corpus.main()

def test_pair_split_is_deterministic():
    assert assign_pair_splits.split_for(42, "pair-7") == assign_pair_splits.split_for(42, "pair-7")


def test_preference_signal_ignores_argument_body_language():
    neutral = {"prompt": [{"type": "human", "content": "Please comment.\nArgument: This is a persuasive argument."}]}
    positive = {"prompt": [{"type": "human", "content": "Please comment. I really like the argument.\nArgument: text"}]}
    assert preference_signal(neutral) is None
    assert preference_signal(positive) == "positive"

def test_candidate_pool_is_provider_neutral_and_preserves_pending_state():
    primary = [row(f"constraint-{index}", "constraint_omission", "pending_review") for index in range(60)]
    primary += [row(f"hedge-{index}", "hedging_or_excessive_caveating", "pending_review") for index in range(29)]
    primary += [row(f"false-{index}", "false_completion", "pending_review") for index in range(26)]
    primary += [row(f"topic-{index}", "topic_shift", "pending_review") for index in range(20)]
    for item in primary:
        item["annotator"] = "gemini_assisted::test-model"
        item["generation_metadata"] = {"model": "gemini-test"}
    gap = [row("hedge-gap", "hedging_or_excessive_caveating", "pending_review")]
    gap += [row(f"false-gap-{index}", "false_completion", "pending_review") for index in range(4)]
    gap += [row(f"topic-gap-{index}", "topic_shift", "pending_review") for index in range(10)]

    result = assemble_instruction_candidate_pool.assemble(primary, gap)

    assert len(result) == 150
    assert all(item["annotation_status"] == "pending_review" for item in result)
    assert all("annotator" not in item for item in result)
    generated = next(item for item in result if item["pair_id"] == "constraint-0")
    assert generated["annotation_provenance"] == {
        "method": "model_assisted",
        "provider": "google",
        "model": "gemini-test",
    }


def test_hygiene_flags_provider_and_versioned_canonical_paths():
    problems = audit_repository_hygiene.findings(
        [
            "scripts/import_reviewed_gemini_drafts.py",
            "data/working/deceptive_candidates_v2.jsonl",
            "docs/notebooks/providers/gemini/ifeval_annotation_colab.ipynb",
        ]
    )
    assert len(problems) == 2

def test_unreviewed_imports_remain_unassigned():
    harmful = harmful_record("jbb-001", "safe", "prompt", "source", "category")
    deceptive = cot_record("cot-001", "safe", "prompt", "source")
    assert harmful["reviewer_status"] == "draft"
    assert deceptive["reviewer_status"] == "draft"
    assert harmful["split"] == deceptive["split"] == "unassigned"


def test_group_split_keeps_distinct_pairs_from_one_source_together():
    rows = [
        {"pair_id": "p1", "source_group_id": "g", "reviewer_status": "approved"},
        {"pair_id": "p2", "source_group_id": "g", "reviewer_status": "approved"},
    ]
    assignments = assign_pair_splits.assign(rows, seed=42)
    assert assignments["p1"] == assignments["p2"]


def test_materializer_excludes_annotation_provider_metadata():
    from scripts.materialize_contrastive_records import materialize

    source = {
        "pair_id": "ifeval-1",
        "archetype": "topic_shift",
        "annotation_status": "approved",
        "english_instruction": "answer directly",
        "safe_completion": "direct answer",
        "naturalistic_evasion": "adjacent tangent",
        "source": "official-source",
        "source_key": 1,
        "annotation_provenance": {"provider": "provider-sentinel"},
        "review_notes": "review-sentinel",
    }
    records = materialize(source)
    serialized = json.dumps(records)
    assert "provider-sentinel" not in serialized
    assert "review-sentinel" not in serialized
    assert {record["completion"] for record in records} == {"direct answer", "adjacent tangent"}

def test_template_audit_rejects_repeated_motivated_frame():
    from scripts.audit_annotation_artifacts import audit

    rows = [
        {"pair_id": f"p{i}", "archetype": "motivated_reasoning", "polarity": "unsafe", "completion": f"same opening frame with topic {i} and the same ending"}
        for i in range(10)
    ]
    errors = audit(rows, strict_archetypes={"motivated_reasoning"})
    assert any("four-word prefix" in error for error in errors)


def test_vector_fit_metadata_rejects_validation_rows():
    from scripts.extract_vector import validate_fit_metadata

    metadata = {
        "sample_ids": ["p1"], "splits": ["validation"],
        "source_group_ids": ["g1"], "token_mode": "response_mean", "layer": 0,
    }
    assert any("train-only" in error for error in validate_fit_metadata(metadata, metadata))


def test_public_source_urls_embed_immutable_commits():
    from scripts.fetch_corpus_sources import SOURCES, source_url

    for spec in SOURCES.values():
        assert spec["commit"] in source_url(spec)
        assert "/main/" not in source_url(spec) and "/master/" not in source_url(spec)

def test_official_ifeval_evaluation_is_order_independent():
    from scripts.validate_ifeval_candidates import evaluate

    row = {
        "source_key": 1,
        "instruction_ids": ["keywords:existence"],
        "english_instruction": "Mention apple.",
        "kwargs": [{"keywords": ["apple"]}],
        "safe_completion": "apple",
        "naturalistic_evasion": "banana",
    }
    first = evaluate(row, "safe_completion")
    evaluate(row, "naturalistic_evasion")
    assert evaluate(row, "safe_completion") == first
    assert first["all_passed"]


def test_group_assignment_is_archetype_stratified():
    rows = [
        {"pair_id": f"a-{i}", "source_group_id": f"ga-{i}", "archetype": "a"}
        for i in range(10)
    ] + [
        {"pair_id": f"b-{i}", "source_group_id": f"gb-{i}", "archetype": "b"}
        for i in range(10)
    ]
    assignments = assign_pair_splits.assign(rows, seed=42)
    for prefix in ("a", "b"):
        values = [assignments[f"{prefix}-{i}"] for i in range(10)]
        assert values.count("train") == 6
        assert values.count("validation") == 2
        assert values.count("test") == 2

def test_final_review_application_fails_closed():
    from scripts.apply_final_review_queues import apply_instruction

    candidate = {
        "pair_id": "p1", "archetype": "constraint_omission",
        "official_ifeval_check": {"declaration_status": "pending"},
        "annotation_status": "pending_review",
    }
    queue = [{
        "pair_id": "p1", "failure_declaration": "pending",
        "annotation_decision": "approved", "review_notes": "checked",
    }]
    with pytest.raises(ValueError, match="unresolved failure declaration"):
        apply_instruction([candidate], queue)