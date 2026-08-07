import json
import sys

import pytest

from scripts import assign_pair_splits, audit_research_corpus, freeze_instruction_corpus, import_reviewed_gemini_drafts
from scripts.build_diverse_deception_draft import preference_signal


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
    import_reviewed_gemini_drafts.main()
    reviewed = json.loads(output.read_text(encoding="utf-8"))
    assert reviewed["annotation_status"] == "approved"
    assert reviewed["review_decision"] == "approved"
    assert reviewed["review_notes"] == "Reviewed completion pair."


def test_freeze_requires_exact_archetype_set_and_writes_120_rows(tmp_path, monkeypatch):
    source = tmp_path / "approved.jsonl"
    output = tmp_path / "frozen.jsonl"
    rows = [row(f"{archetype}-{index}", archetype) for archetype in ARCHETYPES for index in range(30)]
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
