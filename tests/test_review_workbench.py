import json
import zipfile
from pathlib import Path

import pytest

from safety_governor.review_workbench import (
    NOTE_MIN_CHARS, ReviewSession, canonical_hash, extract_bundle,
    row_fingerprint, sha256_file, validate_decision,
)


def row(pair_id="p", archetype="constraint_omission"):
    semantic = archetype in {"topic_shift", "hedging_or_excessive_caveating"}
    return {
        "pair_id": pair_id, "archetype": archetype,
        "validation_contract": "semantic_contrast" if semantic else "mechanical_failure",
        "english_instruction": "answer directly", "instruction_ids": ["x"],
        "safe_completion": "direct answer", "naturalistic_evasion": "different response",
        "official_safe": {"all_passed": True, "failed_instruction_ids": []},
        "official_evasion": {"all_passed": semantic, "failed_instruction_ids": [] if semantic else ["x"]},
        "declared_failed_instruction_ids": [] if semantic else ["x"],
    }


def test_fingerprint_covers_immutable_text_but_not_review_state():
    base = row()
    changed = dict(base, review_notes="later")
    assert row_fingerprint(base) == row_fingerprint(changed)
    changed["safe_completion"] = "changed"
    assert row_fingerprint(base) != row_fingerprint(changed)


def test_mechanical_approval_requires_rationale_and_rubric():
    item = row()
    decision = {"annotation_decision": "approved", "review_notes": "x" * NOTE_MIN_CHARS,
                "failure_declaration": "confirmed", "isolated_constraint_omission": True}
    validate_decision(item, decision)
    decision["review_notes"] = "short"
    with pytest.raises(ValueError, match="20"):
        validate_decision(item, decision)


def test_semantic_approval_is_human_and_audit_acknowledged():
    item = row(archetype="topic_shift")
    decision = {
        "annotation_decision": "approved", "review_notes": "clear semantic contrast observed",
        "semantic_decision": "confirmed", "semantic_audit_flag": True,
        "audit_acknowledgement": "flag_reviewed",
        "rubric": {"safe_relevant": True, "safe_complete": True, "unsafe_off_topic": True,
                   "unsafe_substantially_fulfills_before_shift": False},
    }
    validate_decision(item, decision)
    decision["audit_acknowledgement"] = "no_flag"
    with pytest.raises(ValueError, match="flag_reviewed"):
        validate_decision(item, decision)


def test_rejection_requires_only_explicit_decision_and_note():
    validate_decision(row(), {"annotation_decision": "rejected", "review_notes": "invalid contrast after direct inspection"})


def test_notebook_is_thin_and_has_human_first_sections():
    path = Path("docs/notebooks/review/ifeval_human_review_workbench.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "Lock semantic judgments" in text
    assert "session.attach_audit" in text
    assert text.index("Lock semantic judgments") < text.index("session.attach_audit")
    assert "from safety_governor.review_widgets import launch" in text

def make_bundle(path: Path):
    path.mkdir()
    queues = {
        "mechanical_review_queue.jsonl": [row(f"m-{i}") for i in range(82)],
        "repaired_review_queue.jsonl": [row(f"r-{i}", "false_completion") for i in range(8)],
        "semantic_review_queue.jsonl": [row(f"s-{i}", "topic_shift") for i in range(60)],
    }
    fingerprints = {}
    files = {}
    for name, rows in queues.items():
        target = path / name
        target.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
        files[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
        fingerprints.update({item["pair_id"]: row_fingerprint(item) for item in rows})
    manifest = {"schema_version": 1, "files": files, "immutable_fingerprints": fingerprints}
    (path / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_session_checkpoint_event_and_stale_write_protection(tmp_path):
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    session = ReviewSession(bundle, tmp_path / "session")
    revision = session.save("m-0", {
        "annotation_decision": "approved", "review_notes": "constraint omission directly verified",
        "failure_declaration": "confirmed", "isolated_constraint_omission": True,
    }, 0)
    assert revision == 1
    assert session.manifest["event_count"] == 1
    with pytest.raises(RuntimeError, match="stale"):
        session.save("m-1", {"annotation_decision": "pending"}, 0)
    resumed = ReviewSession(bundle, tmp_path / "session")
    assert resumed.decisions["m-0"]["annotation_decision"] == "approved"
    resumed.undo_last(resumed.revision)
    assert resumed.decisions["m-0"]["annotation_decision"] == "pending"


def test_semantic_audit_cannot_precede_complete_locked_review(tmp_path):
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    session = ReviewSession(bundle, tmp_path / "session")
    scores = tmp_path / "scores.jsonl"
    scores.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="locked"):
        session.attach_audit(scores, "provider", "revision", session.revision)
    with pytest.raises(ValueError, match="all 60"):
        session.lock_semantic(session.revision)
