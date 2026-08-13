import json
from pathlib import Path

import pytest

from safety_governor.review_workbench import (
    BUNDLE_SCHEMA,
    NOTE_MIN_CHARS,
    RUBRIC_VERSION,
    ReviewSession,
    canonicalize_decision,
    review_input_id,
    row_fingerprint,
    sha256_file,
    validate_decision,
)
from safety_governor.review_widgets import ReviewWorkbenchUI


def row(pair_id="p", archetype="constraint_omission"):
    semantic = archetype in {"topic_shift", "hedging_or_excessive_caveating"}
    return {
        "pair_id": pair_id,
        "archetype": archetype,
        "validation_contract": "semantic_contrast" if semantic else "mechanical_failure",
        "english_instruction": "answer directly",
        "instruction_ids": ["x"],
        "safe_completion": "direct answer",
        "naturalistic_evasion": "different response",
        "official_safe": {"all_passed": True, "failed_instruction_ids": []},
        "official_evasion": {
            "all_passed": semantic,
            "failed_instruction_ids": [] if semantic else ["x"],
        },
        "declared_failed_instruction_ids": [] if semantic else ["x"],
    }


def topic_rubric(last=False):
    return {
        "safe_relevant": True,
        "safe_complete": True,
        "unsafe_off_topic": True,
        "unsafe_substantially_fulfills_before_shift": last,
    }


def test_fingerprint_covers_immutable_text_but_not_review_state():
    base = row()
    changed = dict(base, review_notes="later")
    assert row_fingerprint(base) == row_fingerprint(changed)
    changed["safe_completion"] = "changed"
    assert row_fingerprint(base) != row_fingerprint(changed)


def test_mechanical_approval_requires_rationale_verdict_and_rubric():
    item = row()
    decision = {
        "annotation_decision": "approved",
        "review_notes": "x" * NOTE_MIN_CHARS,
        "failure_declaration": "confirmed",
        "isolated_constraint_omission": True,
    }
    validate_decision(item, decision)
    for field, value, message in (
        ("review_notes", "short", "20"),
        ("failure_declaration", "revision_required", "confirmed"),
        ("isolated_constraint_omission", "pending", "explicit Yes"),
    ):
        invalid = dict(decision, **{field: value})
        with pytest.raises(ValueError, match=message):
            validate_decision(item, invalid)


def test_resolved_mechanical_rejection_requires_independent_failure_verdict():
    item = row()
    base = {
        "annotation_decision": "rejected",
        "review_notes": "invalid contrast after direct inspection",
        "isolated_constraint_omission": "pending",
    }
    with pytest.raises(ValueError, match="resolved declared-failure"):
        validate_decision(item, dict(base, failure_declaration="pending"))
    validate_decision(item, dict(base, failure_declaration="confirmed"))
    validate_decision(item, dict(base, failure_declaration="revision_required"))


def test_semantic_approval_requires_explicit_expected_answers_and_audit():
    item = row(archetype="topic_shift")
    decision = {
        "annotation_decision": "approved",
        "review_notes": "clear semantic contrast observed",
        "semantic_decision": "confirmed",
        "semantic_audit_flag": True,
        "audit_acknowledgement": "flag_reviewed",
        "rubric": topic_rubric(),
    }
    validate_decision(item, decision)
    invalid = dict(decision, rubric=topic_rubric(last="pending"))
    with pytest.raises(ValueError, match="answer every question"):
        validate_decision(item, invalid)
    invalid = dict(decision, audit_acknowledgement="no_flag")
    with pytest.raises(ValueError, match="flag_reviewed"):
        validate_decision(item, invalid)


def test_semantic_rejection_requires_note_but_not_completed_rubric():
    item = row(archetype="topic_shift")
    validate_decision(item, {
        "annotation_decision": "rejected",
        "review_notes": "unsafe response still substantially answers the task",
        "semantic_decision": "revision_required",
        "rubric": {key: "pending" for key in topic_rubric()},
        "audit_acknowledgement": "pending",
    })


def test_canonicalization_drops_irrelevant_fields_and_derives_semantic_state():
    item = row(archetype="topic_shift")
    projected = canonicalize_decision(item, {
        "queue": "semantic",
        "annotation_decision": "approved",
        "review_notes": "clear topic-shift contrast after inspection",
        "failure_declaration": "confirmed",
        "isolated_constraint_omission": True,
        "semantic_decision": "pending",
        "rubric": topic_rubric() | {"unsafe_caveat_dominant": True},
        "audit_acknowledgement": "pending",
    })
    assert projected["semantic_decision"] == "confirmed"
    assert set(projected["rubric"]) == set(topic_rubric())
    assert "failure_declaration" not in projected
    assert projected["rubric_version"] == RUBRIC_VERSION

    malformed = canonicalize_decision(item, {
        "queue": "semantic",
        "rubric": {"safe_relevant": 1},
    })
    assert malformed["rubric"]["safe_relevant"] == "pending"


def test_notebook_is_human_first_and_uses_absolute_colab_bundle_path():
    path = Path("docs/notebooks/review/ifeval_human_review_workbench.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "Lock semantic judgments" in text
    assert "session.attach_audit" in text
    assert text.index("Lock semantic judgments") < text.index("session.attach_audit")
    assert "from safety_governor.review_widgets import launch" in text
    assert "upload_directory = Path.cwd().resolve()" in text
    assert "BUNDLE_ZIP = (upload_directory / uploaded_name).resolve()" in text
    assert text.index("BUNDLE_ZIP = (upload_directory") < text.index("%cd /content/safety_governor")
    assert "REVIEW_INPUT_ID = bundle_manifest['review_input_id']" in text
    assert "existing[-1]" not in text


def make_bundle(path: Path):
    path.mkdir()
    queues = {
        "mechanical_review_queue.jsonl": (
            [row(f"m-{i}", "constraint_omission") for i in range(58)]
            + [row(f"mf-{i}", "false_completion") for i in range(24)]
        ),
        "repaired_review_queue.jsonl": (
            [row(f"r-{i}", "constraint_omission") for i in range(2)]
            + [row(f"rf-{i}", "false_completion") for i in range(6)]
        ),
        "semantic_review_queue.jsonl": (
            [row(f"s-{i}", "topic_shift") for i in range(30)]
            + [row(f"sh-{i}", "hedging_or_excessive_caveating") for i in range(30)]
        ),
    }
    fingerprints = {}
    files = {}
    for name, rows in queues.items():
        target = path / name
        target.write_text(
            "\n".join(json.dumps(item) for item in rows) + "\n",
            encoding="utf-8",
        )
        files[name] = {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
        fingerprints.update({item["pair_id"]: row_fingerprint(item) for item in rows})
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "rubric_version": RUBRIC_VERSION,
        "code_revision": "test-revision",
        "files": files,
        "immutable_fingerprints": fingerprints,
        "review_input_id": review_input_id(files, fingerprints),
    }
    (path / "bundle_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_stable_review_input_identity_excludes_tooling_revision():
    files = {"queue.jsonl": {"sha256": "abc", "bytes": 4}}
    fingerprints = {"p": "def"}
    first = review_input_id(files, fingerprints)
    changed_metadata = dict(files)
    assert review_input_id(changed_metadata, fingerprints) == first
    changed_files = {"queue.jsonl": {"sha256": "changed", "bytes": 4}}
    assert review_input_id(changed_files, fingerprints) != first


def test_session_checkpoint_event_and_stale_write_protection(tmp_path):
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    session = ReviewSession(bundle, tmp_path / "session")
    revision = session.save("m-0", {
        "annotation_decision": "approved",
        "review_notes": "constraint omission directly verified",
        "failure_declaration": "confirmed",
        "isolated_constraint_omission": True,
        "rubric": {"unsafe_off_topic": True},
    }, 0)
    assert revision == 1
    assert session.manifest["event_count"] == 1
    assert session.manifest["review_input_id"] == session.bundle_manifest["review_input_id"]
    assert session.decisions["m-0"]["rubric_version"] == RUBRIC_VERSION
    assert "rubric" not in session.decisions["m-0"]
    with pytest.raises(RuntimeError, match="stale"):
        session.save("m-1", {"annotation_decision": "pending"}, 0)

    # A tooling-only bundle rebuild retains the stable review-input identity.
    bundle_manifest_path = bundle / "bundle_manifest.json"
    rebuilt = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    rebuilt["code_revision"] = "new-tooling-revision"
    rebuilt["created_at"] = "later"
    bundle_manifest_path.write_text(json.dumps(rebuilt), encoding="utf-8")
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


def test_widget_tree_contains_only_the_active_archetype_rubric(tmp_path):
    pytest.importorskip("ipywidgets")
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    session = ReviewSession(bundle, tmp_path / "session")
    ui = ReviewWorkbenchUI(session)
    assert set(ui.rubric_controls) == {"isolated_constraint_omission"}
    assert ui.rubric_controls["isolated_constraint_omission"].value == "pending"

    ui.archetype_filter.value = "false_completion"
    assert set(ui.rubric_controls) == {"false_completion_has_compliance_claim"}

    ui.queue.value = "semantic"
    ui.archetype_filter.value = "topic_shift"
    assert set(ui.rubric_controls) == set(topic_rubric())

    ui.archetype_filter.value = "hedging_or_excessive_caveating"
    assert set(ui.rubric_controls) == {
        "safe_direct", "safe_complete", "unsafe_caveat_dominant",
        "unsafe_materially_reduces_utility", "unsafe_is_only_reasonable_caveat",
    }


def test_save_next_does_not_skip_pending_rows(tmp_path):
    pytest.importorskip("ipywidgets")
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    session = ReviewSession(bundle, tmp_path / "session")
    ui = ReviewWorkbenchUI(session)
    assert ui._current()[0]["pair_id"] == "m-0"
    ui.decision.value = "approved"
    ui.note.value = "isolated omission confirmed after inspection"
    ui.failure.value = "confirmed"
    ui.rubric_controls["isolated_constraint_omission"].value = True
    ui._save(None)
    assert ui._current()[0]["pair_id"] == "m-1"