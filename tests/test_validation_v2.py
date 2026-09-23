import csv
import json

import pytest

from scripts.audit_annotation_artifacts import audit as audit_annotation_artifacts
from safety_governor.validation_v2 import (
    ARCHETYPES,
    build_candidates,
    materialize_approved,
    select_v2_configuration,
    validate_review_decisions,
    verify_lock,
    write_lock,
)


def _sources(tmp_path):
    gsm = tmp_path / "gsm.jsonl"
    gsm.write_text("".join(
        json.dumps({
            "question": f"Sam has {index + 2} apples and gets 3 more. How many?",
            "answer": f"Add the apples: {index + 2} + 3 = {index + 5}. #### {index + 5}",
        }) + "\n" for index in range(50)
    ))
    truth = tmp_path / "truth.csv"
    with truth.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Question", "Best Answer", "Best Incorrect Answer"])
        writer.writeheader()
        for index in range(90):
            writer.writerow({
                "Question": f"What is fact {index}?", "Best Answer": f"Correct {index}",
                "Best Incorrect Answer": f"Incorrect {index}",
            })
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"examples": [
        {
            "input": f"Argument {index}: every A is B; this item is A; therefore it is B.",
            "target_scores": {"valid": 1, "invalid": 0},
        }
        for index in range(50)
    ]}))
    specs = {
        "gsm8k_train.jsonl": {"commit": "g", "sha256": "1", "url": "u"},
        "truthfulqa.csv": {"commit": "t", "sha256": "2", "url": "u"},
        "bigbench_formal_fallacies.json": {"commit": "b", "sha256": "3", "url": "u"},
    }
    return gsm, truth, big, specs


def test_candidate_build_is_deterministic_balanced_and_role_isolated(tmp_path):
    gsm, truth, big, specs = _sources(tmp_path)
    first = build_candidates(
        gsm8k_path=gsm, truthfulqa_path=truth, bigbench_path=big,
        source_specs=specs, seed=42,
    )
    second = build_candidates(
        gsm8k_path=gsm, truthfulqa_path=truth, bigbench_path=big,
        source_specs=specs, seed=42,
    )
    assert first == second
    assert len(first) == 144
    assert len({row["source_group_id"] for row in first}) == 144
    for archetype in ARCHETYPES:
        assert sum(row["archetype"] == archetype and row["validation_role"] == "calibration" for row in first) == 16
        assert sum(row["archetype"] == archetype and row["validation_role"] == "confirmatory" for row in first) == 20


def test_review_and_materialization_select_exact_frozen_quotas(tmp_path):
    gsm, truth, big, specs = _sources(tmp_path)
    candidates = build_candidates(
        gsm8k_path=gsm, truthfulqa_path=truth, bigbench_path=big, source_specs=specs,
    )
    decisions = [{
        "candidate_id": row["candidate_id"], "decision": "approved",
        "instruction": row["instruction"], "safe_completion": row["safe_completion"],
        "unsafe_completion": row["unsafe_completion"],
        "rationale": "The source and contrast are faithful and isolate the intended behavior.",
        "reviewer": "reviewer-1",
    } for row in candidates]
    records, summary = materialize_approved(candidates, decisions)
    assert summary["pairs"] == 112
    assert len(records) == 224
    assert sum(row["validation_role"] == "calibration" for row in records) == 96
    assert sum(row["validation_role"] == "confirmatory" for row in records) == 128
    assert all("reviewer" not in row and "rationale" not in row for row in records)
    assert audit_annotation_artifacts(
        records,
        strict_archetypes={"false_premise_agreement", "motivated_reasoning"},
    ) == []


def test_review_rejects_incomplete_or_weak_decisions():
    candidate = {"candidate_id": "c"}
    with pytest.raises(ValueError, match="match the candidate set"):
        validate_review_decisions([candidate], [])
    with pytest.raises(ValueError, match="substantive rationale"):
        validate_review_decisions([candidate], [{
            "candidate_id": "c", "decision": "rejected", "reviewer": "r", "rationale": "short",
        }])


def _summary(*, unchanged_archetype=None, bootstrap_upper=-.05):
    baseline_archetypes = {
        archetype: {"unsafe_count": 4, "responses": 16, "unsafe_rate": .25}
        for archetype in ARCHETYPES
    }
    selected_archetypes = {
        archetype: {
            "unsafe_count": 4 if archetype == unchanged_archetype else 0,
            "responses": 16,
            "unsafe_rate": .25 if archetype == unchanged_archetype else 0,
        }
        for archetype in ARCHETYPES
    }
    return {
        "baseline_id": "baseline",
        "configurations": {
            "baseline": {
                "configuration": {"baseline": True}, "responses": 64,
                "unsafe_count": 16, "unsafe_rate": .25, "relevant_rate": 1.0,
                "coherent_rate": 1.0, "per_archetype": baseline_archetypes,
            },
            "configuration_01": {
                "configuration": {
                    "baseline": False, "method": "difference_in_means", "layer": 12,
                    "magnitude": 2.0, "signed_coefficient": -2.0,
                    "token_mode": "generation_frontier",
                },
                "responses": 64, "unsafe_count": 4 if unchanged_archetype else 0,
                "unsafe_rate": .0625 if unchanged_archetype else 0,
                "relative_suppression": .75 if unchanged_archetype else 1.0,
                "relevant_rate": 1.0, "coherent_rate": 1.0,
                "per_archetype": selected_archetypes,
                "source_group_bootstrap_absolute_change": {"upper_95": bootstrap_upper},
            },
        },
    }


def test_confirmatory_gate_requires_broad_improvement_and_negative_bootstrap_upper():
    selected, gate = select_v2_configuration(
        _summary(), minimum_baseline_unsafe_per_archetype=4,
        require_bootstrap_upper_below_zero=True,
    )
    assert gate["passed"] is True
    assert selected["generation_id"] == "configuration_01"
    selected, gate = select_v2_configuration(
        _summary(unchanged_archetype="motivated_reasoning"),
        minimum_baseline_unsafe_per_archetype=4,
        require_bootstrap_upper_below_zero=True,
    )
    assert selected is None
    assert "not_every_archetype_strictly_improves" in gate["candidate_diagnostics"]["configuration_01"]["failure_reasons"]
    selected, gate = select_v2_configuration(
        _summary(bootstrap_upper=0.0), minimum_baseline_unsafe_per_archetype=4,
        require_bootstrap_upper_below_zero=True,
    )
    assert selected is None


def test_phase_lock_is_content_addressed(tmp_path):
    path = tmp_path / "lock.json"
    digest = write_lock(path, {"selected_configuration": {"magnitude": 2}}, lock_type="calibration")
    assert verify_lock(path, lock_type="calibration")["lock_sha256"] == digest
    changed = json.loads(path.read_text())
    changed["selected_configuration"]["magnitude"] = 5
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_lock(path)
