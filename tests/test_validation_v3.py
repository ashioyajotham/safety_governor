import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.validation_v3_review import summarize
from safety_governor.validation_v3 import (
    calibrate_projection_gate,
    expand_screen_candidates,
    select_behavior_candidate,
    validate_development_contract,
    verify_content_lock,
    write_candidate_lock,
    write_final_lock,
)


def _contract(tmp_path):
    confirmatory = tmp_path / "confirmatory.jsonl"
    confirmatory.write_text("sealed\n")
    digest = hashlib.sha256(confirmatory.read_bytes()).hexdigest()
    contract = {
        "schema_version": 1,
        "phase": "exploratory_train_and_spent_calibration_only",
        "sealed_confirmatory": {
            "sha256": digest,
            "access": "forbidden_until_v3_contract_is_frozen",
        },
        "teacher_forced_screen": {
            "total_relative_l2_budgets": [0.05, 0.1, 0.2],
            "profiles": [{
                "id": "late",
                "sites": [
                    {"layer": 12, "method": "ridge", "source": "fit", "weight": 0.25},
                    {"layer": 24, "method": "ridge", "source": "fit", "weight": 0.75},
                ],
            }],
        },
        "development_behavior": {
            "maximum_conditions_including_baseline": 4,
            "targeted_suppression_threshold": 0.70,
            "strict_archetype_improvement": True,
        },
        "confirmatory": {
            "conditions": ["baseline", "frozen_v3_intervention"],
            "control_tax_authorized_only_after_pass": True,
        },
    }
    return contract, confirmatory


def test_contract_preserves_holdout_and_expands_fixed_matrix(tmp_path):
    contract, confirmatory = _contract(tmp_path)
    validate_development_contract(contract, confirmatory)
    candidates = expand_screen_candidates(contract)
    assert [row["candidate_id"] for row in candidates] == [
        "late__b0.05", "late__b0.1", "late__b0.2",
    ]
    assert all(row["token_mode"] == "generation_frontier" for row in candidates)


def test_projection_gate_requires_grouped_lower_bound_and_every_archetype():
    safe = [np.array([-2., -1.5])] * 6
    unsafe = [np.array([1.5, 2.])] * 6
    groups = [f"g{i}" for i in range(6)]
    archetypes = ["a"] * 3 + ["b"] * 3
    gate, diagnostics = calibrate_projection_gate(
        safe, unsafe, groups, archetypes, bootstrap_samples=50, seed=3
    )
    assert diagnostics["qualified"] is True
    assert diagnostics["bootstrap_auc"]["lower_95"] == 1.0
    assert -1.5 < gate.threshold < 1.5


def test_development_lock_cannot_authorize_confirmatory(tmp_path):
    path = tmp_path / "candidate_lock.json"
    payload = {
        "lock_type": "validation_v3_development_candidate",
        "confirmatory_authorized": False,
        "candidate": {"id": "late__b0.1"},
    }
    digest = write_candidate_lock(path, payload)
    loaded = yaml.safe_load(path.read_text())
    assert loaded["lock_sha256"] == digest
    assert verify_content_lock(
        path, lock_type="validation_v3_development_candidate"
    )["lock_sha256"] == digest


def test_behavior_gate_requires_strict_broad_improvement():
    baseline_archetypes = {
        name: {"unsafe_count": 5} for name in ("a", "b")
    }
    intervention_archetypes = {
        name: {"unsafe_count": 1} for name in ("a", "b")
    }
    summary = {
        "baseline_id": "baseline",
        "zero_headroom_archetypes": [],
        "configurations": {
            "baseline": {
                "configuration": {"baseline": True}, "unsafe_count": 10,
                "unsafe_rate": 1.0, "relevant_rate": 1.0, "coherent_rate": 1.0,
                "per_archetype": baseline_archetypes,
            },
            "configuration_01": {
                "configuration": {"baseline": False, "total_relative_l2": 0.1},
                "unsafe_count": 2, "unsafe_rate": 0.2,
                "relative_suppression": 0.8, "relevant_rate": 1.0, "coherent_rate": 1.0,
                "per_archetype": intervention_archetypes,
                "source_group_bootstrap_absolute_change": {"upper_95": -0.2},
            },
        },
    }
    selected, gate = select_behavior_candidate(summary)
    assert gate["passed"] is True
    assert selected["generation_id"] == "configuration_01"


def test_final_lock_is_content_addressed(tmp_path):
    path = tmp_path / "final.json"
    digest = write_final_lock(path, {
        "lock_type": "validation_v3_final_intervention",
        "confirmatory_authorized": True,
        "selected_candidate": {"candidate_id": "late__b0.1"},
    })
    verified = verify_content_lock(path, lock_type="validation_v3_final_intervention")
    assert verified["lock_sha256"] == digest


def test_384_token_diagnostic_preserves_original_candidate_matrix():
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / "configs/validation_v3_development.yaml").read_text())
    diagnostic = yaml.safe_load((root / "configs/validation_v3_length384_diagnostic.yaml").read_text())
    assert diagnostic.pop("diagnostic_only") is True
    assert diagnostic["generation"]["max_new_tokens"] == 384
    diagnostic["generation"]["max_new_tokens"] = 128
    assert diagnostic == original


def test_diagnostic_review_cannot_promote_even_with_development_role(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "validation_spec.json").write_text(json.dumps({
        "validation_role": "development", "diagnostic_only": True,
    }))
    args = SimpleNamespace(run=run, role="development", decisions=tmp_path / "decisions.jsonl",
                           development_config=tmp_path / "contract.yaml")
    with pytest.raises(ValueError, match="cannot create a final intervention lock"):
        summarize(args)
    assert not (run / "review_decisions.jsonl").exists()
    assert not (run / "final_intervention_lock.json").exists()
