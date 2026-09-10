import json

import numpy as np
import pytest

from safety_governor.validation import (
    audit_validation_run,
    choose_behavior_configuration,
    evaluate_fixed_direction,
    make_blinded_tasks,
    rank_auc,
    select_representation_candidates,
    summarize_behavior_review,
    verify_selection_lock,
    write_selection_lock,
)


def test_validation_audit_fails_closed_on_incomplete_run(tmp_path):
    with pytest.raises(ValueError, match="missing required artifacts"):
        audit_validation_run(tmp_path)


def test_rank_auc_handles_order_and_ties():
    assert rank_auc([0, 0, 1, 1], [0, 1, 2, 3]) == 1.0
    assert rank_auc([0, 1], [1, 1]) == .5


def test_fixed_direction_reports_group_bootstrap_and_archetypes():
    safe = np.zeros((4, 2))
    unsafe = np.array([[1., 0.], [2., 0.], [1., 1.], [2., 1.]])
    result = evaluate_fixed_direction(
        safe, unsafe, np.array([1., 0.]),
        ["a", "a", "b", "b"], ["g1", "g1", "g2", "g3"],
        bootstrap_samples=20,
    )
    assert result["auc"] == 1.0
    assert result["paired_accuracy"] == 1.0
    assert set(result["per_archetype"]) == {"a", "b"}
    assert result["bootstrap"]["auc"]["lower_95"] == 1.0
    assert result["direction_convention"] == "safe_to_unsafe"


def test_fixed_direction_rejects_non_unit_or_misaligned_inputs():
    with pytest.raises(ValueError, match="unit-normalized"):
        evaluate_fixed_direction(
            np.zeros((1, 2)), np.ones((1, 2)), np.array([2., 0.]),
            ["a"], ["g"], bootstrap_samples=2,
        )


def _candidate(method, layer, auc, lower, margins):
    return {
        "method": method,
        "layer": layer,
        "metrics": {
            "bootstrap": {"auc": {"lower_95": lower}},
            "per_archetype": {
                key: {"auc": auc, "mean_margin": margin}
                for key, margin in margins.items()
            },
        },
    }


def test_representation_selection_requires_all_archetypes_positive():
    rows = [
        _candidate("probe", 28, .9, .8, {"a": 1, "b": 1}),
        _candidate("difference_in_means", 12, .9, .8, {"a": 1, "b": 1}),
        _candidate("probe", 24, 1.0, .9, {"a": 1, "b": -1}),
    ]
    selected = select_representation_candidates(rows, limit=2)
    assert [(row["method"], row["layer"]) for row in selected] == [
        ("difference_in_means", 12), ("probe", 28)
    ]


def _generations():
    rows = []
    for pair_id in ("p1", "p2"):
        rows.extend([
            {
                "generation_id": "baseline", "pair_id": pair_id,
                "archetype": "a", "instruction": f"prompt {pair_id}",
                "response": f"baseline {pair_id}", "configuration": {"baseline": True},
            },
            {
                "generation_id": "steered", "pair_id": pair_id,
                "archetype": "a", "instruction": f"prompt {pair_id}",
                "response": f"steered {pair_id}",
                "configuration": {
                    "baseline": False, "method": "difference_in_means", "layer": 12,
                    "magnitude": 1.0, "signed_coefficient": -1.0,
                    "token_mode": "assistant_boundary",
                },
            },
        ])
    return rows


def test_blinded_tasks_hide_configuration_and_are_deterministic():
    tasks, mapping = make_blinded_tasks(_generations(), seed=4)
    again, _ = make_blinded_tasks(_generations(), seed=4)
    assert tasks == again
    assert all("configuration" not in row and "generation_id" not in row for row in tasks)
    assert all("configuration" in row for row in mapping)


def test_behavior_summary_detects_suppression_and_selects_lower_tax():
    tasks, mapping = make_blinded_tasks(_generations())
    generation_by_task = {row["task_id"]: row["generation_id"] for row in mapping}
    decisions = []
    for task in tasks:
        safe = generation_by_task[task["task_id"]] == "steered"
        decisions.append({
            "task_id": task["task_id"],
            "target_safe": "yes" if safe else "no",
            "relevant": "yes", "coherent": "yes",
            "rationale": "The response preserves the requested task while showing the target behavior clearly.",
            "reviewer": "reviewer-1",
        })
    summary = summarize_behavior_review(tasks, mapping, decisions)
    assert summary["configurations"]["steered"]["relative_suppression"] == 1.0
    selected = choose_behavior_configuration(summary)
    assert selected["configuration"]["signed_coefficient"] == -1.0


def test_behavior_summary_refuses_claim_when_baseline_has_no_headroom():
    tasks, mapping = make_blinded_tasks(_generations())
    decisions = [{
        "task_id": task["task_id"], "target_safe": "yes", "relevant": "yes",
        "coherent": "yes", "rationale": "", "reviewer": "reviewer-1",
    } for task in tasks]
    summary = summarize_behavior_review(tasks, mapping, decisions)
    assert choose_behavior_configuration(summary) is None
    assert summary["configurations"]["steered"]["relative_suppression"] is None


def test_review_requires_exact_task_set_and_reason_for_negative():
    tasks, mapping = make_blinded_tasks(_generations())
    with pytest.raises(ValueError, match="match the blinded task set"):
        summarize_behavior_review(tasks, mapping, [])
    decisions = [{
        "task_id": task["task_id"], "target_safe": "no", "relevant": "yes",
        "coherent": "yes", "rationale": "short", "reviewer": "reviewer-1",
    } for task in tasks]
    with pytest.raises(ValueError, match="substantive rationale"):
        summarize_behavior_review(tasks, mapping, decisions)


def test_selection_lock_is_content_addressed_and_negative(tmp_path):
    path = tmp_path / "selection_lock.json"
    payload = {
        "parent_train_sha256": "a", "validation_manifest_sha256": "b",
        "human_review_sha256": "c", "dataset_sha256": "d",
        "model_revision": "model-commit", "method": "difference_in_means",
        "layer": 12, "magnitude": 2, "signed_coefficient": -2,
        "token_mode": "generation_frontier", "metric_contract": {},
        "control_tax": {},
    }
    digest = write_selection_lock(path, payload)
    assert verify_selection_lock(path)["selection_sha256"] == digest
    changed = json.loads(path.read_text())
    changed["signed_coefficient"] = 2
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_selection_lock(path)
