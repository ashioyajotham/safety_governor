import json

import pytest

from scripts.analyze_validation_v2 import analyze, validate_metrics


def _metrics():
    configurations = {}
    for index in range(7):
        generation_id = "baseline" if index == 0 else f"configuration_{index:02d}"
        configurations[generation_id] = {
            "configuration": {"baseline": True} if index == 0 else {
                "baseline": False,
                "magnitude": float(index),
                "token_mode": "generation_frontier",
            },
            "responses": 48,
            "unsafe_count": 39 - (3 if index == 2 else 0),
            "unsafe_rate": (39 - (3 if index == 2 else 0)) / 48,
            "relative_suppression": 3 / 39 if index == 2 else 0.0,
            "paired_change": {"improved": 5, "unchanged": 41, "worsened": 2},
            "source_group_bootstrap_absolute_change": {
                "lower_95": -0.18,
                "mean": -0.06,
                "upper_95": 0.04,
            },
            "per_archetype": {},
        }
    return {
        "schema_version": 3,
        "validation_role": "calibration",
        "diagnostic": "validation_v2_calibration_gate_failed",
        "review_tasks_sha256": "a" * 64,
        "review_mapping_sha256": "b" * 64,
        "review_decisions_sha256": "c" * 64,
        "selected_configuration": None,
        "best_observed_configuration": configurations["configuration_02"] | {
            "generation_id": "configuration_02"
        },
        "behavioral_gate": {
            "passed": False,
            "targeted_suppression_threshold": 0.7,
        },
        "summary": {"configurations": configurations},
    }


def test_analyze_writes_reproducible_negative_result_artifacts(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_metrics()))
    output = tmp_path / "result"
    result = analyze(metrics, output)
    assert result["result"] == "negative_fail_closed"
    assert result["selected_configuration"] is None
    assert {path.name for path in output.iterdir()} == {
        "REPORT.md",
        "calibration_overview.svg",
        "configuration_metrics.csv",
        "diagnostic_summary.json",
    }
    assert "failed closed" in (output / "REPORT.md").read_text()
    assert "&gt;70%" in (output / "calibration_overview.svg").read_text()


def test_analyze_rejects_a_selected_configuration():
    metrics = _metrics()
    metrics["selected_configuration"] = {"generation_id": "configuration_02"}
    with pytest.raises(ValueError, match="must not select"):
        validate_metrics(metrics)
