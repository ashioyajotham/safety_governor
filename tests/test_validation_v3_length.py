import json

import pytest

from scripts.diagnose_validation_v3_length import diagnose


def _write_run(tmp_path, *, stop_reason="token_limit", token_count=384):
    run = tmp_path / "run"
    run.mkdir()
    spec = {
        "phase": "validation_v3_post_review_length_diagnostic",
        "diagnostic_only": True,
        "validation_role": "development",
        "generation_max_new_tokens": 384,
        "pair_ids": ["p1"],
        "configurations": [
            {"baseline": True},
            {"baseline": False, "candidate_id": "ridge"},
        ],
    }
    (run / "validation_spec.json").write_text(json.dumps(spec))
    rows = [
        {"generation_id": "baseline", "pair_id": "p1", "generation_metadata": {
            "stop_reason": "stop_token", "generated_token_count": 15,
        }},
        {"generation_id": "configuration_01", "pair_id": "p1", "generation_metadata": {
            "stop_reason": stop_reason, "generated_token_count": token_count,
        }},
    ]
    (run / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run


def test_length_diagnostic_reports_stops_without_writes(tmp_path):
    run = _write_run(tmp_path)
    result = diagnose(run)
    assert result["promotion_authorized"] is False
    assert result["by_generation"]["baseline"]["stop_reasons"] == {"stop_token": 1}
    assert result["by_generation"]["configuration_01"]["stop_reasons"] == {"token_limit": 1}
    assert result["by_generation"]["configuration_01"]["candidate_id"] == "ridge"
    assert not (run / "final_intervention_lock.json").exists()


def test_length_diagnostic_rejects_inconsistent_stop_count(tmp_path):
    run = _write_run(tmp_path, token_count=128)
    with pytest.raises(ValueError, match="contradicts stop reason"):
        diagnose(run)
