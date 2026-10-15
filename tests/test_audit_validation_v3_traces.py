import json

from scripts.audit_validation_v3_traces import audit_run


def test_audit_detects_missing_hook_and_budget_mismatch(tmp_path):
    root = tmp_path / "generation_shards"
    sites = [{"layer": layer, "weight": 0.25} for layer in (12, 16, 20, 24)]
    config = {"baseline": False, "adaptive": False,
              "token_mode": "generation_frontier", "total_relative_l2": 0.2,
              "sites": sites}
    traces = [dict(layer=site["layer"], position=7 + step, activation_l2=10,
                   projection=0, gate=1, coefficient=0.5, relative_delta=0.05)
              for step in range(2) for site in sites]
    for group, cfg, trace in (
        ("baseline", {"baseline": True}, []),
        ("configuration_01", config, traces),
    ):
        directory = root / group
        directory.mkdir(parents=True)
        (directory / "pair-1.json").write_text(json.dumps({
            "generation_id": group, "pair_id": "pair-1", "configuration": cfg,
            "generation_metadata": {"generated_token_count": 1, "stop_reason": "stop_token"},
            "intervention_trace": trace,
        }))
    assert audit_run(tmp_path)["valid"] is True
    path = root / "configuration_01" / "pair-1.json"
    row = json.loads(path.read_text())
    row["intervention_trace"].pop()
    path.write_text(json.dumps(row))
    assert audit_run(tmp_path)["valid"] is False
    row["intervention_trace"] = traces
    row["intervention_trace"][0]["coefficient"] = 1.0
    path.write_text(json.dumps(row))
    assert audit_run(tmp_path)["valid"] is False
