"""Offline policy preparation must preserve provenance and phase boundaries."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.prepare_validation_v3_policy import prepare, verified_source
from scripts.validation_v3_review import summarize
from safety_governor.validation_v3 import file_sha256, verify_content_lock, write_candidate_lock


def source_run(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    candidate = {
        "candidate_id": "distributed_ridge_uniform__b0.2", "profile_id": "distributed_ridge_uniform",
        "adaptive": False, "token_mode": "generation_frontier", "total_relative_l2": 0.2,
        "sites": [{"layer": l, "weight": 0.25, "method": "balanced_ridge", "source": "v3_direction_fit"}
                  for l in (12, 16, 20, 24)],
    }
    base = yaml.safe_load(Path("configs/validation_v3_development.yaml").read_text())
    spec = {"validation_role": "development", "diagnostic_only": True,
            "generation_max_new_tokens": 768, "pair_ids": [f"p{i}" for i in range(48)],
            "model": {"revision": base["model_revision"]},
            "configurations": [{"baseline": True}, {"baseline": False, **candidate}]}
    (run / "validation_spec.json").write_text(json.dumps(spec))
    write_candidate_lock(run / "phase_lock.json", {
        "lock_type": "validation_v3_development_candidate", "confirmatory_authorized": False,
        "candidates": [candidate],
    })
    rows = [{"generation_id": "baseline" if index == 0 else "configuration_01",
             "pair_id": pair, "configuration": config,
             "generation_metadata": {"stop_reason": "stop_token", "generated_token_count": 20}}
            for index, config in enumerate(spec["configurations"]) for pair in spec["pair_ids"]]
    (run / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def test_prepare_separate_nonpromoting_768_contract(tmp_path):
    run = source_run(tmp_path)
    before = file_sha256(run / "validation_spec.json")
    result = prepare(run, tmp_path / "prepared", Path("configs/validation_v3_development.yaml"))
    contract = yaml.safe_load(Path(result["contract"]).read_text())
    lock = verify_content_lock(result["candidate_lock"], lock_type="validation_v3_development_candidate")
    assert contract["diagnostic_only"] is True
    assert contract["generation"]["max_new_tokens"] == 768
    assert contract["development_behavior"]["targeted_suppression_threshold"] == 0.70
    assert lock["confirmatory_authorized"] is False
    assert [c["token_mode"] for c in lock["candidates"]] == ["generation_frontier", "generated_span"]
    assert result["new_outputs"] == 48 and result["review_tasks"] == 144
    assert file_sha256(run / "validation_spec.json") == before
    with pytest.raises(FileExistsError):
        prepare(run, tmp_path / "prepared", Path("configs/validation_v3_development.yaml"))


def test_source_hash_and_missing_rows_fail_closed(tmp_path):
    run = source_run(tmp_path)
    hashes = {n: file_sha256(run / n) for n in ("validation_spec.json", "generations.jsonl", "phase_lock.json")}
    path = run / "generations.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[1:]))
    with pytest.raises(ValueError, match="source changed"):
        verified_source(run, hashes)
    with pytest.raises(ValueError, match="exactly cover"):
        verified_source(run)


@pytest.mark.parametrize("change", ["cap", "role", "diagnostic", "stop"])
def test_source_rejects_incompatible_archives(tmp_path, change):
    run = source_run(tmp_path)
    path = run / "validation_spec.json"
    spec = json.loads(path.read_text())
    if change == "cap":
        spec["generation_max_new_tokens"] = 384
    elif change == "role":
        spec["validation_role"] = "confirmatory"
    elif change == "diagnostic":
        spec["diagnostic_only"] = False
    else:
        rows = [json.loads(l) for l in (run / "generations.jsonl").read_text().splitlines()]
        rows[0]["generation_metadata"]["stop_reason"] = "token_limit"
        (run / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        verified_source(run)


def test_policy_diagnostic_cannot_enter_promoting_review(tmp_path):
    run = tmp_path / "policy"
    run.mkdir()
    (run / "validation_spec.json").write_text(json.dumps({
        "validation_role": "development", "diagnostic_only": True,
        "phase": "validation_v3_token_policy_diagnostic",
    }))
    with pytest.raises(ValueError, match="cannot create a final intervention lock"):
        summarize(SimpleNamespace(run=run, role="development"))
    assert sorted(p.name for p in run.iterdir()) == ["validation_spec.json"]


def test_runner_reuses_96_and_generates_only_48_with_safe_resume(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import numpy as np
    from scripts import run_validation_v3 as runner
    from safety_governor.config import load

    run = source_run(tmp_path)
    spec = json.loads((run / "validation_spec.json").read_text())
    dataset = Path("datasets/validation_v2/calibration.jsonl")
    records = [json.loads(l) for l in dataset.read_text().splitlines() if l.strip()]
    safe = sorted([r for r in records if r["polarity"] == "safe"], key=lambda r: r["pair_id"])
    assert len(safe) == 48
    spec.update(model=load("configs/llama3_8b.yaml")["model"],
                dataset_sha256=file_sha256(dataset), pair_ids=[r["pair_id"] for r in safe])
    direction = tmp_path / "directions"
    provenance = []
    for site in spec["configurations"][1]["sites"]:
        path = direction / "directions" / f"layer_{site['layer']:02d}" / "balanced_ridge.npy"
        path.parent.mkdir(parents=True)
        np.save(path, np.array([1., 0.]))
        provenance.append({**site, "path": str(path.resolve()), "sha256": file_sha256(path)})
    rows = [{"generation_id": "baseline" if i == 0 else "configuration_01",
             **{k: r[k] for k in ("pair_id", "instruction", "source_group_id", "archetype")},
             "configuration": c, "response": "archived", "sites": None if i == 0 else provenance,
             "intervention_trace": [], "generation_metadata": {
                 "stop_reason": "stop_token", "generated_token_count": 20, "stop_token_id": 128009}}
            for i, c in enumerate(spec["configurations"]) for r in safe]
    (run / "validation_spec.json").write_text(json.dumps(spec))
    (run / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    prepared = prepare(run, tmp_path / "prepared", Path("configs/validation_v3_development.yaml"))
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({"artifact_root": str(tmp_path / "artifacts"),
                                      "device": "cpu", "dtype": "float32",
                                      "hf_cache_root": str(tmp_path / "hf"), "require_clean_git": False}))
    monkeypatch.setattr(runner, "stage1_errors", lambda *a, **k: [])
    monkeypatch.setattr(runner, "runtime_profile_errors", lambda *a, **k: [])
    monkeypatch.setattr(runner, "environment_facts", lambda *a: {"git_dirty": False, "git_sha": "fixture"})
    monkeypatch.setattr(runner, "load_transformerlens_model", lambda *a: object())
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs["token_policy"])
        assert kwargs["max_new_tokens"] == 768
        kwargs["generation_metadata"].update(generated_token_count=1, stop_reason="stop_token", stop_token_id=9)
        return "new span output"
    monkeypatch.setattr(runner, "generate_with_governor", generate)
    monkeypatch.setattr(runner, "generate_unsteered", lambda *a, **k: pytest.fail("baseline must be reused"))
    argv = ["run", "development", "configs/llama3_8b.yaml", "--development-config", prepared["contract"],
            "--runtime-profile", str(runtime), "--train-run", str(tmp_path / "train"),
            "--direction-run", str(direction), "--candidate-lock", prepared["candidate_lock"],
            "--run-id", "policy", "--resume"]
    monkeypatch.setattr("sys.argv", argv)
    runner.main()
    runner.main()
    assert calls == ["generated_span"] * 48
    output = tmp_path / "artifacts" / "policy"
    generated = [json.loads(l) for l in (output / "generations.jsonl").read_text().splitlines()]
    assert len(generated) == 144
    assert sum("reuse_provenance" in r for r in generated) == 96
    assert not (output / "final_intervention_lock.json").exists()
    contract_path = Path(prepared["contract"])
    original_contract = contract_path.read_text()
    modified = yaml.safe_load(original_contract)
    modified["generation"]["max_new_tokens"] = 384
    contract_path.write_text(yaml.safe_dump(modified))
    with pytest.raises(ValueError, match="contract changed after locking"):
        runner.main()
    contract_path.write_text(original_contract)
    monkeypatch.setattr("sys.argv", [argv[0], "confirmatory", *argv[2:]])
    with pytest.raises(ValueError, match="cannot access confirmatory data"):
        runner.main()
    monkeypatch.setattr("sys.argv", argv)
    np.save(direction / "directions/layer_12/balanced_ridge.npy", np.array([0., 1.]))
    with pytest.raises(ValueError, match="vector provenance differs"):
        runner.main()
    assert len(calls) == 48
