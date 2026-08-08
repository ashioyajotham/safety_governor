from safety_governor.domain import Behavior, ContrastiveRecord, Polarity
from safety_governor.preflight import stage1_errors


def record(behavior=Behavior.INSTRUCTION_NONCOMPLIANCE):
    return ContrastiveRecord(
        "p1", behavior, Polarity.SAFE, "en", "", "expected", "source", "approved",
        split="train", instruction="instruction", completion="completion",
        source_group_id="group-1",
    )


def config(**dataset_updates):
    dataset = {"path": "datasets/frozen/data.jsonl", **dataset_updates}
    return {"model": {"revision": "immutable-sha"}, "dataset": dataset}


def test_validation_capture_is_allowed_but_test_requires_authorization():
    assert stage1_errors(config(), [record()], split="validation", allow_test_capture=False) == []
    errors = stage1_errors(config(), [record()], split="test", allow_test_capture=False)
    assert any("allow-test-capture" in error for error in errors)
    assert stage1_errors(config(), [record()], split="test", allow_test_capture=True) == []


def test_harmful_compliance_is_fail_closed_until_eligible():
    errors = stage1_errors(
        config(), [record(Behavior.HARMFUL_COMPLIANCE)], split="train",
        allow_test_capture=False,
    )
    assert any("harmful compliance" in error for error in errors)
    assert stage1_errors(
        config(harmful_compliance_eligible=True),
        [record(Behavior.HARMFUL_COMPLIANCE)], split="train",
        allow_test_capture=False,
    ) == []


def test_quarantined_path_is_rejected_even_for_non_harmful_rows():
    errors = stage1_errors(
        config(path="data/working/quarantined.jsonl"), [record()], split="train",
        allow_test_capture=False,
    )
    assert any("quarantined corpus" in error for error in errors)

def test_runtime_version_mismatch_fails_closed(monkeypatch):
    from safety_governor import preflight

    monkeypatch.setattr(preflight.metadata, "version", lambda _name: "3.2.1")
    configured = config()
    configured["runtime"] = {"exact_versions": {"transformer-lens": "3.1.0"}}
    errors = preflight.runtime_errors(configured)
    assert any("required 3.1.0" in error for error in errors)