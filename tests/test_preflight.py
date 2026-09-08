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


def test_runtime_profile_accepts_exact_qualified_environment(monkeypatch, tmp_path):
    import sys
    from types import ModuleType, SimpleNamespace
    from safety_governor import preflight

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=40 * 1024 ** 3),
        is_bf16_supported=lambda: True,
        device_count=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("VAST_IMAGE", "pinned/image@sha256:example")
    lock = tmp_path / "environment.txt"
    lock.write_text("numpy==2.0.0\n", encoding="utf-8")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="numpy==2.0.0\n"),
    )
    profile = {
        "dtype": "bfloat16",
        "device": "cuda",
        "gpu_count": 1,
        "minimum_vram_gib": 24,
        "allow_quantization": False,
        "allow_cpu_offload": False,
        "environment_lock": str(lock),
    }
    assert preflight.runtime_profile_errors(profile, tmp_path) == []


def test_runtime_profile_rejects_silent_bfloat_fallback(monkeypatch, tmp_path):
    import sys
    from types import ModuleType, SimpleNamespace
    from safety_governor import preflight

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=16 * 1024 ** 3),
        is_bf16_supported=lambda: False,
        device_count=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv("VAST_IMAGE", raising=False)
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    errors = preflight.runtime_profile_errors({
        "dtype": "bfloat16",
        "device": "cuda",
        "gpu_count": 1,
        "minimum_vram_gib": 24,
        "allow_quantization": False,
        "allow_cpu_offload": False,
        "environment_lock": str(tmp_path / "missing.txt"),
    }, tmp_path)
    assert any("unsupported" in error for error in errors)
    assert any("VRAM" in error for error in errors)
    assert any("base image" in error for error in errors)
