"""Fail-closed checks that run before research activation capture.

These checks guard the transition from corpus work into model experiments.
They intentionally reject symbolic model revisions, quarantined datasets, and
unauthorized test captures.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import os
import subprocess
import sys

from .domain import Behavior, ContrastiveRecord

SYMBOLIC_REVISIONS = {"main", "master", "latest"}
SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}


def runtime_errors(config: dict) -> list[str]:
    """Check runtime constraints declared in the experiment config."""

    errors = []
    revision = str(config.get("model", {}).get("revision", ""))
    if not revision or revision.lower() in SYMBOLIC_REVISIONS:
        errors.append("model revision must be an immutable commit or tag")
    for package, expected in config.get("runtime", {}).get("exact_versions", {}).items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "not-installed"
        if actual != str(expected):
            errors.append(f"{package} version {actual}; required {expected}")
    return errors


def runtime_profile_errors(profile: dict, artifact_root: str | Path | None = None) -> list[str]:
    """Validate an explicit single-GPU runtime profile and storage target."""

    errors = []
    dtype = str(profile.get("dtype", ""))
    if dtype not in SUPPORTED_DTYPES:
        errors.append(f"runtime dtype must be one of {sorted(SUPPORTED_DTYPES)}")
    if profile.get("device") != "cuda":
        errors.append("Stage-1 runtime profile must use device=cuda")
    if profile.get("allow_quantization") is not False:
        errors.append("Stage-1 profile must explicitly disable quantization")
    if profile.get("allow_cpu_offload") is not False:
        errors.append("Stage-1 profile must explicitly disable CPU offload")
    if int(profile.get("gpu_count", 0)) != 1:
        errors.append("initial Stage-1 runtime requires gpu_count=1")
    if profile.get("require_container_image", True) and not (
        os.environ.get("VAST_IMAGE") or os.environ.get("CONTAINER_IMAGE")
    ):
        errors.append("VAST_IMAGE or CONTAINER_IMAGE must identify the qualified base image")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA is not available")
        else:
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            available_gib = properties.total_memory / (1024 ** 3)
            minimum_gib = float(profile.get("minimum_vram_gib", 0))
            if available_gib < minimum_gib:
                errors.append(
                    f"GPU VRAM {available_gib:.1f} GiB; profile requires {minimum_gib:.1f} GiB"
                )
            if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                errors.append("selected bfloat16 profile is unsupported by this GPU")
            if torch.cuda.device_count() != 1:
                errors.append(
                    f"exactly one visible GPU is required; found {torch.cuda.device_count()}"
                )
    except ImportError:
        errors.append("torch is not installed")
    root_value = artifact_root or profile.get("artifact_root")
    if not root_value:
        errors.append("runtime profile requires artifact_root")
    else:
        root = Path(root_value)
        if not root.exists():
            errors.append(f"artifact root does not exist: {root}")
        elif not os.access(root, os.W_OK):
            errors.append(f"artifact root is not writable: {root}")
    lock_value = str(profile.get("environment_lock", ""))
    if not lock_value:
        errors.append("runtime profile requires environment_lock")
    else:
        lock_path = Path(lock_value)
        if not lock_path.is_file():
            errors.append(f"qualified environment lock does not exist: {lock_path}")
        else:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            expected = sorted(line.strip() for line in lock_path.read_text(encoding="utf-8").splitlines() if line.strip())
            actual = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
            if result.returncode != 0 or actual != expected:
                errors.append("current Python environment differs from the qualified environment lock")
    return errors


def stage1_errors(
    config: dict,
    records: list[ContrastiveRecord],
    *,
    split: str,
    allow_test_capture: bool,
) -> list[str]:
    """Return Stage-1 blockers for the selected corpus split."""

    errors = runtime_errors(config)
    dataset = config.get("dataset", {})
    path = Path(str(dataset.get("path", "")))
    if path.suffix.lower() != ".jsonl":
        errors.append("Stage-1 dataset must be a JSONL contrastive corpus")
    if "quarantined" in str(path).lower():
        errors.append("quarantined corpus is not Stage-1 eligible")
    # Test capture requires explicit opt-in so final evaluation is not touched
    # during exploratory fitting or validation.
    if split == "test" and not allow_test_capture:
        errors.append("test capture requires --allow-test-capture")
    if any(not record.source_group_id for record in records):
        errors.append("every Stage-1 record requires source_group_id")
    if any(record.behavior is Behavior.HARMFUL_COMPLIANCE for record in records):
        if dataset.get("harmful_compliance_eligible") is not True:
            errors.append("harmful compliance is quarantined until explicitly eligible")
    return errors
