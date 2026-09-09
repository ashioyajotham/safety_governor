"""Regression tests for the reproducible Vast.ai dependency bootstrap."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_torch_is_exactly_pinned_for_stage1() -> None:
    """Prevent an unconstrained future CUDA-major Torch wheel from resolving."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    torch_requirements = [line for line in requirements if line.startswith("torch")]
    assert torch_requirements == ["torch==2.8.0"]


def test_vast_bootstrap_uses_clean_cu128_environment() -> None:
    """Require a clean venv and the official CUDA 12.8 PyTorch wheel index."""
    bootstrap = (ROOT / "scripts" / "bootstrap_vast.sh").read_text(encoding="utf-8")
    assert '-m venv --clear "${ENVIRONMENT_ROOT}"' in bootstrap
    assert 'TORCH_VERSION="2.8.0"' in bootstrap
    assert 'PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"' in bootstrap
    assert '"torch==${TORCH_VERSION}"' in bootstrap
