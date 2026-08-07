"""Automatic environment and repository fingerprints for experiment manifests."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_facts(device: str | None) -> dict:
    try:
        import torch
        torch_facts = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    except ImportError:
        torch_facts = {"torch": "not-installed", "cuda_runtime": None, "cuda_available": False}
    diff = _git("diff", "--binary")
    packages = {}
    for name in ("transformer-lens", "transformers", "torch", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": packages,
        **torch_facts,
    }