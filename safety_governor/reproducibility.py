"""Automatic environment and repository fingerprints for experiment manifests."""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from importlib import metadata


def _git(*args: str) -> str:
    """Run a read-only git command and degrade to 'unknown' outside a repo."""

    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_facts(device: str | None) -> dict:
    """Collect runtime facts stored in experiment manifests."""

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        torch_facts = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
        }
        if cuda_available:
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            torch_facts.update({
                "gpu_name": properties.name,
                "gpu_total_vram_bytes": properties.total_memory,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
                "gpu_count": torch.cuda.device_count(),
            })
    except ImportError:
        torch_facts = {"torch": "not-installed", "cuda_runtime": None, "cuda_available": False}
    # Hash the current diff rather than embedding it directly in manifests.
    diff = _git("diff", "--binary")
    packages = {}
    for name in ("transformer-lens", "transformers", "torch", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    try:
        nvidia = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        nvidia_facts = nvidia.stdout.strip() if nvidia.returncode == 0 else "unavailable"
    except OSError:
        nvidia_facts = "unavailable"
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "hostname": platform.node(),
        "container_image": os.environ.get("VAST_IMAGE") or os.environ.get("CONTAINER_IMAGE") or "unknown",
        "vast_instance_id": os.environ.get("VAST_CONTAINERLABEL", "unknown"),
        "nvidia_smi": nvidia_facts,
        "packages": packages,
        **torch_facts,
    }
