"""Immutable, locally reproducible experiment artifact helpers."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .domain import RunManifest


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def write_manifest(root: str | Path, manifest: RunManifest) -> Path:
    directory = Path(root) / manifest.run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return path
