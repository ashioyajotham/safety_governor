"""Resumable, provider-neutral primitives for Stage-1 activation capture.

The public helpers in this module deliberately separate orchestration state
from the heavyweight model process. Batch shards are immutable once written;
resumption is allowed only when the complete run specification still hashes to
the same value.
"""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


def load_runtime_profile(path: str | Path) -> dict:
    """Load the small execution-only profile used by GPU runners."""

    with Path(path).open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict):
        raise ValueError("runtime profile root must be a mapping")
    return profile


def parse_layers(value: str | Iterable[int]) -> list[int]:
    """Parse a comma-separated layer list and reject ambiguous ordering."""

    layers = [int(item.strip()) for item in value.split(",")] if isinstance(value, str) else list(value)
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("layers must contain non-negative integers")
    if layers != sorted(set(layers)):
        raise ValueError("layers must be unique and sorted")
    return layers


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible object using a stable serialization."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: object) -> None:
    """Atomically replace a JSON control file in its destination directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def initialize_run(directory: str | Path, spec: dict, *, resume: bool) -> str:
    """Create or validate an immutable run specification and return its hash."""

    root = Path(directory)
    spec_hash = canonical_sha256(spec)
    path = root / "run_spec.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"run already exists; pass --resume: {root}")
        if canonical_sha256(existing) != spec_hash:
            raise ValueError("existing run specification does not match requested run")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"non-empty run directory has no run_spec.json: {root}")
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, spec)
    return spec_hash


def save_capture_shard(
    path: str | Path,
    *,
    spec_hash: str,
    batch_index: int,
    pair_ids: list[str],
    source_group_ids: list[str],
    safe_by_layer: dict[int, np.ndarray],
    unsafe_by_layer: dict[int, np.ndarray],
) -> None:
    """Atomically save one aligned safe/unsafe batch for multiple layers."""

    layers = sorted(safe_by_layer)
    if layers != sorted(unsafe_by_layer):
        raise ValueError("safe and unsafe shard layers are not aligned")
    expected_rows = len(pair_ids)
    if len(source_group_ids) != expected_rows:
        raise ValueError("source-group IDs are not aligned with pair IDs")
    arrays: dict[str, np.ndarray] = {}
    for layer in layers:
        safe, unsafe = safe_by_layer[layer], unsafe_by_layer[layer]
        if safe.ndim != 2 or unsafe.shape != safe.shape or safe.shape[0] != expected_rows:
            raise ValueError(f"invalid aligned activation shapes at layer {layer}")
        arrays[f"safe_{layer}"] = safe
        arrays[f"unsafe_{layer}"] = unsafe
    metadata = {
        "spec_hash": spec_hash,
        "batch_index": batch_index,
        "pair_ids": pair_ids,
        "source_group_ids": source_group_ids,
        "layers": layers,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".npz", dir=target.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, metadata=np.array(json.dumps(metadata)), **arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_capture_shard(path: str | Path) -> tuple[dict, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Read a shard without pickle and reconstruct its layer dictionaries."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        safe = {layer: archive[f"safe_{layer}"].copy() for layer in metadata["layers"]}
        unsafe = {layer: archive[f"unsafe_{layer}"].copy() for layer in metadata["layers"]}
    return metadata, safe, unsafe


def validate_shard(
    path: str | Path,
    *,
    spec_hash: str,
    batch_index: int,
    pair_ids: list[str],
    source_group_ids: list[str],
    layers: list[int],
) -> bool:
    """Return whether an existing shard is safe to reuse during resumption."""

    try:
        metadata, safe, unsafe = load_capture_shard(path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "spec_hash": spec_hash,
        "batch_index": batch_index,
        "pair_ids": pair_ids,
        "source_group_ids": source_group_ids,
        "layers": layers,
    }
    if metadata != expected:
        return False
    return all(
        safe[layer].ndim == 2
        and unsafe[layer].shape == safe[layer].shape
        and safe[layer].shape[0] == len(pair_ids)
        and np.isfinite(safe[layer]).all()
        and np.isfinite(unsafe[layer]).all()
        for layer in layers
    )


def consolidate_shards(paths: list[Path], layers: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Concatenate validated shards in caller-supplied batch order."""

    safe_chunks = {layer: [] for layer in layers}
    unsafe_chunks = {layer: [] for layer in layers}
    for path in paths:
        metadata, safe, unsafe = load_capture_shard(path)
        if metadata["layers"] != layers:
            raise ValueError(f"layer mismatch in shard: {path}")
        for layer in layers:
            safe_chunks[layer].append(safe[layer])
            unsafe_chunks[layer].append(unsafe[layer])
    return {
        layer: (
            np.concatenate(safe_chunks[layer], axis=0),
            np.concatenate(unsafe_chunks[layer], axis=0),
        )
        for layer in layers
    }


def export_run(source: str | Path, output: str | Path, *, include_shards: bool = False) -> dict:
    """Create a checksummed run archive, excluding resumable shards by default."""

    source_path, output_path = Path(source), Path(output)
    if not (source_path / "manifest.json").exists():
        raise ValueError("only completed runs with manifest.json can be exported")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    members = [
        path for path in sorted(source_path.rglob("*"))
        if path.is_file() and (include_shards or "shards" not in path.relative_to(source_path).parts)
    ]
    with tarfile.open(output_path, "w:gz") as archive:
        for path in members:
            archive.add(path, arcname=Path(source_path.name) / path.relative_to(source_path))
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    return {
        "archive": str(output_path),
        "sha256": digest,
        "checksum": str(checksum_path),
        "files": len(members),
        "include_shards": include_shards,
    }


def verify_export(archive: str | Path, checksum: str | Path | None = None) -> dict:
    """Verify archive checksum, safe member paths, and required run controls."""

    archive_path = Path(archive)
    checksum_path = Path(checksum) if checksum else archive_path.with_suffix(archive_path.suffix + ".sha256")
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != archive_path.name:
        raise ValueError("invalid checksum sidecar")
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if fields[0] != actual:
        raise ValueError("archive checksum mismatch")
    with tarfile.open(archive_path, "r:gz") as handle:
        members = handle.getmembers()
    names = [Path(member.name) for member in members if member.isfile()]
    if any(path.is_absolute() or ".." in path.parts for path in names):
        raise ValueError("archive contains an unsafe member path")
    basenames = {path.name for path in names}
    required = {"manifest.json", "run_spec.json", "status.json"}
    if not required.issubset(basenames):
        raise ValueError(f"archive is missing required controls: {sorted(required - basenames)}")
    return {"archive": str(archive_path), "sha256": actual, "files": len(names), "valid": True}


def audit_completed_run(directory: str | Path) -> dict:
    """Fail closed unless a completed train run is internally consistent."""

    root = Path(directory)
    errors = []
    try:
        spec = json.loads((root / "run_spec.json").read_text(encoding="utf-8"))
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"run control files are missing or invalid: {exc}") from exc
    spec_hash = canonical_sha256(spec)
    if status.get("state") != "complete":
        errors.append("run status is not complete")
    if status.get("spec_hash") != spec_hash:
        errors.append("status run-spec hash mismatch")
    if manifest.get("config", {}).get("run_spec_sha256") != spec_hash:
        errors.append("manifest run-spec hash mismatch")
    if spec.get("split") != "train" or manifest.get("config", {}).get("capture_split") != "train":
        errors.append("integrated Stage-1 vector run must be train-only")
    pair_ids = spec.get("pair_ids", [])
    group_ids = spec.get("source_group_ids", [])
    if not pair_ids or len(pair_ids) != len(group_ids) or len(set(pair_ids)) != len(pair_ids):
        errors.append("run-spec pair/source-group alignment is invalid")
    if manifest.get("metrics", {}).get("pairs") != float(len(pair_ids)):
        errors.append("manifest pair count does not match run specification")
    environment = manifest.get("config", {}).get("environment", {})
    required_environment = {
        "git_sha", "git_dirty", "python", "device", "torch", "cuda_runtime",
        "gpu_name", "gpu_total_vram_bytes", "gpu_compute_capability",
        "container_image", "nvidia_smi", "packages",
    }
    missing_environment = required_environment - set(environment)
    if missing_environment:
        errors.append(f"manifest environment is missing {sorted(missing_environment)}")
    if environment.get("git_dirty") is not False:
        errors.append("completed Stage-1 run does not record a clean Git checkout")
    if "hf_token" in json.dumps(manifest).lower():
        errors.append("manifest contains forbidden Hugging Face token material")
    methods = spec.get("config", {}).get("extraction", {}).get("methods", [])
    bootstrap_samples = int(spec.get("config", {}).get("extraction", {}).get("bootstrap_samples", 0))
    for layer in spec.get("layers", []):
        layer_root = root / "layers" / f"layer_{layer:02d}"
        try:
            safe = np.load(layer_root / "safe.npy", allow_pickle=False)
            unsafe = np.load(layer_root / "unsafe.npy", allow_pickle=False)
            safe_metadata = json.loads((layer_root / "safe.npy.json").read_text(encoding="utf-8"))
            unsafe_metadata = json.loads((layer_root / "unsafe.npy.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"layer {layer} activation artifacts are unreadable: {exc}")
            continue
        if safe.ndim != 2 or unsafe.shape != safe.shape or safe.shape[0] != len(pair_ids):
            errors.append(f"layer {layer} activation shapes are invalid")
        if not np.isfinite(safe).all() or not np.isfinite(unsafe).all():
            errors.append(f"layer {layer} activations contain non-finite values")
        expected_metadata = {
            "layer": layer,
            "token_mode": spec.get("capture_site"),
            "sample_ids": pair_ids,
            "splits": ["train"] * len(pair_ids),
            "source_group_ids": group_ids,
            "shape": list(safe.shape),
        }
        if safe_metadata != expected_metadata or unsafe_metadata != expected_metadata:
            errors.append(f"layer {layer} activation metadata mismatch")
        for method in methods:
            try:
                vector = np.load(layer_root / f"{method}.npy", allow_pickle=False)
                stability = np.load(layer_root / f"{method}.stability.npy", allow_pickle=False)
            except (OSError, ValueError) as exc:
                errors.append(f"layer {layer} {method} artifacts are unreadable: {exc}")
                continue
            if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.0):
                errors.append(f"layer {layer} {method} vector is not finite and unit-normalized")
            if stability.shape != (bootstrap_samples,) or not np.isfinite(stability).all():
                errors.append(f"layer {layer} {method} stability array is invalid")
    if errors:
        raise ValueError("Stage-1 run audit failed:\n- " + "\n- ".join(errors))
    return {
        "run_id": spec.get("run_id"),
        "layers": spec.get("layers"),
        "pairs": len(pair_ids),
        "source_groups": len(set(group_ids)),
        "valid": True,
    }
