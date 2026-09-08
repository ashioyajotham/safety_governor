import hashlib
import json
import tarfile

import numpy as np
import pytest

from safety_governor.stage1 import (
    audit_completed_run,
    canonical_sha256,
    consolidate_shards,
    export_run,
    initialize_run,
    parse_layers,
    save_capture_shard,
    validate_shard,
    verify_export,
)


def test_layer_parser_requires_sorted_unique_layers():
    assert parse_layers("0,4,8") == [0, 4, 8]
    with pytest.raises(ValueError):
        parse_layers("4,0")
    with pytest.raises(ValueError):
        parse_layers("0,0")


def test_run_spec_resume_requires_exact_contract(tmp_path):
    root = tmp_path / "run"
    spec = {"model": "pinned", "layers": [0, 4]}
    expected = canonical_sha256(spec)
    assert initialize_run(root, spec, resume=False) == expected
    assert initialize_run(root, spec, resume=True) == expected
    with pytest.raises(FileExistsError):
        initialize_run(root, spec, resume=False)
    with pytest.raises(ValueError, match="does not match"):
        initialize_run(root, {**spec, "layers": [0]}, resume=True)


def test_capture_shards_validate_and_consolidate(tmp_path):
    layers = [0, 4]
    paths = []
    for batch_index, pair_id in enumerate(("p1", "p2")):
        path = tmp_path / f"batch_{batch_index}.npz"
        safe = {layer: np.array([[batch_index + layer]], dtype=np.float32) for layer in layers}
        unsafe = {layer: values + 1 for layer, values in safe.items()}
        save_capture_shard(
            path,
            spec_hash="abc",
            batch_index=batch_index,
            pair_ids=[pair_id],
            source_group_ids=[f"g{batch_index}"],
            safe_by_layer=safe,
            unsafe_by_layer=unsafe,
        )
        assert validate_shard(
            path,
            spec_hash="abc",
            batch_index=batch_index,
            pair_ids=[pair_id],
            source_group_ids=[f"g{batch_index}"],
            layers=layers,
        )
        paths.append(path)
    consolidated = consolidate_shards(paths, layers)
    assert consolidated[0][0].ravel().tolist() == [0.0, 1.0]
    assert consolidated[4][1].ravel().tolist() == [5.0, 6.0]


def test_corrupt_shard_fails_validation(tmp_path):
    path = tmp_path / "broken.npz"
    path.write_bytes(b"not an archive")
    assert not validate_shard(
        path,
        spec_hash="abc",
        batch_index=0,
        pair_ids=["p1"],
        source_group_ids=["g1"],
        layers=[0],
    )


def test_export_excludes_shards_and_writes_matching_checksum(tmp_path):
    run = tmp_path / "run-1"
    (run / "layers").mkdir(parents=True)
    (run / "shards").mkdir()
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "run_spec.json").write_text("{}", encoding="utf-8")
    (run / "status.json").write_text("{}", encoding="utf-8")
    (run / "layers" / "vector.npy").write_bytes(b"vector")
    (run / "shards" / "batch.npz").write_bytes(b"shard")
    output = tmp_path / "run-1.tar.gz"
    report = export_run(run, output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == report["sha256"]
    with tarfile.open(output) as archive:
        names = archive.getnames()
    assert any(name.endswith("manifest.json") for name in names)
    assert not any("shards" in name for name in names)
    assert verify_export(output)["valid"] is True


def test_export_verification_rejects_tampering(tmp_path):
    archive = tmp_path / "run.tar.gz"
    archive.write_bytes(b"changed")
    archive.with_suffix(".gz.sha256").write_text(
        f"{'0' * 64}  {archive.name}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_export(archive)


def test_completed_run_audit_checks_train_artifacts(tmp_path):
    run = tmp_path / "run"
    layer = run / "layers" / "layer_00"
    layer.mkdir(parents=True)
    spec = {
        "run_id": "run",
        "split": "train",
        "capture_site": "response_mean",
        "layers": [0],
        "pair_ids": ["p1"],
        "source_group_ids": ["g1"],
        "config": {"extraction": {"methods": ["difference_in_means"], "bootstrap_samples": 2}},
    }
    spec_hash = canonical_sha256(spec)
    (run / "run_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"state": "complete", "spec_hash": spec_hash}), encoding="utf-8")
    environment = {
        "git_sha": "abc", "git_dirty": False, "python": "3.11", "device": "cuda",
        "torch": "2", "cuda_runtime": "12", "gpu_name": "GPU",
        "gpu_total_vram_bytes": 1, "gpu_compute_capability": "8.0",
        "container_image": "image", "nvidia_smi": "uuid, driver", "packages": {},
    }
    manifest = {
        "metrics": {"pairs": 1.0},
        "config": {"run_spec_sha256": spec_hash, "capture_split": "train", "environment": environment},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    safe, unsafe = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    np.save(layer / "safe.npy", safe)
    np.save(layer / "unsafe.npy", unsafe)
    metadata = {
        "layer": 0, "token_mode": "response_mean", "sample_ids": ["p1"],
        "splits": ["train"], "source_group_ids": ["g1"], "shape": [1, 2],
    }
    (layer / "safe.npy.json").write_text(json.dumps(metadata), encoding="utf-8")
    (layer / "unsafe.npy.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(layer / "difference_in_means.npy", np.array([1.0, 0.0]))
    np.save(layer / "difference_in_means.stability.npy", np.array([1.0, 1.0]))
    assert audit_completed_run(run)["valid"] is True


def test_completed_run_audit_rejects_held_out_fit(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    spec = {"run_id": "run", "split": "validation", "pair_ids": [], "source_group_ids": [], "layers": []}
    spec_hash = canonical_sha256(spec)
    (run / "run_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"state": "complete", "spec_hash": spec_hash}), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "metrics": {"pairs": 0.0},
        "config": {"run_spec_sha256": spec_hash, "capture_split": "validation", "environment": {}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        audit_completed_run(run)
