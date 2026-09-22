"""Generate phase-locked Validation-v2 behavioral outputs.

Unlike v1 validation, this runner performs no representation capture or
selection.  It verifies and consumes one fixed train-derived direction.
Calibration explores the predeclared causal interventions; confirmatory uses
only the intervention recorded in a verified calibration lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.run_validation import _load_parent
from safety_governor.config import load
from safety_governor.data import dataset_sha256, load_jsonl, validate_records
from safety_governor.models import generate_unsteered, generate_with_steering, load_transformerlens_model
from safety_governor.preflight import runtime_profile_errors, stage1_errors
from safety_governor.reproducibility import environment_facts
from safety_governor.stage1 import atomic_write_json, canonical_sha256, load_runtime_profile
from safety_governor.validation import file_sha256
from safety_governor.validation_v2 import ARCHETYPES, verify_lock


def _raw_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prepare(args, role: str):
    main_config = load(args.config)
    contract = load_runtime_profile(args.validation_config)
    if int(contract.get("schema_version", 0)) != 3:
        raise ValueError("Validation-v2 requires schema_version 3")
    if contract.get("generation", {}).get("decoding") != "greedy":
        raise ValueError("Validation-v2 generation must use deterministic greedy decoding")
    for phase in ("calibration", "confirmatory"):
        if contract.get(phase, {}).get("gate", {}).get(
            "strict_archetype_improvement"
        ) is not True:
            raise ValueError(
                f"Validation-v2 {phase} must require strict archetype improvement"
            )
    profile = load_runtime_profile(args.runtime_profile)
    dataset_path = Path(contract["datasets"][role])
    records = load_jsonl(dataset_path)
    errors = validate_records(records)
    runtime_config = json.loads(json.dumps(main_config))
    runtime_config["dataset"]["path"] = str(dataset_path)
    errors.extend(stage1_errors(runtime_config, records, split="validation", allow_test_capture=False))
    raw = _raw_rows(dataset_path)
    if any(row.get("validation_role") != role for row in raw):
        errors.append(f"{role} dataset contains rows assigned to another role")
    archetypes = {row.get("archetype") for row in raw}
    if archetypes != set(ARCHETYPES):
        errors.append(f"{role} dataset archetypes are {sorted(archetypes)}")
    artifact_root = Path(args.artifact_root or profile["artifact_root"]).resolve()
    errors.extend(runtime_profile_errors(profile, artifact_root))
    facts = environment_facts(profile["device"])
    if profile.get("require_clean_git", True) and facts["git_dirty"]:
        errors.append("Validation-v2 requires a clean Git checkout")
    parent = Path(args.train_run).resolve()
    parent_spec, _ = _load_parent(parent, main_config)
    fixed = contract["fixed_direction"]
    vector_path = parent / "layers" / f"layer_{int(fixed['layer']):02d}" / f"{fixed['method']}.npy"
    if not vector_path.is_file():
        errors.append(f"fixed direction is missing: {vector_path}")
    elif file_sha256(vector_path) != fixed["vector_sha256"]:
        errors.append("fixed direction hash does not match Validation-v2 contract")
    if file_sha256(parent / "manifest.json") != fixed["parent_train_manifest_sha256"]:
        errors.append("parent train manifest hash does not match Validation-v2 contract")
    if errors:
        raise SystemExit("Validation-v2 preflight failed:\n- " + "\n- ".join(errors))
    return (
        main_config, contract, profile, dataset_path, raw, records, artifact_root,
        facts, parent, parent_spec, vector_path,
    )


def _configurations(contract: dict, role: str, calibration_lock: dict | None) -> list[dict]:
    configurations = [{"baseline": True}]
    fixed = contract["fixed_direction"]
    if role == "calibration":
        generation = contract["calibration"]["generation"]
        for magnitude in map(float, generation["magnitudes"]):
            for token_mode in generation["token_modes"]:
                configurations.append({
                    "baseline": False, "method": fixed["method"],
                    "layer": int(fixed["layer"]), "magnitude": magnitude,
                    "signed_coefficient": -magnitude, "token_mode": token_mode,
                    "vector_sha256": fixed["vector_sha256"],
                })
    else:
        if calibration_lock is None:
            raise ValueError("confirmatory generation requires a calibration lock")
        selected = dict(calibration_lock["selected_configuration"])
        if selected.get("baseline") is not False:
            raise ValueError("calibration lock must select a non-baseline intervention")
        if selected.get("method") != fixed["method"] or int(selected.get("layer", -1)) != int(fixed["layer"]):
            raise ValueError("calibration lock selects a different direction")
        if selected.get("vector_sha256") not in {None, fixed["vector_sha256"]}:
            raise ValueError("calibration lock vector hash mismatch")
        selected["vector_sha256"] = fixed["vector_sha256"]
        configurations.append(selected)
    return configurations


def generate(args, role: str) -> None:
    (
        main_config, contract, profile, dataset_path, raw, records, artifact_root,
        facts, parent, parent_spec, vector_path,
    ) = _prepare(args, role)
    lock = None
    if role == "confirmatory":
        lock = verify_lock(args.calibration_lock, lock_type="validation_v2_calibration")
        if lock["validation_contract_sha256"] != file_sha256(args.validation_config):
            raise ValueError("Validation-v2 contract changed after calibration")
        if lock["fixed_direction_sha256"] != file_sha256(vector_path):
            raise ValueError("fixed direction changed after calibration")
        if lock["model_revision"] != main_config["model"]["revision"]:
            raise ValueError("model revision differs from calibration")
        calibration_path = Path(contract["datasets"]["calibration"])
        if lock["dataset_sha256"] != dataset_sha256(calibration_path):
            raise ValueError("calibration dataset changed after calibration lock")
    configurations = _configurations(contract, role, lock)
    safe_raw = {
        row["pair_id"]: row for row in raw
        if row["polarity"] == "safe" and row["split"] == "validation"
    }
    safe_records = {
        row.pair_id: row for row in records
        if row.polarity.value == "safe" and row.split == "validation"
    }
    pair_ids = sorted(safe_records)
    expected_pairs = int(contract[role]["pairs_per_archetype"]) * len(ARCHETYPES)
    if len(pair_ids) != expected_pairs:
        raise ValueError(f"{role} requires exactly {expected_pairs} pairs; found {len(pair_ids)}")
    spec = {
        "schema_version": 2, "run_id": args.run_id,
        "phase": f"validation_v2_{role}", "validation_role": role,
        "parent_train_run": str(parent),
        "parent_train_manifest_sha256": file_sha256(parent / "manifest.json"),
        "parent_train_spec_sha256": canonical_sha256(parent_spec),
        "dataset_sha256": dataset_sha256(dataset_path),
        "validation_dataset_path": str(dataset_path),
        "model": main_config["model"],
        "capture_site": main_config["extraction"]["capture_site"],
        "split": "validation",
        "fixed_direction": {
            **contract["fixed_direction"], "path": str(vector_path),
        },
        "configurations": configurations, "pair_ids": pair_ids,
        "source_group_ids": [safe_raw[key]["source_group_id"] for key in pair_ids],
        "archetypes": [safe_raw[key]["archetype"] for key in pair_ids],
        "validation_config_sha256": file_sha256(args.validation_config),
        "validation_contract": contract,
        "calibration_lock_sha256": None if lock is None else lock["lock_sha256"],
        "calibration_lock_file_sha256": (
            None if lock is None else file_sha256(args.calibration_lock)
        ),
        "git_sha": facts["git_sha"],
    }
    run_root = artifact_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    spec_path = run_root / "validation_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("existing Validation-v2 specification does not match")
        if not args.resume:
            raise FileExistsError(f"run exists; pass --resume: {run_root}")
    else:
        if any(run_root.iterdir()):
            raise FileExistsError(f"non-empty run directory has no specification: {run_root}")
        atomic_write_json(spec_path, spec)
        atomic_write_json(run_root / "run_spec.json", spec)
        atomic_write_json(run_root / "fixed_direction.json", {
            "schema_version": 1, **spec["fixed_direction"],
            "originating_train_manifest_sha256": spec["parent_train_manifest_sha256"],
        })
        if lock is not None:
            atomic_write_json(run_root / "calibration_lock.json", lock)
    copied_lock = run_root / "calibration_lock.json"
    if lock is None:
        if copied_lock.exists():
            raise ValueError("calibration run must not contain a calibration lock")
    else:
        if not copied_lock.is_file():
            raise ValueError("confirmatory run is missing its copied calibration lock")
        copied = verify_lock(copied_lock, lock_type="validation_v2_calibration")
        if copied != lock:
            raise ValueError("copied calibration lock differs from the supplied lock")
        if file_sha256(copied_lock) != spec["calibration_lock_file_sha256"]:
            raise ValueError("copied calibration-lock file hash mismatch")
    output = run_root / "generations.jsonl"
    expected = len(pair_ids) * len(configurations)
    if output.exists():
        completed = [line for line in output.read_text(encoding="utf-8").splitlines() if line]
        if len(completed) != expected:
            raise ValueError("existing Validation-v2 generations are incomplete")
        print(f"Validation-v2 generation is already complete: {output}")
        return
    os.environ.setdefault("HF_HOME", profile["hf_cache_root"])
    model = load_transformerlens_model(
        main_config["model"]["name"], main_config["model"]["revision"],
        profile["device"], profile["dtype"], main_config["model"]["bridge_weight_mode"],
    )
    vector = np.load(vector_path, allow_pickle=False)
    max_new_tokens = int(contract["generation"]["max_new_tokens"])
    shard_root = run_root / "generation_shards"
    for config_index, generation_config in enumerate(configurations):
        generation_id = "baseline" if generation_config["baseline"] else f"configuration_{config_index:02d}"
        for pair_id in pair_ids:
            row, raw_row = safe_records[pair_id], safe_raw[pair_id]
            shard = shard_root / generation_id / f"{pair_id}.json"
            expected_row = {
                "generation_id": generation_id, "pair_id": pair_id,
                "archetype": raw_row["archetype"], "instruction": row.instruction,
                "source_group_id": row.source_group_id,
                "configuration": generation_config,
            }
            if shard.exists():
                existing = json.loads(shard.read_text(encoding="utf-8"))
                if {key: existing[key] for key in expected_row} != expected_row:
                    raise ValueError(f"incompatible generation shard: {shard}")
                continue
            response = (
                generate_unsteered(model, row.instruction, max_new_tokens=max_new_tokens)
                if generation_config["baseline"]
                else generate_with_steering(
                    model, row.instruction, vector,
                    layer=int(generation_config["layer"]),
                    magnitude=float(generation_config["magnitude"]),
                    token_mode=generation_config["token_mode"],
                    max_new_tokens=max_new_tokens,
                )
            )
            atomic_write_json(shard, {**expected_row, "response": response})
    generations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(shard_root.glob("*/*.json"))
    ]
    if len(generations) != expected:
        raise ValueError("Validation-v2 generation output count is incomplete")
    with output.open("x", encoding="utf-8") as handle:
        for row in generations:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_write_json(run_root / "status.json", {
        "state": f"validation_v2_{role}_generation_complete",
        "responses": expected, "configurations": len(configurations),
    })
    print(f"Validation-v2 {role} generation complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, role in (("generate-calibration", "calibration"), ("generate-confirmatory", "confirmatory")):
        child = subparsers.add_parser(command)
        child.add_argument("config")
        child.add_argument("--validation-config", default="configs/validation_v2.yaml")
        child.add_argument("--runtime-profile", required=True)
        child.add_argument("--train-run", required=True)
        child.add_argument("--run-id", required=True)
        child.add_argument("--artifact-root", default=None)
        child.add_argument("--calibration-lock", required=role == "confirmatory")
        child.add_argument("--resume", action="store_true")
        child.set_defaults(role=role)
    args = parser.parse_args()
    generate(args, args.role)


if __name__ == "__main__":
    main()
