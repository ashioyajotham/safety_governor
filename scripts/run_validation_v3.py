"""Generate phase-locked Validation-v3 development or confirmatory outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safety_governor.config import load
from safety_governor.data import dataset_sha256, load_jsonl, validate_records
from safety_governor.models import (
    ProjectionGate,
    generate_unsteered,
    generate_with_governor,
    load_transformerlens_model,
)
from safety_governor.preflight import runtime_profile_errors, stage1_errors
from safety_governor.reproducibility import environment_facts
from safety_governor.stage1 import atomic_write_json, load_runtime_profile
from safety_governor.validation_v3 import (
    file_sha256,
    load_candidate_sites,
    validate_development_contract,
    verify_content_lock,
)


def _raw(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _candidate_gates(candidate: dict) -> dict[int, ProjectionGate]:
    return {
        int(layer): ProjectionGate(
            threshold=float(values["threshold"]),
            transition_width=float(values["transition_width"]),
        )
        for layer, values in candidate.get("gates", {}).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "confirmatory"))
    parser.add_argument("config")
    parser.add_argument("--development-config", default="configs/validation_v3_development.yaml")
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--direction-run", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path)
    parser.add_argument("--final-lock", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    main_config = load(args.config)
    contract = load_runtime_profile(args.development_config)
    confirmatory_path = Path(contract["sealed_confirmatory"]["path"])
    validate_development_contract(contract, confirmatory_path)
    if contract["generation"]["decoding"] != "greedy":
        raise ValueError("Validation-v3 requires greedy decoding")
    if main_config["model"]["revision"] != contract["model_revision"]:
        raise ValueError("model revision differs from Validation-v3 contract")

    lock_path = args.candidate_lock if args.role == "development" else args.final_lock
    if lock_path is None:
        raise ValueError(f"{args.role} requires its phase lock")
    lock_type = (
        "validation_v3_development_candidate"
        if args.role == "development" else "validation_v3_final_intervention"
    )
    lock = verify_content_lock(lock_path, lock_type=lock_type)
    if lock["development_contract_sha256"] != file_sha256(args.development_config):
        raise ValueError("Validation-v3 development contract changed after locking")
    if args.role == "development":
        if lock.get("confirmatory_authorized") is not False:
            raise ValueError("development candidate lock has invalid authority")
        candidates = lock["candidates"]
        dataset_path = Path(contract["development_dataset"])
    else:
        if lock.get("confirmatory_authorized") is not True:
            raise ValueError("final lock does not authorize confirmatory access")
        candidates = [lock["selected_candidate"]]
        dataset_path = confirmatory_path
    if len(candidates) + 1 > int(contract["development_behavior"]["maximum_conditions_including_baseline"]):
        raise ValueError("phase lock exceeds the behavioral condition cap")

    records = load_jsonl(dataset_path)
    raw = _raw(dataset_path)
    errors = validate_records(records)
    runtime_config = json.loads(json.dumps(main_config))
    runtime_config["dataset"]["path"] = str(dataset_path)
    errors.extend(stage1_errors(runtime_config, records, split="validation", allow_test_capture=False))
    profile = load_runtime_profile(args.runtime_profile)
    artifact_root = Path(args.artifact_root or profile["artifact_root"]).resolve()
    errors.extend(runtime_profile_errors(profile, artifact_root))
    facts = environment_facts(profile["device"])
    if profile.get("require_clean_git", True) and facts["git_dirty"]:
        errors.append("Validation-v3 generation requires a clean Git checkout")
    if errors:
        raise SystemExit("Validation-v3 preflight failed:\n- " + "\n- ".join(errors))

    safe_records = {
        row.pair_id: row for row in records
        if row.split == "validation" and row.polarity.value == "safe"
    }
    safe_raw = {
        row["pair_id"]: row for row in raw
        if row["split"] == "validation" and row["polarity"] == "safe"
    }
    if set(safe_records) != set(safe_raw):
        raise ValueError("raw and typed Validation-v3 pair sets differ")
    if args.role == "confirmatory" and len(safe_records) != int(contract["confirmatory"]["pairs"]):
        raise ValueError("sealed confirmatory pair count changed")

    configurations = [{"baseline": True}] + [{"baseline": False, **row} for row in candidates]
    spec = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": f"validation_v3_{args.role}",
        "validation_role": args.role,
        "model": main_config["model"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "development_contract_sha256": file_sha256(args.development_config),
        "phase_lock": str(lock_path.resolve()),
        "phase_lock_file_sha256": file_sha256(lock_path),
        "phase_lock_sha256": lock["lock_sha256"],
        "train_run": str(args.train_run.resolve()),
        "direction_run": str(args.direction_run.resolve()),
        "configurations": configurations,
        "pair_ids": sorted(safe_records),
        "source_group_ids": [safe_raw[key]["source_group_id"] for key in sorted(safe_records)],
        "archetypes": [safe_raw[key]["archetype"] for key in sorted(safe_records)],
        "git_sha": facts["git_sha"],
    }
    run = artifact_root / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    spec_path = run / "validation_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("existing Validation-v3 run specification differs")
        if not args.resume:
            raise FileExistsError("Validation-v3 run exists; pass --resume")
    else:
        if any(run.iterdir()):
            raise FileExistsError("non-empty Validation-v3 directory has no specification")
        atomic_write_json(spec_path, spec)
        atomic_write_json(run / "run_spec.json", spec)
        atomic_write_json(run / "phase_lock.json", lock)

    os.environ.setdefault("HF_HOME", profile["hf_cache_root"])
    model = load_transformerlens_model(
        main_config["model"]["name"], main_config["model"]["revision"],
        profile["device"], profile["dtype"], main_config["model"]["bridge_weight_mode"],
    )
    max_new_tokens = int(contract["generation"]["max_new_tokens"])
    shard_root = run / "generation_shards"
    for index, configuration in enumerate(configurations):
        generation_id = "baseline" if configuration["baseline"] else f"configuration_{index:02d}"
        sites = provenance = None
        if not configuration["baseline"]:
            sites, provenance = load_candidate_sites(
                configuration, args.train_run.resolve(), args.direction_run.resolve(),
                _candidate_gates(configuration),
            )
        for pair_id in sorted(safe_records):
            record, metadata = safe_records[pair_id], safe_raw[pair_id]
            shard = shard_root / generation_id / f"{pair_id}.json"
            expected = {
                "generation_id": generation_id,
                "pair_id": pair_id,
                "archetype": metadata["archetype"],
                "instruction": record.instruction,
                "source_group_id": record.source_group_id,
                "configuration": configuration,
            }
            if shard.exists():
                existing = json.loads(shard.read_text(encoding="utf-8"))
                if {key: existing[key] for key in expected} != expected:
                    raise ValueError(f"incompatible generation shard: {shard}")
                continue
            trace = []
            response = (
                generate_unsteered(model, record.instruction, max_new_tokens=max_new_tokens)
                if configuration["baseline"] else
                generate_with_governor(
                    model, record.instruction, sites,
                    total_relative_l2=float(configuration["total_relative_l2"]),
                    max_new_tokens=max_new_tokens, trace=trace,
                )
            )
            atomic_write_json(shard, {
                **expected, "response": response, "sites": provenance,
                "intervention_trace": trace,
            })
    generations = [json.loads(path.read_text()) for path in sorted(shard_root.glob("*/*.json"))]
    expected_count = len(configurations) * len(safe_records)
    if len(generations) != expected_count:
        raise ValueError(f"incomplete Validation-v3 generation: {len(generations)}/{expected_count}")
    output = run / "generations.jsonl"
    if not output.exists():
        with output.open("x", encoding="utf-8") as handle:
            for row in generations:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_write_json(run / "status.json", {
        "state": f"validation_v3_{args.role}_generation_complete",
        "responses": expected_count,
        "conditions": len(configurations),
    })
    print(f"Validation-v3 {args.role} generation complete: {output}")


if __name__ == "__main__":
    main()
