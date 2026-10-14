"""Freeze the two prespecified fresh-development finalists, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from scripts.audit_validation_v3_fresh_dataset import audit as audit_fresh_dataset
from safety_governor.stage1 import load_runtime_profile
from safety_governor.validation_v3 import (
    expand_screen_candidates, file_sha256, validate_development_contract,
    write_candidate_lock,
)

SCREEN_RUN_ID = "llama3-v3-frontier-exact-screen-930164d"
SCREEN_ARCHIVE_SHA256 = "9f78ac9201b28c31c2bc6fcb42a35c84fa658336279d9d44a6781742a6da5bb8"
SCREEN_SPEC_SHA256 = "7dc7f4eb7054d9a48115adb12494d164f2f002c488c1f9398cd4b75b1c04a6a9"
FINALISTS = ("distributed_ridge_uniform__b0.2",
             "distributed_ridge_late_dominant__b0.1")


def _archive_json(archive: tarfile.TarFile, name: str) -> tuple[dict, str]:
    member = archive.extractfile(f"artifacts/{SCREEN_RUN_ID}/{name}")
    if member is None:
        raise ValueError(f"screen archive lacks {name}")
    content = member.read()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def freeze(*, contract_path: Path, historical_contract_path: Path,
           candidate_path: Path, attestation_path: Path, screen_archive: Path,
           train_archive: Path, direction_run: Path, output: Path,
           expected_screen_metrics_sha256: str) -> dict:
    if output.exists():
        raise FileExistsError("fresh candidate lock already exists")
    contract = load_runtime_profile(contract_path)
    validate_development_contract(contract, Path(contract["sealed_confirmatory"]["path"]))
    if file_sha256(candidate_path) != contract["development_dataset_sha256"]:
        raise ValueError("approved fresh corpus hash mismatch")
    local_audit = audit_fresh_dataset(
        candidate_path, Path(contract["train_dataset"]),
        Path("datasets/validation_v2/calibration.jsonl"),
        contract["development_dataset_sha256"],
    )
    if not local_audit["local_checks_pass"] or local_audit["candidate_pairs"] != 48:
        raise ValueError("fresh corpus local structural audit failed")
    if file_sha256(attestation_path) != contract["holdout_isolation_attestation_sha256"]:
        raise ValueError("adopted isolation record hash mismatch")
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if (attestation.get("isolated") is not True or
            attestation.get("custodian_signoff") != "adopted_by_user_ashioyajotham" or
            attestation.get("candidate_sha256") != contract["development_dataset_sha256"] or
            attestation.get("holdout_sha256") != contract["sealed_confirmatory"]["sha256"] or
            attestation.get("overlap_counts") != dict.fromkeys(
                ("pair_id", "source_group_id", "source_record_id", "normalized_instruction"), 0)):
        raise ValueError("isolation record is incomplete or not adopted")

    if file_sha256(screen_archive) != SCREEN_ARCHIVE_SHA256:
        raise ValueError("screen archive differs from independently audited report")
    with tarfile.open(screen_archive, "r:gz") as archive:
        spec, spec_sha = _archive_json(archive, "run_spec.json")
        metrics, metrics_sha = _archive_json(archive, "screen_metrics.json")
    if spec_sha != SCREEN_SPEC_SHA256 or metrics_sha != expected_screen_metrics_sha256:
        raise ValueError("screen metrics differ from independently audited report")
    if (spec.get("confirmatory_accessed") is not False or
            spec.get("development_contract_sha256") != file_sha256(historical_contract_path) or
            spec.get("sealed_confirmatory_sha256") != contract["sealed_confirmatory"]["sha256"] or
            spec.get("model", {}).get("revision") != contract["model_revision"]):
        raise ValueError("screen specification differs from historical experiment")
    expanded = expand_screen_candidates(contract)
    if spec.get("candidates") != expanded:
        raise ValueError("fresh contract changes the completed screen grid")
    if metrics.get("ranking", [])[:2] != list(FINALISTS):
        raise ValueError("prespecified finalists no longer lead the completed screen")
    results = {row["candidate_id"]: row for row in metrics.get("candidates", [])}
    if any(results.get(candidate_id, {}).get("eligible_for_behavioral_development") is not True
           for candidate_id in FINALISTS):
        raise ValueError("a finalist fails the completed train-only eligibility rule")
    if file_sha256(direction_run / "manifest.json") != spec.get("direction_manifest_sha256"):
        raise ValueError("recovered direction manifest differs from screened directions")
    with tarfile.open(train_archive, "r:gz") as archive:
        member = archive.extractfile("llama3-stage1-response-mean-hf-native/manifest.json")
        if member is None or hashlib.sha256(member.read()).hexdigest() != spec.get("train_manifest_sha256"):
            raise ValueError("train manifest differs from screened training run")

    available = {row["candidate_id"]: row for row in expanded}
    selected = [available[candidate_id] for candidate_id in FINALISTS]
    vector_hashes = {}
    for candidate in selected:
        for site in candidate["sites"]:
            layer = int(site["layer"])
            vector = direction_run / "directions" / f"layer_{layer:02d}" / "balanced_ridge.npy"
            vector_hashes[str(layer)] = file_sha256(vector)
    payload = {
        "schema_version": 1,
        "lock_type": "validation_v3_development_candidate",
        "confirmatory_authorized": False,
        "development_contract_sha256": file_sha256(contract_path),
        "development_dataset_sha256": contract["development_dataset_sha256"],
        "holdout_isolation_attestation_sha256": contract["holdout_isolation_attestation_sha256"],
        "screen_provenance": {
            "run_id": SCREEN_RUN_ID,
            "archive_sha256": file_sha256(screen_archive),
            "run_spec_sha256": spec_sha,
            "screen_metrics_sha256": metrics_sha,
            "historical_contract_sha256": file_sha256(historical_contract_path),
            "train_manifest_sha256": spec["train_manifest_sha256"],
            "direction_manifest_sha256": spec["direction_manifest_sha256"],
        },
        "vector_sha256_by_layer": vector_hashes,
        "candidates": selected,
        "selection_rule": "fixed_top_two_eligible_train_only_screen_no_development_reranking",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_candidate_lock(output, payload)
    return {"candidate_lock": str(output), "lock_sha256": digest,
            "candidates": list(FINALISTS), "development_dataset_sha256":
            contract["development_dataset_sha256"], "confirmatory_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path,
                        default=Path("configs/validation_v3_fresh_development.yaml"))
    parser.add_argument("--historical-contract", type=Path,
                        default=Path("configs/validation_v3_development.yaml"))
    parser.add_argument("--candidate", type=Path,
                        default=Path("datasets/validation_v3/fresh_development.jsonl"))
    parser.add_argument("--attestation", type=Path,
                        default=Path("data/working/validation_v3_fresh/holdout_isolation_check.json"))
    parser.add_argument("--screen-archive", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--direction-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-screen-metrics-sha256", default=
                        "5408aaef8a5ca13bacf8e3edb4615c641cb04b2bb615ce7fe4cac8f848229bff")
    args = parser.parse_args()
    print(json.dumps(freeze(contract_path=args.contract,
                            historical_contract_path=args.historical_contract,
                            candidate_path=args.candidate,
                            attestation_path=args.attestation,
                            screen_archive=args.screen_archive,
                            train_archive=args.train_archive,
                            direction_run=args.direction_run,
                            output=args.output,
                            expected_screen_metrics_sha256=args.expected_screen_metrics_sha256),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
