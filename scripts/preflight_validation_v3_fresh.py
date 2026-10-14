"""No-inference fresh-v3 launch preflight; never prints corpus examples."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts.audit_validation_v3_fresh_dataset import audit
from safety_governor.stage1 import load_runtime_profile
from safety_governor.validation_v3 import (
    file_sha256, validate_development_contract, verify_content_lock,
    verify_fresh_execution_inputs,
)


def preflight(args: argparse.Namespace) -> dict:
    contract = load_runtime_profile(args.contract)
    validate_development_contract(contract, Path(contract["sealed_confirmatory"]["path"]))
    lock = verify_content_lock(args.lock, lock_type="validation_v3_development_candidate")
    if lock["development_contract_sha256"] != file_sha256(args.contract):
        raise ValueError("development contract differs from frozen lock")
    if file_sha256(args.attestation) != contract["holdout_isolation_attestation_sha256"]:
        raise ValueError("transferred isolation record hash mismatch")
    attestation = json.loads(args.attestation.read_text(encoding="utf-8"))
    if (attestation.get("isolated") is not True or
            attestation.get("custodian_signoff") != "adopted_by_user_ashioyajotham" or
            attestation.get("candidate_sha256") != contract["development_dataset_sha256"] or
            attestation.get("holdout_sha256") != contract["sealed_confirmatory"]["sha256"]):
        raise ValueError("transferred isolation record is not the adopted passing result")
    local = audit(args.dataset, Path(contract["train_dataset"]),
                  Path("datasets/validation_v2/calibration.jsonl"),
                  contract["development_dataset_sha256"])
    if not local["local_checks_pass"]:
        raise ValueError("fresh corpus failed local structural audit")
    verify_fresh_execution_inputs(contract, lock, args.dataset,
                                  args.train_run, args.direction_run)

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if git_sha != args.expected_git_sha:
        raise ValueError("checkout is not the approved code revision")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("checkout is not clean")
    gpu = None
    if args.check_gpu:
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        gpu = {"name": torch.cuda.get_device_name(0),
               "compute_capability": list(torch.cuda.get_device_capability(0)),
               "bfloat16_supported": torch.cuda.is_bf16_supported()}
        if not gpu["bfloat16_supported"] or gpu["compute_capability"][0] < 8:
            raise ValueError("GPU does not meet BF16 Ampere requirement")
    return {
        "valid": True,
        "no_inference_performed": True,
        "git_sha": git_sha,
        "development_contract_sha256": file_sha256(args.contract),
        "candidate_lock_file_sha256": file_sha256(args.lock),
        "candidate_lock_sha256": lock["lock_sha256"],
        "fresh_dataset_sha256": local["candidate_sha256"],
        "holdout_isolation_attestation_sha256": file_sha256(args.attestation),
        "gpu": gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path,
                        default=Path("configs/validation_v3_fresh_development.yaml"))
    parser.add_argument("--lock", type=Path,
                        default=Path("configs/validation_v3_fresh_candidate_lock.json"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("datasets/validation_v3/fresh_development.jsonl"))
    parser.add_argument("--attestation", type=Path,
                        default=Path("data/working/validation_v3_fresh/holdout_isolation_check.json"))
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--direction-run", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--check-gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
