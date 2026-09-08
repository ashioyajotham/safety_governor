"""Qualify a Vast or equivalent single-GPU runtime before model loading."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from safety_governor.config import load
from safety_governor.data import load_jsonl, validate_records
from safety_governor.preflight import runtime_profile_errors, stage1_errors
from safety_governor.reproducibility import environment_facts
from safety_governor.stage1 import load_runtime_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--check-model-access", action="store_true")
    args = parser.parse_args()
    config, profile = load(args.config), load_runtime_profile(args.runtime_profile)
    artifact_root = Path(args.artifact_root or profile["artifact_root"]).resolve()
    records = load_jsonl(config["dataset"]["path"])
    errors = validate_records(records)
    errors.extend(stage1_errors(config, records, split=args.split, allow_test_capture=False))
    errors.extend(runtime_profile_errors(profile, artifact_root))
    facts = environment_facts(profile["device"])
    if profile.get("require_clean_git", True) and facts["git_dirty"]:
        errors.append("Stage-1 run requires a clean Git checkout")
    if args.check_model_access:
        if not os.environ.get("HF_TOKEN"):
            errors.append("HF_TOKEN is required to check gated model access")
        else:
            try:
                from huggingface_hub import HfApi
                HfApi().model_info(
                    config["model"]["name"],
                    revision=config["model"]["revision"],
                    token=os.environ["HF_TOKEN"],
                )
            except Exception as exc:  # network/auth errors are runtime-specific
                errors.append(f"pinned model access failed: {type(exc).__name__}: {exc}")
    if errors:
        raise SystemExit("Stage-1 preflight failed:\n- " + "\n- ".join(errors))
    print("Stage-1 preflight passed")
    for key, value in sorted(facts.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
