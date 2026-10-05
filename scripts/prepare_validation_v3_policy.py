"""Freeze a non-promoting 768-token policy diagnostic from verified archives.

Preparation performs no inference. Missing or incompatible source artifacts
are errors: never silently regenerate the archived comparison conditions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from safety_governor.validation_v3 import file_sha256, write_candidate_lock


def verified_source(run: Path, expected_hashes: dict | None = None) -> tuple[dict, list[dict]]:
    names = ("validation_spec.json", "generations.jsonl", "phase_lock.json")
    if expected_hashes is not None:
        for name in names:
            if file_sha256(run / name) != expected_hashes[name]:
                raise ValueError(f"archived source changed: {name}")
    spec = json.loads((run / names[0]).read_text())
    if (spec.get("validation_role") != "development" or
            spec.get("diagnostic_only") is not True or
            spec.get("generation_max_new_tokens") != 768):
        raise ValueError("source must be a 768-token development diagnostic")
    pairs = spec["pair_ids"]
    if len(pairs) != 48 or len(set(pairs)) != 48:
        raise ValueError("source must contain exactly 48 distinct spent pairs")
    rows = [json.loads(line) for line in (run / names[1]).read_text().splitlines() if line.strip()]
    configurations = spec["configurations"]
    expected = {( "baseline" if c["baseline"] else f"configuration_{i:02d}", p)
                for i, c in enumerate(configurations) for p in pairs}
    keys = [(r["generation_id"], r["pair_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("source generations do not exactly cover locked conditions/pairs")
    for row in rows:
        index = 0 if row["generation_id"] == "baseline" else int(row["generation_id"][-2:])
        if row["configuration"] != configurations[index]:
            raise ValueError("source row configuration differs from specification")
        meta = row.get("generation_metadata", {})
        count, reason = meta.get("generated_token_count"), meta.get("stop_reason")
        if (type(count) is not int or not 0 <= count <= 768 or
                reason not in {"stop_token", "token_limit"} or
                (reason == "token_limit" and count != 768)):
            raise ValueError("source has invalid termination metadata")
    from safety_governor.validation_v3 import verify_content_lock
    lock = verify_content_lock(run / "phase_lock.json", lock_type="validation_v3_development_candidate")
    if lock.get("confirmatory_authorized") is not False:
        raise ValueError("source lock has invalid authority")
    if lock["candidates"] != [{k: v for k, v in c.items() if k != "baseline"}
                              for c in configurations[1:]]:
        raise ValueError("source phase lock and configurations differ")
    return spec, rows


def prepare(source: Path, output: Path, base: Path) -> dict:
    spec, _ = verified_source(source)
    eligible = [c for c in spec["configurations"] if
                c.get("candidate_id") == "distributed_ridge_uniform__b0.2" and
                not c.get("adaptive") and c.get("token_mode") == "generation_frontier"]
    if len(eligible) != 1:
        raise ValueError("source lacks the fixed static uniform frontier candidate")
    frontier = {k: v for k, v in eligible[0].items() if k != "baseline"}
    if (frontier["total_relative_l2"] != 0.20 or
            [(s["layer"], s["weight"], s["method"], s["source"]) for s in frontier["sites"]] !=
            [(l, 0.25, "balanced_ridge", "v3_direction_fit") for l in (12, 16, 20, 24)]):
        raise ValueError("source candidate does not match the fixed comparison")
    span = {**copy.deepcopy(frontier), "candidate_id": "distributed_ridge_uniform__b0.2__generated_span",
            "token_mode": "generated_span"}
    contract = yaml.safe_load(base.read_text())
    contract.update(diagnostic_only=True, diagnostic_kind="token_policy",
                    reuse_source=str(source.resolve()),
                    reuse_source_hashes={n: file_sha256(source / n) for n in
                                        ("validation_spec.json", "generations.jsonl", "phase_lock.json")})
    contract["generation"]["max_new_tokens"] = 768
    if contract["model_revision"] != spec["model"]["revision"]:
        raise ValueError("source and contract model revisions differ")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("preparation output directory must be empty")
    contract_path = output / "policy_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    lock_path = output / "candidate_lock.json"
    write_candidate_lock(lock_path, {
        "lock_type": "validation_v3_development_candidate", "confirmatory_authorized": False,
        "development_contract_sha256": file_sha256(contract_path),
        "candidates": [frontier, span], "diagnostic_only": True,
        "parent_source_spec_sha256": contract["reuse_source_hashes"]["validation_spec.json"],
    })
    return {"contract": str(contract_path.resolve()), "candidate_lock": str(lock_path.resolve()),
            "diagnostic_only": True, "promotion_authorized": False,
            "reused_outputs": 96, "new_outputs": 48, "review_tasks": 144,
            "maximum_new_decode_steps": 48 * 768,
            "billing_note": "Not a time/cost estimate; launch requires separate approval."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, default=Path("configs/validation_v3_development.yaml"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_run, args.output_dir, args.base_contract), indent=2))


if __name__ == "__main__":
    main()
