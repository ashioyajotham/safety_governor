"""Read-only, aggregate audit of saved Validation-v3 generation hook traces.

This checks execution against the shard's locked configuration. It cannot
establish that the intervention changed downstream computation or behavior.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def audit_run(run: Path) -> dict:
    root = run / "generation_shards"
    if not root.is_dir():
        raise ValueError("generation_shards directory is missing")
    groups = sorted(path for path in root.iterdir() if path.is_dir())
    if not groups:
        raise ValueError("no generation shard groups")
    result = {"run_id": run.name, "configurations": {}, "errors": []}
    pair_sets = []
    for group in groups:
        paths = sorted(group.glob("*.json"))
        pairs = set()
        trace_entries = 0
        stop_reasons = Counter()
        expected_layers = None
        expected_config = None
        for path in paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            pair = row["pair_id"]
            if pair in pairs:
                result["errors"].append(f"{group.name}: duplicate pair ID")
            pairs.add(pair)
            if row["generation_id"] != group.name or path.stem != pair:
                result["errors"].append(f"{group.name}: shard identity mismatch")
            config = row["configuration"]
            if expected_config is None:
                expected_config = config
            elif config != expected_config:
                result["errors"].append(f"{group.name}: configuration varies across shards")
            metadata = row["generation_metadata"]
            generated = metadata["generated_token_count"]
            reason = metadata["stop_reason"]
            stop_reasons[reason] += 1
            if not isinstance(generated, int) or generated < 0 or reason != "stop_token":
                result["errors"].append(f"{group.name}: incomplete generation")
            trace = row["intervention_trace"]
            trace_entries += len(trace)
            if config["baseline"]:
                if trace:
                    result["errors"].append(f"{group.name}: baseline contains hook traces")
                continue
            if config.get("token_mode") != "generation_frontier" or config.get("adaptive"):
                result["errors"].append(f"{group.name}: unexpected steering policy")
            sites = sorted(config["sites"], key=lambda site: site["layer"])
            layers = [site["layer"] for site in sites]
            if layers != [12, 16, 20, 24] or len(layers) != len(set(layers)):
                result["errors"].append(f"{group.name}: unexpected steering layers")
            if expected_layers is None:
                expected_layers = layers
            if layers != expected_layers:
                result["errors"].append(f"{group.name}: steering layers vary across shards")
            budget = config["total_relative_l2"]
            if not math.isclose(sum(site["weight"] for site in sites), 1, abs_tol=1e-8):
                result["errors"].append(f"{group.name}: profile weights do not sum to one")
            if len(trace) != (generated + 1) * len(sites):
                result["errors"].append(f"{group.name}: hook count does not match decode steps")
                continue
            first_position = trace[0]["position"]
            for step in range(generated + 1):
                for index, site in enumerate(sites):
                    entry = trace[step * len(sites) + index]
                    expected_ratio = budget * site["weight"]
                    values = [entry.get(key) for key in (
                        "activation_l2", "projection", "gate", "coefficient", "relative_delta")]
                    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                        result["errors"].append(f"{group.name}: nonfinite hook trace")
                        continue
                    norm, _, gate, coefficient, ratio = values
                    if (entry["layer"] != site["layer"] or
                            entry["position"] != first_position + step or
                            norm <= 0 or not math.isclose(gate, 1, abs_tol=1e-8) or
                            not math.isclose(ratio, expected_ratio, abs_tol=0.001) or
                            not math.isclose(coefficient, expected_ratio * norm,
                                             rel_tol=0.02, abs_tol=0.001)):
                        result["errors"].append(f"{group.name}: hook policy/budget mismatch")
        pair_sets.append(pairs)
        result["configurations"][group.name] = {
            "shards": len(paths), "trace_entries": trace_entries,
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "layers": expected_layers or [],
        }
    if any(pairs != pair_sets[0] for pairs in pair_sets[1:]):
        result["errors"].append("pair sets differ across conditions")
    result["valid"] = not result["errors"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
