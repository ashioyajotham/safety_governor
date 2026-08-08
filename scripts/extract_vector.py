"""Extract a train-only vector from pair-aligned activation matrices."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from safety_governor.activations import load_matrix, load_metadata
from safety_governor.vectors import bootstrap_cosine, difference_in_means, paired_delta_pca, probe_direction

METHODS = {"difference_in_means": difference_in_means, "paired_delta_pca": paired_delta_pca, "probe": probe_direction}


def validate_fit_metadata(safe_metadata: dict, unsafe_metadata: dict) -> list[str]:
    errors = []
    for field in ("sample_ids", "splits", "source_group_ids", "token_mode", "layer"):
        if safe_metadata.get(field) != unsafe_metadata.get(field):
            errors.append(f"safe/unsafe activation metadata mismatch: {field}")
    splits = set(safe_metadata.get("splits") or [])
    if splits != {"train"}:
        errors.append(f"vector fitting is train-only; found splits={sorted(splits)}")
    groups = safe_metadata.get("source_group_ids")
    if not groups or any(group is None for group in groups):
        errors.append("vector fitting requires source_group_ids")
    return errors

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe", required=True)
    parser.add_argument("--unsafe", required=True)
    parser.add_argument("--method", choices=METHODS, default="difference_in_means")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    args = parser.parse_args()
    safe, unsafe = load_matrix(args.safe), load_matrix(args.unsafe)
    safe_metadata, unsafe_metadata = load_metadata(args.safe), load_metadata(args.unsafe)
    errors = validate_fit_metadata(safe_metadata, unsafe_metadata)
    if errors:
        raise SystemExit("Activation metadata invalid:\n- " + "\n- ".join(errors))
    group_ids = safe_metadata.get("source_group_ids")
    extractor = METHODS[args.method]
    unique_groups = sorted(set(group_ids))
    safe_grouped = np.stack([safe[[i for i, group in enumerate(group_ids) if group == target]].mean(axis=0) for target in unique_groups])
    unsafe_grouped = np.stack([unsafe[[i for i, group in enumerate(group_ids) if group == target]].mean(axis=0) for target in unique_groups])
    vector = extractor(safe_grouped, unsafe_grouped)
    stability = bootstrap_cosine(
        extractor, safe_grouped, unsafe_grouped, args.bootstrap_samples,
        group_ids=unique_groups,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, vector)
    np.save(output.with_suffix(".stability.npy"), stability)
    print(f"wrote {output}; group-aware bootstrap cosine mean={stability.mean():.4f}")


if __name__ == "__main__":
    main()