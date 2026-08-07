"""Extract a vector from pre-captured safe/unsafe activation matrices."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from safety_governor.activations import load_matrix, load_metadata
from safety_governor.vectors import bootstrap_cosine, difference_in_means, pca_direction, probe_direction

METHODS = {"difference_in_means": difference_in_means, "pca": pca_direction, "probe": probe_direction}


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
    if safe_metadata.get("sample_ids") != unsafe_metadata.get("sample_ids"):
        raise SystemExit("safe/unsafe activation sample IDs are not pair-aligned")
    extractor = METHODS[args.method]
    vector = extractor(safe, unsafe)
    stability = bootstrap_cosine(extractor, safe, unsafe, args.bootstrap_samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, vector)
    np.save(output.with_suffix(".stability.npy"), stability)
    print(f"wrote {output}; bootstrap cosine mean={stability.mean():.4f}")


if __name__ == "__main__":
    main()
