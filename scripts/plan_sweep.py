"""Expand a static-intervention config into a reviewable experiment matrix."""
from __future__ import annotations

import argparse
import itertools
import json

from safety_governor.config import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load(args.config)
    intervention = config["intervention"]
    layers = intervention.get("layers")
    if layers is None:
        raise SystemExit("materialize model layer count before expanding layer_stride")
    rows = [
        {"layer": layer, "coefficient": coefficient, "token_mode": token_mode}
        for layer, coefficient, token_mode in itertools.product(
            layers, intervention["coefficients"], intervention["token_modes"]
        )
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
