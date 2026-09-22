"""Freeze reviewed Validation-v2 candidates into role-specific corpora."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from scripts.audit_annotation_artifacts import audit as audit_annotation_artifacts
from scripts.audit_validation_extension import audit_extension
from safety_governor.data import dataset_sha256
from safety_governor.stage1 import atomic_write_json
from safety_governor.validation_v2 import materialize_approved


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--base", type=Path, default=Path("datasets/frozen/english_contrastive.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/validation_v2"))
    args = parser.parse_args()
    candidates, decisions = _read_jsonl(args.candidates), _read_jsonl(args.decisions)
    records, summary = materialize_approved(candidates, decisions)
    root = args.output_root
    extension = root / "reviewed_extension.jsonl"
    calibration = root / "calibration.jsonl"
    confirmatory = root / "confirmatory.jsonl"
    manifest = root / "manifest.json"
    for path in (extension, calibration, confirmatory, manifest):
        if path.exists():
            raise FileExistsError(f"artifact already exists: {path}")
    lexical_failures = audit_annotation_artifacts(records)
    if lexical_failures:
        raise ValueError(
            "Validation-v2 annotation-artifact audit failed:\n- "
            + "\n- ".join(lexical_failures)
        )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", dir=root, delete=False
    ) as handle:
        temporary_extension = Path(handle.name)
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        audit = audit_extension(
            temporary_extension, args.base, minimum_pairs_per_archetype_role=12
        )
    finally:
        temporary_extension.unlink(missing_ok=True)
    _write_new(extension, records)
    _write_new(calibration, [row for row in records if row["validation_role"] == "calibration"])
    _write_new(confirmatory, [row for row in records if row["validation_role"] == "confirmatory"])
    payload = {
        "schema_version": 1,
        **summary,
        "audit": audit,
        "artifacts": {
            "reviewed_extension": {"path": str(extension), "sha256": dataset_sha256(extension)},
            "calibration": {"path": str(calibration), "sha256": dataset_sha256(calibration)},
            "confirmatory": {"path": str(confirmatory), "sha256": dataset_sha256(confirmatory)},
            "candidates_sha256": dataset_sha256(args.candidates),
            "decisions_sha256": dataset_sha256(args.decisions),
            "base_sha256": dataset_sha256(args.base),
        },
    }
    atomic_write_json(manifest, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
