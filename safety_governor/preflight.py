"""Fail-closed checks that run before research activation capture."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

from .domain import Behavior, ContrastiveRecord

SYMBOLIC_REVISIONS = {"main", "master", "latest"}


def runtime_errors(config: dict) -> list[str]:
    errors = []
    revision = str(config.get("model", {}).get("revision", ""))
    if not revision or revision.lower() in SYMBOLIC_REVISIONS:
        errors.append("model revision must be an immutable commit or tag")
    for package, expected in config.get("runtime", {}).get("exact_versions", {}).items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "not-installed"
        if actual != str(expected):
            errors.append(f"{package} version {actual}; required {expected}")
    return errors


def stage1_errors(
    config: dict,
    records: list[ContrastiveRecord],
    *,
    split: str,
    allow_test_capture: bool,
) -> list[str]:
    errors = runtime_errors(config)
    dataset = config.get("dataset", {})
    path = Path(str(dataset.get("path", "")))
    if path.suffix.lower() != ".jsonl":
        errors.append("Stage-1 dataset must be a JSONL contrastive corpus")
    if "quarantined" in str(path).lower():
        errors.append("quarantined corpus is not Stage-1 eligible")
    if split == "test" and not allow_test_capture:
        errors.append("test capture requires --allow-test-capture")
    if any(not record.source_group_id for record in records):
        errors.append("every Stage-1 record requires source_group_id")
    if any(record.behavior is Behavior.HARMFUL_COMPLIANCE for record in records):
        if dataset.get("harmful_compliance_eligible") is not True:
            errors.append("harmful compliance is quarantined until explicitly eligible")
    return errors