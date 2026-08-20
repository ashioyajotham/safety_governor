"""Dataset loading and safety-oriented validation for contrastive pairs.

The validators here deliberately fail closed. They are not just schema checks:
they encode research-integrity assumptions such as approved-only records,
aligned pair splits, and source-group split isolation.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .domain import Behavior, ContrastiveRecord, Polarity

REQUIRED_REVIEW = "approved"
VALID_SPLITS = {"train", "validation", "test"}
ALL_SPLITS = VALID_SPLITS | {"unassigned"}


def _record(raw: dict) -> ContrastiveRecord:
    """Convert a JSON row into the typed internal contract."""

    return ContrastiveRecord(
        pair_id=raw["pair_id"], behavior=Behavior(raw["behavior"]),
        polarity=Polarity(raw["polarity"]), language=raw["language"].lower(),
        prompt=raw.get("prompt", ""), expected_behavior=raw["expected_behavior"],
        source=raw["source"], reviewer_status=raw["reviewer_status"],
        split=raw.get("split", "unassigned"), translation_of=raw.get("translation_of"),
        translation_notes=raw.get("translation_notes"), instruction=raw.get("instruction"),
        completion=raw.get("completion"), source_group_id=raw.get("source_group_id"),
    )


def load_jsonl(path: str | Path) -> list[ContrastiveRecord]:
    """Load a JSONL contrastive corpus into typed records."""

    with Path(path).open(encoding="utf-8") as handle:
        return [_record(json.loads(line)) for line in handle if line.strip()]


def validate_records(records: Iterable[ContrastiveRecord], require_approved: bool = True) -> list[str]:
    """Return all corpus problems instead of stopping at the first one."""

    records = list(records)
    errors: list[str] = []
    seen: set[tuple[str, Polarity, str]] = set()
    by_pair: dict[str, list[ContrastiveRecord]] = defaultdict(list)
    prompts: set[tuple[str, str]] = set()
    for record in records:
        key = (record.pair_id, record.polarity, record.language)
        if key in seen:
            errors.append(f"duplicate pair/polarity/language: {key}")
        seen.add(key)
        by_pair[record.pair_id].append(record)
        # Records may be stored either as a complete prompt transcript or as an
        # explicit instruction/completion boundary. Stage-1 prefers the latter.
        if not record.prompt.strip() and not (record.instruction and record.completion):
            errors.append(f"empty prompt: {record.pair_id}")
        if not record.source.strip():
            errors.append(f"missing provenance: {record.pair_id}")
        valid_splits = VALID_SPLITS if require_approved else ALL_SPLITS
        if record.split not in valid_splits:
            errors.append(f"invalid split: {record.pair_id}")
        if require_approved and record.reviewer_status != REQUIRED_REVIEW:
            errors.append(f"unreviewed record: {record.pair_id}")
        if record.language == "sw" and not record.translation_of:
            errors.append(f"Swahili record missing translation_of: {record.pair_id}")
        prompt_text = record.prompt or record.instruction or ""
        prompt_key = (record.language, prompt_text.strip(), record.polarity.value)
        if prompt_key in prompts:
            errors.append(f"duplicate prompt: {record.pair_id}")
        prompts.add(prompt_key)
    for pair_id, group in by_pair.items():
        if {item.polarity for item in group} != {Polarity.SAFE, Polarity.UNSAFE}:
            errors.append(f"incomplete contrastive pair: {pair_id}")
        # Safe and unsafe rows in a pair must be evaluated on the same split.
        if len({item.split for item in group}) != 1:
            errors.append(f"split leakage in pair: {pair_id}")
        if len({item.source_group_id for item in group}) != 1:
            errors.append(f"source-group mismatch in pair: {pair_id}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.source_group_id:
            group_splits[record.source_group_id].add(record.split)
    for group_id, splits in group_splits.items():
        # Multiple pairs derived from the same source must not cross splits.
        if len(splits) != 1:
            errors.append(f"source-group split leakage: {group_id}")
    return errors


def dataset_sha256(path: str | Path) -> str:
    """Hash a dataset file for manifests and reconstruction checks."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()