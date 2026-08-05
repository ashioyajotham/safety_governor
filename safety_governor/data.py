"""Dataset loading and safety-oriented validation for contrastive pairs."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .domain import Behavior, ContrastiveRecord, Polarity

REQUIRED_REVIEW = "approved"
VALID_SPLITS = {"train", "validation", "test"}


def _record(raw: dict) -> ContrastiveRecord:
    return ContrastiveRecord(
        pair_id=raw["pair_id"], behavior=Behavior(raw["behavior"]),
        polarity=Polarity(raw["polarity"]), language=raw["language"].lower(),
        prompt=raw["prompt"], expected_behavior=raw["expected_behavior"],
        source=raw["source"], reviewer_status=raw["reviewer_status"],
        split=raw.get("split", "train"), translation_of=raw.get("translation_of"),
        translation_notes=raw.get("translation_notes"),
    )


def load_jsonl(path: str | Path) -> list[ContrastiveRecord]:
    """Load records; raw datasets must stay outside source control."""
    with Path(path).open(encoding="utf-8") as handle:
        return [_record(json.loads(line)) for line in handle if line.strip()]


def validate_records(records: Iterable[ContrastiveRecord], require_approved: bool = True) -> list[str]:
    """Return all validation errors instead of failing on only the first one."""
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
        if not record.prompt.strip():
            errors.append(f"empty prompt: {record.pair_id}")
        if not record.source.strip():
            errors.append(f"missing provenance: {record.pair_id}")
        if record.split not in VALID_SPLITS:
            errors.append(f"invalid split: {record.pair_id}")
        if require_approved and record.reviewer_status != REQUIRED_REVIEW:
            errors.append(f"unreviewed record: {record.pair_id}")
        if record.language == "sw" and not record.translation_of:
            errors.append(f"Swahili record missing translation_of: {record.pair_id}")
        prompt_key = (record.language, record.prompt.strip())
        if prompt_key in prompts:
            errors.append(f"duplicate prompt: {record.pair_id}")
        prompts.add(prompt_key)
    for pair_id, group in by_pair.items():
        polarities = {item.polarity for item in group}
        if polarities != {Polarity.SAFE, Polarity.UNSAFE}:
            errors.append(f"incomplete contrastive pair: {pair_id}")
        if len({item.split for item in group}) != 1:
            errors.append(f"split leakage in pair: {pair_id}")
    return errors


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
