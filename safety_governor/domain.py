"""Stable domain contracts shared by data, vector, and experiment modules.

This module is intentionally boring: it defines the small set of schemas that
other modules rely on. Keeping these contracts centralized prevents subtle
schema drift between curation files, frozen corpora, activation artifacts, and
evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Behavior(str, Enum):
    """Research behaviors represented by contrastive safe/unsafe pairs."""

    DECEPTIVE_REASONING = "deceptive_reasoning"
    INSTRUCTION_NONCOMPLIANCE = "instruction_noncompliance"
    HARMFUL_COMPLIANCE = "harmful_compliance"


class Polarity(str, Enum):
    """Side of a contrastive pair."""

    SAFE = "safe"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class ContrastiveRecord:
    """One row in a frozen or working contrastive corpus.

    A full pair is represented by two rows with the same ``pair_id`` and
    opposite ``polarity`` values. ``source_group_id`` is broader than pair ID:
    it groups paraphrases or source variants that must remain in the same
    train/validation/test split to avoid leakage.
    """

    pair_id: str
    behavior: Behavior
    polarity: Polarity
    language: str
    prompt: str
    expected_behavior: str
    source: str
    reviewer_status: str
    split: str = "unassigned"
    translation_of: str | None = None
    translation_notes: str | None = None
    instruction: str | None = None
    completion: str | None = None
    source_group_id: str | None = None


@dataclass(frozen=True)
class Finding:
    """Pointer to a fitted direction and its stability summary."""

    layer: int
    method: str
    vector_path: str
    stability: float | None = None


@dataclass(frozen=True)
class InterventionSpec:
    """Where and how strongly a steering vector should be applied."""

    layer: int
    coefficient: float
    token_mode: str  # final_response_token | all_response_tokens | all_tokens


@dataclass
class RunManifest:
    """Reproducibility manifest written beside each experiment artifact."""

    run_id: str
    config: dict[str, Any]
    code_revision: str = "unknown"
    seed: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
