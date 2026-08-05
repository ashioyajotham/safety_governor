"""Stable domain contracts shared by data, vector, and experiment modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Behavior(str, Enum):
    DECEPTIVE_REASONING = "deceptive_reasoning"
    INSTRUCTION_NONCOMPLIANCE = "instruction_noncompliance"
    HARMFUL_COMPLIANCE = "harmful_compliance"


class Polarity(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class ContrastiveRecord:
    pair_id: str
    behavior: Behavior
    polarity: Polarity
    language: str
    prompt: str
    expected_behavior: str
    source: str
    reviewer_status: str
    split: str = "train"
    translation_of: str | None = None
    translation_notes: str | None = None


@dataclass(frozen=True)
class Finding:
    layer: int
    method: str
    vector_path: str
    stability: float | None = None


@dataclass(frozen=True)
class InterventionSpec:
    layer: int
    coefficient: float
    token_mode: str  # last_prompt_token | all_tokens


@dataclass
class RunManifest:
    run_id: str
    config: dict[str, Any]
    code_revision: str = "unknown"
    seed: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
