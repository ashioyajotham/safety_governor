"""Deferred RQ4 interface for conditional runtime steering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TriggerDecision:
    trigger: bool
    score: float
    evidence: str


class SafetyTrigger(Protocol):
    def evaluate(self, residual_activation) -> TriggerDecision: ...
