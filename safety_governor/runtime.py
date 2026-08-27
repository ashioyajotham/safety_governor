"""Protocol-only scaffold for the optional conditional-steering stage.

Stage 1 fits and applies unconditional directions. If the core experiments are
successful, a later stage may use a residual-stream trigger to decide whether an
intervention should fire. No production trigger implementation is claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TriggerDecision:
    """Auditable output of a conditional safety trigger."""

    trigger: bool
    score: float
    evidence: str


class SafetyTrigger(Protocol):
    """Interface that future conditional-steering triggers must implement."""

    def evaluate(self, residual_activation) -> TriggerDecision: ...
