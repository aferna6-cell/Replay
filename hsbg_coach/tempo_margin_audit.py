"""Observational audit for Phase 2I seeded opportunity decision-margin diagnostic.

Records policy scores without participating in action selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScoredTransition:
    action_type: str
    candidate_name: Optional[str]
    candidate_slot: Optional[int]
    raw_component: float
    build_gain: float
    build_component: float
    replacement_name: Optional[str]
    replacement_slot: Optional[int]
    replacement_raw: float
    replacement_build_value: float
    replacement_component: float
    net_value: float
    is_target_core: bool
    action_id: Optional[int] = None


@dataclass
class CoreExposureScore:
    core_name: str
    shop_slot: int
    candidate_raw: float
    core_frequency: float
    tier_commitment: float
    build_gain: float
    build_component: float
    board_full: bool
    replacement_name: Optional[str]
    replacement_raw: float
    replacement_build_value: float
    replacement_component: float
    core_transition_raw: float
    core_transition_build: float
    core_net_value: float
    core_free_slot_value: Optional[float]
    core_actual_replacement_value: Optional[float]
    rank_with_build: Optional[int]
    rank_without_build: Optional[int]
    rank_total: int


@dataclass
class DecisionSnapshot:
    lobby: int
    seat: int
    turn: int
    shop_generation: int
    seeded: bool
    lambda_build: float
    tavern_tier: int
    core_have: int
    target_archetype: Optional[str]
    gold: float
    board_full: bool
    pending_stage: Optional[str]
    all_transitions: List[ScoredTransition] = field(default_factory=list)
    chosen: Optional[ScoredTransition] = None
    core_scores: Dict[str, CoreExposureScore] = field(default_factory=dict)
    action_id: Optional[int] = None


class TempoMarginAuditCollector:
    """Append-only decision audit log."""

    def __init__(self) -> None:
        self.snapshots: List[DecisionSnapshot] = []
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def record(self, snap: DecisionSnapshot) -> int:
        if not self._enabled:
            return -1
        self.snapshots.append(snap)
        return len(self.snapshots) - 1

    def clear(self) -> None:
        self.snapshots.clear()


def break_even_lambda(*, core_raw: float, core_build: float,
                      repl_raw: float, repl_build: float,
                      chosen_raw: float, chosen_build: float,
                      chosen_repl_raw: float = 0.0,
                      chosen_repl_build: float = 0.0) -> Optional[float]:
    """λ where core transition net ties chosen transition net."""
    core_slope = core_build - repl_build
    chosen_slope = chosen_build - chosen_repl_build
    denom = core_slope - chosen_slope
    if abs(denom) < 1e-12:
        return None
    core_intercept = core_raw - repl_raw
    chosen_intercept = chosen_raw - chosen_repl_raw
    return (chosen_intercept - core_intercept) / denom


def break_even_lambda_bucket(lam: Optional[float]) -> str:
    if lam is None:
        return "no_finite_helpful_lambda"
    if lam <= 12:
        return "lambda_le_12"
    if lam <= 24:
        return "12_lt_lambda_le_24"
    return "lambda_gt_24"
