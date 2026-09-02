"""Phase 2I decision tree — recommend next branch from margin diagnostic."""

from __future__ import annotations

from typing import Dict, Optional

METHODOLOGY_VERSION = "2i_v2"

DOMINANCE_THRESHOLD = 0.50

BRANCH_MAP = {
    "A_REPLACEMENT_COST_DOMINATES": (
        "Phase 2J: multi-turn / board-slot opportunity-cost model"),
    "B_RAW_STAT_COMPETITOR_DOMINATES": (
        "Phase 2J: normalize immediate tempo vs future build value"),
    "C_BUILD_SIGNAL_TOO_SMALL": (
        "Phase 2J: redesign build-progress representation (not blindly increase λ)"),
    "D_BUILD_SIGNAL_NONDISCRIMINATIVE": (
        "Phase 2J: replace core-frequency signal; λ tuning cannot solve it"),
    "F_ECONOMY_LEGALITY_LOSS": (
        "Phase 2J: include gold/action/roll opportunity cost"),
    "G_TARGET_CHANGED": (
        "Phase 2J: target commitment / hysteresis"),
}


def evaluate_phase_2i_decision(analysis: Dict) -> Dict:
    funnel = analysis.get("funnel") or {}
    comp_by = funnel.get("composition_progress_by_cause") or {}
    comp_failures = funnel.get("composition_progress_failures") or 0
    n_rejected = funnel.get("rejected") or 0

    if comp_failures == 0:
        return {
            "decision_branch": "no_composition_progress_failures",
            "recommended_next_step": (
                "Insufficient rejected composition-progress exposures; "
                "expand diagnostic on additional DEV seeds before Phase 2J."),
            "dominant_cause": None,
            "dominant_fraction": None,
        }

    ranked = sorted(comp_by.items(), key=lambda x: -x[1])
    top_cause, top_count = ranked[0]
    frac = top_count / comp_failures

    if frac > DOMINANCE_THRESHOLD and top_cause in BRANCH_MAP:
        return {
            "decision_branch": top_cause.lower(),
            "recommended_next_step": BRANCH_MAP[top_cause],
            "dominant_cause": top_cause,
            "dominant_fraction": round(frac, 4),
            "composition_progress_failures": comp_failures,
            "rejected_exposures": n_rejected,
        }

    return {
        "decision_branch": "mixed_failure_modes",
        "recommended_next_step": (
            "Mixed rejection causes — do not implement Phase 2J yet; "
            "expand diagnostic on additional DEV seeds or refine taxonomy."),
        "dominant_cause": top_cause,
        "dominant_fraction": round(frac, 4),
        "composition_progress_failures": comp_failures,
        "cause_distribution": dict(comp_by),
    }
