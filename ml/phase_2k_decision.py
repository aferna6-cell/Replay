"""Phase 2K decision tree — route next intervention from gap accounting."""

from __future__ import annotations

from typing import Dict, Optional

METHODOLOGY_VERSION = "2k_v1"
DOMINANCE_THRESHOLD = 0.50

BRANCH_MAP = {
    "A_NEVER_AVAILABLE_POST_ASSEMBLY": (
        "Phase 2L: shop/pool/core-availability fidelity"),
    "B_AVAILABLE_NOT_BOUGHT": (
        "Phase 2L: post-commit valuation / policy (acquisition after assembly)"),
    "C_BOUGHT_NOT_DEPLOYED": (
        "Phase 2L: retention/sell / deploy policy (bought but not deployed)"),
    "D_DEPLOYED_THEN_LOST": (
        "Phase 2L: retention/sell policy (deployed cores lost)"),
    "E_EXISTING_CORE_LOST": (
        "Phase 2L: retention/sell policy (cores present at first-2 lost)"),
    "F_TARGET_SWITCH": (
        "Phase 2L: target commitment / hysteresis"),
    "G_TRANSFORM_TRIPLE_DISCOVER_PATH": (
        "Phase 2L: triple/discover fidelity"),
}


def evaluate_phase_2k_decision(analysis: Dict) -> Dict:
    n = analysis.get("n_states") or 0
    share = analysis.get("missing_coverage_mass_share_by_cause") or {}
    if n == 0:
        return {
            "decision_branch": "insufficient_sample",
            "recommended_next_step": (
                "No post-assembly states — expand DEV seeds before Phase 2L."),
            "dominant_cause": None,
            "dominant_share": None,
        }

    # Combine retention-like causes for routing when individually split
    retention_share = (
        share.get("C_BOUGHT_NOT_DEPLOYED", 0)
        + share.get("D_DEPLOYED_THEN_LOST", 0)
        + share.get("E_EXISTING_CORE_LOST", 0)
    )

    ranked = sorted(share.items(), key=lambda x: -x[1])
    top_cause, top_share = ranked[0] if ranked else (None, 0.0)

    if top_share > DOMINANCE_THRESHOLD and top_cause in BRANCH_MAP:
        return {
            "decision_branch": top_cause.lower(),
            "recommended_next_step": BRANCH_MAP[top_cause],
            "dominant_cause": top_cause,
            "dominant_share": round(top_share, 4),
            "n_states": n,
        }

    if retention_share > DOMINANCE_THRESHOLD:
        return {
            "decision_branch": "retention_loss_combined",
            "recommended_next_step": (
                "Phase 2L: retention/sell policy — combined bought/deployed/"
                "existing-core loss dominates missing mass."),
            "dominant_cause": "RETENTION_COMBINED",
            "dominant_share": round(retention_share, 4),
            "n_states": n,
        }

    # Acquired+retained but coverage still stalls → representation, not effects yet
    wf = analysis.get("weighted_funnel") or {}
    mean_final = wf.get("mean_weight_present_final")
    mean_retained = wf.get("mean_weight_retained_2_turns")
    mean_avail = wf.get("mean_weight_legally_available_after")
    if (mean_avail is not None and mean_retained is not None
            and mean_final is not None
            and mean_avail > 0.3 and mean_retained > 0.25
            and mean_final < 0.35):
        return {
            "decision_branch": "representation_or_core_set",
            "recommended_next_step": (
                "Cores available/acquired/retained but weighted coverage stalls — "
                "investigate representation/core-set definition before card effects."),
            "dominant_cause": top_cause,
            "dominant_share": round(top_share or 0.0, 4),
            "n_states": n,
        }

    return {
        "decision_branch": "mixed_failure_modes",
        "recommended_next_step": (
            "Mixed residual causes — expand DEV diagnostic; "
            "do not implement Phase 2L from a weak story."),
        "dominant_cause": top_cause,
        "dominant_share": round(top_share or 0.0, 4),
        "n_states": n,
        "cause_distribution": dict(share),
    }
