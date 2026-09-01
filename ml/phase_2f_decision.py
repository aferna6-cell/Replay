"""Phase 2F decision: pick exactly one next intervention from lifecycle fates."""

from __future__ import annotations

from typing import Dict, Optional

from ml.core_lifecycle_diagnostic import FATE_LABELS, METHODOLOGY_VERSION

# Pre-specified — not tuned on seeds 1000–1199.
DECISION_THRESHOLDS = {
    "dominant_fate_min_share": 0.35,
    "hand_stuck_min_share": 0.30,
    "sold_min_share": 0.30,
    "seed_loss_min_share": 0.25,
    "target_switch_min_share": 0.25,
    "triple_min_share": 0.25,
    "persistent_two_core_min_share": 0.20,
}


def _share(totals: Dict[str, int], fate: str, n: int) -> float:
    if n <= 0:
        return 0.0
    return totals.get(fate, 0) / n


def evaluate_phase_2f_decision(lifecycle: Dict,
                               thresholds: Optional[Dict] = None) -> Dict:
    th = thresholds or DECISION_THRESHOLDS
    n = lifecycle.get("n_fulfilled_purchases", 0)
    totals = lifecycle.get("fate_totals") or {}
    funnel = lifecycle.get("funnel") or {}

    shares = {f: _share(totals, f, n) for f in FATE_LABELS}

    sold_share = shares["B_PLAYED_THEN_SOLD_SAME_TURN"] + shares["C_PLAYED_THEN_SOLD_LATER"]
    hand_share = shares["A_BOUGHT_STUCK_IN_HAND"]

    if n == 0:
        branch = "no_fulfilled_cohort"
        next_step = (
            "No fulfilled seeded purchases to diagnose — re-run with oracle "
            "treatment or verify Phase 2C latch parity.")
        intervention = "verify_tracing"
    elif hand_share >= th["hand_stuck_min_share"]:
        branch = "board_slot_play_policy"
        intervention = "board_slot_play_policy"
        next_step = (
            "Most fulfilled cores never reach the board — prioritize play-order "
            "and board-slot handling before retention or card-effect work.")
    elif sold_share >= th["sold_min_share"]:
        branch = "retention_aware_sell_policy"
        intervention = "retention_aware_sell_upgrade_policy"
        next_step = (
            "Cores are played then sold — prioritize retention-aware sell/upgrade "
            "policy before triple or card-effect fidelity.")
    elif shares["E_SEED_PIECE_LOST"] >= th["seed_loss_min_share"]:
        branch = "seed_retention_policy"
        intervention = "retention_policy_existing_cores"
        next_step = (
            "Original seeded cores disappear while new cores survive — prioritize "
            "retention around existing core pieces.")
    elif shares["D_TARGET_SWITCH"] >= th["target_switch_min_share"]:
        branch = "target_persistence_hysteresis"
        intervention = "target_persistence_hysteresis"
        next_step = (
            "infer_target churn orphaning cores — prioritize target persistence "
            "or commitment hysteresis.")
    elif shares["F_TRANSFORMED_TRIPLED"] >= th["triple_min_share"]:
        branch = "triple_discover_fidelity"
        intervention = "triple_discover_fidelity"
        next_step = (
            "Triple/discover transformations dominate post-purchase loss — "
            "prioritize triple/discover fidelity.")
    elif shares["H_TWO_CORE_PERSISTENT"] >= th["persistent_two_core_min_share"]:
        branch = "card_effect_fidelity"
        intervention = "card_effect_fidelity"
        next_step = (
            "Two+ cores persist through recruit end but coverage stayed flat in "
            "Phase 2E — card-effect fidelity is now justified.")
    elif shares["G_TWO_CORE_TRANSIENT"] >= th["dominant_fate_min_share"]:
        branch = "transient_assembly_timing"
        intervention = "retention_aware_sell_upgrade_policy"
        next_step = (
            "Two cores coexist transiently but not at recruit end — investigate "
            "same-turn dismantling and sell timing before card effects.")
    else:
        # Dominant fate fallback
        dominant = max(FATE_LABELS, key=lambda f: totals.get(f, 0)) if n else None
        branch = "mixed_lifecycle"
        intervention = "retention_aware_sell_upgrade_policy"
        next_step = (
            f"No single fate exceeds thresholds; dominant={dominant} "
            f"({totals.get(dominant, 0)}/{n}). Default to retention/sell diagnosis.")

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "thresholds": th,
        "n_fulfilled_purchases": n,
        "fate_shares": shares,
        "funnel": funnel,
        "sold_fates_combined_share": sold_share,
        "decision_branch": branch,
        "recommended_intervention": intervention,
        "recommended_next_step": next_step,
        "phase_2e_context": (
            "Phase 2E showed recruit conversion works under oracle stress but "
            "0 end-of-recruit 2+ core assembly — Phase 2F explains post-purchase paths."),
    }
