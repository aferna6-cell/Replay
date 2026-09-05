"""Phase 3Q — play-lifecycle sticky-vs-repaint causal audit (measurement only).

Stacked on Phase 3P (PR #62). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, recruit, or the 2Q recruit-value objective. Confirm 11500–11699
remains reserved. Keep HOLD stack including #62.

For every play on trajectories feeding the T5/T6 3N class-(3) sample,
snapshots pre-play board/body synth, incoming recruit raw/tier, open-slot
vs sell→buy→play replacement, post-play pre-reallocation, post-reallocation,
and post-scale combat-start state. Offline same-state counterfactuals:

* (A) control board under would-be 2S recruit-raw-proportional repaint
* (B) treatment board under sticky incumbent synth / no repaint

Decomposes the T1 22.2→14.6 / T3 7.1→19.2 combat-start shift into:

* (1) same-state repaint vs sticky
* (2) replacement / open-slot lifecycle
* (3) subsequent scaling
* (4) residual
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ml.phase_3p_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_COMPONENTS,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
    EARLY_TURNS,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FIRST_DIVERGENCE_COMPONENTS,
    FLOW_ABS_TOL,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HISTORY_LINK_IDENTITY,
    HP_FLOW_IDENTITY,
    HP_GAP_COMPONENTS,
    HP_GAP_RECONCILE_IDENTITY,
    HP_WALK_FROM_TURN,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_ABS_TOL,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    LOW_WINNER_START_TIERS,
    MATCHMAKING_RECONCILE_IDENTITY,
    MEMBERSHIP_EVENTS,
    NESTED_ALLOCATION_IDENTITY,
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PAIRING_TURNS,
    PAINT_EQUATION_IDENTITY,
    PAINT_RECONCILE_IDENTITY,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3E_CARRY_DELTA,
    PHASE_3E_CARRY_SHARE_OF_A1,
    PHASE_3E_PUNCH_DELTA_CARRY,
    PHASE_3F_SELECTION_SHARE,
    PHASE_3F_UNCOND_SHARE,
    PHASE_3G_MIXTURE,
    PHASE_3G_MIXTURE_SHARE,
    PHASE_3G_MIX_ROLE_SHARE,
    PHASE_3G_N_CONTROL,
    PHASE_3G_N_TREATMENT,
    PHASE_3G_WITHIN_SHARE,
    PHASE_3H_COLLAPSE,
    PHASE_3H_LATE_CONTROL,
    PHASE_3H_LATE_TREATMENT,
    PHASE_3H_LEFTOVER,
    PHASE_3H_SHARE_LEFTOVER,
    PHASE_3I_DIFFERENT_OPPONENT,
    PHASE_3I_KIND_MISMATCH,
    PHASE_3I_OUTCOME_FLIP,
    PHASE_3I_PAIRING_SCHEDULE,
    PHASE_3I_RESIDUAL,
    PHASE_3I_SHARE_PAIRING_SCHEDULE,
    PHASE_3I_SURVIVOR_SUBSTITUTION,
    PHASE_3J_ELIGIBILITY,
    PHASE_3J_ELIG_DIFFERENT_OPPONENT,
    PHASE_3J_ELIG_KIND_MISMATCH,
    PHASE_3J_HISTORY_LEGAL,
    PHASE_3J_RNG_ORDER,
    PHASE_3J_SHARE_ELIGIBILITY,
    PHASE_3K_CONTROL_OPPONENT,
    PHASE_3K_NAMED,
    PHASE_3K_PRIOR_HP,
    PHASE_3K_SHARE_PRIOR_HP,
    PHASE_3K_SHARE_THIRD_PARTY,
    PHASE_3K_THIRD_PARTY,
    PHASE_3K_TREATMENT_EARLIER,
    PHASE_3L_PRIOR_HP,
    PHASE_3L_SAME_SEAT_EARLIER,
    PHASE_3L_SHARE_EARLIER,
    PHASE_3L_SHARE_PRIOR_HP,
    PHASE_3L_VERY_LATE_EARLIER,
    PHASE_3M_CLASS1,
    PHASE_3M_FIRST_SPLIT_T5,
    PHASE_3M_FIRST_SPLIT_T6,
    PHASE_3M_FIRST_SPLIT_T7,
    PHASE_3M_OUTCOME_FLIP,
    PHASE_3M_PRIOR_ALIVE_SET,
    PHASE_3M_SAME_OUTCOME_DAMAGE,
    PHASE_3M_SHARE_DAMAGE,
    PHASE_3M_SHARE_FLIP,
    PHASE_3M_SHARE_PAIRING,
    PHASE_3M_VERY_LATE_CLASS1,
    PHASE_3M_VERY_LATE_DAMAGE,
    PHASE_3M_VERY_LATE_SHARE_DAMAGE,
    PHASE_3N_CLASS3,
    PHASE_3N_CLASS3_T5,
    PHASE_3N_CLASS3_T6,
    PHASE_3N_CLASS3_T7,
    PHASE_3N_CF_DELTA,
    PHASE_3N_P_SURVIVE_T1_CONTROL,
    PHASE_3N_P_SURVIVE_T1_TREATMENT,
    PHASE_3N_P_SURVIVE_T3_CONTROL,
    PHASE_3N_P_SURVIVE_T3_TREATMENT,
    PHASE_3N_P_SURVIVE_T4_CONTROL,
    PHASE_3N_P_SURVIVE_T4_TREATMENT,
    PHASE_3N_SHARE_WITHIN_TIER,
    PHASE_3N_WITHIN_TIER_B,
    PHASE_3O_PRIMARY_N,
    PHASE_3O_SHARE_RECRUIT,
    PHASE_3O_SHARE_START_STATS,
    PHASE_3O_SHARE_SYNTH,
    PHASE_3O_T1_SYNTH_CONTROL,
    PHASE_3O_T1_SYNTH_TREATMENT,
    PHASE_3O_T3_SYNTH_CONTROL,
    PHASE_3O_T3_SYNTH_TREATMENT,
    PHASE_3O_T5T6_B,
    POOL_FLOW_IDENTITY,
    PROXY_ERROR_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    REWEIGHT_ABS_TOL,
    ROW_DAMAGE_RECONCILE_IDENTITY,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    SHARE_DOMINANT,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    share_of_applied,
)

METHODOLOGY_VERSION = "3q_v1"
PHASE_3Q_SEED = 14200
PHASE_3Q_LOBBIES = 500

HOLD_PRS = (
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
)

# Published 3P locks (exact).
PHASE_3P_PRIMARY_N_PAIRS = 4132
PHASE_3P_PRIMARY_N_FIGHTS = 1033
PHASE_3P_SHARE_POOL = 0.006333231838752983
PHASE_3P_SHARE_WEIGHT = 0.0
PHASE_3P_SHARE_TIMING = 0.9857690083784691
PHASE_3P_SHARE_ROUNDING = 0.007897759782777772
PHASE_3P_T1_SYNTH_CONTROL = PHASE_3O_T1_SYNTH_CONTROL
PHASE_3P_T1_SYNTH_TREATMENT = PHASE_3O_T1_SYNTH_TREATMENT
PHASE_3P_T3_SYNTH_CONTROL = PHASE_3O_T3_SYNTH_CONTROL
PHASE_3P_T3_SYNTH_TREATMENT = PHASE_3O_T3_SYNTH_TREATMENT

PRIMARY_TURNS = (5, 6)

LIFECYCLE_COMPONENTS = (
    "same_state_repaint",
    "replacement_lifecycle",
    "subsequent_scaling",
    "residual",
)

PLAY_SUBTYPES = (
    "open_slot",
    "sell_buy_play",
    "sell_play",
    "triple",
)

SAME_STATE_IDENTITY = (
    "(A) control post-play board painted by 2S largest-remainder on the "
    "implicit/on-body pool; (B) treatment post-play board keeps incumbent "
    "and incoming synth with no repaint"
)

NESTED_LIFECYCLE_IDENTITY = (
    "delta_synth = same_state_repaint + replacement_lifecycle + "
    "subsequent_scaling + residual"
)

PLAY_POOL_RECONCILE_IDENTITY = (
    "Σ body synthetic shares = painted_pool on CF-A and on treatment "
    "post-reallocation; sticky boards conserve implicit on-body pool"
)

NEXT_OBSERVABLE_DEFAULT = (
    "the largest T5/T6 play-lifecycle term that is still observable "
    "on the last-play sticky-vs-repaint identity, ranked before any "
    "behavior change"
)


def _rank_parts(parts: List[Tuple[str, Optional[float]]]) -> List[Dict]:
    ranked = []
    for name, share in parts:
        ranked.append({
            "component": name,
            "share": None if share is None else float(share),
            "abs_share": 0.0 if share is None else abs(float(share)),
        })
    ranked.sort(key=lambda r: r["abs_share"], reverse=True)
    return ranked


def diagnose_phase_3q(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the T5/T6 synth shift to a play-lifecycle term."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3q_seed": PHASE_3Q_SEED,
        "phase_3q_lobbies": PHASE_3Q_LOBBIES,
        "reused_seed_range": f"{REUSED_SEED_LO}–{REUSED_SEED_HI}",
        "keep_hold_prs": list(HOLD_PRS),
        "no_merge": True,
        "evaluative": False,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "no_hero_damage_retune": True,
        "no_gate_change": True,
        "no_behavior_change": True,
        "no_2q_rewrite": True,
        "no_scaling_constant_change": True,
        "no_recruit_change": True,
        "share_dominant": SHARE_DOMINANT,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_class3": PHASE_3N_CLASS3,
        "phase_3o_t5t6_B": PHASE_3O_T5T6_B,
        "phase_3o_share_start_stats": PHASE_3O_SHARE_START_STATS,
        "phase_3o_share_synth": PHASE_3O_SHARE_SYNTH,
        "phase_3p_share_timing": PHASE_3P_SHARE_TIMING,
        "phase_3p_n_pairs": PHASE_3P_PRIMARY_N_PAIRS,
        "phase_3m_same_outcome_damage": PHASE_3M_SAME_OUTCOME_DAMAGE,
        "primary_turns": list(PRIMARY_TURNS),
        "lifecycle_components": list(LIFECYCLE_COMPONENTS),
        "play_subtypes": list(PLAY_SUBTYPES),
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_tiers": list(LOW_TIERS),
        "early_turns": list(EARLY_TURNS),
        "late_turns": list(LATE_TURNS),
        "very_late_turns": list(VERY_LATE_TURNS),
        "pairing_turns": list(PAIRING_TURNS),
        "trace_from_turn": TRACE_FROM_TURN,
        "hp_walk_from_turn": HP_WALK_FROM_TURN,
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "lineage_identity": LINEAGE_IDENTITY,
        "paired_seat_identity": PAIRED_SEAT_IDENTITY,
        "pairing_identity": PAIRING_IDENTITY,
        "leftover_reconcile_identity": LEFTOVER_RECONCILE_IDENTITY,
        "candidate_choice_identity": CANDIDATE_CHOICE_IDENTITY,
        "matchmaking_reconcile_identity": MATCHMAKING_RECONCILE_IDENTITY,
        "hp_flow_identity": HP_FLOW_IDENTITY,
        "elimination_identity": ELIMINATION_IDENTITY,
        "eligibility_timing_identity": ELIGIBILITY_TIMING_IDENTITY,
        "hp_gap_reconcile_identity": HP_GAP_RECONCILE_IDENTITY,
        "chain_reconcile_identity": CHAIN_RECONCILE_IDENTITY,
        "chain_hp_reconcile_identity": CHAIN_HP_RECONCILE_IDENTITY,
        "row_elim_hp_identity": ROW_ELIM_HP_IDENTITY,
        "row_damage_reconcile_identity": ROW_DAMAGE_RECONCILE_IDENTITY,
        "paint_equation_identity": PAINT_EQUATION_IDENTITY,
        "paint_reconcile_identity": PAINT_RECONCILE_IDENTITY,
        "nested_allocation_identity": NESTED_ALLOCATION_IDENTITY,
        "same_state_identity": SAME_STATE_IDENTITY,
        "nested_lifecycle_identity": NESTED_LIFECYCLE_IDENTITY,
        "play_pool_reconcile_identity": PLAY_POOL_RECONCILE_IDENTITY,
        "history_filters_applied": False,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    primary = comparison.get("primary") or comparison.get("reweighting") or {}
    shares = {
        name: primary.get(f"share_of_delta_{name}")
        for name in LIFECYCLE_COMPONENTS
    }

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    same_state = _f("same_state_repaint")
    lifecycle = _f("replacement_lifecycle")
    scaling = _f("subsequent_scaling")
    residual = _f("residual")
    ranked = _rank_parts([
        ("same_state_repaint", same_state),
        ("replacement_lifecycle", lifecycle),
        ("subsequent_scaling", scaling),
        ("residual", residual),
    ])
    top = ranked[0]["component"] if ranked else "residual"

    def _no_change_tail() -> str:
        return (
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not change recruit defaults; do not burn confirm."
        )

    if same_state is not None and same_state > SHARE_DOMINANT:
        finding = "same_state_repaint_dominates"
        next_step = (
            "Same-state 2S repaint vs sticky incumbent synth clears ~70% "
            "of the T5/T6 class-(3) body-synth gap. Next hour: evidence "
            "audit of which representation is scientifically defensible "
            "(no implementation yet). " + _no_change_tail()
        )
    elif lifecycle is not None and lifecycle > SHARE_DOMINANT:
        finding = "replacement_lifecycle_dominates"
        next_step = (
            "Replacement / open-slot play lifecycle clears ~70% of the "
            "T5/T6 class-(3) body-synth gap. Next hour: trace that event "
            "subtype (open-slot vs sell→buy→play). " + _no_change_tail()
        )
    elif scaling is not None and scaling > SHARE_DOMINANT:
        finding = "subsequent_scaling_dominates"
        next_step = (
            "Subsequent scaling after the last play clears ~70% of the "
            "T5/T6 class-(3) body-synth gap. Next hour: trace scale-sync "
            "inputs and timing. " + _no_change_tail()
        )
    else:
        finding = "ranked_residual_needs_next_observable"
        next_step = (
            "No same-state / lifecycle / scaling term clears ~70% of the "
            f"T5/T6 class-(3) body-synth gap (top={top}). Rank components "
            f"and pursue the largest residual observable: "
            f"{NEXT_OBSERVABLE_DEFAULT}. " + _no_change_tail()
        )

    out.update({
        "primary_finding": finding,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_same_state_repaint": same_state,
        "share_replacement_lifecycle": lifecycle,
        "share_subsequent_scaling": scaling,
        "share_residual": residual,
        "primary": primary,
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_pr_34_hold": True,
        "keep_pr_35_hold": True,
        "keep_pr_36_hold": True,
        "keep_pr_37_hold": True,
        "keep_pr_38_hold": True,
        "keep_pr_39_hold": True,
        "keep_pr_40_hold": True,
        "keep_pr_41_hold": True,
        "keep_pr_42_hold": True,
        "keep_pr_43_hold": True,
        "keep_pr_44_hold": True,
        "keep_pr_45_hold": True,
        "keep_pr_46_hold": True,
        "keep_pr_47_hold": True,
        "keep_pr_50_hold": True,
        "keep_pr_51_hold": True,
        "keep_pr_52_hold": True,
        "keep_pr_53_hold": True,
        "keep_pr_54_hold": True,
        "keep_pr_55_hold": True,
        "keep_pr_56_hold": True,
        "keep_pr_57_hold": True,
        "keep_pr_58_hold": True,
        "keep_pr_59_hold": True,
        "keep_pr_60_hold": True,
        "keep_pr_61_hold": True,
        "keep_pr_62_hold": True,
    })
    return out
