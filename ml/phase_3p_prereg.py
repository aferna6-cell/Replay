"""Phase 3P — synthetic-pool allocation-input attribution (measurement only).

Stacked on Phase 3O (PR #61). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #61.

For every T5/T6 3N class-(3) starting body, reconstructs the exact 2S
paint equation at combat start and decomposes treatment−control body
synth into:

* (1) player-pool magnitude
* (2) allocation-weight / board-denominator composition
* (3) reallocation timing / membership selection
* (4) integer rounding / largest remainder
* (5) residual
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ml.phase_3o_prereg import (
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
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PAIRING_TURNS,
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

METHODOLOGY_VERSION = "3p_v1"
PHASE_3P_SEED = 14200
PHASE_3P_LOBBIES = 500

HOLD_PRS = (
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
)

# Published 3O locks (exact).
PHASE_3O_PRIMARY_N = 1033
PHASE_3O_T5T6_B = 0.6166505324298197
PHASE_3O_SHARE_START_STATS = 1.2098231585111623
PHASE_3O_SHARE_SYNTH = 1.1847200887085545
PHASE_3O_SHARE_RECRUIT = 0.0
PHASE_3O_T1_SYNTH_CONTROL = 22.23794950267789
PHASE_3O_T1_SYNTH_TREATMENT = 14.631981637337415
PHASE_3O_T3_SYNTH_CONTROL = 7.066298342541437
PHASE_3O_T3_SYNTH_TREATMENT = 19.185082872928177

PRIMARY_TURNS = (5, 6)

ALLOCATION_COMPONENTS = (
    "pool_magnitude",
    "weight_composition",
    "timing_membership",
    "integer_rounding",
    "residual",
)

PAINT_EQUATION_IDENTITY = (
    "painted_pool = round(abstract_pool); per-body share = "
    "largest_remainder(recruit_raw / board_recruit_denom, painted_pool)"
)

PAINT_RECONCILE_IDENTITY = (
    "Σ body synthetic shares = painted_pool; nested allocation parts "
    "sum to treatment−control body synth"
)

NESTED_ALLOCATION_IDENTITY = (
    "delta_synth = pool_magnitude + weight_composition + "
    "timing_membership + integer_rounding + residual"
)

MEMBERSHIP_EVENTS = ("sell", "play", "triple")

NEXT_OBSERVABLE_DEFAULT = (
    "the largest T5/T6 allocation-input term that is still observable "
    "on the 2S paint equation, ranked before any behavior change"
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


def diagnose_phase_3p(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the T5/T6 synth shift to a 2S allocation-input term."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3p_seed": PHASE_3P_SEED,
        "phase_3p_lobbies": PHASE_3P_LOBBIES,
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
        "share_dominant": SHARE_DOMINANT,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_class3": PHASE_3N_CLASS3,
        "phase_3o_t5t6_B": PHASE_3O_T5T6_B,
        "phase_3o_share_start_stats": PHASE_3O_SHARE_START_STATS,
        "phase_3o_share_synth": PHASE_3O_SHARE_SYNTH,
        "phase_3m_same_outcome_damage": PHASE_3M_SAME_OUTCOME_DAMAGE,
        "primary_turns": list(PRIMARY_TURNS),
        "allocation_components": list(ALLOCATION_COMPONENTS),
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
        for name in ALLOCATION_COMPONENTS
    }

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    pool = _f("pool_magnitude")
    weight = _f("weight_composition")
    timing = _f("timing_membership")
    rounding = _f("integer_rounding")
    residual = _f("residual")
    ranked = _rank_parts([
        ("pool_magnitude", pool),
        ("weight_composition", weight),
        ("timing_membership", timing),
        ("integer_rounding", rounding),
        ("residual", residual),
    ])
    top = ranked[0]["component"] if ranked else "residual"

    def _no_change_tail() -> str:
        return (
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )

    if pool is not None and pool > SHARE_DOMINANT:
        finding = "pool_magnitude_dominates"
        next_step = (
            "Player-pool magnitude clears ~70% of the T5/T6 class-(3) "
            "body-synth gap. Next hour: trace when/why the player pool "
            "diverges before T5/T6 without retuning scaling. "
            + _no_change_tail()
        )
    elif weight is not None and weight > SHARE_DOMINANT:
        finding = "weight_composition_dominates"
        next_step = (
            "Allocation-weight / board-denominator composition clears "
            "~70% of the T5/T6 class-(3) body-synth gap. Next hour: "
            "audit whether recruit-raw-proportional painting is a "
            "scientifically defensible representation before any "
            "implementation. " + _no_change_tail()
        )
    elif timing is not None and timing > SHARE_DOMINANT:
        finding = "timing_membership_dominates"
        next_step = (
            "Reallocation timing / membership selection clears ~70% of "
            "the T5/T6 class-(3) body-synth gap. Next hour: trace the "
            "exact sell/play/triple lifecycle event creating it. "
            + _no_change_tail()
        )
    elif rounding is not None and abs(rounding) > SHARE_DOMINANT:
        finding = "rounding_material"
        next_step = (
            "Integer rounding / largest-remainder is materially large "
            "(>~70% of the T5/T6 body-synth gap). Treat as a paint-split "
            "bug only after confirming the remainder term, not a scaling "
            "retune. " + _no_change_tail()
        )
    else:
        finding = "ranked_residual_needs_next_observable"
        next_step = (
            "No pool / weight / timing term clears ~70% of the T5/T6 "
            f"class-(3) body-synth gap (top={top}). Rank components and "
            f"pursue the largest residual observable: "
            f"{NEXT_OBSERVABLE_DEFAULT}. Rounding is a bug only if "
            "materially large. " + _no_change_tail()
        )

    out.update({
        "primary_finding": finding,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_pool_magnitude": pool,
        "share_weight_composition": weight,
        "share_timing_membership": timing,
        "share_integer_rounding": rounding,
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
    })
    return out
