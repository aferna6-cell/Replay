"""Phase 3T — earliest T5 incumbent-synth divergence attribution (measurement only).

Stacked on Phase 3S (PR #65). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, recruit, or the 2Q recruit-value objective. Confirm 11500–11699
remains reserved. Keep HOLD stack including #65.

Starting from the 4,132 paired T5/T6 bodies / 1,033 fights where 3S says
pre-play composition explains 100% of the 44.8% lifecycle term, walks each
affected incumbent backward through T5 from pre-last-play state to
recruit-turn start. Records identity/tier/recruit raw/synth at each
sell→buy→play/triple/transform/paint/scale-sync event and classifies the
FIRST synth-state divergence exclusively into:

* (1) carry-in from pre-T5/T4 state
* (2) earlier T5 membership event
* (3) 2S paint/repaint allocation event
* (4) scale-sync/input event
* (5) residual

and quantifies how each class shares the 3S lifecycle and 3R
membership-allocation increments.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ml.phase_3s_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_COMPONENTS,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
    EARLY_TURNS,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    EXCLUSIVE_FIRST_DIFF_IDENTITY,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FIRST_DIVERGENCE_COMPONENTS,
    FLOW_ABS_TOL,
    FORBIDDEN_RANGES,
    FORMATION_COMPONENTS,
    FORMATION_FLOW_RECONCILE_IDENTITY,
    FROZEN_ALPHA,
    HISTORY_LINK_IDENTITY,
    HP_FLOW_IDENTITY,
    HP_GAP_COMPONENTS,
    HP_GAP_RECONCILE_IDENTITY,
    HP_WALK_FROM_TURN,
    IMPACT_ATTACK_IDENTITY,
    INPUT_FIELDS,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_ABS_TOL,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    LOW_WINNER_START_TIERS,
    MATCHMAKING_RECONCILE_IDENTITY,
    MEMBERSHIP_EVENTS,
    MEMBERSHIP_PROPAGATION_IDENTITY,
    NESTED_ALLOCATION_IDENTITY,
    NESTED_FORMATION_IDENTITY,
    NESTED_LIFECYCLE_IDENTITY,
    NESTED_SCALE_SYNC_IDENTITY,
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
    PHASE_3P_PRIMARY_N_FIGHTS,
    PHASE_3P_PRIMARY_N_PAIRS,
    PHASE_3P_SHARE_POOL,
    PHASE_3P_SHARE_ROUNDING,
    PHASE_3P_SHARE_TIMING,
    PHASE_3P_SHARE_WEIGHT,
    PHASE_3Q_PRIMARY_N_FIGHTS,
    PHASE_3Q_PRIMARY_N_PAIRS,
    PHASE_3Q_SCALING_ABS_MASS,
    PHASE_3Q_SHARE_LIFECYCLE,
    PHASE_3Q_SHARE_RESIDUAL,
    PHASE_3Q_SHARE_SAME_STATE,
    PHASE_3Q_SHARE_SCALING,
    PHASE_3Q_T1_SYNTH_CONTROL,
    PHASE_3Q_T1_SYNTH_TREATMENT,
    PHASE_3Q_T3_SYNTH_CONTROL,
    PHASE_3Q_T3_SYNTH_TREATMENT,
    PHASE_3Q_TOTAL_ABS_MASS,
    PHASE_3R_MEMBERSHIP_ABS_MASS,
    PHASE_3R_PRIMARY_N_FIGHTS,
    PHASE_3R_PRIMARY_N_PAIRS,
    PHASE_3R_SCALING_ABS_MASS,
    PHASE_3R_SHARE_INPUT,
    PHASE_3R_SHARE_MEMBERSHIP,
    PHASE_3R_SHARE_RESIDUAL,
    PHASE_3R_SHARE_ROUNDING,
    PHASE_3R_SHARE_TIMING,
    PHASE_3R_T1_SYNTH_CONTROL,
    PHASE_3R_T1_SYNTH_TREATMENT,
    PHASE_3R_T3_SYNTH_CONTROL,
    PHASE_3R_T3_SYNTH_TREATMENT,
    PLAY_POOL_RECONCILE_IDENTITY,
    PLAY_SUBTYPES,
    POOL_FLOW_IDENTITY,
    PROXY_ERROR_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    REWEIGHT_ABS_TOL,
    ROW_DAMAGE_RECONCILE_IDENTITY,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    SAME_STATE_IDENTITY,
    SAME_TURN_SYNC_IDENTITY,
    SCALE_FLOW_RECONCILE_IDENTITY,
    SCALE_SYNC_COMPONENTS,
    SHARE_DOMINANT,
    SLOT_OPENING_CAUSES,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    share_of_applied,
)

METHODOLOGY_VERSION = "3t_v1"
PHASE_3T_SEED = 14200
PHASE_3T_LOBBIES = 500

HOLD_PRS = (
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
)

# Published 3S locks (exact).
PHASE_3S_PRIMARY_N_PAIRS = 4132
PHASE_3S_PRIMARY_N_FIGHTS = 1033
PHASE_3S_SHARE_PRE_PLAY = 1.0
PHASE_3S_SHARE_INCOMING = 0.0
PHASE_3S_SHARE_OPENING = 0.0
PHASE_3S_SHARE_ORDER = 0.0
PHASE_3S_SHARE_RESIDUAL = 0.0
PHASE_3S_LIFECYCLE_ABS_MASS = 11754.0
PHASE_3S_MEMBERSHIP_PROP_ABS_MASS = 12640.256913085292
PHASE_3S_SHARE_MEMBERSHIP_PROP = 1.0
PHASE_3S_SAME_PRE_PLAY_IDENTITY_RATE = 1.0
PHASE_3S_SAME_PRE_PLAY_STATE_RATE = 0.0
PHASE_3S_MODAL_EARLIEST = 5
PHASE_3S_T1_SYNTH_CONTROL = PHASE_3R_T1_SYNTH_CONTROL
PHASE_3S_T1_SYNTH_TREATMENT = PHASE_3R_T1_SYNTH_TREATMENT
PHASE_3S_T3_SYNTH_CONTROL = PHASE_3R_T3_SYNTH_CONTROL
PHASE_3S_T3_SYNTH_TREATMENT = PHASE_3R_T3_SYNTH_TREATMENT

PRIMARY_TURNS = (5, 6)
WALK_TURN = 5

DIVERGENCE_COMPONENTS = (
    "carry_in",
    "earlier_t5_membership",
    "paint_repaint",
    "scale_sync",
    "residual",
)

MEMBERSHIP_EVENT_KINDS = (
    "sell",
    "buy",
    "play",
    "triple",
    "transform",
    "death_cleanup",
)

NESTED_DIVERGENCE_IDENTITY = (
    "replacement_lifecycle = carry_in + earlier_t5_membership + "
    "paint_repaint + scale_sync + residual"
)

BODY_EVENT_POOL_FLOW_IDENTITY = (
    "T5 start board → each sell/buy/play/triple/transform/paint/"
    "scale-sync post-state; last snapshot in the walk equals "
    "pre-last-play incumbents; Σ sticky shares = implicit on-body "
    "pool; after paint Σ shares = painted pool; scale-sync body "
    "increments sum to the board increment"
)

EXCLUSIVE_T5_FIRST_DIFF_IDENTITY = (
    "each paired body assigns its full 3S replacement_lifecycle and "
    "3R membership_allocation to the first T5 synth-state difference: "
    "T5 recruit-turn start (carry-in), then an earlier T5 membership "
    "event, then a 2S paint/repaint, then a scale-sync/input, else residual"
)

LIFECYCLE_PROPAGATION_IDENTITY = (
    "3S replacement_lifecycle and 3R membership_allocation on the same "
    "paired body are tagged with the same exclusive T5 first-diff "
    "component; within-tier |mass| of those tagged increments is the "
    "propagation share"
)

NEXT_OBSERVABLE_DEFAULT = (
    "the largest T5 incumbent-synth first-diff term that is still "
    "observable on the pre-last-play sticky board, ranked before any "
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


def diagnose_phase_3t(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the 3S pre-play lifecycle leftover to a T5 first-diff term."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3t_seed": PHASE_3T_SEED,
        "phase_3t_lobbies": PHASE_3T_LOBBIES,
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
        "phase_3q_share_lifecycle": PHASE_3Q_SHARE_LIFECYCLE,
        "phase_3q_n_pairs": PHASE_3Q_PRIMARY_N_PAIRS,
        "phase_3r_share_membership": PHASE_3R_SHARE_MEMBERSHIP,
        "phase_3r_n_pairs": PHASE_3R_PRIMARY_N_PAIRS,
        "phase_3s_share_pre_play": PHASE_3S_SHARE_PRE_PLAY,
        "phase_3s_n_pairs": PHASE_3S_PRIMARY_N_PAIRS,
        "phase_3m_same_outcome_damage": PHASE_3M_SAME_OUTCOME_DAMAGE,
        "primary_turns": list(PRIMARY_TURNS),
        "walk_turn": WALK_TURN,
        "divergence_components": list(DIVERGENCE_COMPONENTS),
        "membership_event_kinds": list(MEMBERSHIP_EVENT_KINDS),
        "input_fields": list(INPUT_FIELDS),
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
        "nested_scale_sync_identity": NESTED_SCALE_SYNC_IDENTITY,
        "scale_flow_reconcile_identity": SCALE_FLOW_RECONCILE_IDENTITY,
        "same_turn_sync_identity": SAME_TURN_SYNC_IDENTITY,
        "nested_formation_identity": NESTED_FORMATION_IDENTITY,
        "formation_flow_reconcile_identity": FORMATION_FLOW_RECONCILE_IDENTITY,
        "exclusive_first_diff_identity": EXCLUSIVE_FIRST_DIFF_IDENTITY,
        "membership_propagation_identity": MEMBERSHIP_PROPAGATION_IDENTITY,
        "nested_divergence_identity": NESTED_DIVERGENCE_IDENTITY,
        "body_event_pool_flow_identity": BODY_EVENT_POOL_FLOW_IDENTITY,
        "exclusive_t5_first_diff_identity": EXCLUSIVE_T5_FIRST_DIFF_IDENTITY,
        "lifecycle_propagation_identity": LIFECYCLE_PROPAGATION_IDENTITY,
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
        for name in DIVERGENCE_COMPONENTS
    }

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    carry = _f("carry_in")
    membership = _f("earlier_t5_membership")
    paint = _f("paint_repaint")
    scale = _f("scale_sync")
    residual = _f("residual")
    ranked = _rank_parts([
        ("carry_in", carry),
        ("earlier_t5_membership", membership),
        ("paint_repaint", paint),
        ("scale_sync", scale),
        ("residual", residual),
    ])
    top = ranked[0]["component"] if ranked else "residual"
    modal_kind = (
        (comparison.get("attribution") or comparison).get(
            "modal_first_event_kind"
        )
        or primary.get("modal_first_event_kind")
    )

    def _no_change_tail() -> str:
        return (
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not change recruit defaults; do not burn confirm."
        )

    if carry is not None and carry > SHARE_DOMINANT:
        finding = "carry_in_dominates"
        next_step = (
            "Carry-in from pre-T5/T4 state clears ~70% of the 3S "
            "pre-play lifecycle leftover. Next hour: trace the "
            "originating T4 state/fight. "
            + _no_change_tail()
        )
    elif membership is not None and membership > SHARE_DOMINANT:
        finding = "earlier_t5_membership_dominates"
        kind_note = (
            f" Modal first membership subtype is {modal_kind}."
            if modal_kind else
            " Isolate the lifecycle subtype of that earlier T5 event."
        )
        next_step = (
            "An earlier T5 membership event (sell/buy/play/triple/"
            "transform) clears ~70% of the 3S pre-play lifecycle leftover."
            + kind_note
            + " Next hour: isolate that lifecycle subtype. "
            + _no_change_tail()
        )
    elif paint is not None and paint > SHARE_DOMINANT:
        finding = "paint_repaint_dominates"
        next_step = (
            "A 2S paint/repaint allocation event clears ~70% of the 3S "
            "pre-play lifecycle leftover. Next: independently audit "
            "sticky-vs-repaint representation against the scientific "
            "model before any implementation. "
            + _no_change_tail()
        )
    elif scale is not None and scale > SHARE_DOMINANT:
        finding = "scale_sync_dominates"
        next_step = (
            "A T5 scale-sync/input event clears ~70% of the 3S pre-play "
            "lifecycle leftover. Next hour: trace its first differing "
            "input. "
            + _no_change_tail()
        )
    else:
        finding = "ranked_residual_needs_next_observable"
        next_step = (
            "No carry-in / earlier T5 membership / 2S paint / "
            f"scale-sync term clears ~70% of the 3S leftover (top={top}). "
            f"Rank components and pursue the largest residual "
            f"observable: {NEXT_OBSERVABLE_DEFAULT}. "
            + _no_change_tail()
        )

    hold_flags = {f"keep_pr_{n}_hold": True for n in HOLD_PRS}
    out.update({
        "primary_finding": finding,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_carry_in": carry,
        "share_earlier_t5_membership": membership,
        "share_paint_repaint": paint,
        "share_scale_sync": scale,
        "share_residual": residual,
        "modal_first_event_kind": modal_kind,
        "primary": primary,
        **hold_flags,
    })
    return out
