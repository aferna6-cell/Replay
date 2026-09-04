"""Phase 3N — first-split matched-state damage attribution (measurement only).

Stacked on Phase 3M (PR #59). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #59.

Restricts to the 3M class-(3) same-outcome-damage first splits (1059 of
2082) and decomposes the treatment−control applied `_hero_damage`
difference into winner tavern-tier, survivor-count, survivor-composition
conditional on count, proxy-vs-actual-survivor formula error, and
residual. A second view standardizes on matched pre-fight board raw /
tier mix to tell whether the gap is already in fielded state or created
by within-fight survival.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ml.phase_3m_prereg import (
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
    FIRST_DIVERGENCE_RECONCILE_IDENTITY,
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
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    REWEIGHT_ABS_TOL,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    SHARE_DOMINANT,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    share_of_class1,
)

METHODOLOGY_VERSION = "3n_v1"
PHASE_3N_SEED = 14200
PHASE_3N_LOBBIES = 500

HOLD_PRS = (
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
)

# Published 3M locks (exact).
PHASE_3M_CLASS1 = 2082
PHASE_3M_SAME_OUTCOME_DAMAGE = 1059
PHASE_3M_PRIOR_ALIVE_SET = 597
PHASE_3M_OUTCOME_FLIP = 426
PHASE_3M_INHERITED = 0
PHASE_3M_UNRECONCILED = 0
PHASE_3M_SHARE_DAMAGE = 0.5086455331412104
PHASE_3M_SHARE_PAIRING = 0.28674351585014407
PHASE_3M_SHARE_FLIP = 0.20461095100864554
PHASE_3M_FIRST_SPLIT_T5 = 1239
PHASE_3M_FIRST_SPLIT_T6 = 657
PHASE_3M_FIRST_SPLIT_T7 = 167
PHASE_3M_FIRST_SPLIT_T8 = 12
PHASE_3M_FIRST_SPLIT_T9 = 7
PHASE_3M_VERY_LATE_CLASS1 = 287
PHASE_3M_VERY_LATE_DAMAGE = 164
PHASE_3M_VERY_LATE_SHARE_DAMAGE = 0.5714285714285714

DAMAGE_COMPONENTS = (
    "winner_tavern_tier",
    "survivor_count",
    "survivor_composition",
    "proxy_formula_error",
    "residual",
)

SOURCE_COMPONENTS = (
    "pre_fight_board",
    "within_fight_survival",
    "proxy_formula_error",
    "residual",
)

APPLIED_RECONCILE_IDENTITY = (
    "winner_tavern_tier + survivor_count + survivor_composition + "
    "proxy_formula_error + residual = treatment_applied - control_applied"
)

COUNTERFACTUAL_IDENTITY = (
    "counterfactual = winner_tavern_tier + sum(actual survivor tiers)"
)

PROXY_ERROR_IDENTITY = (
    "proxy_formula_error = applied - counterfactual; "
    "applied = _hero_damage (unchanged board-mean proxy)"
)

FIELD_VS_SURVIVAL_IDENTITY = (
    "pre_fight_board + within_fight_survival + proxy_formula_error + "
    "residual = treatment_applied - control_applied"
)

ROW_DAMAGE_RECONCILE_IDENTITY = (
    "every 3M class-(3) first-split row maps to one paired fight; "
    "applied = pre_hp - post_hp; cf = winner tavern + survivor tier sum; "
    "five-way terms sum to the row applied difference"
)

NEXT_OBSERVABLE_DEFAULT = (
    "the largest applied-damage component of the 3M same-outcome first "
    "splits, ranked before any behavior change"
)


def share_of_applied(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the class-(3) treatment−control applied-damage total."""
    return share_of_class1(part, denom=denom)


def _rank_parts(parts: List[tuple]) -> List[Dict]:
    ranked = []
    for name, share in parts:
        ranked.append({
            "component": name,
            "share": None if share is None else float(share),
            "abs_share": 0.0 if share is None else abs(float(share)),
        })
    ranked.sort(key=lambda r: r["abs_share"], reverse=True)
    return ranked


def classify_row_reconcile(
    *,
    class3: bool = False,
    both_fights: bool = False,
    hp_flow_ok: bool = True,
    cf_ok: bool = True,
    five_way_ok: bool = True,
) -> str:
    """Exclusive row-level reconcile class for one class-(3) first split."""
    if not class3 or not both_fights:
        return "unreconciled"
    if not (hp_flow_ok and cf_ok and five_way_ok):
        return "unreconciled"
    return "reconciled"


def diagnose_phase_3n(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the class-(3) applied-damage gap to a matched-state source."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3n_seed": PHASE_3N_SEED,
        "phase_3n_lobbies": PHASE_3N_LOBBIES,
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
        "phase_3d_board_pool_magnitude": PHASE_3D_BOARD_POOL_MAGNITUDE,
        "phase_3e_carry_delta": PHASE_3E_CARRY_DELTA,
        "phase_3e_carry_share_of_a1": PHASE_3E_CARRY_SHARE_OF_A1,
        "phase_3e_punch_delta_carry": PHASE_3E_PUNCH_DELTA_CARRY,
        "phase_3f_uncond_share": PHASE_3F_UNCOND_SHARE,
        "phase_3f_selection_share": PHASE_3F_SELECTION_SHARE,
        "phase_3g_mixture": PHASE_3G_MIXTURE,
        "phase_3g_mixture_share": PHASE_3G_MIXTURE_SHARE,
        "phase_3g_within_share": PHASE_3G_WITHIN_SHARE,
        "phase_3g_mix_role_share": PHASE_3G_MIX_ROLE_SHARE,
        "phase_3h_leftover": PHASE_3H_LEFTOVER,
        "phase_3h_late_control": PHASE_3H_LATE_CONTROL,
        "phase_3h_late_treatment": PHASE_3H_LATE_TREATMENT,
        "phase_3h_collapse": PHASE_3H_COLLAPSE,
        "phase_3h_share_leftover": PHASE_3H_SHARE_LEFTOVER,
        "phase_3i_pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
        "phase_3i_share_pairing_schedule": PHASE_3I_SHARE_PAIRING_SCHEDULE,
        "phase_3j_eligibility": PHASE_3J_ELIGIBILITY,
        "phase_3j_share_eligibility": PHASE_3J_SHARE_ELIGIBILITY,
        "phase_3k_third_party": PHASE_3K_THIRD_PARTY,
        "phase_3k_share_third_party": PHASE_3K_SHARE_THIRD_PARTY,
        "phase_3l_same_seat_earlier": PHASE_3L_SAME_SEAT_EARLIER,
        "phase_3l_share_earlier": PHASE_3L_SHARE_EARLIER,
        "phase_3m_class1": PHASE_3M_CLASS1,
        "phase_3m_same_outcome_damage": PHASE_3M_SAME_OUTCOME_DAMAGE,
        "phase_3m_share_damage": PHASE_3M_SHARE_DAMAGE,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_tiers": list(LOW_TIERS),
        "early_turns": list(EARLY_TURNS),
        "late_turns": list(LATE_TURNS),
        "very_late_turns": list(VERY_LATE_TURNS),
        "pairing_turns": list(PAIRING_TURNS),
        "trace_from_turn": TRACE_FROM_TURN,
        "hp_walk_from_turn": HP_WALK_FROM_TURN,
        "chain_components": list(CHAIN_COMPONENTS),
        "hp_gap_components": list(HP_GAP_COMPONENTS),
        "first_divergence_components": list(FIRST_DIVERGENCE_COMPONENTS),
        "damage_components": list(DAMAGE_COMPONENTS),
        "source_components": list(SOURCE_COMPONENTS),
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
        "first_divergence_reconcile_identity": FIRST_DIVERGENCE_RECONCILE_IDENTITY,
        "row_history_divergence_identity": ROW_HISTORY_DIVERGENCE_IDENTITY,
        "applied_reconcile_identity": APPLIED_RECONCILE_IDENTITY,
        "counterfactual_identity": COUNTERFACTUAL_IDENTITY,
        "proxy_error_identity": PROXY_ERROR_IDENTITY,
        "field_vs_survival_identity": FIELD_VS_SURVIVAL_IDENTITY,
        "row_damage_reconcile_identity": ROW_DAMAGE_RECONCILE_IDENTITY,
        "history_filters_applied": False,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    attr = comparison.get("attribution") or comparison
    source = comparison.get("source") or attr
    shares = {name: attr.get(f"share_{name}") for name in DAMAGE_COMPONENTS}
    source_shares = {
        name: source.get(f"share_{name}") for name in SOURCE_COMPONENTS
    }
    ranked = _rank_parts([(n, shares[n]) for n in DAMAGE_COMPONENTS])
    ranked_source = _rank_parts(
        [(n, source_shares[n]) for n in SOURCE_COMPONENTS]
    )
    top = ranked[0]["component"] if ranked else "residual"

    def _f(name: str, bag: Dict) -> Optional[float]:
        v = bag.get(name)
        return None if v is None else float(v)

    tavern = _f("winner_tavern_tier", shares)
    count = _f("survivor_count", shares)
    composition = _f("survivor_composition", shares)
    proxy = _f("proxy_formula_error", shares)
    residual = _f("residual", shares)
    pre_fight = _f("pre_fight_board", source_shares)
    within = _f("within_fight_survival", source_shares)
    proxy_src = _f("proxy_formula_error", source_shares)
    represented = [
        s for s in (tavern, count, composition, proxy) if s is not None
    ]
    top_share = max((abs(s) for s in represented), default=0.0)

    def _no_change_tail() -> str:
        return (
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )

    if pre_fight is not None and pre_fight > SHARE_DOMINANT:
        primary = "pre_fight_board_dominates"
        next_step = (
            "Pre-fight board / composition clears ~70% of the class-(3) "
            "applied-damage gap. Next hour: trace the earliest recruit / "
            "fielded-state divergence feeding T5/T6. " + _no_change_tail()
        )
    elif within is not None and within > SHARE_DOMINANT:
        primary = "within_fight_survival_dominates"
        next_step = (
            "Within-fight survivor selection clears ~70% of the class-(3) "
            "applied-damage gap. Next hour: isolate the combat mechanic "
            "causing survivor composition. " + _no_change_tail()
        )
    elif (
        (proxy_src is not None and proxy_src > SHARE_DOMINANT)
        or (proxy is not None and proxy > SHARE_DOMINANT)
    ):
        primary = "proxy_formula_error_dominates"
        next_step = (
            "Proxy-vs-actual-survivor formula error clears ~70% of the "
            "class-(3) applied-damage gap. Independently validate against "
            "2U before any implementation. " + _no_change_tail()
        )
    elif top_share >= 0.30:
        primary = "mixed_route_to_larger"
        next_step = (
            "No single matched-state source clears ~70% of the class-(3) "
            f"applied-damage gap (top={top}). Rank components and pursue "
            "the largest. " + _no_change_tail()
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "No represented matched-state source clears ~70% of the "
            f"class-(3) applied-damage gap (top={top}). Rank the residual "
            f"before any behavior change: {NEXT_OBSERVABLE_DEFAULT}. "
            + _no_change_tail()
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "ranked_source": ranked_source,
        "share_winner_tavern_tier": tavern,
        "share_survivor_count": count,
        "share_survivor_composition": composition,
        "share_proxy_formula_error": proxy,
        "share_residual": None if residual is None else float(residual),
        "share_pre_fight_board": pre_fight,
        "share_within_fight_survival": within,
        "attribution": attr,
        "source": source,
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
    })
    return out
