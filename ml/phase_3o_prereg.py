"""Phase 3O — matched-board survivor-mechanic attribution (measurement only).

Stacked on Phase 3N (PR #60). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #60.

Restricts primary analysis to T5/T6 3N class-(3) first-split fights and
decomposes the reproduced within-tier survival term
(B ≈ +0.688 / split, 100% of the CF gap) into:

* (1) start-body combat strength / HP / synth within printed tier
* (2) attack opportunity / board slot
* (3) target exposure / taunt
* (4) represented keyword / lethal mechanics
* (5) teammate / board protection
* (6) residual
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ml.phase_3n_prereg import (
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
from ml.phase_2z_prereg import target_bin
from ml.phase_3a_prereg import (
    cleave_bin,
    ds_bin,
    ordinary_bin,
    poison_bin,
    slot_bin,
    soc_bin,
)
from ml.phase_2z_prereg import gen_bin

METHODOLOGY_VERSION = "3o_v1"
PHASE_3O_SEED = 14200
PHASE_3O_LOBBIES = 500

HOLD_PRS = (
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
)

# Published 3N locks (exact).
PHASE_3N_CLASS3 = 1059
PHASE_3N_WITHIN_TIER_B = 0.6883852691218131
PHASE_3N_CF_DELTA = 0.6883852691218131
PHASE_3N_SHARE_WITHIN_TIER = 1.0
PHASE_3N_CLASS3_T5 = 884
PHASE_3N_CLASS3_T6 = 149
PHASE_3N_CLASS3_T7 = 26
PHASE_3N_P_SURVIVE_T1_CONTROL = 0.6589724497393894
PHASE_3N_P_SURVIVE_T1_TREATMENT = 0.4058078927773641
PHASE_3N_P_SURVIVE_T3_CONTROL = 0.3748870822041554
PHASE_3N_P_SURVIVE_T3_TREATMENT = 0.6124661246612466
PHASE_3N_P_SURVIVE_T4_CONTROL = 0.3088235294117647
PHASE_3N_P_SURVIVE_T4_TREATMENT = 0.6176470588235294

PRIMARY_TURNS = (5, 6)
N_DECILES = 10
N_TEAM_BINS = 5
SLOT_BIN_CAP = 4
N_TARGET_BINS = 3
N_DS_BINS = 3
N_POISON_BINS = 2
N_CLEAVE_BINS = 3
N_SOC_BINS = 2
N_GEN_BINS = 2

MECHANIC_COMPONENTS = (
    "start_stats",
    "attack_opportunity",
    "target_exposure",
    "represented_keywords",
    "teammate_protection",
    "residual",
)

WALK_LEAF_NAMES = (
    "recruit_mix",
    "synthetic_allocation",
    "start_hp",
    "slot_opportunity",
    "target_exposure",
    "divine_shield",
    "poison_venomous",
    "cleave",
    "start_of_combat",
    "generated",
    "teammate_protection",
    "still_unexplained",
)

START_STATS_PARTS = (
    "recruit_mix",
    "synthetic_allocation",
    "start_hp",
)
KEYWORD_PARTS = (
    "divine_shield",
    "poison_venomous",
    "cleave",
    "start_of_combat",
    "generated",
)

BODY_EVENT_RECONCILE_IDENTITY = (
    "every T5/T6 class-(3) winner starting body maps to one combat-start "
    "row; survivor ids ⊆ start ∪ created; event-count flags reconcile; "
    "Σ synthetic shares = winner pool; nested parts sum to within-tier B"
)

NESTED_SURVIVAL_IDENTITY = (
    "within_tier_B = start_stats + attack_opportunity + target_exposure + "
    "represented_keywords + teammate_protection + residual"
)

NEXT_OBSERVABLE_DEFAULT = (
    "the largest T5/T6 within-tier survival component that is still "
    "observable on matched-board starting bodies, ranked before any "
    "behavior change"
)


def keyword_bin(row: Dict) -> int:
    """1 if any represented DS / poison / cleave / SOC / generated flag is on."""
    if ds_bin(row) > 0:
        return 1
    if poison_bin(row) > 0:
        return 1
    if cleave_bin(row) > 0:
        return 1
    if soc_bin(row) > 0:
        return 1
    if gen_bin(row) > 0:
        return 1
    return 0


def attack_opp_bin(row: Dict) -> int:
    """0 died before first attack / 1 never attacked / 2 attacked."""
    n = int(row.get("n_attacks") or 0)
    if row.get("death_before_first_attack") or (
        (not row.get("survived")) and n == 0
    ):
        return 0
    if n <= 0:
        return 1
    return 2


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


def diagnose_phase_3o(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 3N within-tier survival to a matched-board combat mechanic."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3o_seed": PHASE_3O_SEED,
        "phase_3o_lobbies": PHASE_3O_LOBBIES,
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
        "phase_3m_same_outcome_damage": PHASE_3M_SAME_OUTCOME_DAMAGE,
        "primary_turns": list(PRIMARY_TURNS),
        "mechanic_components": list(MECHANIC_COMPONENTS),
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
        "body_event_reconcile_identity": BODY_EVENT_RECONCILE_IDENTITY,
        "nested_survival_identity": NESTED_SURVIVAL_IDENTITY,
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
        name: primary.get(f"share_of_B_{name}")
        for name in MECHANIC_COMPONENTS
    }

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    start = _f("start_stats")
    attack = _f("attack_opportunity")
    target = _f("target_exposure")
    keywords = _f("represented_keywords")
    teammate = _f("teammate_protection")
    residual = _f("residual")
    ranked = _rank_parts([
        ("start_stats", start),
        ("attack_opportunity", attack),
        ("target_exposure", target),
        ("represented_keywords", keywords),
        ("teammate_protection", teammate),
        ("residual", residual),
    ])
    top = ranked[0]["component"] if ranked else "residual"

    def _no_change_tail() -> str:
        return (
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )

    if start is not None and start > SHARE_DOMINANT:
        finding = "start_stats_synth_dominates"
        next_step = (
            "Within-tier start-body combat strength / HP / synth clears "
            "~70% of the T5/T6 class-(3) survival gap. Next hour: trace "
            "why matched-tier boards allocate stats differently. "
            + _no_change_tail()
        )
    elif attack is not None and attack > SHARE_DOMINANT:
        finding = "attack_opportunity_dominates"
        next_step = (
            "Attack opportunity / slot clears ~70% of the T5/T6 class-(3) "
            "survival gap. Next hour: audit positioning / initiative "
            "fidelity. " + _no_change_tail()
        )
    elif target is not None and target > SHARE_DOMINANT:
        finding = "target_exposure_dominates"
        next_step = (
            "Target exposure / taunt clears ~70% of the T5/T6 class-(3) "
            "survival gap. Next hour: audit targeting / taunt fidelity. "
            + _no_change_tail()
        )
    elif keywords is not None and keywords > SHARE_DOMINANT:
        finding = "represented_keywords_dominates"
        next_step = (
            "Represented keyword / lethal mechanics clear ~70% of the "
            "T5/T6 class-(3) survival gap. Next hour: isolate that "
            "mechanic. " + _no_change_tail()
        )
    else:
        finding = "ranked_residual_needs_next_observable"
        next_step = (
            "No represented start-stat / slot / target / keyword term "
            f"clears ~70% of the T5/T6 class-(3) survival gap (top={top}). "
            f"Rank components and pursue the largest residual observable: "
            f"{NEXT_OBSERVABLE_DEFAULT}. " + _no_change_tail()
        )

    out.update({
        "primary_finding": finding,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_start_stats": start,
        "share_attack_opportunity": attack,
        "share_target_exposure": target,
        "share_represented_keywords": keywords,
        "share_teammate_protection": teammate,
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
    })
    return out
