"""Phase 3J — matchmaking divergence attribution (measurement only).

Stacked on Phase 3I (PR #55). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #55.

Reproduces the 3I pairing-schedule leftover (5952 control punch rows) and
splits those rows exclusively into:

* (1) alive / ghost eligibility divergence
* (2) pairing-history / legal-candidate divergence with the same alive set
* (3) same candidate set but RNG / order choice divergence
* (4) missing / unreconciled
"""

from __future__ import annotations

from typing import Dict, List, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3j_v1"
PHASE_3J_SEED = 14200
PHASE_3J_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
LOW_TIERS = (1, 2, 3)
LOW_WINNER_START_TIERS = LOW_TIERS
EARLY_TURNS = (7, 8, 9)
LATE_TURNS = (10, 11, 12, 13, 14)
VERY_LATE_TURNS = (12, 13, 14)
PAIRING_TURNS = LATE_TURNS

SHARE_DOMINANT = 0.70
FLOW_ABS_TOL = 1.0
REWEIGHT_ABS_TOL = 1e-6
LINEAGE_ABS_TOL = 1e-9

FORBIDDEN_RANGES = (
    (8000, 8199),
    (9000, 9999),
    (10000, 10199),
    (10200, 10699),
    (11000, 11499),
    (11500, 11699),  # confirmation — reserved
    (11700, 12199),  # 2N
    (12200, 12699),  # 2O
    (12700, 13199),  # 2P
    (13200, 13699),  # 2Q
    (13700, 14199),  # 2R
)

HOLD_PRS = (
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55,
)

POOL_FLOW_IDENTITY = (
    "post = pre + add - represented_loss_or_transfer"
)

IMPACT_ATTACK_IDENTITY = (
    "impact_attack = start_recruit + start_pool_share + combat_delta"
)

HISTORY_LINK_IDENTITY = (
    "punch.opp_carry = seat_history[turn].attack_pool_recruit_start"
)

WEIGHT_RECONCILIATION_IDENTITY = (
    "sum_k n_arm(k) = N_arm; sum_k w_arm(k) = 1; "
    "mixture + within_cell + leftover = unpaired_punch_delta_carry"
)

LINEAGE_IDENTITY = (
    "t1t3_end = t1t3_start + t1t3_added - t1t3_removed"
)

PAIRED_SEAT_IDENTITY = (
    "paired (seed, seat) in both arms; first_loss_turn is the first "
    "instrumented turn the seat's combat-start T1–T3 count hits 0"
)

PAIRING_IDENTITY = (
    "same pairing iff both fights are live and opponent_seat matches "
    "at (seed, leftover_winner_seat, turn)"
)

LEFTOVER_RECONCILE_IDENTITY = (
    "pairing_schedule + outcome_flip + survivor_substitution + residual "
    "= 3H leftover (treatment alive and still fields T1–T3)"
)

CANDIDATE_CHOICE_IDENTITY = (
    "chosen opponent is an element of the logged legal candidate set "
    "(other alive seats, plus ghost/bye iff the lobby is odd and eligible)"
)

MATCHMAKING_RECONCILE_IDENTITY = (
    "eligibility + history_legal + rng_order + unreconciled "
    "= 3I pairing_schedule (5952 leftover punch rows)"
)

# Published 3D / 3E / 3F / 3G / 3H / 3I locks (exact).
PHASE_3D_BOARD_POOL_MAGNITUDE = 0.4216721428553852
PHASE_3E_CARRY_DELTA = 0.30513688784757187
PHASE_3E_CARRY_SHARE_OF_A1 = 0.7236353954551374
PHASE_3E_PUNCH_DELTA_CARRY = -196.33317557443002
PHASE_3F_UNCOND_PAIRED_DELTA = -17.83493589743591
PHASE_3F_UNCOND_SHARE = 0.09084015396406948
PHASE_3F_SELECTION_SHARE = 0.9091598460359305
PHASE_3G_MIXTURE = -196.52943934946725
PHASE_3G_MIXTURE_SHARE = 1.0009996465165045
PHASE_3G_WITHIN_CELL = 0.19626377503730166
PHASE_3G_WITHIN_SHARE = -0.0009996465165047867
PHASE_3G_ROLE_ALIVE = 36.35037820310066
PHASE_3G_ROLE_SHARE = -0.1851463874953737
PHASE_3G_MIX_ROLE_SHARE = 0.8158532590211308
PHASE_3G_N_CONTROL = 54223
PHASE_3G_N_TREATMENT = 50116
PHASE_3H_LATE_CONTROL = 17924
PHASE_3H_LATE_TREATMENT = 4273
PHASE_3H_COLLAPSE = 13651
PHASE_3H_LEFTOVER = 7155
PHASE_3H_ELIMINATION = 6550
PHASE_3H_OFFER_SHIFT = 4219
PHASE_3H_SHARE_LEFTOVER = 0.5241374258296095
PHASE_3I_PAIRING_SCHEDULE = 5952
PHASE_3I_OUTCOME_FLIP = 668
PHASE_3I_SURVIVOR_SUBSTITUTION = 292
PHASE_3I_RESIDUAL = 243
PHASE_3I_DIFFERENT_OPPONENT = 5009
PHASE_3I_KIND_MISMATCH = 943
PHASE_3I_SHARE_PAIRING_SCHEDULE = 0.8318658280922432
PHASE_3I_SHARE_OUTCOME_FLIP = 0.093361285814116
PHASE_3I_SHARE_SURVIVOR_SUBSTITUTION = 0.04081062194269741
PHASE_3I_SHARE_RESIDUAL = 0.033962264150943396

MATCHMAKING_COMPONENTS = (
    "eligibility",
    "history_legal",
    "rng_order",
    "unreconciled",
)

REPRESENTED_SOURCE = MATCHMAKING_COMPONENTS

NEXT_OBSERVABLE_DEFAULT = (
    "the largest matchmaking component of the 3I pairing-schedule leftover, "
    "ranked before any behavior change"
)

GHOST_TOKEN = "ghost"
BYE_TOKEN = "bye"


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3J may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3J must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3J seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def share_of_schedule(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the 3I pairing-schedule leftover (5952 rows)."""
    if part is None or denom is None or abs(float(denom)) < 1e-12:
        return None
    return float(part) / float(denom)


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


def classify_matchmaking_gap(
    *,
    control_present: bool = False,
    treatment_present: bool = False,
    leftover_alive_control: bool = False,
    leftover_alive_treatment: bool = False,
    choice_in_candidates_control: bool = False,
    choice_in_candidates_treatment: bool = False,
    alive_sets_equal: bool = False,
    ghost_bye_eligible_equal: bool = False,
    legal_candidates_equal: bool = False,
    chosen_equal: bool = False,
) -> str:
    """Exclusive class for one 3I pairing-schedule leftover punch row.

    Priority: missing / leftover seat absent / choice not in the logged
    candidate set → unreconciled; alive-set or ghost/bye eligibility
    differs → eligibility; same eligibility but legal candidates differ
    → history_legal; same candidates, different chosen opponent →
    rng_order; otherwise unreconciled (same choice, or unparseable).
    """
    if not (
        control_present and treatment_present
        and leftover_alive_control and leftover_alive_treatment
        and choice_in_candidates_control and choice_in_candidates_treatment
    ):
        return "unreconciled"
    if (not alive_sets_equal) or (not ghost_bye_eligible_equal):
        return "eligibility"
    if not legal_candidates_equal:
        return "history_legal"
    if not chosen_equal:
        return "rng_order"
    return "unreconciled"


def diagnose_phase_3j(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the 3I pairing-schedule leftover to a matchmaking component."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3j_seed": PHASE_3J_SEED,
        "phase_3j_lobbies": PHASE_3J_LOBBIES,
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
        "phase_3i_different_opponent": PHASE_3I_DIFFERENT_OPPONENT,
        "phase_3i_kind_mismatch": PHASE_3I_KIND_MISMATCH,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_tiers": list(LOW_TIERS),
        "early_turns": list(EARLY_TURNS),
        "late_turns": list(LATE_TURNS),
        "very_late_turns": list(VERY_LATE_TURNS),
        "pairing_turns": list(PAIRING_TURNS),
        "matchmaking_components": list(MATCHMAKING_COMPONENTS),
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
        "history_filters_applied": False,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    attr = comparison.get("attribution") or comparison
    shares = {name: attr.get(f"share_{name}") for name in MATCHMAKING_COMPONENTS}
    ranked = _rank_parts([(n, shares[n]) for n in MATCHMAKING_COMPONENTS])
    top = ranked[0]["component"] if ranked else "unreconciled"

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    eligibility = _f("eligibility")
    history_legal = _f("history_legal")
    rng_order = _f("rng_order")
    unreconciled = _f("unreconciled")
    represented = [s for s in (eligibility, history_legal, rng_order) if s is not None]
    top_share = max((abs(s) for s in represented), default=0.0)

    if eligibility is not None and eligibility > SHARE_DOMINANT:
        primary = "eligibility_dominates"
        next_step = (
            "Alive-set / ghost-bye eligibility clears ~70% of the 3I "
            "pairing-schedule leftover. Next hour: trace elimination "
            "timing upstream of T10–T14 pairing. Do not apply a scaling "
            "correction; do not rewrite 2Q; do not change `_hero_damage`; "
            "do not retune scaling constants; do not burn confirm."
        )
    elif history_legal is not None and history_legal > SHARE_DOMINANT:
        primary = "history_legal_dominates"
        next_step = (
            "Pairing-history / legal-candidate divergence (same alive set) "
            "clears ~70% of the 3I pairing-schedule leftover. Next hour: "
            "audit no-repeat / history-rule fidelity. Do not apply a "
            "scaling correction; do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not "
            "burn confirm."
        )
    elif rng_order is not None and rng_order > SHARE_DOMINANT:
        primary = "rng_order_dominates"
        next_step = (
            "Same candidate set, different RNG / order choice clears ~70% "
            "of the 3I pairing-schedule leftover. Next hour: audit pairing "
            "RNG coupling / algorithm against BG rules before any change. "
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )
    elif top_share >= 0.30:
        primary = "mixed_route_to_larger"
        next_step = (
            "No single matchmaking class clears ~70% of the 3I "
            f"pairing-schedule leftover (top={top}). Rank components and "
            "pursue the largest. Do not apply a scaling correction; do "
            "not rewrite 2Q; do not change `_hero_damage`; do not retune "
            "scaling constants; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "No represented matchmaking class clears ~70% of the 3I "
            f"pairing-schedule leftover (top={top}). Rank the residual "
            f"before any behavior change: {NEXT_OBSERVABLE_DEFAULT}. Do "
            "not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_eligibility": eligibility,
        "share_history_legal": history_legal,
        "share_rng_order": rng_order,
        "share_unreconciled": None if unreconciled is None else float(unreconciled),
        "attribution": attr,
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
    })
    return out
