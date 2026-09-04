"""Phase 3I — T1–T3 pairing / who-wins attribution (measurement only).

Stacked on Phase 3H (PR #54). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #54.

Reproduces the 3H late T1–T3 leftover (7155 control punch rows whose
paired treatment seat is alive and still fields ≥1 T1–T3 body) and
decomposes those missing treatment low-tier winner-start punches into:

* (1) different opponent / pairing schedule
* (2) same pairing but treatment loses / ties instead of wins
* (3) treatment wins but a different higher-tier survivor generates
  the punch row
* (4) residual
"""

from __future__ import annotations

from typing import Dict, List, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3i_v1"
PHASE_3I_SEED = 14200
PHASE_3I_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
LOW_TIERS = (1, 2, 3)
LOW_WINNER_START_TIERS = LOW_TIERS
EARLY_TURNS = (7, 8, 9)
LATE_TURNS = (10, 11, 12, 13, 14)
VERY_LATE_TURNS = (12, 13, 14)

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
    50, 51, 52, 53, 54,
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

# Published 3D / 3E / 3F / 3G / 3H locks (exact).
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
PHASE_3H_SHARE_ELIMINATION = 0.479818328327595
PHASE_3H_SHARE_OFFER = 0.30906160720826314

PAIRING_COMPONENTS = (
    "pairing_schedule",
    "outcome_flip",
    "survivor_substitution",
    "residual",
)

REPRESENTED_SOURCE = PAIRING_COMPONENTS

NEXT_OBSERVABLE_DEFAULT = (
    "the largest pairing / who-wins component of the 3H leftover, "
    "ranked before any behavior change"
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3I may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3I must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3I seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def share_of_leftover(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the 3H leftover (control rows still fielding T1–T3)."""
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


def classify_pairing_gap(
    *,
    same_pairing: bool = False,
    treatment_wins: bool = False,
    treatment_tie_or_loss: bool = False,
    treatment_t1t3_punches: int = 0,
    uncovered: bool = True,
) -> str:
    """Exclusive class for one leftover control T1–T3 punch row.

    Priority: different pairing → same pairing lose/tie → same pairing
    win but this row is not covered by a treatment T1–T3 punch
    (higher-tier survivor substitution) → residual.
    """
    if not same_pairing:
        return "pairing_schedule"
    if treatment_tie_or_loss or not treatment_wins:
        return "outcome_flip"
    if uncovered or int(treatment_t1t3_punches) <= 0:
        return "survivor_substitution"
    return "residual"


def diagnose_phase_3i(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the 3H leftover to a pairing / who-wins component."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3i_seed": PHASE_3I_SEED,
        "phase_3i_lobbies": PHASE_3I_LOBBIES,
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
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_tiers": list(LOW_TIERS),
        "early_turns": list(EARLY_TURNS),
        "late_turns": list(LATE_TURNS),
        "very_late_turns": list(VERY_LATE_TURNS),
        "pairing_components": list(PAIRING_COMPONENTS),
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
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    attr = comparison.get("attribution") or comparison
    shares = {name: attr.get(f"share_{name}") for name in PAIRING_COMPONENTS}
    ranked = _rank_parts([(n, shares[n]) for n in PAIRING_COMPONENTS])
    top = ranked[0]["component"] if ranked else "residual"

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    pairing = _f("pairing_schedule")
    outcome = _f("outcome_flip")
    survivor = _f("survivor_substitution")
    residual = _f("residual")
    represented = [s for s in (pairing, outcome, survivor) if s is not None]
    top_share = max((abs(s) for s in represented), default=0.0)

    if pairing is not None and pairing > SHARE_DOMINANT:
        primary = "opponent_schedule_dominates"
        next_step = (
            "Different opponent / pairing schedule clears ~70% of the 3H "
            "leftover. Next hour: audit matchmaking / pairing fidelity. "
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )
    elif outcome is not None and outcome > SHARE_DOMINANT:
        primary = "same_pairing_outcome_flip_dominates"
        next_step = (
            "Same-pairing outcome flips (treatment loses / ties) clear "
            "~70% of the 3H leftover. Next hour: trace the earliest "
            "combat-strength / mechanic variable causing those flips. "
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )
    elif survivor is not None and survivor > SHARE_DOMINANT:
        primary = "survivor_substitution_dominates"
        next_step = (
            "Survivor substitution (treatment wins but a higher-tier "
            "survivor generates the punch row) clears ~70% of the 3H "
            "leftover. Next hour: trace within-board survival selection. "
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )
    elif top_share >= 0.30:
        primary = "mixed_route_to_larger"
        next_step = (
            "No single pairing / who-wins class clears ~70% of the 3H "
            f"leftover (top={top}). Rank components and pursue the "
            "largest. Do not apply a scaling correction; do not rewrite "
            "2Q; do not change `_hero_damage`; do not retune scaling "
            "constants; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "No represented pairing / who-wins class clears ~70% of the "
            f"3H leftover (top={top}). Rank the residual before any "
            f"behavior change: {NEXT_OBSERVABLE_DEFAULT}. Do not apply a "
            "scaling correction; do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not "
            "burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_pairing_schedule": pairing,
        "share_outcome_flip": outcome,
        "share_survivor_substitution": survivor,
        "share_residual": None if residual is None else float(residual),
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
    })
    return out
