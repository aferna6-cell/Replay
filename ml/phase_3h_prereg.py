"""Phase 3H — low-tier board-retention lifecycle attribution (measurement only).

Stacked on Phase 3G (PR #53). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #53.

Reproduces the 3G turn × winner-start-tier mixture (−196.53, 100.1% of the
#51 −196 punch-row crater; within-cell carry ~0) and attributes the
treatment collapse of late T1–T3 punch rows to exclusive lifecycle
classes on paired (seed, seat) T7–T14 trajectories:

* (1) full-board sell→buy→play replacement driven by 2Q selection
* (2) open-slot purchase / board fill
* (3) tavern-tier / offer-availability shift
* (4) generated / transform / triple effects if represented
* (5) alive / elimination selection
"""

from __future__ import annotations

from typing import Dict, List, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3h_v1"
PHASE_3H_SEED = 14200
PHASE_3H_LOBBIES = 500
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
    50, 51, 52, 53,
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

# Published 3D / 3E / 3F / 3G locks (exact).
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

LIFECYCLE_COMPONENTS = (
    "full_board_2q_replacement",
    "open_slot_fill",
    "tavern_offer_shift",
    "generated_transform_triple",
    "alive_elimination",
)

REPRESENTED_SOURCE = LIFECYCLE_COMPONENTS

NEXT_OBSERVABLE_DEFAULT = (
    "the largest represented lifecycle component of late T1–T3 punch-row "
    "collapse, ranked before any behavior change"
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3H may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3H must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3H seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def share_of_collapse(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the late T1–T3 punch-row count collapse (C − T)."""
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


def classify_t1t3_exit(
    *,
    sold: bool = False,
    board_full: bool = False,
    replacement_completed: bool = False,
    shop_t1t3_offers: int = 0,
    tavern_tier: int = 1,
    triple: bool = False,
    generated: bool = False,
    open_slot_higher_tier_play: bool = False,
    seat_died: bool = False,
    had_t1t3_at_death: bool = False,
) -> str:
    """Exclusive class for a T1–T3 exit / last-body loss.

    Priority: represented generate/triple → death-with-T1-T3 → shop-empty
    full-board replace (availability) → full-board 2Q replace → open-slot
    fill / non-full sell → leftover tavern/offer residual.
    """
    if triple or generated:
        return "generated_transform_triple"
    if seat_died and had_t1t3_at_death:
        return "alive_elimination"
    if sold and board_full and replacement_completed and int(shop_t1t3_offers) <= 0:
        return "tavern_offer_shift"
    if sold and board_full and replacement_completed:
        return "full_board_2q_replacement"
    if open_slot_higher_tier_play or (sold and not board_full):
        return "open_slot_fill"
    if int(shop_t1t3_offers) <= 0 and int(tavern_tier) >= 4:
        return "tavern_offer_shift"
    if seat_died:
        return "alive_elimination"
    return "tavern_offer_shift"


def diagnose_phase_3h(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route late T1–T3 punch-row collapse to a lifecycle component."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3h_seed": PHASE_3H_SEED,
        "phase_3h_lobbies": PHASE_3H_LOBBIES,
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
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_tiers": list(LOW_TIERS),
        "early_turns": list(EARLY_TURNS),
        "late_turns": list(LATE_TURNS),
        "very_late_turns": list(VERY_LATE_TURNS),
        "lifecycle_components": list(LIFECYCLE_COMPONENTS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "lineage_identity": LINEAGE_IDENTITY,
        "paired_seat_identity": PAIRED_SEAT_IDENTITY,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    attr = comparison.get("attribution") or comparison
    shares = {
        name: attr.get(f"share_{name}")
        for name in LIFECYCLE_COMPONENTS
    }
    share_left = attr.get("share_leftover")
    ranked = _rank_parts(
        [(n, shares[n]) for n in LIFECYCLE_COMPONENTS] + [("leftover", share_left)]
    )
    top = ranked[0]["component"] if ranked else "leftover"

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    repl = _f("full_board_2q_replacement")
    offer = _f("tavern_offer_shift")
    elim = _f("alive_elimination")
    fill = _f("open_slot_fill")
    gen = _f("generated_transform_triple")
    represented = [s for s in (repl, offer, elim, fill, gen) if s is not None]
    top_share = max((abs(s) for s in represented), default=0.0)

    if repl is not None and repl > SHARE_DOMINANT:
        primary = "full_board_2q_replacement_dominates"
        next_step = (
            "Full-board 2Q sell→buy→play replacements clear ~70% of the "
            "late T1–T3 punch-row collapse. Next hour: test whether that "
            "replacement-driven tier turnover is realistic vs the best "
            "available board reference, before any behavior change. Do not "
            "apply a scaling correction; do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )
    elif offer is not None and offer > SHARE_DOMINANT:
        primary = "tavern_offer_availability_dominates"
        next_step = (
            "Tavern-tier / shop-offer availability clears ~70% of the late "
            "T1–T3 punch-row collapse. Next hour: audit progression / "
            "shop-generation fidelity. Do not apply a scaling correction; "
            "do not rewrite 2Q; do not change `_hero_damage`; do not retune "
            "scaling constants; do not burn confirm."
        )
    elif elim is not None and elim > SHARE_DOMINANT:
        primary = "alive_elimination_selection_dominates"
        next_step = (
            "Alive / elimination selection clears ~70% of the late T1–T3 "
            "punch-row collapse. Next hour: trace damage / elimination "
            "mediation. Do not apply a scaling correction; do not rewrite "
            "2Q; do not change `_hero_damage`; do not retune scaling "
            "constants; do not burn confirm."
        )
    elif (
        fill is not None and fill > SHARE_DOMINANT
    ) or (
        gen is not None and gen > SHARE_DOMINANT
    ):
        bigger = "open_slot_fill" if (fill or 0.0) >= (gen or 0.0) else (
            "generated_transform_triple"
        )
        primary = f"{bigger}_dominates"
        next_step = (
            f"{bigger} clears ~70% of the late T1–T3 punch-row collapse. "
            "Rank that component before any behavior change. Do not apply "
            "a scaling correction; do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )
    elif top_share >= 0.30:
        primary = "mixed_route_to_larger"
        larger = top
        next_step = (
            "No single lifecycle class clears ~70% of the late T1–T3 "
            f"punch-row collapse (top={larger}). Rank components and "
            "pursue the largest. Do not apply a scaling correction; do "
            "not rewrite 2Q; do not change `_hero_damage`; do not retune "
            "scaling constants; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "No represented lifecycle class clears ~70% of the late "
            f"T1–T3 punch-row collapse (top={top}). Rank the residual "
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
        "share_full_board_2q_replacement": repl,
        "share_open_slot_fill": fill,
        "share_tavern_offer_shift": offer,
        "share_generated_transform_triple": gen,
        "share_alive_elimination": elim,
        "share_leftover": None if share_left is None else float(share_left),
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
    })
    return out
