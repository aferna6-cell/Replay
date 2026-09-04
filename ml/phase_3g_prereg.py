"""Phase 3G — punch-sample selection decomposition (measurement only).

Stacked on Phase 3F (PR #52). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #52.

Reproduces the #51 / #52 unpaired punch-row Δcarry (−196) and splits
treatment − control with a symmetric Kitagawa/Oaxaca reweight on common
support into:

* (1) winner-start tier × turn mixture weights (who produces punch rows)
* (2) opponent carry conditional on matched turn + winner-start tier
* (3) winner/loser role + alive/elimination selection
* (4) leftover
"""

from __future__ import annotations

from typing import Dict, List, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3g_v1"
PHASE_3G_SEED = 14200
PHASE_3G_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
LOW_WINNER_START_TIERS = (1, 2, 3)
EARLY_TURNS = (7, 8, 9)

SHARE_DOMINANT = 0.70
FLOW_ABS_TOL = 1.0
REWEIGHT_ABS_TOL = 1e-6

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
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52,
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

# Published 3D / 3E / 3F locks (exact).
PHASE_3D_BOARD_POOL_MAGNITUDE = 0.4216721428553852
PHASE_3E_CARRY_DELTA = 0.30513688784757187
PHASE_3E_CARRY_SHARE_OF_A1 = 0.7236353954551374
PHASE_3E_PUNCH_DELTA_CARRY = -196.33317557443002
PHASE_3F_UNCOND_PAIRED_DELTA = -17.83493589743591
PHASE_3F_UNCOND_SHARE = 0.09084015396406948
PHASE_3F_SELECTION_SHARE = 0.9091598460359305

NEXT_OBSERVABLE_DEFAULT = (
    "the upstream gameplay / composition process that changes "
    "winner-tier × turn matchups, or the earliest gameplay state "
    "variable that selects within matched turn × winner-start tier, "
    "ranked by the larger piece of the #51 −196 crater"
)

REPRESENTED_SOURCE = (
    "mixture_turn_winner_tier",
    "within_cell_opponent_carry",
    "role_alive_selection",
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3G may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3G must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3G seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def share_of_crater(
    part: Optional[float],
    *,
    denom: float = PHASE_3E_PUNCH_DELTA_CARRY,
) -> Optional[float]:
    """Signed share: part / Δ. Same sign as the crater → positive contribution."""
    if part is None or abs(float(denom)) < 1e-12:
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


def _alive_bin(n_alive: Optional[int]) -> str:
    """Coarse lobby-survivorship bin for role/elimination selection."""
    try:
        n = int(n_alive or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 3:
        return "alive_2_3"
    if n <= 5:
        return "alive_4_5"
    return "alive_6_plus"


def _role_bin(winner_tavern_tier: Optional[int]) -> str:
    """Winner tavern-tier role: low (T1–T3) vs high (T4–T6) punch producers."""
    try:
        t = int(winner_tavern_tier or 0)
    except (TypeError, ValueError):
        t = 0
    return "winner_tavern_low" if t <= 3 else "winner_tavern_high"


def diagnose_phase_3g(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the #51 −196 crater to mixture/role vs within-cell carry."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3g_seed": PHASE_3G_SEED,
        "phase_3g_lobbies": PHASE_3G_LOBBIES,
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
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_winner_start_tiers": list(LOW_WINNER_START_TIERS),
        "early_turns": list(EARLY_TURNS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    decomp = comparison.get("decomposition") or comparison
    share_mix = decomp.get("share_mixture_turn_winner_tier")
    share_cell = decomp.get("share_within_cell_opponent_carry")
    share_role = decomp.get("share_role_alive_selection")
    share_left = decomp.get("share_leftover")
    share_mix_role = decomp.get("share_mixture_plus_role")
    if share_mix_role is None and (share_mix is not None or share_role is not None):
        share_mix_role = (
            (0.0 if share_mix is None else float(share_mix))
            + (0.0 if share_role is None else float(share_role))
        )

    mix_f = float(share_mix) if share_mix is not None else None
    cell_f = float(share_cell) if share_cell is not None else None
    role_f = float(share_role) if share_role is not None else None
    left_f = float(share_left) if share_left is not None else None
    mix_role_f = float(share_mix_role) if share_mix_role is not None else None

    represented = [
        ("mixture_turn_winner_tier", mix_f),
        ("within_cell_opponent_carry", cell_f),
        ("role_alive_selection", role_f),
    ]
    ranked = _rank_parts(represented + [("leftover", left_f)])
    top = ranked[0]["component"] if ranked else "leftover"

    if mix_role_f is not None and mix_role_f > SHARE_DOMINANT:
        primary = "mixture_role_selection_dominates"
        next_step = (
            "Mixture of winner-start-tier × turn weights and/or "
            "winner/loser role + alive/elimination selection clears ~70% "
            "of the #51 −196 punch-row crater. Next hour: trace the "
            "upstream gameplay / composition process that changes those "
            "winner-tier × turn matchups. Do not apply a scaling "
            "correction; do not rewrite 2Q; do not change `_hero_damage`; "
            "do not retune scaling constants; do not burn confirm."
        )
    elif cell_f is not None and cell_f > SHARE_DOMINANT:
        primary = "within_cell_opponent_carry_dominates"
        next_step = (
            "Opponent carry conditional on matched turn × winner-start "
            "tier clears ~70% of the #51 −196 crater. Next hour: isolate "
            "the earliest gameplay state variable causing that within-cell "
            "selection. Do not apply a scaling correction; do not rewrite "
            "2Q; do not change `_hero_damage`; do not retune scaling "
            "constants; do not burn confirm."
        )
    elif (
        mix_role_f is not None
        and cell_f is not None
        and max(mix_role_f, cell_f) >= 0.30
    ):
        primary = "mixed_route_to_larger"
        larger = "mixture / role selection" if (
            mix_role_f >= cell_f
        ) else "within matched turn × winner-start-tier opponent carry"
        next_step = (
            f"Both mixture/role selection (share={mix_role_f:.3f}) and "
            f"within-cell opponent carry (share={cell_f:.3f}) are present. "
            f"Rank both and pursue the larger piece: {larger}. Do not "
            "apply a scaling correction; do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "Neither mixture/role selection nor within matched "
            f"turn × winner-start-tier opponent carry clears ~70% of the "
            f"#51 −196 crater (top={top}). Rank the residual before any "
            f"behavior change: {NEXT_OBSERVABLE_DEFAULT}. Do not apply a "
            "scaling correction; do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_mixture_turn_winner_tier": mix_f,
        "share_within_cell_opponent_carry": cell_f,
        "share_role_alive_selection": role_f,
        "share_leftover": left_f,
        "share_mixture_plus_role": mix_role_f,
        "decomposition": decomp,
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
    })
    return out
