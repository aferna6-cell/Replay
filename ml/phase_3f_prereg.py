"""Phase 3F — carry divergence timing + outcome-conditioning audit (measurement only).

Stacked on Phase 3E (PR #51). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #51.

Dates the #51 inherited-carry term (+0.305 / hit, 72.4% of 3D A1; punch-row
Δcarry −196) by pairing each (seed, seat) from T7 through that seat's
eventual T7–T14 punch-row appearance, then asking whether a real paired
Δcarry appears *before* punch / winner-start-tier / outcome conditioning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3f_v1"
PHASE_3F_SEED = 14200
PHASE_3F_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive

SHARE_DOMINANT = 0.70
FLOW_ABS_TOL = 1.0
# First turn a paired path "materially separates": |Δ| ≥ max(floor, rel × scale).
MATERIAL_ABS = 8.0
MATERIAL_REL = 0.10
LOW_WINNER_START_TIERS = (1,)

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
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51,
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

# Published 3E / 3D locks (exact).
PHASE_2V_WITHIN_TIER_B = 1.6782901818400895
PHASE_3C_ATTACKER_ATTACK_STRENGTH = 0.5120447786800975
PHASE_3D_BOARD_POOL_MAGNITUDE = 0.4216721428553852
PHASE_3D_SHARE_BOARD_POOL = 0.8235063814972068
PHASE_3E_CARRY_DELTA = 0.30513688784757187
PHASE_3E_CARRY_SHARE_OF_A1 = 0.7236353954551374
PHASE_3E_PUNCH_DELTA_CARRY = -196.33317557443002
PHASE_3E_PUNCH_DELTA_COMBAT = -274.83106360865395
PHASE_3E_PUNCH_SHARE_CARRY = 0.7143776725836163

NEXT_OBSERVABLE_DEFAULT = (
    "exact upstream scaling inputs at the first paired-divergence turn, "
    "or the punch/winner/low-tier selection mechanism, ranked by the larger "
    "piece of the #51 carry term"
)

REPRESENTED_SOURCE = (
    "paired_divergence_before_conditioning",
    "punch_winner_outcome_selection",
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3F may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3F must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3F seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def _float_or_none(row: Optional[Dict], *keys: str) -> Optional[float]:
    if not row:
        return None
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def carry_value(row: Optional[Dict]) -> Optional[float]:
    """Seat-turn inherited / recruit-start attack pool."""
    return _float_or_none(
        row,
        "attack_pool_recruit_start",
        "opp_carry_attack_pool",
        "opp_attack_pool_recruit_start",
    )


def scale_add_value(row: Optional[Dict]) -> Optional[float]:
    return _float_or_none(
        row, "scale_add_attack", "opp_scale_add_attack", "opp_attack_pool_scale_add",
    )


def materially_separated(
    control_carry: Optional[float],
    treatment_carry: Optional[float],
    *,
    abs_floor: float = MATERIAL_ABS,
    rel: float = MATERIAL_REL,
) -> bool:
    """True when paired |Δcarry| clears max(abs_floor, rel × larger |carry|)."""
    if control_carry is None or treatment_carry is None:
        return False
    delta = abs(float(treatment_carry) - float(control_carry))
    scale = max(abs(float(control_carry)), abs(float(treatment_carry)), 1.0)
    return delta >= max(float(abs_floor), float(rel) * scale)


def first_separation_turn(
    by_turn: Dict[int, Tuple[Optional[float], Optional[float]]],
    *,
    abs_floor: float = MATERIAL_ABS,
    rel: float = MATERIAL_REL,
) -> Optional[int]:
    """Earliest instrumented turn whose paired carries materially separate."""
    for turn in INSTRUMENT_TURNS:
        if turn not in by_turn:
            continue
        c, t = by_turn[turn]
        if materially_separated(c, t, abs_floor=abs_floor, rel=rel):
            return int(turn)
    return None


def share_of_carry_term(
    delta_carry: Optional[float],
    *,
    denom: float = PHASE_3E_PUNCH_DELTA_CARRY,
) -> Optional[float]:
    """|paired Δcarry| / |#51 punch-row Δcarry|. None if denom is 0."""
    if delta_carry is None or abs(float(denom)) < 1e-12:
        return None
    return abs(float(delta_carry)) / abs(float(denom))


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


def diagnose_phase_3f(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route #51 carry term to pre-conditioning paired Δ vs selection."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3f_seed": PHASE_3F_SEED,
        "phase_3f_lobbies": PHASE_3F_LOBBIES,
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
        "instrument_turns": list(INSTRUMENT_TURNS),
        "material_abs": MATERIAL_ABS,
        "material_rel": MATERIAL_REL,
        "low_winner_start_tiers": list(LOW_WINNER_START_TIERS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "history_link_identity": HISTORY_LINK_IDENTITY,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    timing = comparison.get("timing") or comparison
    share_uncond = timing.get("share_of_3e_carry_unconditional")
    share_punch = timing.get("share_of_3e_carry_punch_included")
    share_lowtier = timing.get("share_of_3e_carry_low_winner_start")
    share_outcome = timing.get("share_of_3e_carry_outcome_conditioned")
    share_pre = timing.get("share_of_3e_carry_before_conditioning")
    share_sel = timing.get("share_of_3e_carry_from_selection")
    if share_pre is None:
        share_pre = share_uncond
    if share_sel is None:
        # Selection increment: extra gap that appears only after filters.
        if share_uncond is None:
            share_sel = None
        else:
            after = [
                s for s in (share_punch, share_lowtier, share_outcome)
                if s is not None
            ]
            share_sel = None if not after else max(0.0, max(after) - float(share_uncond))

    uncond_f = float(share_uncond) if share_uncond is not None else None
    punch_f = float(share_punch) if share_punch is not None else None
    low_f = float(share_lowtier) if share_lowtier is not None else None
    out_f = float(share_outcome) if share_outcome is not None else None
    pre_f = float(share_pre) if share_pre is not None else None
    sel_f = float(share_sel) if share_sel is not None else None

    represented = [
        ("paired_divergence_before_conditioning", pre_f),
        ("punch_winner_outcome_selection", sel_f),
    ]
    ranked = _rank_parts(represented)
    top = ranked[0]["component"] if ranked else "paired_divergence_before_conditioning"

    if pre_f is not None and pre_f > SHARE_DOMINANT:
        primary = "paired_divergence_precedes_conditioning"
        next_step = (
            "A real paired (seed, seat) carry split appears before punch / "
            "winner-start-tier / outcome filters and clears ~70% of the #51 "
            "carry term. Next hour: audit the exact upstream scaling inputs "
            "(Firestone target, pre-scale stats, growth factor, clamp) at the "
            "first divergence turn — not the constants. Do not rewrite 2Q; "
            "do not change `_hero_damage`; do not retune scaling constants; "
            "do not burn confirm."
        )
    elif sel_f is not None and sel_f > SHARE_DOMINANT:
        primary = "selection_outcome_conditioning_dominates"
        next_step = (
            "The #51 carry crater appears mainly after conditioning on punch "
            "inclusion / winner-start tier / eventual outcome. Treat 3E carry "
            "dominance as selection / outcome-conditioning. Next hour: isolate "
            "that selection mechanism rather than changing scaling. Do not "
            "rewrite 2Q; do not change `_hero_damage`; do not retune scaling "
            "constants; do not burn confirm."
        )
    elif pre_f is not None and sel_f is not None:
        primary = "mixed_route_to_larger"
        larger = "paired pre-conditioning divergence" if (
            pre_f >= sel_f
        ) else "punch / winner / outcome selection"
        next_step = (
            f"Both a paired pre-conditioning carry split (share={pre_f:.3f}) "
            f"and post-filter selection (share={sel_f:.3f}) are present. "
            f"Route next hour to the larger piece: {larger}. Do not rewrite "
            "2Q; do not change `_hero_damage`; do not retune scaling "
            "constants; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "Neither pre-conditioning paired Δcarry nor punch/winner/outcome "
            f"selection clears ~70% of the #51 carry term (top={top}). "
            f"Rank the residual before any behavior change: "
            f"{NEXT_OBSERVABLE_DEFAULT}. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "share_of_3e_carry_unconditional": uncond_f,
        "share_of_3e_carry_punch_included": punch_f,
        "share_of_3e_carry_low_winner_start": low_f,
        "share_of_3e_carry_outcome_conditioned": out_f,
        "share_of_3e_carry_before_conditioning": pre_f,
        "share_of_3e_carry_from_selection": sel_f,
        "timing": timing,
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
    })
    return out
