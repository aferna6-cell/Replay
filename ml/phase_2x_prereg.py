"""Phase 2X — within-tier survival vs synthetic allocation (measurement only).

Stacked on Phase 2W (PR #41). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #41.

Isolates the leftover 2V within-tier survival term (B ≈ +1.678 / hit, 41.9%)
by tavern tier and synthetic abstract-pool share. Do not rewrite 2Q.
"""

from __future__ import annotations

from typing import Dict, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2x_v1"
PHASE_2X_SEED = 14200
PHASE_2X_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
N_DECILES = 10

# 2V published Kitagawa within-tier survival (exclusive T6 assigned to A).
PHASE_2V_WITHIN_TIER_B = 1.6782901818400895
PHASE_2V_SHARE_WITHIN_TIER = 0.4185551426754372
PHASE_2V_SURVIVOR_TIER_SUM_DELTA = 4.009722998772222

# Decision: synthetic allocation vs residual position/combat-order.
SHARE_SYNTHETIC_DOMINANT = 0.70

# Confirm + prior DEV bands. 14200–14699 is reused, not forbidden.
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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40, 41)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """2X may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 2X must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2X seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def diagnose_phase_2x(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route leftover 2V B to synthetic allocation vs position/order."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2x_seed": PHASE_2X_SEED,
        "phase_2x_lobbies": PHASE_2X_LOBBIES,
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
        "share_synthetic_dominant": SHARE_SYNTHETIC_DOMINANT,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2v_share_within_tier": PHASE_2V_SHARE_WITHIN_TIER,
        "instrument_turns": list(INSTRUMENT_TURNS),
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    rw = comparison.get("reweighting") or {}
    share_s = rw.get("share_of_B_synthetic")
    share_r = rw.get("share_of_B_residual_position")
    share_m = rw.get("share_of_B_recruit_mix")
    b_hat = rw.get("within_tier_B")
    s_f = float(share_s) if share_s is not None else None
    r_f = float(share_r) if share_r is not None else None

    if s_f is not None and s_f > SHARE_SYNTHETIC_DOMINANT:
        primary = "synthetic_allocation_dominates"
        next_step = (
            "Most of the leftover 2V within-tier survival term is extra "
            "synthetic abstract-pool allocation on same-tier / same-recruit "
            "bodies. Next hour: preregister alternative board-level pool "
            "allocation rules without retuning total scaling. Do not rewrite "
            "2Q; do not change `_hero_damage`; do not burn confirm."
        )
    elif r_f is not None and r_f > SHARE_SYNTHETIC_DOMINANT:
        primary = "position_combat_order_dominates"
        next_step = (
            "Most of the leftover 2V within-tier survival term remains after "
            "holding tier + recruit/raw and synthetic share fixed — residual "
            "combat-order / board-slot effects. Next hour: diagnose "
            "positioning / combat fidelity. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    else:
        primary = "mixed_or_missing_within_tier_feature"
        next_step = (
            "Neither synthetic allocation nor residual position/combat-order "
            "clears ~70% of the 2V within-tier term. Identify the missing "
            "within-tier feature (including recruit-mix leftovers) before "
            "implementation. Do not rewrite 2Q; do not change `_hero_damage`; "
            "do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "reweighting": rw,
        "share_of_B_synthetic": s_f,
        "share_of_B_residual_position": r_f,
        "share_of_B_recruit_mix": share_m,
        "within_tier_B": b_hat,
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
    })
    return out
