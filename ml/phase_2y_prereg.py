"""Phase 2Y — board-slot / attack-order vs teammate protection (measurement only).

Stacked on Phase 2X (PR #42). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #42.

Splits the leftover 2X residual position / combat-order term
(R ≈ +1.372 / hit, 81.7% of 2V B) into:

* (A) earlier slot / more attack opportunities
* (B) stronger teammates / board-size protection
* (C) unexplained combat mechanics
"""

from __future__ import annotations

from typing import Dict, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2y_v1"
PHASE_2Y_SEED = 14200
PHASE_2Y_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
N_DECILES = 10
N_TEAM_BINS = 5
SLOT_BIN_CAP = 4  # slots 0,1,2,3,4+

# 2V published Kitagawa within-tier survival (exclusive T6 assigned to A).
PHASE_2V_WITHIN_TIER_B = 1.6782901818400895
PHASE_2V_SHARE_WITHIN_TIER = 0.4185551426754372
PHASE_2V_SURVIVOR_TIER_SUM_DELTA = 4.009722998772222

# 2X published leftover after holding tier + recruit/raw + synth share.
PHASE_2X_RESIDUAL_POSITION = 1.3719447683362298
PHASE_2X_SHARE_RESIDUAL = 0.8174657655638667
PHASE_2X_SHARE_SYNTHETIC = 0.1571005720968943
PHASE_2X_SHARE_RECRUIT_MIX = 0.02543366234070279

# Decision: which piece of the 2X residual clears ~70%.
SHARE_DOMINANT = 0.70

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """2Y may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 2Y must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2Y seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def slot_bin(slot) -> int:
    """Coarse starting board slot: 0, 1, 2, 3, 4+."""
    try:
        s = int(slot)
    except (TypeError, ValueError):
        s = 0
    return min(max(s, 0), SLOT_BIN_CAP)


def diagnose_phase_2y(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 2X residual to slot/opportunity vs teammate protection vs leftover."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2y_seed": PHASE_2Y_SEED,
        "phase_2y_lobbies": PHASE_2Y_LOBBIES,
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
        "share_dominant": SHARE_DOMINANT,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2x_share_residual": PHASE_2X_SHARE_RESIDUAL,
        "instrument_turns": list(INSTRUMENT_TURNS),
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    rw = comparison.get("reweighting") or {}
    share_a = rw.get("share_of_residual_slot_opportunity")
    share_b = rw.get("share_of_residual_teammate_protection")
    share_c = rw.get("share_of_residual_unexplained")
    resid = rw.get("phase_2x_residual_position_hat")
    a_f = float(share_a) if share_a is not None else None
    b_f = float(share_b) if share_b is not None else None
    c_f = float(share_c) if share_c is not None else None

    if a_f is not None and a_f > SHARE_DOMINANT:
        primary = "slot_attack_opportunity_dominates"
        next_step = (
            "Most of the leftover 2X residual is earlier board slot / more "
            "attack opportunities on same-tier / same-recruit / same-synth "
            "bodies. Next hour: audit the recruit/play positioning policy "
            "against real BG positioning evidence before implementation. "
            "Do not rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )
    elif b_f is not None and b_f > SHARE_DOMINANT:
        primary = "teammate_protection_dominates"
        next_step = (
            "Most of the leftover 2X residual is stronger teammates / larger "
            "boards protecting the same body. Next hour: diagnose board-level "
            "combat composition / effect fidelity. Do not rewrite 2Q; do not "
            "change `_hero_damage`; do not burn confirm."
        )
    else:
        primary = "unexplained_combat_mechanics"
        next_step = (
            "Neither slot/attack opportunity nor teammate protection clears "
            "~70% of the 2X residual. Isolate the specific combat mechanic "
            "(taunt / targeting / deathrattle / attack-cursor) before any "
            "behavior change. Do not rewrite 2Q; do not change `_hero_damage`; "
            "do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "reweighting": rw,
        "share_of_residual_slot_opportunity": a_f,
        "share_of_residual_teammate_protection": b_f,
        "share_of_residual_unexplained": c_f,
        "phase_2x_residual_position_hat": resid,
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
    })
    return out
