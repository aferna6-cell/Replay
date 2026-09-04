"""Phase 2U — survivor-tier damage fidelity (measurement only).

Stacked on Phase 2T (PR #38). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
or defaults. Confirm 11500–11699 remains reserved. Keep HOLD stack including #38.
"""

from __future__ import annotations

from typing import Dict, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2u_v1"
PHASE_2U_SEED = 14200
PHASE_2U_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive

# "Most" of the treatment−control +2.78 amplification is gone under the
# rules-faithful survivor-tier counterfactual.
SHARE_REMOVED_MOST = 0.55
# 2T published treatment−control amplification when hit (applied − count-only).
PHASE_2T_AMP_DELTA_WHEN_HIT = 2.7756345999047585

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """2U may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 2U must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2U seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def diagnose_phase_2u(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route: preregister default-OFF formula vs isolate survivor composition."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2u_seed": PHASE_2U_SEED,
        "phase_2u_lobbies": PHASE_2U_LOBBIES,
        "reused_seed_range": f"{REUSED_SEED_LO}–{REUSED_SEED_HI}",
        "keep_hold_prs": list(HOLD_PRS),
        "no_merge": True,
        "evaluative": False,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "no_hero_damage_retune": True,
        "no_gate_change": True,
        "no_behavior_change": True,
        "share_removed_most_threshold": SHARE_REMOVED_MOST,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    fid = comparison.get("fidelity") or {}
    share_removed = fid.get("share_of_amp_delta_removed")
    share_remaining = fid.get("share_of_amp_delta_remaining")
    cf_amp_delta = fid.get("counterfactual_amplification_delta_when_hit")
    proxy_amp_delta = fid.get("proxy_amplification_delta_when_hit")

    if share_removed is not None and float(share_removed) >= SHARE_REMOVED_MOST:
        primary = "preregister_default_off_damage_formula"
        next_step = (
            "Actual-survivor (rules-faithful) damage removes most of the "
            "treatment−control `_hero_damage` amplification. Next hour: "
            "preregister a default-OFF damage-formula treatment. Do not "
            "change `_hero_damage` this hour; do not burn confirm."
        )
    else:
        primary = "isolate_survivor_composition"
        next_step = (
            "Substantial treatment−control amplification remains after "
            "replacing the board-mean proxy with actual combat-survivor "
            "tiers. Isolate survivor composition / tier distribution "
            "before changing `_hero_damage`. Do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "fidelity": fid,
        "share_of_amp_delta_removed": share_removed,
        "share_of_amp_delta_remaining": share_remaining,
        "proxy_amplification_delta_when_hit": proxy_amp_delta,
        "counterfactual_amplification_delta_when_hit": cf_amp_delta,
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_pr_34_hold": True,
        "keep_pr_35_hold": True,
        "keep_pr_36_hold": True,
        "keep_pr_37_hold": True,
        "keep_pr_38_hold": True,
    })
    return out
