"""Phase 2V — survivor-composition attribution (measurement only).

Stacked on Phase 2U (PR #39). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
or defaults. Confirm 11500–11699 remains reserved. Keep HOLD stack including #39.
"""

from __future__ import annotations

from typing import Dict, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2v_v1"
PHASE_2V_SEED = 14200
PHASE_2V_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
HIGH_TIER_MIN = 4
CHAFF_TIER_MAX = 2
SHARE_DOMINANT = 0.55

# 2U published treatment−control survivor-tier-sum gap when hit.
PHASE_2U_SURVIVOR_TIER_SUM_DELTA = 4.009722998772222

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """2V may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 2V must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2V seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def diagnose_phase_2v(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route +4.01 survivor-tier-sum gap to fielded / survival / tokens."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2v_seed": PHASE_2V_SEED,
        "phase_2v_lobbies": PHASE_2V_LOBBIES,
        "reused_seed_range": f"{REUSED_SEED_LO}–{REUSED_SEED_HI}",
        "keep_hold_prs": list(HOLD_PRS),
        "no_merge": True,
        "evaluative": False,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "no_hero_damage_retune": True,
        "no_gate_change": True,
        "no_behavior_change": True,
        "share_dominant_threshold": SHARE_DOMINANT,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
        "high_tier_min": HIGH_TIER_MIN,
        "chaff_tier_max": CHAFF_TIER_MAX,
        "phase_2u_survivor_tier_sum_delta": PHASE_2U_SURVIVOR_TIER_SUM_DELTA,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    decomp = comparison.get("decomposition") or {}
    share_a = decomp.get("share_fielded_composition")
    share_b = decomp.get("share_within_tier_survival")
    share_c = decomp.get("share_token_generated")
    shares = {
        "fielded_composition": share_a,
        "within_tier_survival": share_b,
        "token_generated": share_c,
    }
    ranked = [
        (name, float(val))
        for name, val in shares.items()
        if val is not None
    ]
    ranked.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top_name, top_share = (ranked[0] if ranked else (None, None))

    if top_name == "fielded_composition" and top_share >= SHARE_DOMINANT:
        primary = "fielded_composition_dominates"
        next_step = (
            "Most of the treatment−control survivor-tier-sum gap is higher-tier "
            "cards being recruited/fielded. Next hour: measure whether the 2Q "
            "replacement policy over-selects tavern tier / raw stats versus "
            "real Firestone board composition. Do not change `_hero_damage`; "
            "do not burn confirm."
        )
    elif top_name == "within_tier_survival" and top_share >= SHARE_DOMINANT:
        primary = "within_tier_survival_dominates"
        next_step = (
            "Most of the gap is same-tier cards surviving more often "
            "(combat strength / scaling). Next hour: diagnose combat/scaling "
            "allocation by tavern tier. Do not change `_hero_damage`; "
            "do not burn confirm."
        )
    elif top_name == "token_generated" and top_share >= SHARE_DOMINANT:
        primary = "token_generated_dominates"
        next_step = (
            "Most of the gap is token/generated combat bodies. Next hour: "
            "audit token tier and creation fidelity. Do not change "
            "`_hero_damage`; do not burn confirm."
        )
    else:
        primary = "mixed_survivor_composition"
        next_step = (
            "No single bucket (fielded composition / within-tier survival / "
            "tokens) clears the dominant-share threshold. Report the mix; "
            "do not change `_hero_damage`; do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "decomposition": decomp,
        "dominant_component": top_name,
        "dominant_share": top_share,
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_pr_34_hold": True,
        "keep_pr_35_hold": True,
        "keep_pr_36_hold": True,
        "keep_pr_37_hold": True,
        "keep_pr_38_hold": True,
        "keep_pr_39_hold": True,
    })
    return out
