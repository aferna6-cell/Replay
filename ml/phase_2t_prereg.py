"""Phase 2T — game-length / damage attribution (measurement only).

Reuses the already-consumed Phase 2S DEV band 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
or defaults. Confirm 11500–11699 remains reserved. Keep HOLD stack.
"""

from __future__ import annotations

from typing import Dict, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2t_v1"
PHASE_2T_SEED = 14200
PHASE_2T_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
SHARE_DOMINANT = 0.55

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

HOLD_PRS = (29, 33, 34, 35, 36, 37)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """2T may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 2T must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2T seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def diagnose_phase_2t(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route extra game-length shortening to damage-model / combat / lifecycle."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2t_seed": PHASE_2T_SEED,
        "phase_2t_lobbies": PHASE_2T_LOBBIES,
        "reused_seed_range": f"{REUSED_SEED_LO}–{REUSED_SEED_HI}",
        "keep_hold_prs": list(HOLD_PRS),
        "no_merge": True,
        "evaluative": False,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "no_hero_damage_retune": True,
        "no_gate_change": True,
        "share_dominant_threshold": SHARE_DOMINANT,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    attr = comparison.get("attribution") or {}
    share_amp = attr.get("share_of_extra_hp_from_amplification")
    share_combat = attr.get("share_of_extra_hp_from_combat_outcome")
    share_life = attr.get("share_of_shortening_unexplained_lifecycle")
    healthy = bool(attr.get("combat_strength_fidelity_healthy"))
    shortening = attr.get("actual_shortening_turns")

    if shortening is None or float(shortening) <= 0.25:
        primary = "no_material_shortening"
        next_step = "Game length did not shorten materially; re-check wiring."
    elif (
        share_amp is not None
        and float(share_amp) >= SHARE_DOMINANT
        and healthy
    ):
        primary = "damage_model_fidelity"
        next_step = (
            "Survivor tier/count or the `_hero_damage` amplification explains "
            "most of the shortening while combat-strength fidelity is healthy. "
            "Next step is a separate damage-model fidelity phase. Do not "
            "retune recruit/scaling; do not burn confirm."
        )
    elif share_combat is not None and float(share_combat) >= SHARE_DOMINANT:
        primary = "combat_outcome_dominance"
        next_step = (
            "More decisive / one-sided combat outcomes explain most of the "
            "shortening. Next phase should diagnose board composition / "
            "effect fidelity. Do not retune α or residual scaling."
        )
    else:
        primary = "lifecycle_or_unexplained"
        next_step = (
            "Neither survivor-tier/count amplification nor combat-outcome "
            "dominance explains most of the shortening"
            + (
                f" (lifecycle residual share={share_life})."
                if share_life is not None
                else "."
            )
            + " Identify the lifecycle source before any implementation."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "attribution": attr,
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_pr_34_hold": True,
        "keep_pr_35_hold": True,
        "keep_pr_36_hold": True,
        "keep_pr_37_hold": True,
    })
    return out
