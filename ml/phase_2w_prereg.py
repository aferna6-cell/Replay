"""Phase 2W — Firestone final-board composition vs 2Q selection (measurement only).

Stacked on Phase 2V (PR #40). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
or defaults. Confirm 11500–11699 remains reserved. Keep HOLD stack including #40.

Firestone ``firestone_final_boards.json`` is **final-board** data. Compare
primarily to each simulated player's last/alive late-game board, with
T12–T14 snapshots as sensitivity — not early turns.
"""

from __future__ import annotations

from typing import Dict, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2w_v1"
PHASE_2W_SEED = 14200
PHASE_2W_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
LATE_TURNS = (12, 13, 14)
HIGH_TIER_MIN = 4

# Coverage: join/example floors below which we refuse a composition call.
COVERAGE_JOIN_MIN = 0.80
COVERAGE_N_BOARDS_MIN = 20
COVERAGE_N_UNIQUE_CARDS_MIN = 30
COVERAGE_POOL_NAME_MIN = 0.70

# Material shift vs Firestone (and vs control) on last/alive boards.
MATERIAL_T4_SHARE_DELTA = 0.08
MATERIAL_MEAN_TIER_DELTA = 0.25
MATERIAL_MEAN_PRINTED_RAW_DELTA = 2.0
MATCH_T4_SHARE_TOL = 0.08
MATCH_MEAN_TIER_TOL = 0.25

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """2W may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 2W must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2W seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def _shifted_vs(delta_t4, delta_tier, delta_raw, *, floor_t4, floor_tier, floor_raw):
    """True if at least two of {T4+ share, mean tier, printed raw} clear floors."""
    hits = []
    if delta_t4 is not None:
        hits.append(float(delta_t4) >= floor_t4)
    if delta_tier is not None:
        hits.append(float(delta_tier) >= floor_tier)
    if delta_raw is not None:
        hits.append(float(delta_raw) >= floor_raw)
    return sum(1 for h in hits if h) >= 2


def _matches(delta_t4, delta_tier, *, tol_t4, tol_tier):
    if delta_t4 is None or delta_tier is None:
        return False
    return abs(float(delta_t4)) <= tol_t4 and abs(float(delta_tier)) <= tol_tier


def diagnose_phase_2w(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 2Q last-board mix vs Firestone final-board reference."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2w_seed": PHASE_2W_SEED,
        "phase_2w_lobbies": PHASE_2W_LOBBIES,
        "reused_seed_range": f"{REUSED_SEED_LO}–{REUSED_SEED_HI}",
        "keep_hold_prs": list(HOLD_PRS),
        "no_merge": True,
        "evaluative": False,
        "no_scaling_retune": True,
        "no_alpha_retune": True,
        "no_hero_damage_retune": True,
        "no_gate_change": True,
        "no_behavior_change": True,
        "confirm_seeds_reserved": "11500–11699",
        "frozen_alpha": FROZEN_ALPHA,
        "high_tier_min": HIGH_TIER_MIN,
        "late_turns": list(LATE_TURNS),
        "firestone_is_final_board_data": True,
        "primary_sim_window": "last_alive_late_board",
        "sensitivity_windows": ["t12", "t13", "t14"],
        "material_t4_share_delta": MATERIAL_T4_SHARE_DELTA,
        "material_mean_tier_delta": MATERIAL_MEAN_TIER_DELTA,
        "material_mean_printed_raw_delta": MATERIAL_MEAN_PRINTED_RAW_DELTA,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    cov = comparison.get("coverage") or {}
    join_rate = cov.get("join_rate")
    n_boards = cov.get("n_example_boards")
    n_unique = cov.get("n_unique_joined_cards")
    pool_name = cov.get("pool_name_rate")
    coverage_ok = (
        join_rate is not None and float(join_rate) >= COVERAGE_JOIN_MIN
        and n_boards is not None and int(n_boards) >= COVERAGE_N_BOARDS_MIN
        and n_unique is not None and int(n_unique) >= COVERAGE_N_UNIQUE_CARDS_MIN
        and pool_name is not None and float(pool_name) >= COVERAGE_POOL_NAME_MIN
    )
    if not coverage_ok:
        out.update({
            "primary_finding": "firestone_coverage_too_weak",
            "evaluative": True,
            "coverage": cov,
            "recommended_next_step": (
                "Firestone final-board coverage is too thin for a valid "
                "over-select call. Do not change 2Q behavior. Smallest "
                "stronger reference: a turn-labeled final-board dump with "
                ">>3 examples per archetype (or HSReplay/Firestone full "
                "finalComp boards), still joined to the active pool."
            ),
            "keep_pr_40_hold": True,
        })
        return out

    last = comparison.get("last_alive") or {}
    d_t4_fs = last.get("t4_share_treatment_minus_firestone")
    d_t4_c = last.get("t4_share_treatment_minus_control")
    d_tier_fs = last.get("mean_printed_tier_treatment_minus_firestone")
    d_tier_c = last.get("mean_printed_tier_treatment_minus_control")
    d_raw_fs = last.get("mean_printed_raw_treatment_minus_firestone")
    d_raw_c = last.get("mean_printed_raw_treatment_minus_control")

    over_fs = _shifted_vs(
        d_t4_fs, d_tier_fs, d_raw_fs,
        floor_t4=MATERIAL_T4_SHARE_DELTA,
        floor_tier=MATERIAL_MEAN_TIER_DELTA,
        floor_raw=MATERIAL_MEAN_PRINTED_RAW_DELTA,
    )
    over_c = _shifted_vs(
        d_t4_c, d_tier_c, d_raw_c,
        floor_t4=MATERIAL_T4_SHARE_DELTA,
        floor_tier=MATERIAL_MEAN_TIER_DELTA,
        floor_raw=MATERIAL_MEAN_PRINTED_RAW_DELTA,
    )
    matches_fs = _matches(
        d_t4_fs, d_tier_fs,
        tol_t4=MATCH_T4_SHARE_TOL, tol_tier=MATCH_MEAN_TIER_TOL,
    )

    if over_fs and over_c:
        primary = "treatment_high_tier_raw_vs_control_and_firestone"
        next_step = (
            "Treatment last/alive boards are materially higher-tier / "
            "higher printed-raw than both control and Firestone finals. "
            "Next hour: diagnose/revise the 2Q recruit-value objective "
            "(full-board replace scoring). Do not change `_hero_damage`; "
            "do not burn confirm."
        )
    elif matches_fs:
        primary = "treatment_matches_firestone"
        next_step = (
            "Treatment last/alive mix matches Firestone finals on printed "
            "tier; 2V within-tier survival (42%) remains the leftover. "
            "Next hour: isolate combat/scaling allocation by tavern tier. "
            "Do not change `_hero_damage`; do not burn confirm."
        )
    else:
        primary = "mixed_or_undershoots_firestone"
        next_step = (
            "Treatment is not a clean over-select vs both control and "
            "Firestone, and does not match Firestone within tolerance. "
            "Report the mix; do not change 2Q or `_hero_damage`; "
            "do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "coverage": cov,
        "last_alive_deltas": last,
        "overshoots_firestone": over_fs,
        "overshoots_control": over_c,
        "matches_firestone": matches_fs,
        "keep_pr_29_hold": True,
        "keep_pr_33_hold": True,
        "keep_pr_34_hold": True,
        "keep_pr_35_hold": True,
        "keep_pr_36_hold": True,
        "keep_pr_37_hold": True,
        "keep_pr_38_hold": True,
        "keep_pr_39_hold": True,
        "keep_pr_40_hold": True,
    })
    return out
