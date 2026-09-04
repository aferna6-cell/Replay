"""Phase 2Z — combat-mechanic split of the 2Y leftover residual (measurement only).

Stacked on Phase 2Y (PR #43). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #43.

Splits the leftover 2Y unexplained combat-mechanics term
(C ≈ +0.946 / hit, 68.9% of 2X residual R) into:

* (A) targeting / taunt (forced vs open)
* (B) attack-cursor / initiative
* (C) represented generated-body / deathrattle effects
* (D) unsupported-effect coverage (placeholders / approximate DRs; marked, not fitted)
* (E) still unexplained
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "2z_v1"
PHASE_2Z_SEED = 14200
PHASE_2Z_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
N_DECILES = 10
N_TEAM_BINS = 5
SLOT_BIN_CAP = 4  # slots 0,1,2,3,4+
N_TARGET_BINS = 3
N_CURSOR_BINS = 3
N_GEN_BINS = 2
N_UNSUP_BINS = 2

# 2V published Kitagawa within-tier survival (exclusive T6 assigned to A).
PHASE_2V_WITHIN_TIER_B = 1.6782901818400895
PHASE_2V_SHARE_WITHIN_TIER = 0.4185551426754372
PHASE_2V_SURVIVOR_TIER_SUM_DELTA = 4.009722998772222

# 2X published leftover after holding tier + recruit/raw + synth share.
PHASE_2X_RESIDUAL_POSITION = 1.3719447683362298
PHASE_2X_SHARE_RESIDUAL = 0.8174657655638667

# 2Y published leftover after holding slot then teammate-raw.
PHASE_2Y_UNEXPLAINED = 0.9456715648873479
PHASE_2Y_SHARE_UNEXPLAINED = 0.6892927373703044
PHASE_2Y_SLOT_OPPORTUNITY = -0.3219135497293096
PHASE_2Y_TEAMMATE_PROTECTION = 0.7481867531781913

SHARE_DOMINANT = 0.70

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)

NEXT_OBSERVABLE_DEFAULT = (
    "per-swing divine-shield / poisonous / cleave lethal cause, or "
    "start-of-combat hits, beyond taunt-forced vs open targeting"
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """2Z may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 2Z must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 2Z seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def slot_bin(slot) -> int:
    """Coarse starting board slot: 0, 1, 2, 3, 4+."""
    try:
        s = int(slot)
    except (TypeError, ValueError):
        s = 0
    return min(max(s, 0), SLOT_BIN_CAP)


def target_bin(row: Dict) -> int:
    """0 never / 1 open-only / 2 taunt or taunt-forced targeting."""
    if row.get("taunt") or int(row.get("n_targeted_forced") or 0) > 0:
        return 2
    if (
        int(row.get("n_targeted_open") or 0) > 0
        or row.get("was_targeted")
        or row.get("open_target")
    ):
        return 1
    return 0


def cursor_bin(row: Dict) -> int:
    """0 never reached / 1 side-first no wrap / 2 wrap or second-side."""
    idx = row.get("first_attack_index")
    attacked = bool(row.get("attacked")) or (idx is not None)
    if not attacked:
        return 0
    if row.get("side_first") and not row.get("cursor_wrapped_before_first"):
        return 1
    return 2


def gen_bin(row: Dict) -> int:
    """1 if exposed to a faithfully represented generated-body / DR effect."""
    if row.get("has_represented_generated_effect"):
        return 1
    if int(row.get("spawned_represented") or 0) > 0:
        return 1
    if int(row.get("n_board_generated_represented") or 0) > 0:
        return 1
    return 0


def unsupported_bin(row: Dict) -> int:
    """1 if this body carries a marked placeholder or approximate effect."""
    if row.get("has_unsupported_effect"):
        return 1
    status = str(row.get("effect_status") or "")
    if status in ("unsupported_placeholder", "represented_approximate"):
        return 1
    return 0


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


def diagnose_phase_2z(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 2Y leftover to targeting / cursor / represented DR / unsupported / leftover."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_2z_seed": PHASE_2Z_SEED,
        "phase_2z_lobbies": PHASE_2Z_LOBBIES,
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
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "phase_2y_share_unexplained": PHASE_2Y_SHARE_UNEXPLAINED,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "unsupported_marked_not_approximated": True,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    rw = comparison.get("reweighting") or {}
    share_a = rw.get("share_of_leftover_targeting_taunt")
    share_b = rw.get("share_of_leftover_attack_cursor")
    share_c = rw.get("share_of_leftover_represented_generated")
    share_d = rw.get("share_of_leftover_unsupported_coverage")
    share_e = rw.get("share_of_leftover_still_unexplained")
    resid = rw.get("phase_2y_unexplained_hat")
    a_f = float(share_a) if share_a is not None else None
    b_f = float(share_b) if share_b is not None else None
    c_f = float(share_c) if share_c is not None else None
    d_f = float(share_d) if share_d is not None else None
    e_f = float(share_e) if share_e is not None else None

    ranked = _rank_parts([
        ("targeting_taunt", a_f),
        ("attack_cursor_initiative", b_f),
        ("represented_generated_deathrattle", c_f),
        ("unsupported_effect_coverage", d_f),
        ("still_unexplained", e_f),
    ])
    top = ranked[0]["component"] if ranked else "still_unexplained"

    if a_f is not None and a_f > SHARE_DOMINANT:
        primary = "targeting_taunt_dominates"
        next_step = (
            "Most of the 2Y leftover is taunt / targeting (forced vs open) on "
            "same-slot / same-teammate bodies. Next hour: preregister only a "
            "taunt/targeting correction. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    elif b_f is not None and b_f > SHARE_DOMINANT:
        primary = "attack_cursor_initiative_dominates"
        next_step = (
            "Most of the 2Y leftover is attack-cursor / initiative on "
            "same-slot / same-teammate bodies. Next hour: preregister only an "
            "attack-cursor / first-attacker correction. Do not rewrite 2Q; "
            "do not change `_hero_damage`; do not burn confirm."
        )
    elif c_f is not None and c_f > SHARE_DOMINANT:
        primary = "represented_generated_deathrattle_dominates"
        next_step = (
            "Most of the 2Y leftover is faithfully represented generated-body "
            "/ deathrattle effects. Next hour: preregister only that "
            "represented correction. Do not approximate unsupported DRs; do "
            "not rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )
    elif d_f is not None and d_f > SHARE_DOMINANT:
        primary = "unsupported_effect_coverage_dominates"
        next_step = (
            "Unsupported-effect coverage (registry placeholders and marked "
            "approximate deathrattles) dominates the 2Y leftover. Next hour: "
            "audit that missing effect class against card data. Do not "
            "approximate the missing mechanic; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            f"No represented mechanic clears ~70% of the 2Y leftover "
            f"(top={top}). Rank the residual and collect the smallest extra "
            f"observable before any behavior change: {NEXT_OBSERVABLE_DEFAULT}. "
            "Do not rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "smallest_additional_observable": NEXT_OBSERVABLE_DEFAULT,
        "reweighting": rw,
        "share_of_leftover_targeting_taunt": a_f,
        "share_of_leftover_attack_cursor": b_f,
        "share_of_leftover_represented_generated": c_f,
        "share_of_leftover_unsupported_coverage": d_f,
        "share_of_leftover_still_unexplained": e_f,
        "phase_2y_unexplained_hat": resid,
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
    })
    return out
