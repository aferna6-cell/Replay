"""Phase 3B — per-hit HP depletion / overkill / hit-count (measurement only).

Stacked on Phase 3A (PR #45). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #45.

Splits the leftover 3A unexplained term
(F ≈ +0.828 / hit, 103.5% of 2Z leftover E) into:

* (A) fewer damaging hits / exposure count
* (B) lower damage per hit / HP depletion margin
* (C) overkill / death-threshold effects
* (D) still unexplained
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3b_v1"
PHASE_3B_SEED = 14200
PHASE_3B_LOBBIES = 500
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
N_DS_BINS = 3
N_POISON_BINS = 2
N_CLEAVE_BINS = 3
N_SOC_BINS = 2
N_ORDINARY_BINS = 3
N_HIT_BINS = 3
N_MARGIN_BINS = 5
N_OVERKILL_BINS = 3

# Published nested leftovers from prior HOLD hours (exact).
PHASE_2V_WITHIN_TIER_B = 1.6782901818400895
PHASE_2V_SHARE_WITHIN_TIER = 0.4185551426754372
PHASE_2V_SURVIVOR_TIER_SUM_DELTA = 4.009722998772222

PHASE_2X_RESIDUAL_POSITION = 1.3719447683362298
PHASE_2X_SHARE_RESIDUAL = 0.8174657655638667

PHASE_2Y_UNEXPLAINED = 0.9456715648873479
PHASE_2Y_SHARE_UNEXPLAINED = 0.6892927373703044

PHASE_2Z_UNEXPLAINED = 0.7993514476549548
PHASE_2Z_SHARE_UNEXPLAINED = 0.84527385334905

PHASE_3A_UNEXPLAINED = 0.8275878344476644
PHASE_3A_SHARE_UNEXPLAINED = 1.0353241204173036
PHASE_3A_DIVINE_SHIELD = -0.019057282937965794
PHASE_3A_POISON = -0.009179103854743919
PHASE_3A_CLEAVE = 0.0
PHASE_3A_SOC = 0.0
PHASE_3A_ORDINARY = 0.0

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)

NEXT_OBSERVABLE_DEFAULT = (
    "windfury extra swings / reborn-token survival / death-burst HP, "
    "beyond damaging-hit count / damage-per-hit / overkill"
)

REPRESENTED_HP = (
    "damaging_hits",
    "damage_per_hit",
    "overkill_threshold",
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3B may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3B must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3B seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def slot_bin(slot) -> int:
    """Coarse starting board slot: 0, 1, 2, 3, 4+."""
    try:
        s = int(slot)
    except (TypeError, ValueError):
        s = 0
    return min(max(s, 0), SLOT_BIN_CAP)


def hit_count_bin(row: Dict) -> int:
    """0 never damaging / 1 one damaging hit / 2 two or more."""
    n = int(row.get("n_damaging_hits") or 0)
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return 2


def overkill_bin(row: Dict) -> int:
    """0 none / 1 small (1–4) / 2 large (5+). Survivors and exact kills are 0."""
    try:
        ok = float(row.get("overkill_on_death") or 0)
    except (TypeError, ValueError):
        ok = 0.0
    if ok <= 0:
        return 0
    if ok < 5:
        return 1
    return 2


def hp_margin_value(row: Dict) -> float:
    """Start HP per unit mean incoming damage (depletion margin)."""
    try:
        start = float(row.get("start_health") or 0)
    except (TypeError, ValueError):
        start = 0.0
    try:
        mean_in = float(row.get("mean_incoming_dmg") or 0)
    except (TypeError, ValueError):
        mean_in = 0.0
    if mean_in <= 0:
        n = int(row.get("n_hits") or 0)
        tot = float(row.get("cumulative_incoming") or 0)
        mean_in = (tot / n) if n else 0.0
    return float(start) / max(mean_in, 1.0)


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


def diagnose_phase_3b(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 3A leftover to hit-count / damage-per-hit / overkill / leftover."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3b_seed": PHASE_3B_SEED,
        "phase_3b_lobbies": PHASE_3B_LOBBIES,
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
        "phase_2z_unexplained": PHASE_2Z_UNEXPLAINED,
        "phase_3a_unexplained": PHASE_3A_UNEXPLAINED,
        "phase_3a_share_unexplained": PHASE_3A_SHARE_UNEXPLAINED,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    rw = comparison.get("reweighting") or {}
    share_hits = rw.get("share_of_leftover_damaging_hits")
    share_dmg = rw.get("share_of_leftover_damage_per_hit")
    share_ok = rw.get("share_of_leftover_overkill_threshold")
    share_un = rw.get("share_of_leftover_still_unexplained")
    resid = rw.get("phase_3a_unexplained_hat")
    hits_f = float(share_hits) if share_hits is not None else None
    dmg_f = float(share_dmg) if share_dmg is not None else None
    ok_f = float(share_ok) if share_ok is not None else None
    un_f = float(share_un) if share_un is not None else None

    represented = [
        ("damaging_hits", hits_f),
        ("damage_per_hit", dmg_f),
        ("overkill_threshold", ok_f),
    ]
    ranked = _rank_parts(represented + [("still_unexplained", un_f)])
    ranked_repr = _rank_parts(represented)
    top = ranked[0]["component"] if ranked else "still_unexplained"
    top_repr = ranked_repr[0]["component"] if ranked_repr else "damaging_hits"
    repr_sum = sum(0.0 if s is None else float(s) for _, s in represented)

    def _correction(name: str) -> Tuple[str, str]:
        labels = {
            "damaging_hits": (
                "damaging-hit / exposure count",
                "the upstream represented mechanic that changes how often "
                "winner-start bodies take damaging hits",
            ),
            "damage_per_hit": (
                "damage-per-hit / HP depletion margin",
                "the upstream represented mechanic that changes incoming "
                "punch size relative to starting HP",
            ),
            "overkill_threshold": (
                "overkill / death-threshold",
                "the upstream represented mechanic that changes how far "
                "past 0 HP lethal hits land",
            ),
        }
        return labels[name]

    one_dom = None
    for name, share in represented:
        if share is not None and share > SHARE_DOMINANT:
            one_dom = name
            break

    if one_dom is not None:
        pretty, nxt = _correction(one_dom)
        primary = f"{one_dom}_dominates"
        next_step = (
            f"Most of the 3A leftover is {pretty} on same-keyword bodies. "
            f"Next hour: audit {nxt} before any behavior change. Do not "
            "rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )
    elif repr_sum > SHARE_DOMINANT:
        pretty, nxt = _correction(top_repr)
        primary = "jointly_explained_rank_largest"
        next_step = (
            f"Several HP/depletion mechanisms jointly clear ~70% of the "
            f"3A leftover (sum={repr_sum:.3f}; largest={top_repr} = {pretty}). "
            f"Next hour: isolate the largest upstream cause first and audit "
            f"{nxt}. Do not rewrite 2Q; do not change `_hero_damage`; do not "
            "burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            f"HP/depletion mechanisms explain <~70% of the 3A leftover "
            f"(top={top}, represented_sum={repr_sum:.3f}). Rank the residual "
            f"and collect the smallest extra observable before any behavior "
            f"change: {NEXT_OBSERVABLE_DEFAULT}. Do not rewrite 2Q; do not "
            "change `_hero_damage`; do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "ranked_represented": ranked_repr,
        "represented_share_sum": repr_sum,
        "smallest_additional_observable": NEXT_OBSERVABLE_DEFAULT,
        "reweighting": rw,
        "share_of_leftover_damaging_hits": hits_f,
        "share_of_leftover_damage_per_hit": dmg_f,
        "share_of_leftover_overkill_threshold": ok_f,
        "share_of_leftover_still_unexplained": un_f,
        "phase_3a_unexplained_hat": resid,
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
    })
    return out
