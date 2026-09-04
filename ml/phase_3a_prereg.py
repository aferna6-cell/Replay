"""Phase 3A — lethal-cause / keyword attribution of the 2Z leftover (measurement only).

Stacked on Phase 2Z (PR #44). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #44.

Splits the leftover 2Z unexplained term
(E ≈ +0.799 / hit, 84.5% of 2Y leftover C) into:

* (A) divine-shield exposure / pop
* (B) poisonous / venomous hit
* (C) cleave primary vs secondary
* (D) represented start-of-combat hits
* (E) ordinary attack / counterattack hit
* (F) still unexplained
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3a_v1"
PHASE_3A_SEED = 14200
PHASE_3A_LOBBIES = 500
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

# 2Z published leftover after holding target / cursor / gen-DR / unsupported.
PHASE_2Z_UNEXPLAINED = 0.7993514476549548
PHASE_2Z_SHARE_UNEXPLAINED = 0.84527385334905
PHASE_2Z_TARGETING = 0.04779538299192791
PHASE_2Z_CURSOR = 0.032450468782126644
PHASE_2Z_GENERATED = 0.0660742654583368
PHASE_2Z_UNSUPPORTED = 0.0

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44)

NEXT_OBSERVABLE_DEFAULT = (
    "per-hit remaining HP / overkill / windfury / reborn-token survival, "
    "beyond DS / poison / cleave / SOC / ordinary lethal"
)

REPRESENTED_LETHAL = (
    "divine_shield",
    "poison_venomous",
    "cleave",
    "start_of_combat",
    "ordinary_combat",
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3A may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3A must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3A seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def slot_bin(slot) -> int:
    """Coarse starting board slot: 0, 1, 2, 3, 4+."""
    try:
        s = int(slot)
    except (TypeError, ValueError):
        s = 0
    return min(max(s, 0), SLOT_BIN_CAP)


def ds_bin(row: Dict) -> int:
    """0 never DS / 1 started with DS never popped / 2 shield popped."""
    if int(row.get("n_shield_pops") or 0) > 0:
        return 2
    if row.get("start_divine_shield") or row.get("divine_shield"):
        return 1
    return 0


def poison_bin(row: Dict) -> int:
    """0 never hit by poisonous/venomous / 1 hit (including shield-eaten)."""
    if int(row.get("n_hits_poison") or 0) > 0 or row.get("poison_lethal"):
        return 1
    return 0


def cleave_bin(row: Dict) -> int:
    """0 never / 1 primary of a cleave attacker / 2 secondary splash."""
    if int(row.get("n_cleave_secondary") or 0) > 0 or (
        row.get("cleave_lethal") and int(row.get("n_cleave_primary") or 0) == 0
    ):
        return 2
    if int(row.get("n_cleave_primary") or 0) > 0:
        return 1
    return 0


def soc_bin(row: Dict) -> int:
    """0 never hit by represented SOC / 1 hit."""
    if int(row.get("n_soc_hits") or 0) > 0 or row.get("soc_lethal"):
        return 1
    return 0


def ordinary_bin(row: Dict) -> int:
    """0 never ordinary hit / 1 attack-hit as defender / 2 counterattack-hit."""
    if int(row.get("n_ordinary_counter_hits") or 0) > 0:
        return 2
    if int(row.get("n_ordinary_attack_hits") or 0) > 0 or row.get("ordinary_lethal"):
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


def diagnose_phase_3a(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 2Z leftover to DS / poison / cleave / SOC / ordinary / leftover."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3a_seed": PHASE_3A_SEED,
        "phase_3a_lobbies": PHASE_3A_LOBBIES,
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
        "phase_2z_share_unexplained": PHASE_2Z_SHARE_UNEXPLAINED,
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
    share_ds = rw.get("share_of_leftover_divine_shield")
    share_poi = rw.get("share_of_leftover_poison_venomous")
    share_cl = rw.get("share_of_leftover_cleave")
    share_soc = rw.get("share_of_leftover_start_of_combat")
    share_ord = rw.get("share_of_leftover_ordinary_combat")
    share_un = rw.get("share_of_leftover_still_unexplained")
    resid = rw.get("phase_2z_unexplained_hat")
    ds_f = float(share_ds) if share_ds is not None else None
    poi_f = float(share_poi) if share_poi is not None else None
    cl_f = float(share_cl) if share_cl is not None else None
    soc_f = float(share_soc) if share_soc is not None else None
    ord_f = float(share_ord) if share_ord is not None else None
    un_f = float(share_un) if share_un is not None else None

    represented = [
        ("divine_shield", ds_f),
        ("poison_venomous", poi_f),
        ("cleave", cl_f),
        ("start_of_combat", soc_f),
        ("ordinary_combat", ord_f),
    ]
    ranked = _rank_parts(represented + [("still_unexplained", un_f)])
    ranked_repr = _rank_parts(represented)
    top = ranked[0]["component"] if ranked else "still_unexplained"
    top_repr = ranked_repr[0]["component"] if ranked_repr else "divine_shield"
    repr_sum = sum(0.0 if s is None else float(s) for _, s in represented)

    def _correction(name: str) -> str:
        labels = {
            "divine_shield": (
                "divine-shield pop / absorb",
                "a divine-shield correction/audit",
            ),
            "poison_venomous": (
                "poisonous / venomous lethal",
                "a poison/venomous correction/audit",
            ),
            "cleave": (
                "cleave primary/secondary lethal",
                "a cleave correction/audit",
            ),
            "start_of_combat": (
                "represented start-of-combat hits",
                "a start-of-combat correction/audit",
            ),
            "ordinary_combat": (
                "ordinary attack/counterattack lethal",
                "an ordinary-combat lethal correction/audit",
            ),
        }
        pretty, nxt = labels[name]
        return pretty, nxt

    one_dom = None
    for name, share in represented:
        if share is not None and share > SHARE_DOMINANT:
            one_dom = name
            break

    if one_dom is not None:
        pretty, nxt = _correction(one_dom)
        primary = f"{one_dom}_dominates"
        next_step = (
            f"Most of the 2Z leftover is {pretty} on same-target / same-cursor "
            f"/ same-DR bodies. Next hour: preregister only {nxt}. Do not "
            "rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )
    elif repr_sum > SHARE_DOMINANT:
        pretty, nxt = _correction(top_repr)
        primary = "jointly_explained_rank_largest"
        next_step = (
            f"Several represented lethal mechanisms jointly clear ~70% of the "
            f"2Z leftover (sum={repr_sum:.3f}; largest={top_repr}). Next hour: "
            f"isolate the largest first and preregister only {nxt}. Do not "
            "rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            f"Represented lethal mechanisms explain <~70% of the 2Z leftover "
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
        "share_of_leftover_divine_shield": ds_f,
        "share_of_leftover_poison_venomous": poi_f,
        "share_of_leftover_cleave": cl_f,
        "share_of_leftover_start_of_combat": soc_f,
        "share_of_leftover_ordinary_combat": ord_f,
        "share_of_leftover_still_unexplained": un_f,
        "phase_2z_unexplained_hat": resid,
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
    })
    return out
