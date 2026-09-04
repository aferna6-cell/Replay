"""Phase 3C — attacker-punch attribution (measurement only).

Stacked on Phase 3B (PR #46). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #46.

Splits the 3B damage-per-hit / HP-margin term
(B ≈ +0.939 / hit, 113.4% of 3A leftover F) into:

* (A) attacker attack-strength mix at impact
* (B) attacker synthetic-vs-recruit attack composition
* (C) attacker/defender pairing or attack-order | same attacker strength
* (D) still unexplained
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3c_v1"
PHASE_3C_SEED = 14200
PHASE_3C_LOBBIES = 500
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
N_ATK_BINS = 5
N_SYNTH_ATK_BINS = 5
N_PAIR_BINS = 5

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

PHASE_3B_DAMAGE_PER_HIT = 0.9385531501941458
PHASE_3B_SHARE_DAMAGE_PER_HIT = 1.1340828261697928
PHASE_3B_DAMAGING_HITS = -0.11048217273562631
PHASE_3B_OVERKILL = -0.009564230345383801
PHASE_3B_UNEXPLAINED = 0.009081087334528948

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46)

NEXT_OBSERVABLE_DEFAULT = (
    "windfury extra swings / reborn-token survival / death-burst HP / "
    "per-keyword punch, beyond attacker attack-strength / synth-vs-recruit / "
    "pairing-order"
)

REPRESENTED_PUNCH = (
    "attacker_attack_strength",
    "attacker_synth_composition",
    "pairing_order",
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3C may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3C must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3C seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def slot_bin(slot) -> int:
    """Coarse starting board slot: 0, 1, 2, 3, 4+."""
    try:
        s = int(slot)
    except (TypeError, ValueError):
        s = 0
    return min(max(s, 0), SLOT_BIN_CAP)


def attacker_attack_value(row: Dict) -> float:
    """Mean dealer attack at damaging-hit impact. 0 if none."""
    try:
        return float(row.get("mean_attacker_attack") or 0)
    except (TypeError, ValueError):
        return 0.0


def attacker_synth_share_value(row: Dict) -> float:
    """Mean dealer synthetic-attack share at impact. 0 if none."""
    try:
        return float(row.get("mean_attacker_synth_share") or 0)
    except (TypeError, ValueError):
        return 0.0


def pairing_order_value(row: Dict) -> float:
    """Joint attack-order + relative slot. −1 when the body took no punch."""
    try:
        n = int(row.get("n_damaging_hits") or row.get("n_punch_hits") or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return -1.0
    raw = row.get("pairing_order_value")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    idx = row.get("mean_attacker_first_attack_index")
    try:
        idx_f = 0.0 if idx is None else float(idx)
    except (TypeError, ValueError):
        idx_f = 0.0
    try:
        rel = float(row.get("mean_relative_slot") or 0)
    except (TypeError, ValueError):
        rel = 0.0
    return float(idx_f) + 0.05 * rel


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


def diagnose_phase_3c(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 3B damage-per-hit to attack-strength / synth / pairing / leftover."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3c_seed": PHASE_3C_SEED,
        "phase_3c_lobbies": PHASE_3C_LOBBIES,
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
        "phase_3b_damage_per_hit": PHASE_3B_DAMAGE_PER_HIT,
        "phase_3b_share_damage_per_hit": PHASE_3B_SHARE_DAMAGE_PER_HIT,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    rw = comparison.get("reweighting") or {}
    share_atk = rw.get("share_of_b_attacker_attack_strength")
    share_syn = rw.get("share_of_b_attacker_synth_composition")
    share_pair = rw.get("share_of_b_pairing_order")
    share_un = rw.get("share_of_b_still_unexplained")
    resid = rw.get("phase_3b_damage_per_hit_hat")
    atk_f = float(share_atk) if share_atk is not None else None
    syn_f = float(share_syn) if share_syn is not None else None
    pair_f = float(share_pair) if share_pair is not None else None
    un_f = float(share_un) if share_un is not None else None

    represented = [
        ("attacker_attack_strength", atk_f),
        ("attacker_synth_composition", syn_f),
        ("pairing_order", pair_f),
    ]
    ranked = _rank_parts(represented + [("still_unexplained", un_f)])
    ranked_repr = _rank_parts(represented)
    top = ranked[0]["component"] if ranked else "still_unexplained"
    top_repr = ranked_repr[0]["component"] if ranked_repr else "attacker_attack_strength"
    repr_sum = sum(0.0 if s is None else float(s) for _, s in represented)

    def _correction(name: str) -> Tuple[str, str]:
        labels = {
            "attacker_attack_strength": (
                "attacker attack-strength mix at impact",
                "the upstream board-strength / allocation source producing "
                "that attacker-strength distribution (without tuning)",
            ),
            "attacker_synth_composition": (
                "attacker synthetic-vs-recruit attack composition",
                "the upstream synthetic-allocation source of incoming punch "
                "(without retuning total scaling)",
            ),
            "pairing_order": (
                "same-strength attacker/defender pairing or attack-order",
                "targeting / initiative fidelity on same-attack bodies",
            ),
        }
        return labels[name]

    one_dom = None
    for name, share in represented:
        if share is not None and share > SHARE_DOMINANT:
            one_dom = name
            break

    if one_dom == "attacker_attack_strength":
        pretty, nxt = _correction(one_dom)
        primary = "attacker_attack_strength_dominates"
        next_step = (
            f"Most of the 3B +0.939 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    elif one_dom == "pairing_order":
        pretty, nxt = _correction(one_dom)
        primary = "pairing_order_dominates"
        next_step = (
            f"Most of the 3B +0.939 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    elif one_dom is not None:
        pretty, nxt = _correction(one_dom)
        primary = f"{one_dom}_dominates"
        next_step = (
            f"Most of the 3B +0.939 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    elif repr_sum > SHARE_DOMINANT:
        pretty, nxt = _correction(top_repr)
        primary = "jointly_explained_rank_largest"
        next_step = (
            f"Several punch mechanisms jointly clear ~70% of the 3B +0.939 "
            f"(sum={repr_sum:.3f}; largest={top_repr} = {pretty}). Next hour: "
            f"isolate the largest upstream cause first and audit {nxt}. Do not "
            "rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            f"Neither attack-strength mix nor same-strength pairing/order "
            f"clears ~70% of the 3B +0.939 (top={top}, "
            f"represented_sum={repr_sum:.3f}). Rank the residual and collect "
            f"the smallest extra observable before any behavior change: "
            f"{NEXT_OBSERVABLE_DEFAULT}. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
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
        "share_of_b_attacker_attack_strength": atk_f,
        "share_of_b_attacker_synth_composition": syn_f,
        "share_of_b_pairing_order": pair_f,
        "share_of_b_still_unexplained": un_f,
        "phase_3b_damage_per_hit_hat": resid,
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
    })
    return out
