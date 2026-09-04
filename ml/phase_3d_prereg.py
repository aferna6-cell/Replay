"""Phase 3D — upstream attacker-punch source attribution (measurement only).

Stacked on Phase 3C (PR #47). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #47.

Splits the 3C attacker-attack-strength mix term
(A ≈ +0.512 / hit, 54.6% of 3B B) into:

* (A1) opposing board-level abstract-pool magnitude at combat start
* (A2) allocation concentration onto bodies that actually attack
* (A3) in-combat attack growth / effect delta
* (A4) still unexplained (residual attack-strength mix)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3d_v1"
PHASE_3D_SEED = 14200
PHASE_3D_LOBBIES = 500
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
N_POOL_BINS = 5
N_CONC_BINS = 5
N_DELTA_BINS = 5

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

PHASE_3C_ATTACKER_ATTACK_STRENGTH = 0.5120447786800975
PHASE_3C_SHARE_ATTACKER_ATTACK = 0.5455682276216085
PHASE_3C_SYNTH_COMPOSITION = 0.20105428114752724
PHASE_3C_PAIRING_ORDER = 0.0996625571539457
PHASE_3C_UNEXPLAINED = 0.12530839020172058

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

HOLD_PRS = (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)

IMPACT_ATTACK_IDENTITY = (
    "impact_attack = start_recruit + start_pool_share + combat_delta"
)

NEXT_OBSERVABLE_DEFAULT = (
    "why opposing board-level synthetic strength differs across arms, "
    "or pool-allocation fidelity, or represented in-combat attack effects, "
    "ranked by the largest 3D piece of +0.512"
)

REPRESENTED_SOURCE = (
    "board_pool_magnitude",
    "allocation_concentration",
    "combat_mutation",
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3D may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3D must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3D seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def _float_or_zero(row: Dict, *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def board_pool_value(row: Dict) -> float:
    """Opposing board total abstract-pool attack at combat start."""
    return _float_or_zero(
        row, "opp_board_pool_attack", "mean_attacker_board_pool",
    )


def allocation_concentration_value(row: Dict) -> float:
    """Share of opposing pool sitting on bodies that actually attacked."""
    return _float_or_zero(
        row, "opp_pool_on_attackers_share", "mean_attacker_pool_share_of_board",
    )


def combat_delta_value(row: Dict) -> float:
    """Mean in-combat attack mutation of damaging dealers. 0 if none."""
    return _float_or_zero(row, "mean_attacker_combat_delta")


def attacker_start_recruit_value(row: Dict) -> float:
    return _float_or_zero(
        row, "mean_attacker_start_recruit", "mean_attacker_recruit_attack",
    )


def attacker_start_pool_value(row: Dict) -> float:
    return _float_or_zero(row, "mean_attacker_start_pool")


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


def diagnose_phase_3d(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 3C A=+0.512 to pool magnitude / concentration / combat / leftover."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3d_seed": PHASE_3D_SEED,
        "phase_3d_lobbies": PHASE_3D_LOBBIES,
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
        "phase_3c_attacker_attack_strength": PHASE_3C_ATTACKER_ATTACK_STRENGTH,
        "phase_3c_share_attacker_attack": PHASE_3C_SHARE_ATTACKER_ATTACK,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    rw = comparison.get("reweighting") or {}
    share_pool = rw.get("share_of_a_board_pool_magnitude")
    share_conc = rw.get("share_of_a_allocation_concentration")
    share_delta = rw.get("share_of_a_combat_mutation")
    share_un = rw.get("share_of_a_still_unexplained")
    resid = rw.get("phase_3c_attacker_attack_strength_hat")
    pool_f = float(share_pool) if share_pool is not None else None
    conc_f = float(share_conc) if share_conc is not None else None
    delta_f = float(share_delta) if share_delta is not None else None
    un_f = float(share_un) if share_un is not None else None

    represented = [
        ("board_pool_magnitude", pool_f),
        ("allocation_concentration", conc_f),
        ("combat_mutation", delta_f),
    ]
    ranked = _rank_parts(represented + [("still_unexplained", un_f)])
    ranked_repr = _rank_parts(represented)
    top = ranked[0]["component"] if ranked else "still_unexplained"
    top_repr = (
        ranked_repr[0]["component"] if ranked_repr else "board_pool_magnitude"
    )
    repr_sum = sum(0.0 if s is None else float(s) for _, s in represented)

    def _correction(name: str) -> Tuple[str, str]:
        labels = {
            "board_pool_magnitude": (
                "opposing board-level abstract-pool magnitude",
                "why opposing board-level synthetic strength differs so "
                "sharply across arms (without retuning total scaling)",
            ),
            "allocation_concentration": (
                "allocation concentration onto bodies that actually attack",
                "pool allocation fidelity (who receives the painted share)",
            ),
            "combat_mutation": (
                "in-combat attack growth / effect delta",
                "represented in-combat attack-effect fidelity",
            ),
        }
        return labels[name]

    one_dom = None
    for name, share in represented:
        if share is not None and share > SHARE_DOMINANT:
            one_dom = name
            break

    if one_dom == "board_pool_magnitude":
        pretty, nxt = _correction(one_dom)
        primary = "board_pool_magnitude_dominates"
        next_step = (
            f"Most of the 3C +0.512 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    elif one_dom == "allocation_concentration":
        pretty, nxt = _correction(one_dom)
        primary = "allocation_concentration_dominates"
        next_step = (
            f"Most of the 3C +0.512 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    elif one_dom == "combat_mutation":
        pretty, nxt = _correction(one_dom)
        primary = "combat_mutation_dominates"
        next_step = (
            f"Most of the 3C +0.512 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not burn confirm."
        )
    elif repr_sum > SHARE_DOMINANT:
        pretty, nxt = _correction(top_repr)
        primary = "jointly_explained_rank_largest"
        next_step = (
            f"Several source mechanisms jointly clear ~70% of the 3C +0.512 "
            f"(sum={repr_sum:.3f}; largest={top_repr} = {pretty}). Next hour: "
            f"isolate the largest piece first and audit {nxt}. Do not "
            "rewrite 2Q; do not change `_hero_damage`; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            f"Neither board-pool magnitude, allocation concentration, nor "
            f"combat mutation clears ~70% of the 3C +0.512 (top={top}, "
            f"represented_sum={repr_sum:.3f}). Rank the residual and isolate "
            f"the largest piece before any behavior change: "
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
        "share_of_a_board_pool_magnitude": pool_f,
        "share_of_a_allocation_concentration": conc_f,
        "share_of_a_combat_mutation": delta_f,
        "share_of_a_still_unexplained": un_f,
        "phase_3c_attacker_attack_strength_hat": resid,
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
        "keep_pr_47_hold": True,
    })
    return out
