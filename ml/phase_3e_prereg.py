"""Phase 3E — board-pool lifecycle attribution (measurement only).

Stacked on Phase 3D (PR #50). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #50.

Reproduces 3D A1 (board-pool magnitude ≈ +0.422 / hit, 82.4% of 3C A)
then splits the arm gap in opposing board-pool magnitude into:

* (1) inherited / carry pool entering the turn
* (2) current-turn scaling increment (identical existing scaling math)
* (3) replacement / churn retention-vs-loss
* (4) lifecycle / elimination / board-state selection + leftover
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3e_v1"
PHASE_3E_SEED = 14200
PHASE_3E_LOBBIES = 500
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
N_CARRY_BINS = 5
N_SCALE_BINS = 5
N_REPLACE_BINS = 5
N_SELECT_BINS = 5

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

# Published 3D A1 (board-pool magnitude of 3C A). Reproduce exactly.
PHASE_3D_BOARD_POOL_MAGNITUDE = 0.4216721428553852
PHASE_3D_SHARE_BOARD_POOL = 0.8235063814972068
PHASE_3D_ALLOCATION_CONCENTRATION = 0.2586937634696737
PHASE_3D_SHARE_ALLOCATION = 0.5052170713204244
PHASE_3D_COMBAT_MUTATION = 0.0
PHASE_3D_SHARE_COMBAT_MUTATION = 0.0
PHASE_3D_UNEXPLAINED = 0.08033207704471289
PHASE_3D_SHARE_UNEXPLAINED = 0.15688486708483898

SHARE_DOMINANT = 0.70
FLOW_ABS_TOL = 1.0  # integer attack-pool rounding from residual paint

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

HOLD_PRS = (
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50,
)

POOL_FLOW_IDENTITY = (
    "post = pre + add - represented_loss_or_transfer"
)

IMPACT_ATTACK_IDENTITY = (
    "impact_attack = start_recruit + start_pool_share + combat_delta"
)

NEXT_OBSERVABLE_DEFAULT = (
    "when the board-pool divergence first appears, or the upstream scaling "
    "inputs, or 2S pool lifecycle fidelity, or matched alive/turn/board state, "
    "ranked by the largest 3E piece of +0.422"
)

REPRESENTED_SOURCE = (
    "inherited_carry_pool",
    "current_turn_scaling_add",
    "replacement_churn",
    "lifecycle_selection",
)


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3E may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3E must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3E seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
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


def carry_pool_value(row: Dict) -> float:
    """Opposing attack-pool at recruit start (inherited / carry)."""
    return _float_or_zero(
        row, "opp_carry_attack_pool", "opp_attack_pool_recruit_start",
    )


def scaling_add_value(row: Dict) -> float:
    """Opposing current-turn attack-pool increment from residual/ratio paint."""
    return _float_or_zero(
        row, "opp_scale_add_attack", "opp_attack_pool_scale_add",
    )


def replacement_loss_value(row: Dict) -> float:
    """Opposing recruit-phase attack-pool loss (positive = lost / not retained)."""
    return _float_or_zero(
        row, "opp_replace_loss_attack", "opp_attack_pool_replace_loss",
    )


def selection_state_value(row: Dict) -> float:
    """Board-state selection proxy: opposing combat-start board size."""
    return _float_or_zero(
        row, "opp_board_size", "opp_select_board_size",
    )


def board_pool_value(row: Dict) -> float:
    """Opposing board total abstract-pool attack at combat start (3D A1)."""
    return _float_or_zero(
        row, "opp_board_pool_attack", "mean_attacker_board_pool",
    )


def flow_post_value(row: Dict) -> float:
    return _float_or_zero(
        row, "opp_attack_pool_post_scale", "opp_attack_pool_combat_start",
        "opp_board_pool_attack",
    )


def flow_pre_value(row: Dict) -> float:
    return _float_or_zero(row, "opp_attack_pool_pre_scale")


def flow_identity_residual(row: Dict) -> float:
    """post - (carry + add - loss). 0 when the seat-turn flow closes."""
    if row.get("opp_flow_residual") not in (None, ""):
        try:
            return float(row["opp_flow_residual"])
        except (TypeError, ValueError):
            pass
    post = flow_post_value(row)
    carry = carry_pool_value(row)
    add = scaling_add_value(row)
    loss = replacement_loss_value(row)
    return float(post - (carry + add - loss))


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


def diagnose_phase_3e(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route 3D A1=+0.422 to carry / scale-add / replacement / leftover."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3e_seed": PHASE_3E_SEED,
        "phase_3e_lobbies": PHASE_3E_LOBBIES,
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
        "no_scaling_constant_change": True,
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
        "phase_3d_board_pool_magnitude": PHASE_3D_BOARD_POOL_MAGNITUDE,
        "phase_3d_share_board_pool": PHASE_3D_SHARE_BOARD_POOL,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    rw = comparison.get("reweighting") or {}
    share_carry = rw.get("share_of_a1_inherited_carry_pool")
    share_scale = rw.get("share_of_a1_current_turn_scaling_add")
    share_repl = rw.get("share_of_a1_replacement_churn")
    share_sel = rw.get("share_of_a1_lifecycle_selection")
    share_left = rw.get("share_of_a1_still_unexplained")
    share_sel_left = rw.get("share_of_a1_lifecycle_selection_plus_leftover")
    a1_hat = rw.get("phase_3d_board_pool_magnitude_hat")
    a1_3d = rw.get("reproduced_3d_board_pool_magnitude")
    carry_f = float(share_carry) if share_carry is not None else None
    scale_f = float(share_scale) if share_scale is not None else None
    repl_f = float(share_repl) if share_repl is not None else None
    sel_f = float(share_sel) if share_sel is not None else None
    left_f = float(share_left) if share_left is not None else None
    sel_left_f = float(share_sel_left) if share_sel_left is not None else None
    if sel_left_f is None:
        sel_left_f = None if sel_f is None and left_f is None else (
            (0.0 if sel_f is None else sel_f) + (0.0 if left_f is None else left_f)
        )

    represented = [
        ("inherited_carry_pool", carry_f),
        ("current_turn_scaling_add", scale_f),
        ("replacement_churn", repl_f),
        ("lifecycle_selection", sel_f),
    ]
    ranked = _rank_parts(
        represented + [("still_unexplained", left_f)]
    )
    ranked_repr = _rank_parts(represented)
    top = ranked[0]["component"] if ranked else "still_unexplained"
    top_repr = (
        ranked_repr[0]["component"] if ranked_repr else "inherited_carry_pool"
    )
    repr_sum = sum(0.0 if s is None else float(s) for _, s in represented)

    def _correction(name: str) -> Tuple[str, str]:
        labels = {
            "inherited_carry_pool": (
                "inherited / carry pool entering the turn",
                "when the board-pool divergence first appears "
                "(history / earlier turns), without retuning total scaling",
            ),
            "current_turn_scaling_add": (
                "current-turn scaling increment from identical existing math",
                "the upstream scaling *inputs* (Firestone target, pre-scale "
                "stats, growth factor, clamp) — not the constants",
            ),
            "replacement_churn": (
                "replacement / churn retention-vs-loss",
                "2S pool lifecycle fidelity (sell→buy→play conservation)",
            ),
            "lifecycle_selection": (
                "lifecycle / elimination / board-state selection",
                "matched alive / turn / board state before any implementation",
            ),
        }
        return labels[name]

    one_dom = None
    checks = [
        ("inherited_carry_pool", carry_f),
        ("current_turn_scaling_add", scale_f),
        ("replacement_churn", repl_f),
        ("lifecycle_selection", sel_left_f if sel_left_f is not None else sel_f),
    ]
    for name, share in checks:
        if share is not None and share > SHARE_DOMINANT:
            one_dom = name
            break

    if one_dom == "inherited_carry_pool":
        pretty, nxt = _correction(one_dom)
        primary = "carry_history_dominates"
        next_step = (
            f"Most of the 3D A1 +0.422 is {pretty}. Next hour: trace {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )
    elif one_dom == "current_turn_scaling_add":
        pretty, nxt = _correction(one_dom)
        primary = "current_turn_scaling_add_dominates"
        next_step = (
            f"Most of the 3D A1 +0.422 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )
    elif one_dom == "replacement_churn":
        pretty, nxt = _correction(one_dom)
        primary = "replacement_retention_dominates"
        next_step = (
            f"Most of the 3D A1 +0.422 is {pretty}. Next hour: audit {nxt} "
            "before any behavior change. Do not rewrite 2Q; do not change "
            "`_hero_damage`; do not retune scaling constants; do not burn "
            "confirm."
        )
    elif one_dom == "lifecycle_selection":
        pretty, nxt = _correction(one_dom)
        primary = "lifecycle_selection_dominates"
        next_step = (
            f"Most of the 3D A1 +0.422 is {pretty}. Next hour: condition on "
            f"{nxt}. Do not rewrite 2Q; do not change `_hero_damage`; do not "
            "retune scaling constants; do not burn confirm."
        )
    elif repr_sum > SHARE_DOMINANT:
        pretty, nxt = _correction(top_repr)
        primary = "jointly_explained_rank_largest"
        next_step = (
            f"Several lifecycle mechanisms jointly clear ~70% of the 3D A1 "
            f"+0.422 (sum={repr_sum:.3f}; largest={top_repr} = {pretty}). "
            f"Next hour: isolate the largest piece first and audit {nxt}. "
            "Do not rewrite 2Q; do not change `_hero_damage`; do not retune "
            "scaling constants; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            f"Neither carry, current-turn scaling-add, replacement "
            f"retention/loss, nor lifecycle selection clears ~70% of the "
            f"3D A1 +0.422 (top={top}, represented_sum={repr_sum:.3f}). "
            f"Rank the residual and isolate the largest piece before any "
            f"behavior change: {NEXT_OBSERVABLE_DEFAULT}. Do not rewrite "
            "2Q; do not change `_hero_damage`; do not retune scaling "
            "constants; do not burn confirm."
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
        "share_of_a1_inherited_carry_pool": carry_f,
        "share_of_a1_current_turn_scaling_add": scale_f,
        "share_of_a1_replacement_churn": repl_f,
        "share_of_a1_lifecycle_selection": sel_f,
        "share_of_a1_lifecycle_selection_plus_leftover": sel_left_f,
        "share_of_a1_still_unexplained": left_f,
        "phase_3d_board_pool_magnitude_hat": a1_hat,
        "reproduced_3d_board_pool_magnitude": a1_3d,
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
        "keep_pr_50_hold": True,
    })
    return out
