"""Phase 3K — elimination-timing attribution (measurement only).

Stacked on Phase 3J (PR #56). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #56.

Reproduces the 3J eligibility leftover (5648 pairing-schedule punch rows)
and splits first T7+ eligibility divergence exclusively into:

* (1) treatment seat eliminated earlier
* (2) control comparison/opponent eliminated earlier (different live set)
* (3) ghost/bye transition from an earlier third-party elimination
* (4) missing / unreconciled

For (1)/(2), further attributes the decisive HP gap into accumulated
prior HP vs current-fight hit/no-hit vs current-fight damage magnitude.
"""

from __future__ import annotations

from typing import Dict, List, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3k_v1"
PHASE_3K_SEED = 14200
PHASE_3K_LOBBIES = 500
REUSED_SEED_LO = 14200
REUSED_SEED_HI = 14699
FROZEN_ALPHA = 0.5
INSTRUMENT_TURNS = tuple(range(7, 15))  # T7–T14 inclusive
LOW_TIERS = (1, 2, 3)
LOW_WINNER_START_TIERS = LOW_TIERS
EARLY_TURNS = (7, 8, 9)
LATE_TURNS = (10, 11, 12, 13, 14)
VERY_LATE_TURNS = (12, 13, 14)
PAIRING_TURNS = LATE_TURNS
TRACE_FROM_TURN = 7

SHARE_DOMINANT = 0.70
FLOW_ABS_TOL = 1.0
REWEIGHT_ABS_TOL = 1e-6
LINEAGE_ABS_TOL = 1e-9

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
    29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    50, 51, 52, 53, 54, 55, 56,
)

POOL_FLOW_IDENTITY = (
    "post = pre + add - represented_loss_or_transfer"
)

IMPACT_ATTACK_IDENTITY = (
    "impact_attack = start_recruit + start_pool_share + combat_delta"
)

HISTORY_LINK_IDENTITY = (
    "punch.opp_carry = seat_history[turn].attack_pool_recruit_start"
)

WEIGHT_RECONCILIATION_IDENTITY = (
    "sum_k n_arm(k) = N_arm; sum_k w_arm(k) = 1; "
    "mixture + within_cell + leftover = unpaired_punch_delta_carry"
)

LINEAGE_IDENTITY = (
    "t1t3_end = t1t3_start + t1t3_added - t1t3_removed"
)

PAIRED_SEAT_IDENTITY = (
    "paired (seed, seat) in both arms; first_loss_turn is the first "
    "instrumented turn the seat's combat-start T1–T3 count hits 0"
)

PAIRING_IDENTITY = (
    "same pairing iff both fights are live and opponent_seat matches "
    "at (seed, leftover_winner_seat, turn)"
)

LEFTOVER_RECONCILE_IDENTITY = (
    "pairing_schedule + outcome_flip + survivor_substitution + residual "
    "= 3H leftover (treatment alive and still fields T1–T3)"
)

CANDIDATE_CHOICE_IDENTITY = (
    "chosen opponent is an element of the logged legal candidate set "
    "(other alive seats, plus ghost/bye iff the lobby is odd and eligible)"
)

MATCHMAKING_RECONCILE_IDENTITY = (
    "eligibility + history_legal + rng_order + unreconciled "
    "= 3I pairing_schedule (5952 leftover punch rows)"
)

HP_FLOW_IDENTITY = (
    "post_hp = pre_hp - applied_to_seat; "
    "applied_fight = sum of seat HP losses"
)

ELIMINATION_IDENTITY = (
    "elimination_turn is the first combat turn after which the seat is "
    "dead (hp<=0); survived seats have no combat elimination"
)

ELIGIBILITY_TIMING_IDENTITY = (
    "treatment_eliminated_earlier + control_opponent_eliminated_earlier "
    "+ ghost_bye_third_party + unreconciled = 3J eligibility (5648)"
)

HP_GAP_RECONCILE_IDENTITY = (
    "accumulated_prior_hp + current_fight_hit + current_fight_damage_magnitude "
    "+ hp_unreconciled = treatment_eliminated_earlier "
    "+ control_opponent_eliminated_earlier"
)

# Published 3D / 3E / 3F / 3G / 3H / 3I / 3J locks (exact).
PHASE_3D_BOARD_POOL_MAGNITUDE = 0.4216721428553852
PHASE_3E_CARRY_DELTA = 0.30513688784757187
PHASE_3E_CARRY_SHARE_OF_A1 = 0.7236353954551374
PHASE_3E_PUNCH_DELTA_CARRY = -196.33317557443002
PHASE_3F_UNCOND_PAIRED_DELTA = -17.83493589743591
PHASE_3F_UNCOND_SHARE = 0.09084015396406948
PHASE_3F_SELECTION_SHARE = 0.9091598460359305
PHASE_3G_MIXTURE = -196.52943934946725
PHASE_3G_MIXTURE_SHARE = 1.0009996465165045
PHASE_3G_WITHIN_CELL = 0.19626377503730166
PHASE_3G_WITHIN_SHARE = -0.0009996465165047867
PHASE_3G_ROLE_ALIVE = 36.35037820310066
PHASE_3G_ROLE_SHARE = -0.1851463874953737
PHASE_3G_MIX_ROLE_SHARE = 0.8158532590211308
PHASE_3G_N_CONTROL = 54223
PHASE_3G_N_TREATMENT = 50116
PHASE_3H_LATE_CONTROL = 17924
PHASE_3H_LATE_TREATMENT = 4273
PHASE_3H_COLLAPSE = 13651
PHASE_3H_LEFTOVER = 7155
PHASE_3H_ELIMINATION = 6550
PHASE_3H_OFFER_SHIFT = 4219
PHASE_3H_SHARE_LEFTOVER = 0.5241374258296095
PHASE_3I_PAIRING_SCHEDULE = 5952
PHASE_3I_OUTCOME_FLIP = 668
PHASE_3I_SURVIVOR_SUBSTITUTION = 292
PHASE_3I_RESIDUAL = 243
PHASE_3I_DIFFERENT_OPPONENT = 5009
PHASE_3I_KIND_MISMATCH = 943
PHASE_3I_SHARE_PAIRING_SCHEDULE = 0.8318658280922432
PHASE_3I_SHARE_OUTCOME_FLIP = 0.093361285814116
PHASE_3I_SHARE_SURVIVOR_SUBSTITUTION = 0.04081062194269741
PHASE_3I_SHARE_RESIDUAL = 0.033962264150943396
PHASE_3J_ELIGIBILITY = 5648
PHASE_3J_HISTORY_LEGAL = 0
PHASE_3J_RNG_ORDER = 304
PHASE_3J_UNRECONCILED = 0
PHASE_3J_SHARE_ELIGIBILITY = 0.9489247311827957
PHASE_3J_SHARE_RNG_ORDER = 0.051075268817204304
PHASE_3J_ELIG_DIFFERENT_OPPONENT = 4771
PHASE_3J_ELIG_KIND_MISMATCH = 877

TIMING_COMPONENTS = (
    "treatment_eliminated_earlier",
    "control_opponent_eliminated_earlier",
    "ghost_bye_third_party",
    "unreconciled",
)

HP_GAP_COMPONENTS = (
    "accumulated_prior_hp",
    "current_fight_hit",
    "current_fight_damage_magnitude",
    "hp_unreconciled",
)

REPRESENTED_SOURCE = TIMING_COMPONENTS

NEXT_OBSERVABLE_DEFAULT = (
    "the largest first-eligibility-divergence component of the 3J "
    "eligibility leftover, ranked before any behavior change"
)

GHOST_TOKEN = "ghost"
BYE_TOKEN = "bye"


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3K may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3K must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3K seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def share_of_eligibility(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the 3J eligibility leftover (5648 rows)."""
    if part is None or denom is None or abs(float(denom)) < 1e-12:
        return None
    return float(part) / float(denom)


def share_of_hp_gap(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the (1)+(2) decisive-HP-gap rows."""
    return share_of_eligibility(part, denom=denom)


def _rank_parts(parts: List[tuple]) -> List[Dict]:
    ranked = []
    for name, share in parts:
        ranked.append({
            "component": name,
            "share": None if share is None else float(share),
            "abs_share": 0.0 if share is None else abs(float(share)),
        })
    ranked.sort(key=lambda r: r["abs_share"], reverse=True)
    return ranked


def classify_first_eligibility(
    *,
    control_present: bool = False,
    treatment_present: bool = False,
    leftover_alive_control: bool = False,
    leftover_alive_treatment: bool = False,
    leftover_in_only_control: bool = False,
    leftover_in_only_treatment: bool = False,
    named_in_only_control: bool = False,
    named_in_only_treatment: bool = False,
    third_party_alive_diff: bool = False,
    ghost_bye_eligible_equal: bool = True,
    alive_sets_equal: bool = True,
) -> str:
    """Exclusive class for one 3J eligibility leftover punch row.

    Walk T7 through the leftover pairing turn; classify the *first*
    alive-set / ghost-bye divergence.

    Priority: missing pairing / leftover seat absent at the leftover
    turn → unreconciled; leftover or leftover's pairing opponent died
    earlier in treatment → treatment_eliminated_earlier; leftover's
    comparison / opponent died earlier in control →
    control_opponent_eliminated_earlier; only a third-party death (or
    same live named set with ghost/bye flipped) →
    ghost_bye_third_party; otherwise unreconciled.
    """
    if not (control_present and treatment_present):
        return "unreconciled"
    if leftover_in_only_control or leftover_in_only_treatment:
        # Leftover itself dead in one arm at first divergence. Leftover
        # rows require the seat alive at the leftover turn, so this is
        # only reachable if the first-divergence snapshot is missing
        # the leftover seat; still classify by which arm died first.
        if leftover_in_only_control:
            return "treatment_eliminated_earlier"
        return "control_opponent_eliminated_earlier"
    if not leftover_alive_control or not leftover_alive_treatment:
        return "unreconciled"
    if named_in_only_control:
        return "treatment_eliminated_earlier"
    if named_in_only_treatment:
        return "control_opponent_eliminated_earlier"
    if third_party_alive_diff or (not ghost_bye_eligible_equal):
        return "ghost_bye_third_party"
    if not alive_sets_equal:
        return "unreconciled"
    return "unreconciled"


def classify_hp_gap(
    *,
    timing_class: Optional[str] = None,
    control_fight_present: bool = False,
    treatment_fight_present: bool = False,
    pre_hp_equal: bool = True,
    control_hit: bool = False,
    treatment_hit: bool = False,
    applied_equal: bool = True,
) -> Optional[str]:
    """Exclusive HP-gap class for a (1)/(2) first-divergence row.

    Priority: pre-combat HP already differs → accumulated_prior_hp;
    same pre-HP, one arm hit and the other did not → current_fight_hit;
    both hit (or both missed) but applied damage differs →
    current_fight_damage_magnitude; missing fight → hp_unreconciled.
    """
    if timing_class not in (
        "treatment_eliminated_earlier",
        "control_opponent_eliminated_earlier",
    ):
        return None
    if not (control_fight_present and treatment_fight_present):
        return "hp_unreconciled"
    if not pre_hp_equal:
        return "accumulated_prior_hp"
    if bool(control_hit) != bool(treatment_hit):
        return "current_fight_hit"
    if not applied_equal:
        return "current_fight_damage_magnitude"
    return "hp_unreconciled"


def diagnose_phase_3k(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the 3J eligibility leftover to an elimination-timing / HP class."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3k_seed": PHASE_3K_SEED,
        "phase_3k_lobbies": PHASE_3K_LOBBIES,
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
        "phase_3d_board_pool_magnitude": PHASE_3D_BOARD_POOL_MAGNITUDE,
        "phase_3e_carry_delta": PHASE_3E_CARRY_DELTA,
        "phase_3e_carry_share_of_a1": PHASE_3E_CARRY_SHARE_OF_A1,
        "phase_3e_punch_delta_carry": PHASE_3E_PUNCH_DELTA_CARRY,
        "phase_3f_uncond_share": PHASE_3F_UNCOND_SHARE,
        "phase_3f_selection_share": PHASE_3F_SELECTION_SHARE,
        "phase_3g_mixture": PHASE_3G_MIXTURE,
        "phase_3g_mixture_share": PHASE_3G_MIXTURE_SHARE,
        "phase_3g_within_share": PHASE_3G_WITHIN_SHARE,
        "phase_3g_mix_role_share": PHASE_3G_MIX_ROLE_SHARE,
        "phase_3h_leftover": PHASE_3H_LEFTOVER,
        "phase_3h_late_control": PHASE_3H_LATE_CONTROL,
        "phase_3h_late_treatment": PHASE_3H_LATE_TREATMENT,
        "phase_3h_collapse": PHASE_3H_COLLAPSE,
        "phase_3h_share_leftover": PHASE_3H_SHARE_LEFTOVER,
        "phase_3i_pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
        "phase_3i_share_pairing_schedule": PHASE_3I_SHARE_PAIRING_SCHEDULE,
        "phase_3i_different_opponent": PHASE_3I_DIFFERENT_OPPONENT,
        "phase_3i_kind_mismatch": PHASE_3I_KIND_MISMATCH,
        "phase_3j_eligibility": PHASE_3J_ELIGIBILITY,
        "phase_3j_share_eligibility": PHASE_3J_SHARE_ELIGIBILITY,
        "phase_3j_rng_order": PHASE_3J_RNG_ORDER,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_tiers": list(LOW_TIERS),
        "early_turns": list(EARLY_TURNS),
        "late_turns": list(LATE_TURNS),
        "very_late_turns": list(VERY_LATE_TURNS),
        "pairing_turns": list(PAIRING_TURNS),
        "trace_from_turn": TRACE_FROM_TURN,
        "timing_components": list(TIMING_COMPONENTS),
        "hp_gap_components": list(HP_GAP_COMPONENTS),
        "unsupported_marked_not_approximated": True,
        "venomous_equals_poisonous_in_sim": True,
        "ordinary_hp_loss_identity": "min(pre_hit_hp, effective_incoming_attack)",
        "impact_attack_identity": IMPACT_ATTACK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "lineage_identity": LINEAGE_IDENTITY,
        "paired_seat_identity": PAIRED_SEAT_IDENTITY,
        "pairing_identity": PAIRING_IDENTITY,
        "leftover_reconcile_identity": LEFTOVER_RECONCILE_IDENTITY,
        "candidate_choice_identity": CANDIDATE_CHOICE_IDENTITY,
        "matchmaking_reconcile_identity": MATCHMAKING_RECONCILE_IDENTITY,
        "hp_flow_identity": HP_FLOW_IDENTITY,
        "elimination_identity": ELIMINATION_IDENTITY,
        "eligibility_timing_identity": ELIGIBILITY_TIMING_IDENTITY,
        "hp_gap_reconcile_identity": HP_GAP_RECONCILE_IDENTITY,
        "history_filters_applied": False,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    attr = comparison.get("attribution") or comparison
    shares = {name: attr.get(f"share_{name}") for name in TIMING_COMPONENTS}
    ranked = _rank_parts([(n, shares[n]) for n in TIMING_COMPONENTS])
    top = ranked[0]["component"] if ranked else "unreconciled"

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    treat = _f("treatment_eliminated_earlier")
    ctrl = _f("control_opponent_eliminated_earlier")
    third = _f("ghost_bye_third_party")
    unrec = _f("unreconciled")
    hp_shares = {name: attr.get(f"share_{name}") for name in HP_GAP_COMPONENTS}
    hp_ranked = _rank_parts([(n, hp_shares[n]) for n in HP_GAP_COMPONENTS])
    prior = hp_shares.get("accumulated_prior_hp")
    hit = hp_shares.get("current_fight_hit")
    mag = hp_shares.get("current_fight_damage_magnitude")
    prior_f = None if prior is None else float(prior)
    hit_f = None if hit is None else float(hit)
    mag_f = None if mag is None else float(mag)
    treat_f = 0.0 if treat is None else treat
    ctrl_f = 0.0 if ctrl is None else ctrl
    named_share = treat_f + ctrl_f
    represented = [s for s in (treat, ctrl, third) if s is not None]
    top_share = max((abs(s) for s in represented), default=0.0)

    def _hp_next(kind: str) -> str:
        if kind == "accumulated_prior_hp":
            return (
                "Accumulated prior-HP divergence clears ~70% of the "
                "(1)+(2) decisive HP gaps. Next hour: trace the earliest "
                "turn the HP paths separate. Do not apply a scaling "
                "correction; do not rewrite 2Q; do not change "
                "`_hero_damage`; do not retune scaling constants; do not "
                "burn confirm."
            )
        if kind == "current_fight_hit":
            return (
                "Current-fight hit / no-hit (outcome) clears ~70% of the "
                "(1)+(2) decisive HP gaps. Next hour: isolate combat-"
                "outcome fidelity for the decisive fight. Do not apply a "
                "scaling correction; do not rewrite 2Q; do not change "
                "`_hero_damage`; do not retune scaling constants; do not "
                "burn confirm."
            )
        if kind == "current_fight_damage_magnitude":
            return (
                "Current-fight damage magnitude clears ~70% of the (1)+(2) "
                "decisive HP gaps. Next hour: route back to the already-"
                "measured survivor-composition / damage mechanism with "
                "matched-state conditioning. Do not apply a scaling "
                "correction; do not rewrite 2Q; do not change "
                "`_hero_damage`; do not retune scaling constants; do not "
                "burn confirm."
            )
        return (
            "No represented HP-gap class clears ~70% of the (1)+(2) "
            "decisive HP gaps. Rank the residual before any behavior "
            "change. Do not apply a scaling correction; do not rewrite "
            "2Q; do not change `_hero_damage`; do not retune scaling "
            "constants; do not burn confirm."
        )

    if third is not None and third > SHARE_DOMINANT:
        primary = "third_party_elimination_dominates"
        next_step = (
            "Third-party elimination (ghost/bye transition) clears ~70% "
            "of the 3J eligibility leftover. Next hour: trace that "
            "elimination chain one hop upstream. Do not apply a scaling "
            "correction; do not rewrite 2Q; do not change `_hero_damage`; "
            "do not retune scaling constants; do not burn confirm."
        )
    elif named_share > SHARE_DOMINANT:
        if prior_f is not None and prior_f > SHARE_DOMINANT:
            primary = "accumulated_prior_hp_dominates"
            next_step = _hp_next("accumulated_prior_hp")
        elif hit_f is not None and hit_f > SHARE_DOMINANT:
            primary = "current_fight_hit_dominates"
            next_step = _hp_next("current_fight_hit")
        elif mag_f is not None and mag_f > SHARE_DOMINANT:
            primary = "current_fight_damage_dominates"
            next_step = _hp_next("current_fight_damage_magnitude")
        elif max(
            (abs(s) for s in (prior_f, hit_f, mag_f) if s is not None),
            default=0.0,
        ) >= 0.30:
            hp_top = hp_ranked[0]["component"] if hp_ranked else "hp_unreconciled"
            primary = "mixed_hp_route_to_larger"
            next_step = (
                "No single HP-gap class clears ~70% of the (1)+(2) "
                f"decisive HP gaps (top={hp_top}). Rank components and "
                "pursue the largest. Do not apply a scaling correction; "
                "do not rewrite 2Q; do not change `_hero_damage`; do not "
                "retune scaling constants; do not burn confirm."
            )
        else:
            primary = "ranked_hp_residual_needs_next_observable"
            next_step = _hp_next("residual")
    elif top_share >= 0.30:
        primary = "mixed_route_to_larger"
        next_step = (
            "No single first-eligibility-divergence class clears ~70% "
            f"of the 3J eligibility leftover (top={top}). Rank "
            "components and pursue the largest. Do not apply a scaling "
            "correction; do not rewrite 2Q; do not change `_hero_damage`; "
            "do not retune scaling constants; do not burn confirm."
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "No represented first-eligibility-divergence class clears "
            f"~70% of the 3J eligibility leftover (top={top}). Rank the "
            f"residual before any behavior change: {NEXT_OBSERVABLE_DEFAULT}. "
            "Do not apply a scaling correction; do not rewrite 2Q; do "
            "not change `_hero_damage`; do not retune scaling constants; "
            "do not burn confirm."
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "ranked_hp_gap": hp_ranked,
        "share_treatment_eliminated_earlier": treat,
        "share_control_opponent_eliminated_earlier": ctrl,
        "share_ghost_bye_third_party": third,
        "share_unreconciled": None if unrec is None else float(unrec),
        "share_accumulated_prior_hp": prior_f,
        "share_current_fight_hit": hit_f,
        "share_current_fight_damage_magnitude": mag_f,
        "share_named_eliminations": named_share,
        "attribution": attr,
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
        "keep_pr_51_hold": True,
        "keep_pr_52_hold": True,
        "keep_pr_53_hold": True,
        "keep_pr_54_hold": True,
        "keep_pr_55_hold": True,
        "keep_pr_56_hold": True,
    })
    return out
