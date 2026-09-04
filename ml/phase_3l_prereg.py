"""Phase 3L — third-party elimination-chain attribution (measurement only).

Stacked on Phase 3K (PR #57). Reuses consumed DEV 14200–14699. No new seeds.
Does not change simulator behavior, α, scaling math, `_hero_damage`, gates,
defaults, or the 2Q recruit-value objective. Confirm 11500–11699 remains
reserved. Keep HOLD stack including #57.

Reproduces the 3K ghost/bye third-party leftover (3701 eligibility punch
rows) and reconciles each row to one causal third-party elimination,
split exclusively into:

* (1) same third-party seat but earlier elimination in one arm
* (2) different third-party seat eliminated because the alive set already diverged
* (3) same fight pairing but outcome flips
* (4) same outcome but lethal threshold / damage magnitude differs
* (5) missing / unreconciled

For (1), further attributes the decisive HP gap into accumulated prior
HP vs current-fight hit/no-hit vs current-fight damage magnitude.
"""

from __future__ import annotations

from typing import Dict, List, Optional

FEATURE_TOGGLE = "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
FEATURE_TOGGLE_DEFAULT = False

METHODOLOGY_VERSION = "3l_v1"
PHASE_3L_SEED = 14200
PHASE_3L_LOBBIES = 500
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
    50, 51, 52, 53, 54, 55, 56, 57,
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

CHAIN_RECONCILE_IDENTITY = (
    "same_seat_earlier_elimination + different_seat_alive_set_cascade "
    "+ same_fight_outcome_flip + same_outcome_damage_threshold "
    "+ unreconciled = 3K ghost_bye_third_party (3701)"
)

CHAIN_HP_RECONCILE_IDENTITY = (
    "accumulated_prior_hp + current_fight_hit + current_fight_damage_magnitude "
    "+ hp_unreconciled = same_seat_earlier_elimination"
)

ROW_ELIM_HP_IDENTITY = (
    "every 3K third-party row maps to one causal elimination event; "
    "that event's fight has post_hp<=0; post_hp = pre_hp - applied_to_seat"
)

# Published 3D / 3E / 3F / 3G / 3H / 3I / 3J / 3K locks (exact).
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
PHASE_3K_TREATMENT_EARLIER = 1108
PHASE_3K_CONTROL_OPPONENT = 839
PHASE_3K_THIRD_PARTY = 3701
PHASE_3K_UNRECONCILED = 0
PHASE_3K_NAMED = 1947
PHASE_3K_SHARE_THIRD_PARTY = 0.6552762039660056
PHASE_3K_SHARE_TREATMENT_EARLIER = 0.1961756373937677
PHASE_3K_SHARE_CONTROL_OPPONENT = 0.14854815864022664
PHASE_3K_PRIOR_HP = 1818
PHASE_3K_HIT = 57
PHASE_3K_MAG = 72
PHASE_3K_SHARE_PRIOR_HP = 0.9337442218798151
PHASE_3K_SHARE_HIT = 0.029275808936825885
PHASE_3K_SHARE_MAG = 0.03697996918335902

CHAIN_COMPONENTS = (
    "same_seat_earlier_elimination",
    "different_seat_alive_set_cascade",
    "same_fight_outcome_flip",
    "same_outcome_damage_threshold",
    "unreconciled",
)

HP_GAP_COMPONENTS = (
    "accumulated_prior_hp",
    "current_fight_hit",
    "current_fight_damage_magnitude",
    "hp_unreconciled",
)

REPRESENTED_SOURCE = CHAIN_COMPONENTS

NEXT_OBSERVABLE_DEFAULT = (
    "the largest causal-elimination class of the 3K third-party leftover, "
    "ranked before any behavior change"
)

GHOST_TOKEN = "ghost"
BYE_TOKEN = "bye"


def assert_seed_range_allowed(seed: int, lobbies: int) -> None:
    """3L may only reuse 14200–14699. Confirm and all other consumed bands fail."""
    lo, hi = seed, seed + lobbies - 1
    if lo < REUSED_SEED_LO or hi > REUSED_SEED_HI:
        raise ValueError(
            f"Phase 3L must reuse {REUSED_SEED_LO}–{REUSED_SEED_HI}, "
            f"got {lo}–{hi}"
        )
    for flo, fhi in FORBIDDEN_RANGES:
        if lo <= fhi and hi >= flo:
            raise ValueError(
                f"Phase 3L seed range {lo}–{hi} overlaps forbidden {flo}–{fhi}"
            )


def share_of_third_party(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the 3K third-party leftover (3701 rows)."""
    if part is None or denom is None or abs(float(denom)) < 1e-12:
        return None
    return float(part) / float(denom)


def share_of_hp_gap(
    part: Optional[float],
    *,
    denom: Optional[float],
) -> Optional[float]:
    """Signed share of the class-(1) decisive-HP-gap rows."""
    return share_of_third_party(part, denom=denom)


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


def classify_third_party_chain(
    *,
    control_present: bool = False,
    treatment_present: bool = False,
    causal_seat_control: Optional[int] = None,
    causal_seat_treatment: Optional[int] = None,
    same_fight_pairing: bool = False,
    outcomes_equal: bool = True,
    one_died_other_alive_at_elim_turn: bool = False,
    applied_equal: bool = True,
    elim_turns_equal: bool = True,
) -> str:
    """Exclusive class for one 3K third-party leftover punch row.

    Identify the third-party seat whose elimination first changes
    ghost/bye / alive-set eligibility, then classify that event.

    Priority: missing pairing / no causal seat → unreconciled;
    different causal seats in the two arms →
    different_seat_alive_set_cascade; same seat and same fight pairing
    with outcome flip → same_fight_outcome_flip; same pairing and
    same outcome but lethal threshold / applied damage differs →
    same_outcome_damage_threshold; same seat but earlier elimination
    (different fight or only one arm dead) →
    same_seat_earlier_elimination; otherwise unreconciled.
    """
    if not (control_present and treatment_present):
        return "unreconciled"
    c_s = causal_seat_control
    t_s = causal_seat_treatment
    if c_s is None and t_s is None:
        return "unreconciled"
    if (
        c_s is not None and t_s is not None
        and int(c_s) != int(t_s)
    ):
        return "different_seat_alive_set_cascade"
    if same_fight_pairing:
        if not outcomes_equal:
            return "same_fight_outcome_flip"
        if one_died_other_alive_at_elim_turn or not applied_equal:
            return "same_outcome_damage_threshold"
        return "unreconciled"
    if (
        not elim_turns_equal
        or one_died_other_alive_at_elim_turn
        or ((c_s is None) != (t_s is None))
    ):
        return "same_seat_earlier_elimination"
    return "unreconciled"


def classify_chain_hp_gap(
    *,
    chain_class: Optional[str] = None,
    control_fight_present: bool = False,
    treatment_fight_present: bool = False,
    pre_hp_equal: bool = True,
    control_hit: bool = False,
    treatment_hit: bool = False,
    applied_equal: bool = True,
) -> Optional[str]:
    """Exclusive HP-gap class for a class-(1) same-seat earlier row.

    Priority: pre-combat HP already differs → accumulated_prior_hp;
    same pre-HP, one arm hit and the other did not → current_fight_hit;
    both hit (or both missed) but applied damage differs →
    current_fight_damage_magnitude; missing fight → hp_unreconciled.
    """
    if chain_class != "same_seat_earlier_elimination":
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


def diagnose_phase_3l(
    comparison: Optional[Dict] = None,
    *,
    non_evaluative: bool = False,
) -> Dict:
    """Route the 3K third-party leftover to a causal-elimination class."""
    out = {
        "methodology_version": METHODOLOGY_VERSION,
        "primary_finding": "implemented_not_evaluated",
        "feature_toggle": FEATURE_TOGGLE,
        "feature_toggle_default_off": True,
        "phase_3l_seed": PHASE_3L_SEED,
        "phase_3l_lobbies": PHASE_3L_LOBBIES,
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
        "phase_3k_third_party": PHASE_3K_THIRD_PARTY,
        "phase_3k_share_third_party": PHASE_3K_SHARE_THIRD_PARTY,
        "phase_3k_named": PHASE_3K_NAMED,
        "phase_3k_share_prior_hp": PHASE_3K_SHARE_PRIOR_HP,
        "instrument_turns": list(INSTRUMENT_TURNS),
        "low_tiers": list(LOW_TIERS),
        "early_turns": list(EARLY_TURNS),
        "late_turns": list(LATE_TURNS),
        "very_late_turns": list(VERY_LATE_TURNS),
        "pairing_turns": list(PAIRING_TURNS),
        "trace_from_turn": TRACE_FROM_TURN,
        "chain_components": list(CHAIN_COMPONENTS),
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
        "chain_reconcile_identity": CHAIN_RECONCILE_IDENTITY,
        "chain_hp_reconcile_identity": CHAIN_HP_RECONCILE_IDENTITY,
        "row_elim_hp_identity": ROW_ELIM_HP_IDENTITY,
        "history_filters_applied": False,
    }
    if comparison is None:
        return out
    if non_evaluative or comparison.get("non_evaluative"):
        out["primary_finding"] = "measurement_smoke_non_evaluative"
        out["note"] = "tiny smoke only; do not route the 500-lobby DEV"
        return out

    attr = comparison.get("attribution") or comparison
    shares = {name: attr.get(f"share_{name}") for name in CHAIN_COMPONENTS}
    ranked = _rank_parts([(n, shares[n]) for n in CHAIN_COMPONENTS])
    top = ranked[0]["component"] if ranked else "unreconciled"

    def _f(name: str) -> Optional[float]:
        v = shares.get(name)
        return None if v is None else float(v)

    earlier = _f("same_seat_earlier_elimination")
    cascade = _f("different_seat_alive_set_cascade")
    flip = _f("same_fight_outcome_flip")
    thresh = _f("same_outcome_damage_threshold")
    unrec = _f("unreconciled")
    hp_shares = {name: attr.get(f"share_{name}") for name in HP_GAP_COMPONENTS}
    hp_ranked = _rank_parts([(n, hp_shares[n]) for n in HP_GAP_COMPONENTS])
    prior = hp_shares.get("accumulated_prior_hp")
    hit = hp_shares.get("current_fight_hit")
    mag = hp_shares.get("current_fight_damage_magnitude")
    prior_f = None if prior is None else float(prior)
    hit_f = None if hit is None else float(hit)
    mag_f = None if mag is None else float(mag)
    earlier_f = 0.0 if earlier is None else earlier
    represented = [s for s in (earlier, cascade, flip, thresh) if s is not None]
    top_share = max((abs(s) for s in represented), default=0.0)

    def _no_change_tail() -> str:
        return (
            "Do not apply a scaling correction; do not rewrite 2Q; do not "
            "change `_hero_damage`; do not retune scaling constants; do "
            "not burn confirm."
        )

    def _hp_next(kind: str) -> str:
        if kind == "accumulated_prior_hp":
            return (
                "Same-seat earlier elimination clears ~70% of the 3K "
                "third-party leftover and accumulated prior HP dominates "
                "the class-(1) HP gap. Next hour: trace the earliest "
                "turn that third-party seat's HP paths separate. "
                + _no_change_tail()
            )
        if kind == "current_fight_hit":
            return (
                "Same-seat earlier elimination clears ~70% of the 3K "
                "third-party leftover and current-fight hit / no-hit "
                "dominates the class-(1) HP gap. Next hour: isolate "
                "combat-outcome fidelity for that fight. "
                + _no_change_tail()
            )
        if kind == "current_fight_damage_magnitude":
            return (
                "Same-seat earlier elimination clears ~70% of the 3K "
                "third-party leftover and current-fight damage magnitude "
                "dominates the class-(1) HP gap. Next hour: matched-state "
                "damage attribution. "
                + _no_change_tail()
            )
        return (
            "Same-seat earlier elimination clears ~70% of the 3K "
            "third-party leftover but no HP-gap class clears ~70%. "
            "Rank the residual before any behavior change. "
            + _no_change_tail()
        )

    if earlier is not None and earlier > SHARE_DOMINANT:
        if prior_f is not None and prior_f > SHARE_DOMINANT:
            primary = "same_seat_earlier_prior_hp_dominates"
            next_step = _hp_next("accumulated_prior_hp")
        elif hit_f is not None and hit_f > SHARE_DOMINANT:
            primary = "same_seat_earlier_hit_dominates"
            next_step = _hp_next("current_fight_hit")
        elif mag_f is not None and mag_f > SHARE_DOMINANT:
            primary = "same_seat_earlier_damage_dominates"
            next_step = _hp_next("current_fight_damage_magnitude")
        elif max(
            (abs(s) for s in (prior_f, hit_f, mag_f) if s is not None),
            default=0.0,
        ) >= 0.30:
            hp_top = hp_ranked[0]["component"] if hp_ranked else "hp_unreconciled"
            primary = "mixed_hp_route_to_larger"
            next_step = (
                "Same-seat earlier elimination clears ~70% of the 3K "
                "third-party leftover but no single HP-gap class clears "
                f"~70% (top={hp_top}). Rank components and pursue the "
                "largest. " + _no_change_tail()
            )
        else:
            primary = "ranked_hp_residual_needs_next_observable"
            next_step = _hp_next("residual")
    elif cascade is not None and cascade > SHARE_DOMINANT:
        primary = "different_seat_cascade_dominates"
        next_step = (
            "Different-seat / alive-set cascade clears ~70% of the 3K "
            "third-party leftover. Next hour: trace one earlier "
            "eligibility divergence recursively. " + _no_change_tail()
        )
    elif flip is not None and flip > SHARE_DOMINANT:
        primary = "same_fight_outcome_dominates"
        next_step = (
            "Same-fight outcome flip clears ~70% of the 3K third-party "
            "leftover. Next hour: isolate combat-outcome fidelity. "
            + _no_change_tail()
        )
    elif thresh is not None and thresh > SHARE_DOMINANT:
        primary = "damage_threshold_dominates"
        next_step = (
            "Same-outcome lethal threshold / damage magnitude clears "
            "~70% of the 3K third-party leftover. Next hour: "
            "matched-state damage attribution. " + _no_change_tail()
        )
    elif top_share >= 0.30:
        primary = "mixed_route_to_larger"
        next_step = (
            "No single causal-elimination class clears ~70% of the 3K "
            f"third-party leftover (top={top}). Rank components and "
            "pursue the largest. " + _no_change_tail()
        )
    else:
        primary = "ranked_residual_needs_next_observable"
        next_step = (
            "No represented causal-elimination class clears ~70% of the "
            f"3K third-party leftover (top={top}). Rank the residual "
            f"before any behavior change: {NEXT_OBSERVABLE_DEFAULT}. "
            + _no_change_tail()
        )

    out.update({
        "primary_finding": primary,
        "evaluative": True,
        "recommended_next_step": next_step,
        "ranked_residual": ranked,
        "ranked_hp_gap": hp_ranked,
        "share_same_seat_earlier_elimination": earlier,
        "share_different_seat_alive_set_cascade": cascade,
        "share_same_fight_outcome_flip": flip,
        "share_same_outcome_damage_threshold": thresh,
        "share_unreconciled": None if unrec is None else float(unrec),
        "share_accumulated_prior_hp": prior_f,
        "share_current_fight_hit": hit_f,
        "share_current_fight_damage_magnitude": mag_f,
        "share_same_seat_earlier": earlier_f,
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
        "keep_pr_57_hold": True,
    })
    return out
