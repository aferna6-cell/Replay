"""Phase 3N — observational first-split matched-state damage attribution.

Reuses the 3M first-divergence walk on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

Restricts to 3M class-(3) same-outcome-damage rows. At each first split
records both combat-start boards, winner tavern tier, actual survivors,
applied `_hero_damage`, the rules-faithful counterfactual, board raw
fields, and combat margin. Decomposes treatment−control applied damage
and standardizes on matched pre-fight board mix.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.carry_divergence_diagnostic import reconcile_history_links
from ml.elimination_chain_diagnostic import compare_chain
from ml.elimination_timing_diagnostic import (
    EliminationTimingTracer,
    _applied_to_seat,
    _hp_flow_ok,
    _seat_hp,
)
from ml.hp_divergence_diagnostic import (
    attribute_first_divergence,
    compare_first_divergence,
    find_first_hp_divergence,
    iter_class1_chain_records,
)
from ml.matchmaking_divergence_diagnostic import _index_decisions
from ml.pairing_who_wins_diagnostic import (
    _index_seat_fights,
    _seat_side_fields,
)
from ml.phase_3n_prereg import (
    APPLIED_RECONCILE_IDENTITY,
    COUNTERFACTUAL_IDENTITY,
    DAMAGE_COMPONENTS,
    FIELD_VS_SURVIVAL_IDENTITY,
    FIRST_DIVERGENCE_RECONCILE_IDENTITY,
    FLOW_ABS_TOL,
    HP_WALK_FROM_TURN,
    LATE_TURNS,
    PHASE_3G_MIXTURE,
    PHASE_3G_MIXTURE_SHARE,
    PHASE_3G_N_CONTROL,
    PHASE_3G_N_TREATMENT,
    PHASE_3G_WITHIN_SHARE,
    PHASE_3H_COLLAPSE,
    PHASE_3H_LATE_CONTROL,
    PHASE_3H_LATE_TREATMENT,
    PHASE_3H_LEFTOVER,
    PHASE_3H_SHARE_LEFTOVER,
    PHASE_3I_PAIRING_SCHEDULE,
    PHASE_3I_SHARE_PAIRING_SCHEDULE,
    PHASE_3J_ELIGIBILITY,
    PHASE_3J_SHARE_ELIGIBILITY,
    PHASE_3K_SHARE_THIRD_PARTY,
    PHASE_3K_THIRD_PARTY,
    PHASE_3L_SAME_SEAT_EARLIER,
    PHASE_3L_SHARE_EARLIER,
    PHASE_3M_CLASS1,
    PHASE_3M_FIRST_SPLIT_T5,
    PHASE_3M_SAME_OUTCOME_DAMAGE,
    PHASE_3M_SHARE_DAMAGE,
    PHASE_3M_VERY_LATE_CLASS1,
    PHASE_3M_VERY_LATE_DAMAGE,
    PROXY_ERROR_IDENTITY,
    REWEIGHT_ABS_TOL,
    ROW_DAMAGE_RECONCILE_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    SOURCE_COMPONENTS,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    classify_row_reconcile,
    share_of_applied,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.survivor_composition_diagnostic import (
    TIERS,
    classify_env_minion,
    decompose_gap,
    tier_histogram,
    tier_sum,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage

METHODOLOGY_VERSION = "3n_v1"

_N_EXAMPLES = 8
_ROW_ABS_TOL = 1e-9
_CLASS3 = "same_outcome_damage"

_SLIM_KEYS = (
    "name", "card_id", "tier", "golden", "token", "generated", "origin",
    "body_id", "board_slot", "recruit_raw", "combat_raw", "attack",
    "health",
)


def _safe_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def slim_body(body) -> Dict:
    """Observational identity/tier/raw for one combat-start or survivor body."""
    if body is None:
        return {}
    if isinstance(body, dict):
        out = {k: body.get(k) for k in _SLIM_KEYS if body.get(k) is not None}
        if "tier" not in out:
            out["tier"] = int(body.get("tier") or 1)
        return out
    try:
        return classify_env_minion(body, 0)
    except Exception:
        return {
            "name": str(getattr(body, "name", "") or ""),
            "card_id": str(getattr(body, "card_id", "") or ""),
            "tier": int(getattr(body, "tier", 1) or 1),
            "recruit_raw": None,
            "combat_raw": None,
        }


def slim_board(bodies: Optional[Sequence]) -> List[Dict]:
    return [slim_body(b) for b in list(bodies or [])]


def _tier_list(bodies: Sequence[Dict]) -> List[int]:
    return [int(b.get("tier") or 1) for b in bodies]


def _mean_or_unit(total: float, n: float) -> float:
    if n > 0:
        return float(total) / float(n)
    return 1.0


def _winner_seat(fight: Optional[Dict]):
    if not fight:
        return None
    return _safe_int(fight.get("winner_seat"))


def _loser_seat(fight: Optional[Dict]):
    if not fight:
        return None
    return _safe_int(fight.get("loser_seat"))


def _board_for_seat(fight: Optional[Dict], seat) -> List[Dict]:
    if not fight or seat in (None, ""):
        return []
    seat_i = _safe_int(seat)
    sa = _safe_int(fight.get("seat_a"))
    sb = _safe_int(fight.get("seat_b"))
    if seat_i is not None and seat_i == sa and fight.get("starting_a"):
        return slim_board(fight.get("starting_a"))
    if seat_i is not None and seat_i == sb and fight.get("starting_b"):
        return slim_board(fight.get("starting_b"))
    winner = _winner_seat(fight)
    loser = _loser_seat(fight)
    if seat_i is not None and winner is not None and seat_i == winner:
        return slim_board(
            fight.get("start_board")
            or fight.get("start_combat_bodies")
            or fight.get("start_minions")
            or fight.get("starting_winner")
        )
    if seat_i is not None and loser is not None and seat_i == loser:
        return slim_board(
            fight.get("starting_loser") or fight.get("loser_board")
        )
    return []


def extract_fight_state(fight: Optional[Dict], seat) -> Dict:
    """Winner-relative damage fields plus both combat-start boards."""
    empty = {
        "winner_tavern_tier": None,
        "survivors": [],
        "survivor_count": 0,
        "survivor_tier_sum": 0,
        "applied": None,
        "counterfactual": 0,
        "proxy_error": 0,
        "board_a": [],
        "board_b": [],
        "winner_board": [],
        "loser_board": [],
        "board_recruit_raw": None,
        "abstract_pool_raw": None,
        "total_combat_raw": None,
        "winner_board_recruit_raw": None,
        "winner_abstract_pool_raw": None,
        "winner_total_combat_raw": None,
        "winner_start_tier_sum": 0,
        "winner_start_n": 0,
        "winner_start_tier_mean": 1.0,
        "winner_start_tier_hist": {str(t): 0 for t in TIERS},
        "combat_margin_raw": None,
        "hp_flow_ok": True,
        "cf_ok": True,
        "kind": None,
        "outcome": None,
        "pre_hp": None,
        "post_hp": None,
    }
    if not fight:
        return empty
    winner = _winner_seat(fight)
    loser = _loser_seat(fight)
    winner_tavern = _safe_int(fight.get("winner_tavern_tier"), default=1) or 1
    survivors = slim_board(
        fight.get("actual_survivors") or fight.get("survivors") or []
    )
    n = _safe_int(
        fight.get("actual_survivor_count",
                  fight.get("survivor_count_actual", len(survivors))),
        default=len(survivors),
    ) or 0
    recorded_sum = _safe_int(
        fight.get("actual_survivor_tier_sum", fight.get("survivor_tier_sum")),
        default=None,
    )
    tier_sum_actual = int(sum(_tier_list(survivors))) if survivors else (
        int(recorded_sum or 0)
    )
    applied = _applied_to_seat(fight, seat)
    if applied is None:
        applied = _safe_int(fight.get("applied_hp_loss"), default=0) or 0
    cf = rules_faithful_hero_damage(winner_tavern, _tier_list(survivors))
    if not survivors and recorded_sum is not None:
        cf = int(winner_tavern) + int(recorded_sum)
        tier_sum_actual = int(recorded_sum)
    proxy_err = int(applied) - int(cf)
    winner_board = _board_for_seat(fight, winner) if winner is not None else slim_board(
        fight.get("start_board")
        or fight.get("start_combat_bodies")
        or fight.get("start_minions")
    )
    loser_board = _board_for_seat(fight, loser) if loser is not None else slim_board(
        fight.get("starting_loser")
    )
    board_a = slim_board(fight.get("starting_a")) or (
        winner_board if _safe_int(fight.get("seat_a")) == winner else loser_board
    )
    board_b = slim_board(fight.get("starting_b")) or (
        winner_board if _safe_int(fight.get("seat_b")) == winner else loser_board
    )
    causal_side = _seat_side_fields(fight, seat)
    winner_side = _seat_side_fields(fight, winner) if winner is not None else {}
    start_sum = tier_sum(winner_board) if winner_board else int(
        fight.get("start_tier_sum") or 0
    )
    start_n = len(winner_board) if winner_board else int(fight.get("n_start") or 0)
    start_mean = _mean_or_unit(start_sum, start_n)
    pre, post = _seat_hp(fight, seat)
    hp_ok = True if fight is None else _hp_flow_ok(fight)
    cf_ok = True
    if survivors:
        cf_ok = tier_sum_actual == int(sum(_tier_list(survivors)))
    return {
        "winner_tavern_tier": int(winner_tavern),
        "survivors": survivors,
        "survivor_identities": [
            {
                "name": s.get("name"),
                "card_id": s.get("card_id"),
                "tier": int(s.get("tier") or 1),
            }
            for s in survivors
        ],
        "survivor_count": int(n),
        "survivor_tier_sum": int(tier_sum_actual),
        "applied": int(applied),
        "counterfactual": int(cf),
        "proxy_error": int(proxy_err),
        "board_a": board_a,
        "board_b": board_b,
        "winner_board": winner_board,
        "loser_board": loser_board,
        "board_recruit_raw": causal_side.get("recruit_raw"),
        "abstract_pool_raw": causal_side.get("abstract_pool_raw"),
        "total_combat_raw": causal_side.get("combat_raw"),
        "winner_board_recruit_raw": winner_side.get("recruit_raw"),
        "winner_abstract_pool_raw": winner_side.get("abstract_pool_raw"),
        "winner_total_combat_raw": winner_side.get("combat_raw"),
        "winner_start_tier_sum": int(start_sum),
        "winner_start_n": int(start_n),
        "winner_start_tier_mean": float(start_mean),
        "winner_start_tier_hist": (
            fight.get("start_tier_hist") or tier_histogram(winner_board)
        ),
        "combat_margin_raw": fight.get(
            "combat_margin_raw", fight.get("raw")
        ),
        "hp_flow_ok": bool(hp_ok),
        "cf_ok": bool(cf_ok),
        "kind": fight.get("kind"),
        "outcome": None,
        "pre_hp": pre,
        "post_hp": post,
        "winner_seat": winner,
        "loser_seat": loser,
    }


def decompose_applied_row(control: Dict, treatment: Dict) -> Dict:
    """Five-way + fielded/survival split of one paired applied-damage gap.

    Sequential count-first Kitagawa on actual survivor tier sum:
    count = (n_T − n_C) × mean_C (mean_C = 1 when n_C = 0);
    composition | count = Δsum − count.
    Fielded mix uses winner start-board mean × actual survivor count.
    """
    tavern_c = int(control.get("winner_tavern_tier") or 0)
    tavern_t = int(treatment.get("winner_tavern_tier") or 0)
    n_c = float(control.get("survivor_count") or 0)
    n_t = float(treatment.get("survivor_count") or 0)
    sum_c = float(control.get("survivor_tier_sum") or 0)
    sum_t = float(treatment.get("survivor_tier_sum") or 0)
    applied_c = float(control.get("applied") or 0)
    applied_t = float(treatment.get("applied") or 0)
    cf_c = float(control.get("counterfactual") or 0)
    cf_t = float(treatment.get("counterfactual") or 0)
    start_mean_c = float(control.get("winner_start_tier_mean") or 1.0)
    start_mean_t = float(treatment.get("winner_start_tier_mean") or 1.0)

    delta_applied = applied_t - applied_c
    delta_tavern = float(tavern_t - tavern_c)
    delta_sum = sum_t - sum_c
    mean_c = _mean_or_unit(sum_c, n_c)
    count_term = (n_t - n_c) * mean_c
    composition_term = delta_sum - count_term
    proxy_term = (applied_t - cf_t) - (applied_c - cf_c)
    residual = delta_applied - (
        delta_tavern + count_term + composition_term + proxy_term
    )
    fielded_term = (n_t * start_mean_t) - (n_c * start_mean_c)
    survival_term = delta_sum - fielded_term
    pre_fight = delta_tavern + fielded_term
    five_way_ok = abs(residual) <= max(1e-6, 1e-9 * (1.0 + abs(delta_applied)))
    return {
        "delta_applied": float(delta_applied),
        "winner_tavern_tier": float(delta_tavern),
        "survivor_count": float(count_term),
        "survivor_composition": float(composition_term),
        "proxy_formula_error": float(proxy_term),
        "residual": float(residual),
        "pre_fight_board": float(pre_fight),
        "within_fight_survival": float(survival_term),
        "fielded_expected_sum_delta": float(fielded_term),
        "five_way_ok": bool(five_way_ok),
        "mean_survivor_tier_control": float(mean_c),
        "mean_survivor_tier_treatment": _mean_or_unit(sum_t, n_t),
    }


class MatchedStateDamageTracer(EliminationTimingTracer):
    """3M HP/pairing stamps plus both combat-start boards."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        rec["starting_a"] = slim_board(fight.get("starting_a"))
        rec["starting_b"] = slim_board(fight.get("starting_b"))
        rec["starting_winner"] = slim_board(
            fight.get("starting_winner") or rec.get("start_combat_bodies")
        )
        rec["starting_loser"] = slim_board(fight.get("starting_loser"))
        if fight.get("loser_board") and not rec.get("starting_loser"):
            rec["starting_loser"] = [
                classify_env_minion(m, i)
                for i, m in enumerate(list(fight.get("loser_board") or []))
            ]
        if fight.get("winner_board") and not rec.get("start_board"):
            rec["start_board"] = [
                classify_env_minion(m, i)
                for i, m in enumerate(list(fight.get("winner_board") or []))
            ]


def run_matched_state_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    from ml.phase_3n_prereg import assert_seed_range_allowed
    assert_seed_range_allowed(seed, lobbies)
    fights: List[Dict] = []
    lengths: List[float] = []
    turn_rows: List[Dict] = []
    replacement_events: List[Dict] = []
    board_snapshots: List[Dict] = []
    t1t3_events: List[Dict] = []
    last_t1t3_losses: List[Dict] = []
    pairing_decisions: List[Dict] = []
    hp_rows: List[Dict] = []
    eliminations: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = MatchedStateDamageTracer(i, seed + i, arm)
                env = BGEnv(seed=seed + i)
                tracer.attach_to_env(env)
                recs = env.play_scripted(
                    [greedy_policy] * env.n_players, recruit_tracer=tracer
                )
                game_length = max((r["turn"] for r in recs), default=0)
                if tracer.game_length is None:
                    tracer.game_length = game_length
                lengths.append(float(game_length))
                fights.extend(tracer.fights)
                turn_rows.extend(tracer.turn_rows)
                replacement_events.extend(tracer.replacement_events)
                board_snapshots.extend(tracer.board_snapshots)
                t1t3_events.extend(tracer.t1t3_events)
                last_t1t3_losses.extend(tracer.last_t1t3_losses)
                pairing_decisions.extend(tracer.pairing_decisions)
                hp_rows.extend(tracer.hp_rows)
                eliminations.extend(tracer.eliminations)
                del env
                del tracer

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "board_level_abstract_scaling": bool(board_level_abstract_scaling),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "fights": fights,
        "game_lengths": lengths,
        "turn_rows": turn_rows,
        "replacement_events": replacement_events,
        "board_snapshots": board_snapshots,
        "t1t3_events": t1t3_events,
        "last_t1t3_losses": last_t1t3_losses,
        "pairing_decisions": pairing_decisions,
        "hp_rows": hp_rows,
        "eliminations": eliminations,
    }


def run_greedy_control_matched(lobbies: int, seed: int) -> Dict:
    return run_matched_state_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_matched(lobbies: int, seed: int) -> Dict:
    return run_matched_state_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _arm_summary(states: Sequence[Dict]) -> Dict:
    n = len(states)
    if n == 0:
        return {
            "n": 0,
            "mean_applied": None,
            "mean_counterfactual": None,
            "mean_proxy_error": None,
            "mean_winner_tavern_tier": None,
            "mean_survivor_count": None,
            "mean_survivor_tier_sum": None,
            "mean_winner_start_tier_sum": None,
            "mean_combat_margin_raw": None,
            "start_combat_tier_histogram": {str(t): 0.0 for t in TIERS},
            "survival_prob_by_tier": {str(t): None for t in TIERS},
            "mean_surv_generated_tier_sum": 0.0,
            "mean_survivor_tier_sum": None,
        }
    applied = [float(s.get("applied") or 0) for s in states]
    cf = [float(s.get("counterfactual") or 0) for s in states]
    err = [float(s.get("proxy_error") or 0) for s in states]
    tavern = [float(s.get("winner_tavern_tier") or 0) for s in states]
    count = [float(s.get("survivor_count") or 0) for s in states]
    tsum = [float(s.get("survivor_tier_sum") or 0) for s in states]
    start = [float(s.get("winner_start_tier_sum") or 0) for s in states]
    margin = [
        float(s["combat_margin_raw"])
        for s in states if s.get("combat_margin_raw") is not None
    ]
    start_hist = {str(t): 0.0 for t in TIERS}
    surv_hist = {str(t): 0.0 for t in TIERS}
    for s in states:
        hist = s.get("winner_start_tier_hist") or {}
        for t in TIERS:
            start_hist[str(t)] += float(hist.get(str(t)) or 0)
        for body in s.get("survivors") or []:
            surv_hist[str(int(body.get("tier") or 1))] += 1.0
    p_surv = {}
    for t in TIERS:
        k = str(t)
        den = start_hist[k]
        p_surv[k] = None if den < 1e-12 else surv_hist[k] / den
        start_hist[k] = start_hist[k] / float(n)
    return {
        "n": n,
        "mean_applied": sum(applied) / n,
        "mean_counterfactual": sum(cf) / n,
        "mean_proxy_error": sum(err) / n,
        "mean_winner_tavern_tier": sum(tavern) / n,
        "mean_survivor_count": sum(count) / n,
        "mean_survivor_tier_sum": sum(tsum) / n,
        "mean_winner_start_tier_sum": sum(start) / n,
        "mean_combat_margin_raw": (
            sum(margin) / len(margin) if margin else None
        ),
        "start_combat_tier_histogram": start_hist,
        "survival_prob_by_tier": p_surv,
        "mean_surv_generated_tier_sum": 0.0,
    }


def _start_hist_key(state: Dict) -> Tuple:
    hist = state.get("winner_start_tier_hist") or {}
    return tuple(int(hist.get(str(t)) or 0) for t in TIERS)


def enrich_first_split(
    event: Dict,
    c_fight: Optional[Dict],
    t_fight: Optional[Dict],
) -> Dict:
    """Attach boards / CF / five-way terms to one class-(3) first split."""
    seat = event.get("causal_seat")
    control = extract_fight_state(c_fight, seat)
    treatment = extract_fight_state(t_fight, seat)
    decomp = decompose_applied_row(control, treatment)
    both = c_fight is not None and t_fight is not None
    hp_ok = bool(control.get("hp_flow_ok") and treatment.get("hp_flow_ok"))
    cf_ok = bool(control.get("cf_ok") and treatment.get("cf_ok"))
    row_class = classify_row_reconcile(
        class3=event.get("class") == _CLASS3,
        both_fights=both,
        hp_flow_ok=hp_ok,
        cf_ok=cf_ok,
        five_way_ok=bool(decomp.get("five_way_ok")),
    )
    matched_board = (
        both
        and _start_hist_key(control) == _start_hist_key(treatment)
        and int(control.get("winner_tavern_tier") or 0)
        == int(treatment.get("winner_tavern_tier") or 0)
    )
    out = dict(event)
    out.update({
        "control_state": control,
        "treatment_state": treatment,
        "decomposition": decomp,
        "row_class": row_class,
        "matched_pre_fight_board": matched_board,
        "control_winner_tavern_tier": control.get("winner_tavern_tier"),
        "treatment_winner_tavern_tier": treatment.get("winner_tavern_tier"),
        "control_counterfactual": control.get("counterfactual"),
        "treatment_counterfactual": treatment.get("counterfactual"),
        "control_proxy_error": control.get("proxy_error"),
        "treatment_proxy_error": treatment.get("proxy_error"),
        "control_combat_margin_raw": control.get("combat_margin_raw"),
        "treatment_combat_margin_raw": treatment.get("combat_margin_raw"),
        "control_winner_board": control.get("winner_board"),
        "treatment_winner_board": treatment.get("winner_board"),
        "control_board_a": control.get("board_a"),
        "control_board_b": control.get("board_b"),
        "treatment_board_a": treatment.get("board_a"),
        "treatment_board_b": treatment.get("board_b"),
        "control_survivors": control.get("survivor_identities"),
        "treatment_survivors": treatment.get("survivor_identities"),
    })
    return out


def iter_class3_events(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Iterable[Dict]:
    """Yield enriched 3M class-(3) first-split events."""
    window = tuple(turns or LATE_TURNS)
    c_dec = _index_decisions(control_raw.get("pairing_decisions") or [])
    t_dec = _index_decisions(treatment_raw.get("pairing_decisions") or [])
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    from ml.elimination_chain_diagnostic import _hp_row_index
    hp_c = _hp_row_index(control_raw.get("hp_rows") or [])
    hp_t = _hp_row_index(treatment_raw.get("hp_rows") or [])
    for chain in iter_class1_chain_records(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    ):
        event = find_first_hp_divergence(
            chain, c_fights, t_fights, hp_c, hp_t, c_dec, t_dec,
        )
        if event.get("class") != _CLASS3:
            continue
        seed = _safe_int(event.get("seed"))
        seat = _safe_int(event.get("causal_seat"))
        turn = _safe_int(event.get("first_divergence_turn"))
        c_fight = None
        t_fight = None
        if seed is not None and seat is not None and turn is not None:
            c_fight = c_fights.get((seed, seat, turn))
            t_fight = t_fights.get((seed, seat, turn))
        yield enrich_first_split(event, c_fight, t_fight)


def attribute_matched_state_damage(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Dict:
    """Decompose class-(3) first-split applied-damage difference."""
    window = tuple(turns or LATE_TURNS)
    first = attribute_first_divergence(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    )
    rows = list(iter_class3_events(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    ))
    totals = {name: 0.0 for name in DAMAGE_COMPONENTS}
    source_totals = {name: 0.0 for name in SOURCE_COMPONENTS}
    turn_counts = Counter()
    examples: List[Dict] = []
    n_recon = 0
    n_hp = 0
    n_cf = 0
    n_five = 0
    n_matched = 0
    matched_delta = 0.0
    control_states = []
    treatment_states = []

    for rec in rows:
        decomp = rec.get("decomposition") or {}
        for name in DAMAGE_COMPONENTS:
            totals[name] += float(decomp.get(name) or 0.0)
        source_totals["pre_fight_board"] += float(
            decomp.get("pre_fight_board") or 0.0
        )
        source_totals["within_fight_survival"] += float(
            decomp.get("within_fight_survival") or 0.0
        )
        source_totals["proxy_formula_error"] += float(
            decomp.get("proxy_formula_error") or 0.0
        )
        source_totals["residual"] += float(decomp.get("residual") or 0.0)
        if rec.get("first_divergence_turn") is not None:
            turn_counts[str(rec["first_divergence_turn"])] += 1
        if rec.get("row_class") == "reconciled":
            n_recon += 1
        if (rec.get("control_state") or {}).get("hp_flow_ok") and (
            rec.get("treatment_state") or {}
        ).get("hp_flow_ok"):
            n_hp += 1
        if (rec.get("control_state") or {}).get("cf_ok") and (
            rec.get("treatment_state") or {}
        ).get("cf_ok"):
            n_cf += 1
        if decomp.get("five_way_ok"):
            n_five += 1
        if rec.get("matched_pre_fight_board"):
            n_matched += 1
            matched_delta += float(decomp.get("delta_applied") or 0.0)
        if len(examples) < _N_EXAMPLES:
            examples.append(rec)
        control_states.append(rec.get("control_state") or {})
        treatment_states.append(rec.get("treatment_state") or {})

    n_rows = float(len(rows))
    delta_applied = sum(float((r.get("decomposition") or {}).get("delta_applied") or 0.0) for r in rows)
    reconstructed = sum(totals.values())
    shares = {
        name: share_of_applied(totals[name], denom=delta_applied)
        for name in DAMAGE_COMPONENTS
    }
    source_shares = {
        name: share_of_applied(source_totals[name], denom=delta_applied)
        for name in SOURCE_COMPONENTS
    }
    control_arm = _arm_summary(control_states)
    treatment_arm = _arm_summary(treatment_states)
    kitagawa = decompose_gap(
        control_arm, treatment_arm,
        observed_delta=(
            None if control_arm.get("mean_survivor_tier_sum") is None
            else (
                float(treatment_arm.get("mean_survivor_tier_sum") or 0.0)
                - float(control_arm.get("mean_survivor_tier_sum") or 0.0)
            )
        ),
    )
    # Convert per-hit Kitagawa A/B into a share of total Δ_applied.
    n_hits = n_rows if n_rows else 0.0
    fielded_a = float(kitagawa.get("fielded_composition_A") or 0.0) * n_hits
    survival_b = float(kitagawa.get("within_tier_survival_B") or 0.0) * n_hits
    tokens_c = float(kitagawa.get("token_generated_C") or 0.0) * n_hits
    tavern_total = totals["winner_tavern_tier"]
    proxy_total = totals["proxy_formula_error"]
    kit_pre = tavern_total + fielded_a
    kit_within = survival_b + tokens_c
    kit_residual = delta_applied - (kit_pre + kit_within + proxy_total)
    kit_source = {
        "pre_fight_board": kit_pre,
        "within_fight_survival": kit_within,
        "proxy_formula_error": proxy_total,
        "residual": kit_residual,
    }
    kit_shares = {
        name: share_of_applied(kit_source[name], denom=delta_applied)
        for name in SOURCE_COMPONENTS
    }
    return {
        "turns": list(window),
        "trace_from_turn": TRACE_FROM_TURN,
        "hp_walk_from_turn": HP_WALK_FROM_TURN,
        "n_same_seat_earlier": first.get("n_same_seat_earlier"),
        "n_same_outcome_damage": int(n_rows),
        "published_same_seat_earlier": PHASE_3L_SAME_SEAT_EARLIER,
        "published_same_outcome_damage": PHASE_3M_SAME_OUTCOME_DAMAGE,
        "same_seat_earlier_reproduced": first.get("same_seat_earlier_reproduced"),
        "same_outcome_damage_reproduced": (
            int(n_rows) == PHASE_3M_SAME_OUTCOME_DAMAGE
            if window == LATE_TURNS else int(n_rows) == PHASE_3M_VERY_LATE_DAMAGE
        ),
        "first_divergence_3m": {
            "counts": first.get("counts"),
            "shares": {
                k: first.get(f"share_{k}")
                for k in (
                    "prior_alive_set_or_pairing",
                    "same_pairing_outcome_flip",
                    "same_outcome_damage",
                    "inherited_hp_carry",
                    "unreconciled",
                )
            },
            "first_divergence_turn_counts": first.get(
                "first_divergence_turn_counts"
            ),
            "reconciliation_ok": first.get("reconciliation_ok"),
            "row_divergence_ok": first.get("row_divergence_ok"),
        },
        "first_divergence_turn_counts": dict(turn_counts),
        "delta_applied_total": float(delta_applied),
        "attributed": dict(totals),
        "reconstructed_delta_applied": float(reconstructed),
        "reconciliation_gap": float(delta_applied - reconstructed),
        "reconciliation_ok": abs(delta_applied - reconstructed) <= max(
            1.0, 1e-9 * (1.0 + abs(delta_applied))
        ),
        "n_row_checked": int(n_rows),
        "n_row_reconciled": n_recon,
        "n_row_hp_flow_ok": n_hp,
        "n_row_cf_ok": n_cf,
        "n_row_five_way_ok": n_five,
        "row_damage_ok": int(n_rows) == 0 or n_recon == int(n_rows),
        "row_hp_flow_ok": int(n_rows) == 0 or n_hp == int(n_rows),
        "row_cf_ok": int(n_rows) == 0 or n_cf == int(n_rows),
        "row_five_way_ok": int(n_rows) == 0 or n_five == int(n_rows),
        **{f"share_{k}": v for k, v in shares.items()},
        "source_totals": dict(source_totals),
        **{f"share_{k}": v for k, v in source_shares.items() if f"share_{k}" not in shares},
        "share_pre_fight_board": source_shares["pre_fight_board"],
        "share_within_fight_survival": source_shares["within_fight_survival"],
        "control": control_arm,
        "treatment": treatment_arm,
        "kitagawa": kitagawa,
        "kitagawa_source": kit_source,
        "share_pre_fight_board_kitagawa": kit_shares["pre_fight_board"],
        "share_within_fight_survival_kitagawa": kit_shares["within_fight_survival"],
        "share_proxy_formula_error_kitagawa": kit_shares["proxy_formula_error"],
        "n_matched_pre_fight_board": n_matched,
        "matched_board_delta_applied": float(matched_delta),
        "share_matched_board_of_delta": share_of_applied(
            matched_delta, denom=delta_applied,
        ),
        "examples": examples,
        "candidate_choice_ok": first.get("candidate_choice_ok"),
        "matchmaking_reconciliation_ok": first.get("matchmaking_reconciliation_ok"),
        "timing_reconciliation_ok": first.get("timing_reconciliation_ok"),
        "chain_reconciliation_ok": first.get("chain_reconciliation_ok"),
        "first_divergence_reconciliation_ok": first.get("reconciliation_ok"),
    }


def compare_matched_state_damage(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    selection: Optional[Dict] = None,
    retention: Optional[Dict] = None,
    pairing: Optional[Dict] = None,
    matchmaking: Optional[Dict] = None,
    timing: Optional[Dict] = None,
    chain: Optional[Dict] = None,
    first: Optional[Dict] = None,
) -> Dict:
    """3M class-(1)/class-(3) lock + matched-state damage split."""
    if first is None:
        first = compare_first_divergence(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, retention=retention, pairing=pairing,
            matchmaking=matchmaking, timing=timing, chain=chain,
        )
    if chain is None:
        chain = compare_chain(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, retention=retention, pairing=pairing,
            matchmaking=matchmaking, timing=timing,
        )
    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch, turns=LATE_TURNS,
        still_fields_t1t3=False,
    )
    very_late_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch, turns=VERY_LATE_TURNS,
        still_fields_t1t3=False,
    )
    late = attribute_matched_state_damage(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late = attribute_matched_state_damage(
        control_raw, treatment_raw,
        leftover_rows=very_late_rows, treatment_punch=t_punch,
        turns=VERY_LATE_TURNS,
    )
    first_rec = first.get("reconciliation") or {}
    rec = dict(first_rec)
    rec.update({
        "applied_reconcile_identity": APPLIED_RECONCILE_IDENTITY,
        "counterfactual_identity": COUNTERFACTUAL_IDENTITY,
        "proxy_error_identity": PROXY_ERROR_IDENTITY,
        "field_vs_survival_identity": FIELD_VS_SURVIVAL_IDENTITY,
        "row_damage_reconcile_identity": ROW_DAMAGE_RECONCILE_IDENTITY,
        "first_divergence_reconcile_identity": FIRST_DIVERGENCE_RECONCILE_IDENTITY,
        "row_history_divergence_identity": ROW_HISTORY_DIVERGENCE_IDENTITY,
        "same_outcome_damage_n": late.get("n_same_outcome_damage"),
        "same_outcome_damage_reproduced": late.get("same_outcome_damage_reproduced"),
        "applied_reconciliation_ok": late.get("reconciliation_ok"),
        "row_damage_ok": late.get("row_damage_ok"),
        "row_cf_ok": late.get("row_cf_ok"),
        "row_five_way_ok": late.get("row_five_way_ok"),
        "flow_abs_tol": FLOW_ABS_TOL,
        "reweight_abs_tol": REWEIGHT_ABS_TOL,
    })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": late,
        "source": {
            "share_pre_fight_board": late.get("share_pre_fight_board_kitagawa"),
            "share_within_fight_survival": late.get(
                "share_within_fight_survival_kitagawa"
            ),
            "share_proxy_formula_error": late.get(
                "share_proxy_formula_error_kitagawa"
            ),
            "share_residual": share_of_applied(
                (late.get("kitagawa_source") or {}).get("residual"),
                denom=late.get("delta_applied_total"),
            ),
            "kitagawa": late.get("kitagawa"),
            "kitagawa_source": late.get("kitagawa_source"),
            "n_matched_pre_fight_board": late.get("n_matched_pre_fight_board"),
            "matched_board_delta_applied": late.get("matched_board_delta_applied"),
            "row_fielded_share_pre_fight_board": late.get("share_pre_fight_board"),
            "row_fielded_share_within_fight_survival": late.get(
                "share_within_fight_survival"
            ),
        },
        "very_late_attribution": very_late,
        "first_divergence_3m": first.get("attribution"),
        "very_late_first_divergence_3m": first.get("very_late_attribution"),
        "chain_3l": first.get("chain_3l"),
        "timing_3k": first.get("timing_3k"),
        "matchmaking_3j": first.get("matchmaking_3j"),
        "pairing_3i": first.get("pairing_3i"),
        "leftover_3h": first.get("leftover_3h"),
        "reconciliation": rec,
        "decomposition_3g": first.get("decomposition_3g"),
        "paired_seats": first.get("paired_seats"),
        "timing_3f": first.get("timing_3f"),
        "lifecycle": first.get("lifecycle"),
        "published_3g_locks": first.get("published_3g_locks") or {
            "mixture": PHASE_3G_MIXTURE,
            "mixture_share": PHASE_3G_MIXTURE_SHARE,
            "within_share": PHASE_3G_WITHIN_SHARE,
            "n_control": PHASE_3G_N_CONTROL,
            "n_treatment": PHASE_3G_N_TREATMENT,
        },
        "published_3h_locks": first.get("published_3h_locks") or {
            "leftover": PHASE_3H_LEFTOVER,
            "late_control": PHASE_3H_LATE_CONTROL,
            "late_treatment": PHASE_3H_LATE_TREATMENT,
            "collapse": PHASE_3H_COLLAPSE,
            "share_leftover": PHASE_3H_SHARE_LEFTOVER,
        },
        "published_3i_locks": first.get("published_3i_locks") or {
            "pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
            "share_pairing_schedule": PHASE_3I_SHARE_PAIRING_SCHEDULE,
        },
        "published_3j_locks": first.get("published_3j_locks") or {
            "eligibility": PHASE_3J_ELIGIBILITY,
            "share_eligibility": PHASE_3J_SHARE_ELIGIBILITY,
        },
        "published_3k_locks": first.get("published_3k_locks") or {
            "third_party": PHASE_3K_THIRD_PARTY,
            "share_third_party": PHASE_3K_SHARE_THIRD_PARTY,
        },
        "published_3l_locks": first.get("published_3l_locks") or {
            "same_seat_earlier": PHASE_3L_SAME_SEAT_EARLIER,
            "share_earlier": PHASE_3L_SHARE_EARLIER,
        },
        "published_3m_locks": {
            "class1": PHASE_3M_CLASS1,
            "same_outcome_damage": PHASE_3M_SAME_OUTCOME_DAMAGE,
            "share_damage": PHASE_3M_SHARE_DAMAGE,
            "first_split_t5": PHASE_3M_FIRST_SPLIT_T5,
            "very_late_class1": PHASE_3M_VERY_LATE_CLASS1,
            "very_late_damage": PHASE_3M_VERY_LATE_DAMAGE,
        },
    }


__all__ = [
    "MatchedStateDamageTracer",
    "attribute_matched_state_damage",
    "compare_matched_state_damage",
    "decompose_applied_row",
    "enrich_first_split",
    "extract_fight_state",
    "iter_class3_events",
    "run_greedy_2s_treatment_matched",
    "run_greedy_control_matched",
    "rules_faithful_hero_damage",
    "slim_board",
    "slim_body",
]
