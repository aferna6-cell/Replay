"""Phase 3K — observational elimination-timing attribution.

Reuses the 3J MatchmakingDivergenceTracer on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For each 3J eligibility leftover punch row, walks T7 through the first
alive-set / ghost-bye divergence and splits those 5648 rows exclusively.
For treatment-earlier / control-opponent-earlier rows, attributes the
decisive HP gap into prior HP vs current-fight hit vs damage magnitude.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.matchmaking_divergence_diagnostic import (
    BYE_TOKEN,
    GHOST_TOKEN,
    MatchmakingDivergenceTracer,
    _canon_set,
    _index_decisions,
    _seat_view,
    _slim_decision,
    attribute_matchmaking,
    compare_matchmaking,
    iter_pairing_schedule_rows,
)
from ml.pairing_who_wins_diagnostic import (
    _index_seat_fights,
    _kind_of,
    _opponent_of,
    _seat_side_fields,
)
from ml.phase_3k_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    FLOW_ABS_TOL,
    HISTORY_LINK_IDENTITY,
    HP_FLOW_IDENTITY,
    HP_GAP_COMPONENTS,
    HP_GAP_RECONCILE_IDENTITY,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    MATCHMAKING_RECONCILE_IDENTITY,
    PAIRED_SEAT_IDENTITY,
    PAIRING_IDENTITY,
    PHASE_3E_PUNCH_DELTA_CARRY,
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
    PHASE_3I_DIFFERENT_OPPONENT,
    PHASE_3I_KIND_MISMATCH,
    PHASE_3I_OUTCOME_FLIP,
    PHASE_3I_PAIRING_SCHEDULE,
    PHASE_3I_RESIDUAL,
    PHASE_3I_SHARE_PAIRING_SCHEDULE,
    PHASE_3I_SURVIVOR_SUBSTITUTION,
    PHASE_3J_ELIGIBILITY,
    PHASE_3J_ELIG_DIFFERENT_OPPONENT,
    PHASE_3J_ELIG_KIND_MISMATCH,
    PHASE_3J_HISTORY_LEGAL,
    PHASE_3J_RNG_ORDER,
    PHASE_3J_SHARE_ELIGIBILITY,
    PHASE_3J_UNRECONCILED,
    POOL_FLOW_IDENTITY,
    REWEIGHT_ABS_TOL,
    TIMING_COMPONENTS,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_first_eligibility,
    classify_hp_gap,
    share_of_eligibility,
    share_of_hp_gap,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.carry_divergence_diagnostic import reconcile_history_links

METHODOLOGY_VERSION = "3k_v1"

_N_EXAMPLES = 8


def _safe_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _seat_hp(fight: Optional[Dict], seat) -> Tuple[Optional[int], Optional[int]]:
    if not fight or seat in (None, ""):
        return None, None
    try:
        seat_i = int(seat)
    except (TypeError, ValueError):
        return None, None
    sa = fight.get("seat_a")
    sb = fight.get("seat_b")
    try:
        if sa not in (None, "") and int(sa) == seat_i:
            return _safe_int(fight.get("pre_hp_a")), _safe_int(fight.get("post_hp_a"))
        if sb not in (None, "") and int(sb) == seat_i:
            return _safe_int(fight.get("pre_hp_b")), _safe_int(fight.get("post_hp_b"))
    except (TypeError, ValueError):
        return None, None
    return None, None


def _applied_to_seat(fight: Optional[Dict], seat) -> Optional[int]:
    pre, post = _seat_hp(fight, seat)
    if pre is None or post is None:
        return None
    return int(pre) - int(post)


def _seat_hit(fight: Optional[Dict], seat) -> bool:
    applied = _applied_to_seat(fight, seat)
    return bool(applied is not None and applied > 0)


def _seat_outcome(fight: Optional[Dict], seat) -> Optional[str]:
    if not fight or seat in (None, ""):
        return None
    kind = _kind_of(fight)
    if kind == "bye":
        return "bye"
    try:
        seat_i = int(seat)
    except (TypeError, ValueError):
        return None
    winner = fight.get("winner_seat")
    loser = fight.get("loser_seat")
    try:
        if winner not in (None, "") and int(winner) == seat_i:
            return "win"
        if loser not in (None, "") and int(loser) == seat_i:
            return "loss"
    except (TypeError, ValueError):
        pass
    if int(fight.get("raw") or 0) == 0:
        return "tie"
    return kind


def _seat_trace(fight: Optional[Dict], seat, *, elimination_turn=None) -> Dict:
    side = _seat_side_fields(fight, seat)
    pre, post = _seat_hp(fight, seat)
    pairing = (fight or {}).get("pairing") or {}
    return {
        "turn": None if not fight else fight.get("turn"),
        "kind": _kind_of(fight),
        "opponent": _opponent_of(fight, seat),
        "outcome": _seat_outcome(fight, seat),
        "pre_combat_hp": pre if pre is not None else side.get("pre_fight_hp"),
        "post_combat_hp": post,
        "applied_damage": _applied_to_seat(fight, seat),
        "applied_hp_loss": None if not fight else fight.get("applied_hp_loss"),
        "survivor_count": (
            None if not fight else
            fight.get("actual_survivor_count",
                      pairing.get("survivor_count", fight.get("survivor_count")))
        ),
        "survivor_tier_sum": (
            None if not fight else
            fight.get("actual_survivor_tier_sum",
                      pairing.get("survivor_tier_sum", fight.get("survivor_tier_sum")))
        ),
        "tavern_tier": side.get("tavern_tier"),
        "board_recruit_raw": side.get("recruit_raw"),
        "abstract_pool_raw": side.get("abstract_pool_raw"),
        "total_combat_raw": side.get("combat_raw"),
        "elimination_turn": elimination_turn,
    }


def _alive_set(decision: Optional[Dict]) -> set:
    out = set()
    for s in list((decision or {}).get("alive_seats") or []):
        v = _safe_int(s)
        if v is not None:
            out.add(v)
    return out


def _ghost_bye_equal(c_dec: Optional[Dict], t_dec: Optional[Dict]) -> bool:
    if not c_dec or not t_dec:
        return False
    return (
        bool(c_dec.get("ghost_eligible")) == bool(t_dec.get("ghost_eligible"))
        and bool(c_dec.get("bye_eligible")) == bool(t_dec.get("bye_eligible"))
    )


def _eligibility_differs(c_dec: Optional[Dict], t_dec: Optional[Dict]) -> bool:
    if not c_dec or not t_dec:
        return True
    if _canon_set(c_dec.get("alive_seats") or []) != _canon_set(
        t_dec.get("alive_seats") or []
    ):
        return True
    return not _ghost_bye_equal(c_dec, t_dec)


def _token_or_int(v):
    if v in (GHOST_TOKEN, BYE_TOKEN, None, ""):
        return v
    return _safe_int(v, default=v)


def _named_seats(leftover, *opponents) -> set:
    named = set()
    leftover_i = _safe_int(leftover)
    if leftover_i is not None:
        named.add(leftover_i)
    for opp in opponents:
        v = _token_or_int(opp)
        if isinstance(v, int):
            named.add(v)
    return named


def _chosen_of(decision: Optional[Dict], seat):
    view = _seat_view(decision, seat)
    if not view:
        return None
    return view.get("chosen")


def first_eligibility_turn(
    leftover_turn: int,
    c_decisions: Dict[Tuple[int, int], Dict],
    t_decisions: Dict[Tuple[int, int], Dict],
    seed: int,
    *,
    from_turn: int = TRACE_FROM_TURN,
) -> Optional[int]:
    """First T>=from_turn through leftover_turn where eligibility differs."""
    try:
        end = int(leftover_turn)
        start = int(from_turn)
        seed_i = int(seed)
    except (TypeError, ValueError):
        return None
    for turn in range(start, end + 1):
        if _eligibility_differs(c_decisions.get((seed_i, turn)),
                                t_decisions.get((seed_i, turn))):
            return turn
    return None


def _elim_turn_map(eliminations: Sequence[Dict]) -> Dict[Tuple[int, int], Optional[int]]:
    out: Dict[Tuple[int, int], Optional[int]] = {}
    for rec in eliminations or []:
        if rec.get("survived"):
            continue
        seed = _safe_int(rec.get("seed"))
        seat = _safe_int(rec.get("seat"))
        turn = _safe_int(rec.get("turn"))
        if seed is None or seat is None or turn is None:
            continue
        prev = out.get((seed, seat))
        if prev is None or turn < prev:
            out[(seed, seat)] = turn
    return out


def _hp_flow_ok(fight: Dict) -> bool:
    kind = _kind_of(fight)
    applied = _safe_int(fight.get("applied_hp_loss"), 0) or 0
    if applied is None:
        applied = _safe_int(fight.get("applied"), 0) or 0
    delta = 0
    pre_a, post_a = fight.get("pre_hp_a"), fight.get("post_hp_a")
    pre_b, post_b = fight.get("pre_hp_b"), fight.get("post_hp_b")
    if pre_a is not None and post_a is not None:
        delta += int(pre_a) - int(post_a)
    if kind == "live" and pre_b is not None and post_b is not None:
        delta += int(pre_b) - int(post_b)
    return int(delta) == int(applied)


class EliminationTimingTracer(MatchmakingDivergenceTracer):
    """3J pairing stamps plus per-seat HP / elimination trajectories."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.hp_rows: List[Dict] = []
        self.eliminations: List[Dict] = []
        self._alive_last: Dict[int, bool] = {}

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._alive_last.clear()

    def on_pairing(self, env: BGEnv, rec: Dict) -> None:
        super().on_pairing(env, rec)
        hp_at_pair = {}
        alive_at_pair = {}
        for p in env.players:
            hp_at_pair[str(int(p.idx))] = int(p.hp)
            alive_at_pair[str(int(p.idx))] = bool(p.alive)
        self.pairing_decisions[-1]["hp_at_pair"] = hp_at_pair
        self.pairing_decisions[-1]["alive_at_pair"] = alive_at_pair

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        pairing = rec.get("pairing")
        if isinstance(pairing, dict):
            pairing["post_hp_a"] = fight.get("post_hp_a")
            pairing["post_hp_b"] = fight.get("post_hp_b")
            pairing["applied"] = fight.get("applied")

    def after_combat(self, env: BGEnv) -> None:
        super().after_combat(env)
        turn = int(getattr(env, "turn", 0) or 0)
        n_alive = sum(1 for p in env.players if p.alive)
        for p in env.players:
            seat = int(p.idx)
            was = self._alive_last.get(seat, True)
            alive = bool(p.alive)
            self.hp_rows.append({
                "lobby": self.lobby_id,
                "seed": self.seed,
                "arm": self.arm,
                "seat": seat,
                "turn": turn,
                "hp": int(p.hp),
                "alive": alive,
                "tier": int(p.tier),
                "players_alive": n_alive,
            })
            if was and not alive:
                self.eliminations.append({
                    "lobby": self.lobby_id,
                    "seed": self.seed,
                    "arm": self.arm,
                    "seat": seat,
                    "turn": turn,
                    "hp": int(p.hp),
                    "placement": p.placement,
                    "survived": False,
                })
            self._alive_last[seat] = alive

    def end_lobby(self, players) -> None:
        turns = [int(r["turn"]) for r in self.hp_rows]
        self.game_length = max(turns) if turns else 0
        seen = {e["seat"] for e in self.eliminations}
        for p in players:
            if p.idx not in seen:
                self.eliminations.append({
                    "lobby": self.lobby_id,
                    "seed": self.seed,
                    "arm": self.arm,
                    "seat": p.idx,
                    "turn": self.game_length,
                    "hp": int(p.hp),
                    "placement": p.placement,
                    "survived": True,
                })


def run_elimination_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
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
                tracer = EliminationTimingTracer(i, seed + i, arm)
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


def run_greedy_control_elimination(lobbies: int, seed: int) -> Dict:
    return run_elimination_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_elimination(lobbies: int, seed: int) -> Dict:
    return run_elimination_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _decisive_seat(
    timing_class: str,
    only_c: set,
    only_t: set,
    named: set,
    elim_c: Dict[Tuple[int, int], Optional[int]],
    elim_t: Dict[Tuple[int, int], Optional[int]],
    seed: int,
) -> Optional[int]:
    if timing_class == "treatment_eliminated_earlier":
        pool = (only_c & named) or only_c
        key_fn = lambda s: elim_t.get((seed, s))
    elif timing_class == "control_opponent_eliminated_earlier":
        pool = (only_t & named) or only_t
        key_fn = lambda s: elim_c.get((seed, s))
    else:
        return None
    if not pool:
        return None
    dated = [(s, key_fn(s)) for s in pool]
    dated = [(s, t) for s, t in dated if t is not None]
    if not dated:
        return min(pool)
    # Latest earlier-arm elimination among candidates caused T_div.
    dated.sort(key=lambda kv: (kv[1], kv[0]))
    return dated[-1][0]


def _walk_traces(
    leftover_seat,
    seed: int,
    from_turn: int,
    through_turn: int,
    c_fights: Dict,
    t_fights: Dict,
    elim_c: Dict,
    elim_t: Dict,
) -> List[Dict]:
    out = []
    leftover_i = _safe_int(leftover_seat)
    if leftover_i is None:
        return out
    for turn in range(int(from_turn), int(through_turn) + 1):
        c_f = c_fights.get((seed, leftover_i, turn))
        t_f = t_fights.get((seed, leftover_i, turn))
        out.append({
            "turn": turn,
            "control": _seat_trace(c_f, leftover_i,
                                   elimination_turn=elim_c.get((seed, leftover_i))),
            "treatment": _seat_trace(t_f, leftover_i,
                                     elimination_turn=elim_t.get((seed, leftover_i))),
        })
    return out


def _classify_one_eligibility(
    row: Dict,
    c_decisions: Dict[Tuple[int, int], Dict],
    t_decisions: Dict[Tuple[int, int], Dict],
    c_fights: Dict,
    t_fights: Dict,
    elim_c: Dict[Tuple[int, int], Optional[int]],
    elim_t: Dict[Tuple[int, int], Optional[int]],
) -> Dict:
    seed = _safe_int(row.get("seed"))
    leftover = _safe_int(row.get("winner_seat", row.get("leftover_seat")))
    leftover_turn = _safe_int(row.get("turn"))
    if seed is None or leftover is None or leftover_turn is None:
        return {
            "seed": row.get("seed"),
            "turn": row.get("turn"),
            "leftover_seat": leftover,
            "class": "unreconciled",
            "hp_gap_class": None,
            "first_divergence_turn": None,
            "pairing_schedule_subtype": row.get("subtype"),
        }
    t_div = first_eligibility_turn(
        leftover_turn, c_decisions, t_decisions, seed,
        from_turn=TRACE_FROM_TURN,
    )
    c_dec = None if t_div is None else c_decisions.get((seed, t_div))
    t_dec = None if t_div is None else t_decisions.get((seed, t_div))
    c_alive = _alive_set(c_dec)
    t_alive = _alive_set(t_dec)
    only_c = c_alive - t_alive
    only_t = t_alive - c_alive
    leftover_c = leftover in c_alive
    leftover_t = leftover in t_alive
    c_row_dec = c_decisions.get((seed, leftover_turn))
    t_row_dec = t_decisions.get((seed, leftover_turn))
    c_opp_div = _chosen_of(c_dec, leftover)
    t_opp_div = _chosen_of(t_dec, leftover)
    c_opp_row = row.get("control_opponent_seat", _chosen_of(c_row_dec, leftover))
    t_opp_row = row.get("treatment_opponent_seat", _chosen_of(t_row_dec, leftover))
    named = _named_seats(leftover, c_opp_div, t_opp_div, c_opp_row, t_opp_row)
    third = bool((only_c - named) or (only_t - named))
    timing = classify_first_eligibility(
        control_present=c_dec is not None,
        treatment_present=t_dec is not None,
        leftover_alive_control=leftover_c,
        leftover_alive_treatment=leftover_t,
        leftover_in_only_control=leftover in only_c,
        leftover_in_only_treatment=leftover in only_t,
        named_in_only_control=bool(only_c & named),
        named_in_only_treatment=bool(only_t & named),
        third_party_alive_diff=third,
        ghost_bye_eligible_equal=_ghost_bye_equal(c_dec, t_dec),
        alive_sets_equal=not only_c and not only_t,
    )
    decisive = _decisive_seat(
        timing, only_c, only_t, named, elim_c, elim_t, seed,
    )
    c_elim = None if decisive is None else elim_c.get((seed, decisive))
    t_elim = None if decisive is None else elim_t.get((seed, decisive))
    if timing == "treatment_eliminated_earlier":
        fight_turn = t_elim
    elif timing == "control_opponent_eliminated_earlier":
        fight_turn = c_elim
    else:
        fight_turn = None
    c_fight = None
    t_fight = None
    if decisive is not None and fight_turn is not None:
        c_fight = c_fights.get((seed, decisive, int(fight_turn)))
        t_fight = t_fights.get((seed, decisive, int(fight_turn)))
    pre_c, _ = _seat_hp(c_fight, decisive)
    pre_t, _ = _seat_hp(t_fight, decisive)
    applied_c = _applied_to_seat(c_fight, decisive)
    applied_t = _applied_to_seat(t_fight, decisive)
    hp_cls = classify_hp_gap(
        timing_class=timing,
        control_fight_present=c_fight is not None,
        treatment_fight_present=t_fight is not None,
        pre_hp_equal=(pre_c is not None and pre_t is not None and pre_c == pre_t),
        control_hit=_seat_hit(c_fight, decisive),
        treatment_hit=_seat_hit(t_fight, decisive),
        applied_equal=(
            applied_c is not None and applied_t is not None
            and applied_c == applied_t
        ),
    )
    traces = _walk_traces(
        leftover, seed, TRACE_FROM_TURN,
        leftover_turn if t_div is None else t_div,
        c_fights, t_fights, elim_c, elim_t,
    )
    return {
        "seed": seed,
        "turn": leftover_turn,
        "leftover_seat": leftover,
        "class": timing,
        "hp_gap_class": hp_cls,
        "first_divergence_turn": t_div,
        "pairing_schedule_subtype": row.get("subtype"),
        "control_kind": row.get("control_kind"),
        "treatment_kind": row.get("treatment_kind"),
        "control_fight_opponent": row.get("control_opponent_seat"),
        "treatment_fight_opponent": row.get("treatment_opponent_seat"),
        "alive_sets_equal": not only_c and not only_t,
        "ghost_bye_eligible_equal": _ghost_bye_equal(c_dec, t_dec),
        "only_control_alive": sorted(only_c),
        "only_treatment_alive": sorted(only_t),
        "named_seats": sorted(named),
        "decisive_seat": decisive,
        "control_elimination_turn": c_elim,
        "treatment_elimination_turn": t_elim,
        "decisive_fight_turn": fight_turn,
        "control_pre_hp": pre_c,
        "treatment_pre_hp": pre_t,
        "control_applied": applied_c,
        "treatment_applied": applied_t,
        "control_hit": _seat_hit(c_fight, decisive),
        "treatment_hit": _seat_hit(t_fight, decisive),
        "control_decisive": _seat_trace(
            c_fight, decisive, elimination_turn=c_elim,
        ) if decisive is not None else {},
        "treatment_decisive": _seat_trace(
            t_fight, decisive, elimination_turn=t_elim,
        ) if decisive is not None else {},
        "control": _slim_decision(c_dec, leftover),
        "treatment": _slim_decision(t_dec, leftover),
        "leftover_trace": traces,
    }


def attribute_elimination_timing(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Dict:
    """Split 3J eligibility leftover rows into elimination-timing classes."""
    window = tuple(turns or LATE_TURNS)
    schedule_rows = iter_pairing_schedule_rows(
        leftover_rows, control_raw, treatment_raw,
        treatment_punch=treatment_punch, turns=window,
    )
    c_dec = _index_decisions(control_raw.get("pairing_decisions") or [])
    t_dec = _index_decisions(treatment_raw.get("pairing_decisions") or [])
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    elim_c = _elim_turn_map(control_raw.get("eliminations") or [])
    elim_t = _elim_turn_map(treatment_raw.get("eliminations") or [])

    mm = attribute_matchmaking(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    )
    from ml.matchmaking_divergence_diagnostic import _compare_one_row
    elig_schedule = []
    for row in schedule_rows:
        try:
            key = (int(row["seed"]), int(row["turn"]))
        except (KeyError, TypeError, ValueError):
            continue
        rec = _compare_one_row(row, c_dec.get(key), t_dec.get(key))
        if rec.get("class") == "eligibility":
            elig_schedule.append(row)

    counts = Counter()
    hp_counts = Counter()
    subtype_x_class = Counter()
    first_turn_counts = Counter()
    examples: Dict[str, List[Dict]] = {name: [] for name in TIMING_COMPONENTS}
    hp_examples: Dict[str, List[Dict]] = {name: [] for name in HP_GAP_COMPONENTS}
    compared: List[Dict] = []

    for row in elig_schedule:
        rec = _classify_one_eligibility(
            row, c_dec, t_dec, c_fights, t_fights, elim_c, elim_t,
        )
        cls = rec["class"]
        counts[cls] += 1
        subtype_x_class[f"{row.get('subtype')}:{cls}"] += 1
        if rec.get("first_divergence_turn") is not None:
            first_turn_counts[str(rec["first_divergence_turn"])] += 1
        hp_cls = rec.get("hp_gap_class")
        if hp_cls:
            hp_counts[hp_cls] += 1
            if len(hp_examples[hp_cls]) < _N_EXAMPLES:
                hp_examples[hp_cls].append(rec)
        if len(examples[cls]) < _N_EXAMPLES:
            examples[cls].append(rec)
        if len(compared) < 32 or cls != "treatment_eliminated_earlier":
            if len(compared) < 64:
                compared.append(rec)

    elig_n = float(len(elig_schedule))
    attributed = {name: float(counts.get(name, 0)) for name in TIMING_COMPONENTS}
    reconstructed = sum(attributed.values())
    shares = {
        name: share_of_eligibility(attributed[name], denom=elig_n)
        for name in TIMING_COMPONENTS
    }
    named_n = (
        attributed["treatment_eliminated_earlier"]
        + attributed["control_opponent_eliminated_earlier"]
    )
    hp_attributed = {name: float(hp_counts.get(name, 0)) for name in HP_GAP_COMPONENTS}
    hp_reconstructed = sum(hp_attributed.values())
    hp_shares = {
        name: share_of_hp_gap(hp_attributed[name], denom=named_n)
        for name in HP_GAP_COMPONENTS
    }
    return {
        "turns": list(window),
        "trace_from_turn": TRACE_FROM_TURN,
        "n_pairing_schedule": mm.get("n_pairing_schedule"),
        "n_eligibility": int(elig_n),
        "n_leftover_input": len(leftover_rows),
        "published_eligibility": PHASE_3J_ELIGIBILITY,
        "eligibility_reproduced": int(elig_n) == PHASE_3J_ELIGIBILITY,
        "published_pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
        "pairing_schedule_reproduced": mm.get("pairing_schedule_reproduced"),
        "matchmaking_counts": mm.get("counts"),
        "counts": dict(counts),
        "hp_counts": dict(hp_counts),
        "subtype_x_class": dict(subtype_x_class),
        "first_divergence_turn_counts": dict(first_turn_counts),
        "attributed": attributed,
        "hp_attributed": hp_attributed,
        "reconstructed_eligibility_rows": reconstructed,
        "reconciliation_gap": elig_n - reconstructed,
        "reconciliation_ok": abs(elig_n - reconstructed) <= max(
            1.0, 1e-9 * (1 + elig_n)
        ),
        "hp_reconstructed_named_rows": hp_reconstructed,
        "hp_reconciliation_gap": named_n - hp_reconstructed,
        "hp_reconciliation_ok": abs(named_n - hp_reconstructed) <= max(
            1.0, 1e-9 * (1 + named_n)
        ),
        "n_named_eliminations": int(named_n),
        **{f"share_{k}": v for k, v in shares.items()},
        **{f"share_{k}": v for k, v in hp_shares.items()},
        "examples": examples,
        "hp_examples": hp_examples,
        "n_compared_kept": len(compared),
        "candidate_choice_ok": mm.get("candidate_choice_ok"),
        "matchmaking_reconciliation_ok": mm.get("reconciliation_ok"),
    }


def reconcile_hp_flow(fights: Sequence[Dict], *, turns=None) -> Dict:
    window = set(turns or INSTRUMENT_TURNS)
    n = 0
    n_ok = 0
    for fight in fights or []:
        turn = _safe_int(fight.get("turn"))
        if turn is None or turn not in window:
            continue
        n += 1
        if _hp_flow_ok(fight):
            n_ok += 1
    return {
        "identity": HP_FLOW_IDENTITY,
        "n_fights": n,
        "n_ok": n_ok,
        "n_mismatch": n - n_ok,
        "ok": n == 0 or n_ok == n,
    }


def reconcile_eliminations(
    fights: Sequence[Dict],
    eliminations: Sequence[Dict],
    *,
    n_lobbies: Optional[int] = None,
) -> Dict:
    combat = [e for e in (eliminations or []) if not e.get("survived")]
    survived = [e for e in (eliminations or []) if e.get("survived")]
    fight_idx = _index_seat_fights(fights or [])
    n_linked = 0
    n_checked = 0
    for rec in combat:
        seed = _safe_int(rec.get("seed"))
        seat = _safe_int(rec.get("seat"))
        turn = _safe_int(rec.get("turn"))
        if seed is None or seat is None or turn is None:
            continue
        n_checked += 1
        fight = fight_idx.get((seed, seat, turn))
        pre, post = _seat_hp(fight, seat)
        if fight is not None and post is not None and post <= 0 and (
            pre is None or pre > 0
        ):
            n_linked += 1
        elif fight is not None and post is not None and post <= 0:
            n_linked += 1
    n_players = None if n_lobbies is None else int(n_lobbies) * 8
    closed = (
        n_players is None
        or (len(combat) + len(survived) == n_players)
    )
    return {
        "identity": ELIMINATION_IDENTITY,
        "n_combat_eliminations": len(combat),
        "n_survived": len(survived),
        "n_checked": n_checked,
        "n_linked_to_fight": n_linked,
        "n_unlinked": n_checked - n_linked,
        "n_players_expected": n_players,
        "census_ok": closed,
        "link_ok": n_checked == 0 or n_linked == n_checked,
        "ok": closed and (n_checked == 0 or n_linked == n_checked),
    }


def compare_elimination(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    selection: Optional[Dict] = None,
    retention: Optional[Dict] = None,
    pairing: Optional[Dict] = None,
    matchmaking: Optional[Dict] = None,
) -> Dict:
    """3J eligibility lock + first-divergence elimination-timing split."""
    if matchmaking is None:
        matchmaking = compare_matchmaking(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, retention=retention, pairing=pairing,
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
    late = attribute_elimination_timing(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late = attribute_elimination_timing(
        control_raw, treatment_raw,
        leftover_rows=very_late_rows, treatment_punch=t_punch,
        turns=VERY_LATE_TURNS,
    )
    hp_c = reconcile_hp_flow(control_raw.get("fights") or [])
    hp_t = reconcile_hp_flow(treatment_raw.get("fights") or [])
    elim_c = reconcile_eliminations(
        control_raw.get("fights") or [],
        control_raw.get("eliminations") or [],
        n_lobbies=control_raw.get("n_lobbies"),
    )
    elim_t = reconcile_eliminations(
        treatment_raw.get("fights") or [],
        treatment_raw.get("eliminations") or [],
        n_lobbies=treatment_raw.get("n_lobbies"),
    )
    hist_c = reconcile_history_links(
        control_raw.get("fights") or [], control_raw.get("turn_rows") or [],
    )
    hist_t = reconcile_history_links(
        treatment_raw.get("fights") or [], treatment_raw.get("turn_rows") or [],
    )
    mm_rec = matchmaking.get("reconciliation") or {}
    rec = {
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
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
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "matchmaking_reconciliation_ok": mm_rec.get("matchmaking_reconciliation_ok"),
        "matchmaking_n": mm_rec.get("matchmaking_n"),
        "eligibility_n": late.get("n_eligibility"),
        "eligibility_reproduced": late.get("eligibility_reproduced"),
        "timing_reconciliation_ok": late.get("reconciliation_ok"),
        "hp_gap_reconciliation_ok": late.get("hp_reconciliation_ok"),
        "hp_flow_control": hp_c,
        "hp_flow_treatment": hp_t,
        "hp_flow_ok": bool(hp_c.get("ok") and hp_t.get("ok")),
        "elimination_control": elim_c,
        "elimination_treatment": elim_t,
        "elimination_ok": bool(elim_c.get("ok") and elim_t.get("ok")),
        "candidate_choice_ok": late.get("candidate_choice_ok"),
        "phase_3g_mixture_reproduced": (
            matchmaking.get("decomposition_3g") or {}
        ).get("mixture_turn_winner_tier"),
        "flow_abs_tol": FLOW_ABS_TOL,
        "reweight_abs_tol": REWEIGHT_ABS_TOL,
        "lineage_control": mm_rec.get("lineage_control"),
        "lineage_treatment": mm_rec.get("lineage_treatment"),
        "paired": mm_rec.get("paired"),
        "reproduced_3d_board_pool_magnitude": mm_rec.get(
            "reproduced_3d_board_pool_magnitude"
        ),
        "reproduced_3e_carry_share": mm_rec.get("reproduced_3e_carry_share"),
        "flow_mismatches_control": mm_rec.get("flow_mismatches_control"),
        "flow_mismatches_treatment": mm_rec.get("flow_mismatches_treatment"),
    }
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": late,
        "very_late_attribution": very_late,
        "matchmaking_3j": matchmaking.get("attribution"),
        "pairing_3i": matchmaking.get("pairing_3i"),
        "leftover_3h": matchmaking.get("leftover_3h"),
        "reconciliation": rec,
        "decomposition_3g": matchmaking.get("decomposition_3g"),
        "paired_seats": matchmaking.get("paired_seats"),
        "timing_3f": matchmaking.get("timing_3f"),
        "lifecycle": matchmaking.get("lifecycle"),
        "published_3g_locks": {
            "mixture": PHASE_3G_MIXTURE,
            "mixture_share": PHASE_3G_MIXTURE_SHARE,
            "within_share": PHASE_3G_WITHIN_SHARE,
            "n_control": PHASE_3G_N_CONTROL,
            "n_treatment": PHASE_3G_N_TREATMENT,
            "punch_delta": PHASE_3E_PUNCH_DELTA_CARRY,
        },
        "published_3h_locks": {
            "leftover": PHASE_3H_LEFTOVER,
            "late_control": PHASE_3H_LATE_CONTROL,
            "late_treatment": PHASE_3H_LATE_TREATMENT,
            "collapse": PHASE_3H_COLLAPSE,
            "share_leftover": PHASE_3H_SHARE_LEFTOVER,
        },
        "published_3i_locks": {
            "pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
            "outcome_flip": PHASE_3I_OUTCOME_FLIP,
            "survivor_substitution": PHASE_3I_SURVIVOR_SUBSTITUTION,
            "residual": PHASE_3I_RESIDUAL,
            "different_opponent": PHASE_3I_DIFFERENT_OPPONENT,
            "kind_mismatch": PHASE_3I_KIND_MISMATCH,
            "share_pairing_schedule": PHASE_3I_SHARE_PAIRING_SCHEDULE,
        },
        "published_3j_locks": {
            "eligibility": PHASE_3J_ELIGIBILITY,
            "history_legal": PHASE_3J_HISTORY_LEGAL,
            "rng_order": PHASE_3J_RNG_ORDER,
            "unreconciled": PHASE_3J_UNRECONCILED,
            "share_eligibility": PHASE_3J_SHARE_ELIGIBILITY,
            "different_opponent": PHASE_3J_ELIG_DIFFERENT_OPPONENT,
            "kind_mismatch": PHASE_3J_ELIG_KIND_MISMATCH,
        },
    }
