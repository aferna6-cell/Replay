"""Phase 3L — observational third-party elimination-chain attribution.

Reuses the 3K EliminationTimingTracer on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For each 3K ghost/bye third-party leftover punch row, identifies the
exact third-party seat whose elimination first changes ghost/bye
eligibility and splits those 3701 rows exclusively. For same-seat
earlier-elimination rows, attributes the decisive HP gap into prior HP
vs current-fight hit vs damage magnitude.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ml.elimination_timing_diagnostic import (
    EliminationTimingTracer,
    _alive_set,
    _applied_to_seat,
    _classify_one_eligibility,
    _elim_turn_map,
    _hp_flow_ok,
    _seat_hit,
    _seat_hp,
    _seat_outcome,
    _seat_trace,
    compare_elimination,
    reconcile_eliminations,
    reconcile_hp_flow,
    run_greedy_2s_treatment_elimination,
    run_greedy_control_elimination,
)
from ml.matchmaking_divergence_diagnostic import (
    _index_decisions,
    _slim_decision,
    attribute_matchmaking,
    iter_pairing_schedule_rows,
)
from ml.pairing_who_wins_diagnostic import (
    _index_seat_fights,
    same_pairing,
)
from ml.phase_3l_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_COMPONENTS,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
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
    PHASE_3K_CONTROL_OPPONENT,
    PHASE_3K_NAMED,
    PHASE_3K_PRIOR_HP,
    PHASE_3K_SHARE_PRIOR_HP,
    PHASE_3K_SHARE_THIRD_PARTY,
    PHASE_3K_THIRD_PARTY,
    PHASE_3K_TREATMENT_EARLIER,
    PHASE_3K_UNRECONCILED,
    POOL_FLOW_IDENTITY,
    REWEIGHT_ABS_TOL,
    ROW_ELIM_HP_IDENTITY,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    classify_chain_hp_gap,
    classify_third_party_chain,
    share_of_hp_gap,
    share_of_third_party,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.carry_divergence_diagnostic import reconcile_history_links

METHODOLOGY_VERSION = "3l_v1"

_N_EXAMPLES = 8


def _safe_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _hp_row_index(hp_rows: Sequence[Dict]) -> Dict[Tuple[int, int, int], Dict]:
    out: Dict[Tuple[int, int, int], Dict] = {}
    for rec in hp_rows or []:
        seed = _safe_int(rec.get("seed"))
        seat = _safe_int(rec.get("seat"))
        turn = _safe_int(rec.get("turn"))
        if seed is None or seat is None or turn is None:
            continue
        out[(seed, seat, turn)] = rec
    return out


def _prior_turn_hp(
    hp_index: Dict[Tuple[int, int, int], Dict],
    seed: Optional[int],
    seat: Optional[int],
    turn: Optional[int],
) -> Optional[int]:
    if seed is None or seat is None or turn is None:
        return None
    rec = hp_index.get((int(seed), int(seat), int(turn) - 1))
    if not rec:
        return None
    return _safe_int(rec.get("hp"))


def _paired_status(
    hp_index: Dict[Tuple[int, int, int], Dict],
    elim_map: Dict[Tuple[int, int], Optional[int]],
    seed: Optional[int],
    seat: Optional[int],
    turn: Optional[int],
) -> Dict:
    if seed is None or seat is None or turn is None:
        return {
            "alive": None,
            "eliminated": None,
            "hp": None,
            "elimination_turn": None,
        }
    rec = hp_index.get((int(seed), int(seat), int(turn)))
    elim_t = elim_map.get((int(seed), int(seat)))
    if rec is not None:
        alive = bool(rec.get("alive"))
        return {
            "alive": alive,
            "eliminated": not alive,
            "hp": _safe_int(rec.get("hp")),
            "elimination_turn": elim_t,
        }
    if elim_t is None:
        return {
            "alive": None,
            "eliminated": False,
            "hp": None,
            "elimination_turn": None,
        }
    eliminated = int(elim_t) <= int(turn)
    return {
        "alive": not eliminated,
        "eliminated": eliminated,
        "hp": None,
        "elimination_turn": elim_t,
    }


def _dead_board(decision: Optional[Dict]) -> set:
    out = set()
    for s in list((decision or {}).get("dead_with_board_seats") or []):
        v = _safe_int(s)
        if v is not None:
            out.add(v)
    return out


def _latest_elim_seat(
    seats,
    elim_map: Dict[Tuple[int, int], Optional[int]],
    seed: int,
    *,
    before_turn: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    pool = {int(s) for s in seats if _safe_int(s) is not None}
    if not pool:
        return None, None
    dated = []
    for s in pool:
        t = elim_map.get((seed, s))
        if t is None:
            continue
        if before_turn is not None and int(t) >= int(before_turn):
            continue
        dated.append((s, int(t)))
    if not dated:
        # Infer the combat immediately before first pairing divergence.
        inferred = None if before_turn is None else int(before_turn) - 1
        if inferred is not None and inferred >= 1:
            return min(pool), inferred
        return min(pool), None
    dated.sort(key=lambda kv: (kv[1], kv[0]))
    return dated[-1]


def _enrich_trace(
    fight: Optional[Dict],
    seat,
    *,
    elimination_turn=None,
    prior_turn_hp=None,
    paired_status=None,
) -> Dict:
    rec = _seat_trace(fight, seat, elimination_turn=elimination_turn)
    rec["prior_turn_hp"] = prior_turn_hp
    rec["paired_arm"] = paired_status or {
        "alive": None, "eliminated": None, "hp": None, "elimination_turn": None,
    }
    return rec


def _causal_seats(
    timing: Dict,
    elim_c: Dict[Tuple[int, int], Optional[int]],
    elim_t: Dict[Tuple[int, int], Optional[int]],
    c_dec: Optional[Dict],
    t_dec: Optional[Dict],
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return (control_seat, control_elim_turn, treatment_seat, treatment_elim_turn)."""
    seed = _safe_int(timing.get("seed"))
    t_div = _safe_int(timing.get("first_divergence_turn"))
    if seed is None:
        return None, None, None, None
    named = {
        int(s) for s in (timing.get("named_seats") or [])
        if _safe_int(s) is not None
    }
    only_c = {
        int(s) for s in (timing.get("only_control_alive") or [])
        if _safe_int(s) is not None
    }
    only_t = {
        int(s) for s in (timing.get("only_treatment_alive") or [])
        if _safe_int(s) is not None
    }
    # Alive only in control → died in treatment; alive only in treatment → died in control.
    third_died_t = only_c - named
    third_died_c = only_t - named
    seat_c, turn_c = _latest_elim_seat(
        third_died_c, elim_c, seed, before_turn=t_div,
    )
    seat_t, turn_t = _latest_elim_seat(
        third_died_t, elim_t, seed, before_turn=t_div,
    )
    if seat_c is None and seat_t is None:
        extra_c = (_dead_board(c_dec) - _dead_board(t_dec)) - named
        extra_t = (_dead_board(t_dec) - _dead_board(c_dec)) - named
        seat_c, turn_c = _latest_elim_seat(
            extra_c, elim_c, seed, before_turn=t_div,
        )
        seat_t, turn_t = _latest_elim_seat(
            extra_t, elim_t, seed, before_turn=t_div,
        )
    return seat_c, turn_c, seat_t, turn_t


def _died_at(elim_map, seed, seat, turn) -> bool:
    if seed is None or seat is None or turn is None:
        return False
    t = elim_map.get((int(seed), int(seat)))
    return t is not None and int(t) == int(turn)


def iter_third_party_timing_records(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Iterable[Dict]:
    """Yield 3K-classified eligibility records that are third-party."""
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
    from ml.matchmaking_divergence_diagnostic import _compare_one_row
    for row in schedule_rows:
        try:
            key = (int(row["seed"]), int(row["turn"]))
        except (KeyError, TypeError, ValueError):
            continue
        mm = _compare_one_row(row, c_dec.get(key), t_dec.get(key))
        if mm.get("class") != "eligibility":
            continue
        rec = _classify_one_eligibility(
            row, c_dec, t_dec, c_fights, t_fights, elim_c, elim_t,
        )
        if rec.get("class") == "ghost_bye_third_party":
            yield rec


def _classify_one_chain(
    timing: Dict,
    c_decisions: Dict[Tuple[int, int], Dict],
    t_decisions: Dict[Tuple[int, int], Dict],
    c_fights: Dict,
    t_fights: Dict,
    elim_c: Dict[Tuple[int, int], Optional[int]],
    elim_t: Dict[Tuple[int, int], Optional[int]],
    hp_c: Dict[Tuple[int, int, int], Dict],
    hp_t: Dict[Tuple[int, int, int], Dict],
) -> Dict:
    seed = _safe_int(timing.get("seed"))
    leftover = _safe_int(timing.get("leftover_seat"))
    leftover_turn = _safe_int(timing.get("turn"))
    t_div = _safe_int(timing.get("first_divergence_turn"))
    c_dec = None if seed is None or t_div is None else c_decisions.get((seed, t_div))
    t_dec = None if seed is None or t_div is None else t_decisions.get((seed, t_div))
    if seed is None or leftover is None:
        return {
            "seed": timing.get("seed"),
            "turn": timing.get("turn"),
            "leftover_seat": leftover,
            "class": "unreconciled",
            "hp_gap_class": None,
            "first_divergence_turn": t_div,
            "timing_class": timing.get("class"),
            "causal_seat": None,
        }
    seat_c, turn_c, seat_t, turn_t = _causal_seats(
        timing, elim_c, elim_t, c_dec, t_dec,
    )
    if seat_c is not None and turn_c is None:
        turn_c = elim_c.get((seed, seat_c))
    if seat_t is not None and turn_t is None:
        turn_t = elim_t.get((seed, seat_t))

    same_seat = (
        seat_c is not None and seat_t is not None and int(seat_c) == int(seat_t)
    ) or ((seat_c is None) != (seat_t is None) and (seat_c is not None or seat_t is not None))
    primary = None
    fight_turn = None
    if same_seat:
        primary = seat_t if seat_t is not None else seat_c
        fight_turn = turn_t if seat_t is not None else turn_c
        if fight_turn is None and primary is not None:
            fight_turn = elim_t.get((seed, primary)) or elim_c.get((seed, primary))
    elif seat_c is not None and seat_t is not None:
        # Different seats: pick the later of the two causal deaths as primary.
        if (turn_t or -1) >= (turn_c or -1):
            primary, fight_turn = seat_t, turn_t
        else:
            primary, fight_turn = seat_c, turn_c

    c_fight = None
    t_fight = None
    if primary is not None and fight_turn is not None:
        c_fight = c_fights.get((seed, int(primary), int(fight_turn)))
        t_fight = t_fights.get((seed, int(primary), int(fight_turn)))
    same = same_pairing(c_fight, t_fight, primary) if primary is not None else False
    out_c = _seat_outcome(c_fight, primary)
    out_t = _seat_outcome(t_fight, primary)
    applied_c = _applied_to_seat(c_fight, primary)
    applied_t = _applied_to_seat(t_fight, primary)
    pre_c, post_c = _seat_hp(c_fight, primary)
    pre_t, post_t = _seat_hp(t_fight, primary)
    died_c = _died_at(elim_c, seed, primary, fight_turn) or (
        post_c is not None and int(post_c) <= 0
    )
    died_t = _died_at(elim_t, seed, primary, fight_turn) or (
        post_t is not None and int(post_t) <= 0
    )
    c_elim = None if primary is None else elim_c.get((seed, int(primary)))
    t_elim = None if primary is None else elim_t.get((seed, int(primary)))
    chain = classify_third_party_chain(
        control_present=c_dec is not None,
        treatment_present=t_dec is not None,
        causal_seat_control=seat_c,
        causal_seat_treatment=seat_t,
        same_fight_pairing=same,
        outcomes_equal=(out_c is not None and out_t is not None and out_c == out_t),
        one_died_other_alive_at_elim_turn=bool(died_c) != bool(died_t),
        applied_equal=(
            applied_c is not None and applied_t is not None
            and applied_c == applied_t
        ),
        elim_turns_equal=(
            c_elim is not None and t_elim is not None and int(c_elim) == int(t_elim)
        ),
    )

    hp_turn = fight_turn
    if chain == "same_seat_earlier_elimination" and primary is not None:
        dated = [t for t in (c_elim, t_elim) if t is not None]
        if dated:
            hp_turn = min(int(t) for t in dated)
        c_hp_fight = None if hp_turn is None else c_fights.get(
            (seed, int(primary), int(hp_turn))
        )
        t_hp_fight = None if hp_turn is None else t_fights.get(
            (seed, int(primary), int(hp_turn))
        )
    else:
        c_hp_fight = c_fight
        t_hp_fight = t_fight
    pre_hp_c, _ = _seat_hp(c_hp_fight, primary)
    pre_hp_t, _ = _seat_hp(t_hp_fight, primary)
    applied_hp_c = _applied_to_seat(c_hp_fight, primary)
    applied_hp_t = _applied_to_seat(t_hp_fight, primary)
    hp_cls = classify_chain_hp_gap(
        chain_class=chain,
        control_fight_present=c_hp_fight is not None,
        treatment_fight_present=t_hp_fight is not None,
        pre_hp_equal=(
            pre_hp_c is not None and pre_hp_t is not None and pre_hp_c == pre_hp_t
        ),
        control_hit=_seat_hit(c_hp_fight, primary),
        treatment_hit=_seat_hit(t_hp_fight, primary),
        applied_equal=(
            applied_hp_c is not None and applied_hp_t is not None
            and applied_hp_c == applied_hp_t
        ),
    )

    c_status = _paired_status(hp_c, elim_c, seed, primary, fight_turn)
    t_status = _paired_status(hp_t, elim_t, seed, primary, fight_turn)
    prior_c = _prior_turn_hp(hp_c, seed, primary, fight_turn)
    prior_t = _prior_turn_hp(hp_t, seed, primary, fight_turn)
    c_trace = _enrich_trace(
        c_fight, primary, elimination_turn=c_elim,
        prior_turn_hp=prior_c,
        paired_status=_paired_status(hp_t, elim_t, seed, primary, fight_turn),
    ) if primary is not None else {}
    t_trace = _enrich_trace(
        t_fight, primary, elimination_turn=t_elim,
        prior_turn_hp=prior_t,
        paired_status=_paired_status(hp_c, elim_c, seed, primary, fight_turn),
    ) if primary is not None else {}

    c_other = None
    t_other = None
    if seat_c is not None and (primary is None or int(seat_c) != int(primary or -1)):
        c_other = _enrich_trace(
            c_fights.get((seed, int(seat_c), int(turn_c))) if turn_c is not None else None,
            seat_c, elimination_turn=elim_c.get((seed, int(seat_c))),
            prior_turn_hp=_prior_turn_hp(hp_c, seed, seat_c, turn_c),
            paired_status=_paired_status(hp_t, elim_t, seed, seat_c, turn_c),
        )
    if seat_t is not None and (primary is None or int(seat_t) != int(primary or -1)):
        t_other = _enrich_trace(
            t_fights.get((seed, int(seat_t), int(turn_t))) if turn_t is not None else None,
            seat_t, elimination_turn=elim_t.get((seed, int(seat_t))),
            prior_turn_hp=_prior_turn_hp(hp_t, seed, seat_t, turn_t),
            paired_status=_paired_status(hp_c, elim_c, seed, seat_t, turn_t),
        )

    causal_ok = False
    if primary is not None and fight_turn is not None:
        dying_fight = t_fight if died_t else (c_fight if died_c else None)
        if dying_fight is None:
            dying_fight = t_fight or c_fight
        _, post = _seat_hp(dying_fight, primary)
        causal_ok = post is not None and int(post) <= 0
    hp_flow_ok = True
    for fight, seat in ((c_fight, primary), (t_fight, primary)):
        if fight is None or seat is None:
            continue
        if not _hp_flow_ok(fight):
            hp_flow_ok = False

    return {
        "seed": seed,
        "turn": leftover_turn,
        "leftover_seat": leftover,
        "class": chain,
        "hp_gap_class": hp_cls,
        "first_divergence_turn": t_div,
        "timing_class": timing.get("class"),
        "pairing_schedule_subtype": timing.get("pairing_schedule_subtype"),
        "control_kind": timing.get("control_kind"),
        "treatment_kind": timing.get("treatment_kind"),
        "named_seats": list(timing.get("named_seats") or []),
        "only_control_alive": list(timing.get("only_control_alive") or []),
        "only_treatment_alive": list(timing.get("only_treatment_alive") or []),
        "causal_seat": primary,
        "control_causal_seat": seat_c,
        "treatment_causal_seat": seat_t,
        "control_elimination_turn": turn_c if seat_c is not None else c_elim,
        "treatment_elimination_turn": turn_t if seat_t is not None else t_elim,
        "decisive_fight_turn": fight_turn,
        "hp_compare_turn": hp_turn,
        "same_fight_pairing": same,
        "control_outcome": out_c,
        "treatment_outcome": out_t,
        "control_pre_hp": pre_c,
        "treatment_pre_hp": pre_t,
        "control_post_hp": post_c,
        "treatment_post_hp": post_t,
        "control_applied": applied_c,
        "treatment_applied": applied_t,
        "control_prior_turn_hp": prior_c,
        "treatment_prior_turn_hp": prior_t,
        "control_hit": _seat_hit(c_fight, primary),
        "treatment_hit": _seat_hit(t_fight, primary),
        "control_alive_at_elim_turn": c_status.get("alive"),
        "treatment_alive_at_elim_turn": t_status.get("alive"),
        "control_eliminated_at_elim_turn": c_status.get("eliminated"),
        "treatment_eliminated_at_elim_turn": t_status.get("eliminated"),
        "causal_elimination_linked": causal_ok,
        "hp_flow_ok": hp_flow_ok,
        "control_decisive": c_trace,
        "treatment_decisive": t_trace,
        "control_other_causal": c_other,
        "treatment_other_causal": t_other,
        "control": _slim_decision(c_dec, leftover),
        "treatment": _slim_decision(t_dec, leftover),
    }


def attribute_elimination_chain(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Dict:
    """Split 3K third-party leftover rows into elimination-chain classes."""
    window = tuple(turns or LATE_TURNS)
    timing_rows = list(iter_third_party_timing_records(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    ))
    c_dec = _index_decisions(control_raw.get("pairing_decisions") or [])
    t_dec = _index_decisions(treatment_raw.get("pairing_decisions") or [])
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    elim_c = _elim_turn_map(control_raw.get("eliminations") or [])
    elim_t = _elim_turn_map(treatment_raw.get("eliminations") or [])
    hp_c = _hp_row_index(control_raw.get("hp_rows") or [])
    hp_t = _hp_row_index(treatment_raw.get("hp_rows") or [])

    mm = attribute_matchmaking(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    )
    from ml.elimination_timing_diagnostic import attribute_elimination_timing
    timing_attr = attribute_elimination_timing(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    )

    counts = Counter()
    hp_counts = Counter()
    subtype_x_class = Counter()
    first_turn_counts = Counter()
    elim_turn_counts = Counter()
    examples: Dict[str, List[Dict]] = {name: [] for name in CHAIN_COMPONENTS}
    hp_examples: Dict[str, List[Dict]] = {name: [] for name in HP_GAP_COMPONENTS}
    compared: List[Dict] = []
    n_linked = 0
    n_hp_flow = 0
    n_checked = 0

    for timing in timing_rows:
        rec = _classify_one_chain(
            timing, c_dec, t_dec, c_fights, t_fights,
            elim_c, elim_t, hp_c, hp_t,
        )
        cls = rec["class"]
        counts[cls] += 1
        subtype_x_class[f"{timing.get('pairing_schedule_subtype')}:{cls}"] += 1
        if rec.get("first_divergence_turn") is not None:
            first_turn_counts[str(rec["first_divergence_turn"])] += 1
        if rec.get("decisive_fight_turn") is not None:
            elim_turn_counts[str(rec["decisive_fight_turn"])] += 1
        hp_cls = rec.get("hp_gap_class")
        if hp_cls:
            hp_counts[hp_cls] += 1
            if len(hp_examples[hp_cls]) < _N_EXAMPLES:
                hp_examples[hp_cls].append(rec)
        if len(examples[cls]) < _N_EXAMPLES:
            examples[cls].append(rec)
        if len(compared) < 64:
            compared.append(rec)
        n_checked += 1
        if rec.get("causal_elimination_linked"):
            n_linked += 1
        if rec.get("hp_flow_ok"):
            n_hp_flow += 1

    third_n = float(len(timing_rows))
    attributed = {name: float(counts.get(name, 0)) for name in CHAIN_COMPONENTS}
    reconstructed = sum(attributed.values())
    shares = {
        name: share_of_third_party(attributed[name], denom=third_n)
        for name in CHAIN_COMPONENTS
    }
    earlier_n = attributed["same_seat_earlier_elimination"]
    hp_attributed = {name: float(hp_counts.get(name, 0)) for name in HP_GAP_COMPONENTS}
    hp_reconstructed = sum(hp_attributed.values())
    hp_shares = {
        name: share_of_hp_gap(hp_attributed[name], denom=earlier_n)
        for name in HP_GAP_COMPONENTS
    }
    return {
        "turns": list(window),
        "trace_from_turn": TRACE_FROM_TURN,
        "n_pairing_schedule": mm.get("n_pairing_schedule"),
        "n_eligibility": timing_attr.get("n_eligibility"),
        "n_third_party": int(third_n),
        "n_leftover_input": len(leftover_rows),
        "published_third_party": PHASE_3K_THIRD_PARTY,
        "third_party_reproduced": int(third_n) == PHASE_3K_THIRD_PARTY,
        "published_eligibility": PHASE_3J_ELIGIBILITY,
        "eligibility_reproduced": timing_attr.get("eligibility_reproduced"),
        "published_pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
        "pairing_schedule_reproduced": mm.get("pairing_schedule_reproduced"),
        "timing_3k_counts": timing_attr.get("counts"),
        "matchmaking_counts": mm.get("counts"),
        "counts": dict(counts),
        "hp_counts": dict(hp_counts),
        "subtype_x_class": dict(subtype_x_class),
        "first_divergence_turn_counts": dict(first_turn_counts),
        "causal_elimination_turn_counts": dict(elim_turn_counts),
        "attributed": attributed,
        "hp_attributed": hp_attributed,
        "reconstructed_third_party_rows": reconstructed,
        "reconciliation_gap": third_n - reconstructed,
        "reconciliation_ok": abs(third_n - reconstructed) <= max(
            1.0, 1e-9 * (1 + third_n)
        ),
        "hp_reconstructed_earlier_rows": hp_reconstructed,
        "hp_reconciliation_gap": earlier_n - hp_reconstructed,
        "hp_reconciliation_ok": abs(earlier_n - hp_reconstructed) <= max(
            1.0, 1e-9 * (1 + earlier_n)
        ),
        "n_same_seat_earlier": int(earlier_n),
        "n_row_elim_checked": n_checked,
        "n_causal_elimination_linked": n_linked,
        "n_row_hp_flow_ok": n_hp_flow,
        "row_elim_ok": n_checked == 0 or n_linked == n_checked,
        "row_hp_flow_ok": n_checked == 0 or n_hp_flow == n_checked,
        **{f"share_{k}": v for k, v in shares.items()},
        **{f"share_{k}": v for k, v in hp_shares.items()},
        "examples": examples,
        "hp_examples": hp_examples,
        "n_compared_kept": len(compared),
        "candidate_choice_ok": mm.get("candidate_choice_ok"),
        "matchmaking_reconciliation_ok": mm.get("reconciliation_ok"),
        "timing_reconciliation_ok": timing_attr.get("reconciliation_ok"),
    }


def compare_chain(
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
) -> Dict:
    """3K third-party lock + causal elimination-chain split."""
    if timing is None:
        timing = compare_elimination(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, retention=retention, pairing=pairing,
            matchmaking=matchmaking,
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
    late = attribute_elimination_chain(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late = attribute_elimination_chain(
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
    mm_rec = (timing.get("reconciliation") or {})
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
        "chain_reconcile_identity": CHAIN_RECONCILE_IDENTITY,
        "chain_hp_reconcile_identity": CHAIN_HP_RECONCILE_IDENTITY,
        "row_elim_hp_identity": ROW_ELIM_HP_IDENTITY,
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "matchmaking_reconciliation_ok": mm_rec.get("matchmaking_reconciliation_ok"),
        "eligibility_n": late.get("n_eligibility"),
        "eligibility_reproduced": late.get("eligibility_reproduced"),
        "third_party_n": late.get("n_third_party"),
        "third_party_reproduced": late.get("third_party_reproduced"),
        "timing_reconciliation_ok": late.get("timing_reconciliation_ok"),
        "chain_reconciliation_ok": late.get("reconciliation_ok"),
        "hp_gap_reconciliation_ok": late.get("hp_reconciliation_ok"),
        "row_elim_ok": late.get("row_elim_ok"),
        "row_hp_flow_ok": late.get("row_hp_flow_ok"),
        "hp_flow_control": hp_c,
        "hp_flow_treatment": hp_t,
        "hp_flow_ok": bool(hp_c.get("ok") and hp_t.get("ok")),
        "elimination_control": elim_c,
        "elimination_treatment": elim_t,
        "elimination_ok": bool(elim_c.get("ok") and elim_t.get("ok")),
        "candidate_choice_ok": late.get("candidate_choice_ok"),
        "phase_3g_mixture_reproduced": (
            timing.get("decomposition_3g") or {}
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
        "timing_3k": timing.get("attribution"),
        "matchmaking_3j": timing.get("matchmaking_3j"),
        "pairing_3i": timing.get("pairing_3i"),
        "leftover_3h": timing.get("leftover_3h"),
        "reconciliation": rec,
        "decomposition_3g": timing.get("decomposition_3g"),
        "paired_seats": timing.get("paired_seats"),
        "timing_3f": timing.get("timing_3f"),
        "lifecycle": timing.get("lifecycle"),
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
        "published_3k_locks": {
            "third_party": PHASE_3K_THIRD_PARTY,
            "treatment_earlier": PHASE_3K_TREATMENT_EARLIER,
            "control_opponent": PHASE_3K_CONTROL_OPPONENT,
            "unreconciled": PHASE_3K_UNRECONCILED,
            "named": PHASE_3K_NAMED,
            "share_third_party": PHASE_3K_SHARE_THIRD_PARTY,
            "prior_hp": PHASE_3K_PRIOR_HP,
            "share_prior_hp": PHASE_3K_SHARE_PRIOR_HP,
        },
    }
