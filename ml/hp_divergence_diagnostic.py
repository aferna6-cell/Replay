"""Phase 3M — observational earliest same-seat HP divergence attribution.

Reuses the 3L elimination-chain tracer on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For each 3L same-seat earlier-elimination punch row, walks the causal
seat from the earliest recorded turn through the earlier elimination
and stops at the first paired pre-combat or post-combat HP split.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ml.elimination_chain_diagnostic import (
    _classify_one_chain,
    _hp_row_index,
    compare_chain,
    iter_third_party_timing_records,
)
from ml.elimination_timing_diagnostic import (
    EliminationTimingTracer,
    _applied_to_seat,
    _eligibility_differs,
    _elim_turn_map,
    _hp_flow_ok,
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
)
from ml.pairing_who_wins_diagnostic import (
    _index_seat_fights,
    _kind_of,
    _opponent_of,
    same_pairing,
)
from ml.phase_3m_prereg import (
    CANDIDATE_CHOICE_IDENTITY,
    CHAIN_HP_RECONCILE_IDENTITY,
    CHAIN_RECONCILE_IDENTITY,
    ELIGIBILITY_TIMING_IDENTITY,
    ELIMINATION_IDENTITY,
    FIRST_DIVERGENCE_COMPONENTS,
    FIRST_DIVERGENCE_RECONCILE_IDENTITY,
    FLOW_ABS_TOL,
    HISTORY_LINK_IDENTITY,
    HP_FLOW_IDENTITY,
    HP_GAP_RECONCILE_IDENTITY,
    HP_WALK_FROM_TURN,
    IMPACT_ATTACK_IDENTITY,
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
    PHASE_3L_CASCADE,
    PHASE_3L_DAMAGE_THRESHOLD,
    PHASE_3L_HIT,
    PHASE_3L_MAG,
    PHASE_3L_OUTCOME_FLIP,
    PHASE_3L_PRIOR_HP,
    PHASE_3L_SAME_SEAT_EARLIER,
    PHASE_3L_SHARE_EARLIER,
    PHASE_3L_SHARE_PRIOR_HP,
    PHASE_3L_UNRECONCILED,
    POOL_FLOW_IDENTITY,
    REWEIGHT_ABS_TOL,
    ROW_ELIM_HP_IDENTITY,
    ROW_HISTORY_DIVERGENCE_IDENTITY,
    TRACE_FROM_TURN,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    classify_first_divergence,
    share_of_class1,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.carry_divergence_diagnostic import reconcile_history_links

METHODOLOGY_VERSION = "3m_v1"

_N_EXAMPLES = 8


def _safe_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _hp_observed(pre, post) -> bool:
    return pre is not None or post is not None


def _vals_differ(a, b) -> bool:
    if a is None or b is None:
        return False
    return int(a) != int(b)


def _pairing_equal(
    c_fight: Optional[Dict],
    t_fight: Optional[Dict],
    seat,
    c_dec: Optional[Dict],
    t_dec: Optional[Dict],
) -> bool:
    """Same live opponent, or same ghost/bye kind, without an alive-set split."""
    if c_fight is None or t_fight is None:
        return False
    alive_set_differs = (
        c_dec is not None and t_dec is not None
        and _eligibility_differs(c_dec, t_dec)
    )
    if same_pairing(c_fight, t_fight, seat):
        return not alive_set_differs
    kc, kt = _kind_of(c_fight), _kind_of(t_fight)
    if kc in ("bye", "ghost") and kc == kt:
        return not alive_set_differs
    return False


def _obs_at_turn(
    fights: Dict,
    hp_index: Dict[Tuple[int, int, int], Dict],
    seed: int,
    seat: int,
    turn: int,
    prev_post: Optional[int],
) -> Tuple[Optional[Dict], Optional[int], Optional[int]]:
    fight = fights.get((seed, seat, turn))
    pre, post = _seat_hp(fight, seat)
    row = hp_index.get((seed, seat, turn))
    row_hp = None if not row else _safe_int(row.get("hp"))
    if pre is None:
        pre = prev_post
    if post is None:
        if row_hp is not None:
            post = row_hp
        elif fight is None:
            post = pre
    return fight, pre, post


def _earlier_elim_turn(chain: Dict) -> Optional[int]:
    dated = [
        _safe_int(chain.get("control_elimination_turn")),
        _safe_int(chain.get("treatment_elimination_turn")),
    ]
    dated = [t for t in dated if t is not None]
    if dated:
        return min(dated)
    return _safe_int(chain.get("hp_compare_turn") or chain.get("decisive_fight_turn"))


def find_first_hp_divergence(
    chain: Dict,
    c_fights: Dict,
    t_fights: Dict,
    hp_c: Dict[Tuple[int, int, int], Dict],
    hp_t: Dict[Tuple[int, int, int], Dict],
    c_decisions: Dict[Tuple[int, int], Dict],
    t_decisions: Dict[Tuple[int, int], Dict],
    *,
    from_turn: int = HP_WALK_FROM_TURN,
) -> Dict:
    """Walk the causal seat to the first paired pre/post HP split."""
    seed = _safe_int(chain.get("seed"))
    seat = _safe_int(chain.get("causal_seat"))
    earlier = _earlier_elim_turn(chain)
    leftover = _safe_int(chain.get("leftover_seat"))
    leftover_turn = _safe_int(chain.get("turn"))
    empty = {
        "seed": seed,
        "turn": leftover_turn,
        "leftover_seat": leftover,
        "causal_seat": seat,
        "earlier_elimination_turn": earlier,
        "chain_class": chain.get("class"),
        "hp_gap_class": chain.get("hp_gap_class"),
        "class": "unreconciled",
        "first_divergence_turn": None,
        "found_event": False,
        "control_obs": False,
        "treatment_obs": False,
        "history_linked": seat is not None and earlier is not None,
        "hp_flow_ok": True,
        "control": {},
        "treatment": {},
    }
    if seed is None or seat is None or earlier is None:
        return empty

    start = int(from_turn)
    end = int(earlier)
    if end < start:
        return empty

    prev_c = None
    prev_t = None
    for turn in range(start, end + 1):
        c_fight, pre_c, post_c = _obs_at_turn(
            c_fights, hp_c, seed, seat, turn, prev_c,
        )
        t_fight, pre_t, post_t = _obs_at_turn(
            t_fights, hp_t, seed, seat, turn, prev_t,
        )
        c_obs = _hp_observed(pre_c, post_c)
        t_obs = _hp_observed(pre_t, post_t)
        pre_differs = _vals_differ(pre_c, pre_t)
        post_differs = _vals_differ(post_c, post_t)
        if c_obs and t_obs and (pre_differs or post_differs):
            c_dec = c_decisions.get((seed, turn))
            t_dec = t_decisions.get((seed, turn))
            both_fights = c_fight is not None and t_fight is not None
            one_fight = (c_fight is None) != (t_fight is None)
            if both_fights:
                pairing_eq = _pairing_equal(c_fight, t_fight, seat, c_dec, t_dec)
            elif one_fight:
                pairing_eq = False
            else:
                pairing_eq = True
            out_c = _seat_outcome(c_fight, seat)
            out_t = _seat_outcome(t_fight, seat)
            applied_c = _applied_to_seat(c_fight, seat)
            applied_t = _applied_to_seat(t_fight, seat)
            if applied_c is None and pre_c is not None and post_c is not None:
                applied_c = int(pre_c) - int(post_c)
            if applied_t is None and pre_t is not None and post_t is not None:
                applied_t = int(pre_t) - int(post_t)
            cls = classify_first_divergence(
                found_event=True,
                control_obs=c_obs,
                treatment_obs=t_obs,
                pre_hp_equal=not pre_differs,
                pairing_equal=pairing_eq,
                outcomes_equal=(
                    out_c is not None and out_t is not None and out_c == out_t
                ),
                applied_equal=(
                    applied_c is not None and applied_t is not None
                    and applied_c == applied_t
                ),
                paired_fights_present=both_fights,
            )
            hp_flow_ok = True
            for fight in (c_fight, t_fight):
                if fight is None:
                    continue
                if not _hp_flow_ok(fight):
                    hp_flow_ok = False
            c_trace = _seat_trace(
                c_fight, seat,
                elimination_turn=_safe_int(chain.get("control_elimination_turn")),
            )
            t_trace = _seat_trace(
                t_fight, seat,
                elimination_turn=_safe_int(chain.get("treatment_elimination_turn")),
            )
            if c_fight is None:
                c_trace["pre_combat_hp"] = pre_c
                c_trace["post_combat_hp"] = post_c
                c_trace["applied_damage"] = applied_c
                c_trace["turn"] = turn
            if t_fight is None:
                t_trace["pre_combat_hp"] = pre_t
                t_trace["post_combat_hp"] = post_t
                t_trace["applied_damage"] = applied_t
                t_trace["turn"] = turn
            return {
                "seed": seed,
                "turn": leftover_turn,
                "leftover_seat": leftover,
                "causal_seat": seat,
                "earlier_elimination_turn": earlier,
                "chain_class": chain.get("class"),
                "hp_gap_class": chain.get("hp_gap_class"),
                "class": cls,
                "first_divergence_turn": turn,
                "found_event": True,
                "control_obs": True,
                "treatment_obs": True,
                "history_linked": True,
                "pre_hp_equal": not pre_differs,
                "post_hp_equal": not post_differs,
                "pairing_equal": pairing_eq,
                "same_fight_pairing": same_pairing(c_fight, t_fight, seat),
                "control_kind": _kind_of(c_fight),
                "treatment_kind": _kind_of(t_fight),
                "control_opponent": _opponent_of(c_fight, seat),
                "treatment_opponent": _opponent_of(t_fight, seat),
                "control_outcome": out_c,
                "treatment_outcome": out_t,
                "control_pre_hp": pre_c,
                "treatment_pre_hp": pre_t,
                "control_post_hp": post_c,
                "treatment_post_hp": post_t,
                "control_applied": applied_c,
                "treatment_applied": applied_t,
                "control_survivor_count": c_trace.get("survivor_count"),
                "treatment_survivor_count": t_trace.get("survivor_count"),
                "control_survivor_tier_sum": c_trace.get("survivor_tier_sum"),
                "treatment_survivor_tier_sum": t_trace.get("survivor_tier_sum"),
                "control_tavern_tier": c_trace.get("tavern_tier"),
                "treatment_tavern_tier": t_trace.get("tavern_tier"),
                "control_board_recruit_raw": c_trace.get("board_recruit_raw"),
                "treatment_board_recruit_raw": t_trace.get("board_recruit_raw"),
                "control_abstract_pool_raw": c_trace.get("abstract_pool_raw"),
                "treatment_abstract_pool_raw": t_trace.get("abstract_pool_raw"),
                "control_total_combat_raw": c_trace.get("total_combat_raw"),
                "treatment_total_combat_raw": t_trace.get("total_combat_raw"),
                "hp_flow_ok": hp_flow_ok,
                "control_decisive": c_trace,
                "treatment_decisive": t_trace,
                "control": _slim_decision(c_dec, leftover),
                "treatment": _slim_decision(t_dec, leftover),
            }
        if post_c is not None:
            prev_c = post_c
        elif pre_c is not None:
            prev_c = pre_c
        if post_t is not None:
            prev_t = post_t
        elif pre_t is not None:
            prev_t = pre_t

    empty["history_linked"] = True
    return empty


def iter_class1_chain_records(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Iterable[Dict]:
    """Yield 3L-classified third-party records that are class-(1)."""
    window = tuple(turns or LATE_TURNS)
    c_dec = _index_decisions(control_raw.get("pairing_decisions") or [])
    t_dec = _index_decisions(treatment_raw.get("pairing_decisions") or [])
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    elim_c = _elim_turn_map(control_raw.get("eliminations") or [])
    elim_t = _elim_turn_map(treatment_raw.get("eliminations") or [])
    hp_c = _hp_row_index(control_raw.get("hp_rows") or [])
    hp_t = _hp_row_index(treatment_raw.get("hp_rows") or [])
    for timing in iter_third_party_timing_records(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    ):
        rec = _classify_one_chain(
            timing, c_dec, t_dec, c_fights, t_fights,
            elim_c, elim_t, hp_c, hp_t,
        )
        if rec.get("class") == "same_seat_earlier_elimination":
            yield rec


def attribute_first_divergence(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Dict:
    """Split 3L class-(1) rows into first-divergence classes."""
    window = tuple(turns or LATE_TURNS)
    chain_cmp = None
    class1_rows = list(iter_class1_chain_records(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    ))
    c_dec = _index_decisions(control_raw.get("pairing_decisions") or [])
    t_dec = _index_decisions(treatment_raw.get("pairing_decisions") or [])
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    hp_c = _hp_row_index(control_raw.get("hp_rows") or [])
    hp_t = _hp_row_index(treatment_raw.get("hp_rows") or [])

    from ml.elimination_chain_diagnostic import attribute_elimination_chain
    chain_attr = attribute_elimination_chain(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
        turns=window,
    )

    counts = Counter()
    turn_counts = Counter()
    examples: Dict[str, List[Dict]] = {
        name: [] for name in FIRST_DIVERGENCE_COMPONENTS
    }
    compared: List[Dict] = []
    n_linked = 0
    n_hp_flow = 0
    n_history = 0
    n_checked = 0

    for chain in class1_rows:
        rec = find_first_hp_divergence(
            chain, c_fights, t_fights, hp_c, hp_t, c_dec, t_dec,
        )
        cls = rec["class"]
        counts[cls] += 1
        if rec.get("first_divergence_turn") is not None:
            turn_counts[str(rec["first_divergence_turn"])] += 1
        if len(examples[cls]) < _N_EXAMPLES:
            examples[cls].append(rec)
        if len(compared) < 64:
            compared.append(rec)
        n_checked += 1
        if rec.get("found_event"):
            n_linked += 1
        if rec.get("hp_flow_ok"):
            n_hp_flow += 1
        if rec.get("history_linked"):
            n_history += 1

    class1_n = float(len(class1_rows))
    attributed = {
        name: float(counts.get(name, 0)) for name in FIRST_DIVERGENCE_COMPONENTS
    }
    reconstructed = sum(attributed.values())
    shares = {
        name: share_of_class1(attributed[name], denom=class1_n)
        for name in FIRST_DIVERGENCE_COMPONENTS
    }
    return {
        "turns": list(window),
        "trace_from_turn": TRACE_FROM_TURN,
        "hp_walk_from_turn": HP_WALK_FROM_TURN,
        "n_pairing_schedule": chain_attr.get("n_pairing_schedule"),
        "n_eligibility": chain_attr.get("n_eligibility"),
        "n_third_party": chain_attr.get("n_third_party"),
        "n_same_seat_earlier": int(class1_n),
        "n_leftover_input": len(leftover_rows),
        "published_third_party": PHASE_3K_THIRD_PARTY,
        "third_party_reproduced": chain_attr.get("third_party_reproduced"),
        "published_same_seat_earlier": PHASE_3L_SAME_SEAT_EARLIER,
        "same_seat_earlier_reproduced": int(class1_n) == PHASE_3L_SAME_SEAT_EARLIER,
        "published_eligibility": PHASE_3J_ELIGIBILITY,
        "eligibility_reproduced": chain_attr.get("eligibility_reproduced"),
        "published_pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
        "pairing_schedule_reproduced": chain_attr.get("pairing_schedule_reproduced"),
        "chain_3l_counts": chain_attr.get("counts"),
        "chain_3l_hp_counts": chain_attr.get("hp_counts"),
        "timing_3k_counts": chain_attr.get("timing_3k_counts"),
        "matchmaking_counts": chain_attr.get("matchmaking_counts"),
        "counts": dict(counts),
        "first_divergence_turn_counts": dict(turn_counts),
        "attributed": attributed,
        "reconstructed_class1_rows": reconstructed,
        "reconciliation_gap": class1_n - reconstructed,
        "reconciliation_ok": abs(class1_n - reconstructed) <= max(
            1.0, 1e-9 * (1 + class1_n)
        ),
        "n_row_checked": n_checked,
        "n_first_divergence_linked": n_linked,
        "n_row_hp_flow_ok": n_hp_flow,
        "n_history_linked": n_history,
        "row_divergence_ok": n_checked == 0 or n_linked == n_checked,
        "row_hp_flow_ok": n_checked == 0 or n_hp_flow == n_checked,
        "row_history_ok": n_checked == 0 or n_history == n_checked,
        **{f"share_{k}": v for k, v in shares.items()},
        "examples": examples,
        "n_compared_kept": len(compared),
        "candidate_choice_ok": chain_attr.get("candidate_choice_ok"),
        "matchmaking_reconciliation_ok": chain_attr.get("matchmaking_reconciliation_ok"),
        "timing_reconciliation_ok": chain_attr.get("timing_reconciliation_ok"),
        "chain_reconciliation_ok": chain_attr.get("reconciliation_ok"),
        "hp_gap_reconciliation_ok": chain_attr.get("hp_reconciliation_ok"),
        "_chain_cmp": chain_cmp,
    }


def compare_first_divergence(
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
) -> Dict:
    """3L class-(1) lock + earliest HP-divergence split."""
    if chain is None:
        chain = compare_chain(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, retention=retention, pairing=pairing,
            matchmaking=matchmaking, timing=timing,
        )
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
    late = attribute_first_divergence(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late = attribute_first_divergence(
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
    chain_rec = chain.get("reconciliation") or {}
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
        "first_divergence_reconcile_identity": FIRST_DIVERGENCE_RECONCILE_IDENTITY,
        "row_history_divergence_identity": ROW_HISTORY_DIVERGENCE_IDENTITY,
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "matchmaking_reconciliation_ok": chain_rec.get("matchmaking_reconciliation_ok"),
        "eligibility_n": late.get("n_eligibility"),
        "eligibility_reproduced": late.get("eligibility_reproduced"),
        "third_party_n": late.get("n_third_party"),
        "third_party_reproduced": late.get("third_party_reproduced"),
        "same_seat_earlier_n": late.get("n_same_seat_earlier"),
        "same_seat_earlier_reproduced": late.get("same_seat_earlier_reproduced"),
        "timing_reconciliation_ok": late.get("timing_reconciliation_ok"),
        "chain_reconciliation_ok": late.get("chain_reconciliation_ok"),
        "hp_gap_reconciliation_ok": late.get("hp_gap_reconciliation_ok"),
        "first_divergence_reconciliation_ok": late.get("reconciliation_ok"),
        "row_divergence_ok": late.get("row_divergence_ok"),
        "row_hp_flow_ok": late.get("row_hp_flow_ok"),
        "row_history_ok": late.get("row_history_ok"),
        "row_elim_ok": chain_rec.get("row_elim_ok"),
        "hp_flow_control": hp_c,
        "hp_flow_treatment": hp_t,
        "hp_flow_ok": bool(hp_c.get("ok") and hp_t.get("ok")),
        "elimination_control": elim_c,
        "elimination_treatment": elim_t,
        "elimination_ok": bool(elim_c.get("ok") and elim_t.get("ok")),
        "candidate_choice_ok": late.get("candidate_choice_ok"),
        "phase_3g_mixture_reproduced": (
            chain.get("decomposition_3g") or {}
        ).get("mixture_turn_winner_tier"),
        "flow_abs_tol": FLOW_ABS_TOL,
        "reweight_abs_tol": REWEIGHT_ABS_TOL,
        "lineage_control": chain_rec.get("lineage_control"),
        "lineage_treatment": chain_rec.get("lineage_treatment"),
        "paired": chain_rec.get("paired"),
        "reproduced_3d_board_pool_magnitude": chain_rec.get(
            "reproduced_3d_board_pool_magnitude"
        ),
        "reproduced_3e_carry_share": chain_rec.get("reproduced_3e_carry_share"),
        "flow_mismatches_control": chain_rec.get("flow_mismatches_control"),
        "flow_mismatches_treatment": chain_rec.get("flow_mismatches_treatment"),
    }
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": late,
        "very_late_attribution": very_late,
        "chain_3l": chain.get("attribution"),
        "timing_3k": chain.get("timing_3k"),
        "matchmaking_3j": chain.get("matchmaking_3j"),
        "pairing_3i": chain.get("pairing_3i"),
        "leftover_3h": chain.get("leftover_3h"),
        "reconciliation": rec,
        "decomposition_3g": chain.get("decomposition_3g"),
        "paired_seats": chain.get("paired_seats"),
        "timing_3f": chain.get("timing_3f"),
        "lifecycle": chain.get("lifecycle"),
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
        "published_3l_locks": {
            "same_seat_earlier": PHASE_3L_SAME_SEAT_EARLIER,
            "cascade": PHASE_3L_CASCADE,
            "outcome_flip": PHASE_3L_OUTCOME_FLIP,
            "damage_threshold": PHASE_3L_DAMAGE_THRESHOLD,
            "unreconciled": PHASE_3L_UNRECONCILED,
            "share_earlier": PHASE_3L_SHARE_EARLIER,
            "prior_hp": PHASE_3L_PRIOR_HP,
            "hit": PHASE_3L_HIT,
            "mag": PHASE_3L_MAG,
            "share_prior_hp": PHASE_3L_SHARE_PRIOR_HP,
        },
    }


__all__ = [
    "EliminationTimingTracer",
    "attribute_first_divergence",
    "compare_first_divergence",
    "find_first_hp_divergence",
    "iter_class1_chain_records",
    "run_greedy_2s_treatment_elimination",
    "run_greedy_control_elimination",
]
