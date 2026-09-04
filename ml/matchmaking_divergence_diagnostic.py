"""Phase 3J — observational matchmaking divergence attribution.

Reuses the 3I PairingWhoWinsTracer on consumed DEV 14200–14699. Does not
change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

At every T10–T14 pairing decision, logs the pre-pair alive-seat set,
ghost/bye eligibility, prior live-opponent history, legal candidate set
(the algorithm's actual options: other alive seats plus ghost/bye when
the lobby is odd), pairing RNG digest/index, shuffled order, and chosen
opponent. Restricts attribution to the 3I pairing-schedule leftover
(5952 punch rows) and splits those rows exclusively.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.board_retention_diagnostic import (
    _late_t1t3_rows,
    attribute_late_t1t3_collapse,
    collect_3h_leftover_rows,
    compare_retention,
)
from ml.carry_divergence_diagnostic import (
    compare_divergence,
    reconcile_history_links,
)
from ml.pairing_who_wins_diagnostic import (
    PairingWhoWinsTracer,
    _index_seat_fights,
    _kind_of,
    _opponent_of,
    _punch_key,
    attribute_leftover_pairing,
    classify_pairing_gap,
    same_pairing,
    treatment_won,
)
from ml.phase_3j_prereg import (
    BYE_TOKEN,
    CANDIDATE_CHOICE_IDENTITY,
    FLOW_ABS_TOL,
    GHOST_TOKEN,
    HISTORY_LINK_IDENTITY,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    MATCHMAKING_COMPONENTS,
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
    POOL_FLOW_IDENTITY,
    REWEIGHT_ABS_TOL,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_matchmaking_gap,
    share_of_schedule,
)
from ml.pool_lifecycle_diagnostic import compare_lifecycle, summarize_lifecycle_arm
from ml.punch_selection_diagnostic import collect_punch_sample_rows, compare_selection

METHODOLOGY_VERSION = "3j_v1"

_N_EXAMPLES = 8


def rng_state_meta(state) -> Dict:
    """Observational digest + MT index. Does not consume RNG."""
    digest = hashlib.sha256(repr(state).encode("utf-8")).hexdigest()
    index = None
    if isinstance(state, tuple) and len(state) >= 3:
        try:
            index = int(state[2])
        except (TypeError, ValueError):
            index = None
    return {"rng_state_digest": digest, "rng_index": index}


def ghost_bye_eligibility(alive_seats: Sequence, dead_with_board_seats: Sequence) -> Dict:
    n_alive = len(list(alive_seats))
    odd = n_alive % 2 == 1
    has_ghost_board = len(list(dead_with_board_seats)) > 0
    return {
        "n_alive": n_alive,
        "odd_alive": odd,
        "ghost_eligible": bool(odd and has_ghost_board),
        "bye_eligible": bool(odd and not has_ghost_board),
        "history_filters_applied": False,
    }


def legal_candidates_for_seat(
    seat,
    *,
    alive_seats: Sequence,
    ghost_eligible: bool,
    bye_eligible: bool,
) -> List:
    """Algorithm legal set: other alive seats + ghost/bye iff the lobby is odd."""
    try:
        seat_i = int(seat)
    except (TypeError, ValueError):
        return []
    cands: List = []
    for other in alive_seats:
        try:
            other_i = int(other)
        except (TypeError, ValueError):
            continue
        if other_i != seat_i:
            cands.append(other_i)
    cands.sort()
    if ghost_eligible:
        cands.append(GHOST_TOKEN)
    elif bye_eligible:
        cands.append(BYE_TOKEN)
    return cands


def history_constrained_candidates(
    seat,
    *,
    alive_seats: Sequence,
    prior_opponents: Sequence,
    ghost_eligible: bool,
    bye_eligible: bool,
) -> List:
    """Observational no-repeat set. The pairing algorithm does not apply this."""
    try:
        prior = {int(x) for x in (prior_opponents or [])}
    except (TypeError, ValueError):
        prior = set()
    legal = legal_candidates_for_seat(
        seat,
        alive_seats=alive_seats,
        ghost_eligible=ghost_eligible,
        bye_eligible=bye_eligible,
    )
    out = []
    for c in legal:
        if c in (GHOST_TOKEN, BYE_TOKEN):
            out.append(c)
            continue
        try:
            if int(c) not in prior:
                out.append(int(c))
        except (TypeError, ValueError):
            continue
    return out


def chosen_opponent_for_seat(seat, pairs: Sequence, *, ghost_eligible: bool,
                             bye_eligible: bool):
    try:
        seat_i = int(seat)
    except (TypeError, ValueError):
        return None
    for pair in pairs or []:
        if not pair or len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        try:
            a_i = None if a in (None, "") else int(a)
        except (TypeError, ValueError):
            a_i = None
        try:
            b_i = None if b in (None, "") else int(b)
        except (TypeError, ValueError):
            b_i = None
        if a_i == seat_i:
            if b_i is None:
                if ghost_eligible:
                    return GHOST_TOKEN
                if bye_eligible:
                    return BYE_TOKEN
                return None
            return b_i
        if b_i == seat_i:
            return a_i
    return None


def choice_in_candidates(chosen, candidates: Sequence) -> bool:
    if chosen is None:
        return False
    for c in candidates or []:
        if c == chosen:
            return True
        try:
            if int(c) == int(chosen):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _canon_set(xs: Sequence) -> Tuple:
    out = []
    for x in xs or []:
        if x in (GHOST_TOKEN, BYE_TOKEN):
            out.append(x)
            continue
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            out.append(str(x))
    return tuple(sorted(out, key=lambda v: (isinstance(v, str), str(v))))


class MatchmakingDivergenceTracer(PairingWhoWinsTracer):
    """3I pairing stamps plus pre-combat pairing-decision logs."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.pairing_decisions: List[Dict] = []
        self._prior_live_opponents: Dict[int, List[int]] = defaultdict(list)

    def attach_to_env(self, env: BGEnv) -> None:
        super().attach_to_env(env)
        env.pairing_audit_hook = self.on_pairing

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._prior_live_opponents.clear()
        # PairingWhoWinsTracer does not clear fights; decisions are per-lobby
        # appends on the shared list across lobbies, same as fights.

    def on_pairing(self, env: BGEnv, rec: Dict) -> None:
        turn = int(rec.get("turn") or getattr(env, "turn", 0) or 0)
        alive = [int(s) for s in (rec.get("alive_seats") or [])]
        dead_boards = [int(s) for s in (rec.get("dead_with_board_seats") or [])]
        elig = ghost_bye_eligibility(alive, dead_boards)
        pre_meta = rng_state_meta(rec.get("rng_state_pre"))
        post_meta = rng_state_meta(rec.get("rng_state_post"))
        pairs = [tuple(p) for p in (rec.get("pairs") or [])]
        order = [int(s) for s in (rec.get("shuffled_order") or [])]
        per_seat: Dict[str, Dict] = {}
        for seat in alive:
            legal = legal_candidates_for_seat(
                seat,
                alive_seats=alive,
                ghost_eligible=elig["ghost_eligible"],
                bye_eligible=elig["bye_eligible"],
            )
            prior = list(self._prior_live_opponents.get(int(seat), []))
            chosen = chosen_opponent_for_seat(
                seat, pairs,
                ghost_eligible=elig["ghost_eligible"],
                bye_eligible=elig["bye_eligible"],
            )
            try:
                p_index = order.index(int(seat))
            except ValueError:
                p_index = None
            per_seat[str(int(seat))] = {
                "legal_candidates": legal,
                "prior_opponents": prior,
                "history_constrained_candidates": history_constrained_candidates(
                    seat,
                    alive_seats=alive,
                    prior_opponents=prior,
                    ghost_eligible=elig["ghost_eligible"],
                    bye_eligible=elig["bye_eligible"],
                ),
                "chosen": chosen,
                "choice_in_candidates": choice_in_candidates(chosen, legal),
                "pairing_index": p_index,
            }
        self.pairing_decisions.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "turn": turn,
            "alive_seats": alive,
            "dead_with_board_seats": dead_boards,
            "n_alive": elig["n_alive"],
            "odd_alive": elig["odd_alive"],
            "ghost_eligible": elig["ghost_eligible"],
            "bye_eligible": elig["bye_eligible"],
            "history_filters_applied": False,
            "rng_state_digest_pre": pre_meta["rng_state_digest"],
            "rng_index_pre": pre_meta["rng_index"],
            "rng_state_digest_post": post_meta["rng_state_digest"],
            "rng_index_post": post_meta["rng_index"],
            "shuffled_order": order,
            "pairs": [list(p) for p in pairs],
            "per_seat": per_seat,
        })

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        if _kind_of(fight) != "live":
            return
        sa = fight.get("seat_a")
        sb = fight.get("seat_b")
        try:
            a_i = int(sa)
            b_i = int(sb)
        except (TypeError, ValueError):
            return
        self._prior_live_opponents[a_i].append(b_i)
        self._prior_live_opponents[b_i].append(a_i)


def run_matchmaking_arm(
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

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = MatchmakingDivergenceTracer(i, seed + i, arm)
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
    }


def run_greedy_control_matchmaking(lobbies: int, seed: int) -> Dict:
    return run_matchmaking_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_matchmaking(lobbies: int, seed: int) -> Dict:
    return run_matchmaking_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _index_decisions(decisions: Sequence[Dict]) -> Dict[Tuple[int, int], Dict]:
    out: Dict[Tuple[int, int], Dict] = {}
    for rec in decisions or []:
        try:
            seed = int(rec["seed"])
            turn = int(rec["turn"])
        except (KeyError, TypeError, ValueError):
            continue
        out[(seed, turn)] = rec
    return out


def _seat_view(decision: Optional[Dict], seat) -> Optional[Dict]:
    if not decision or seat in (None, ""):
        return None
    try:
        return (decision.get("per_seat") or {}).get(str(int(seat)))
    except (TypeError, ValueError):
        return None


def _slim_decision(decision: Optional[Dict], seat) -> Dict:
    if not decision:
        return {}
    view = _seat_view(decision, seat) or {}
    return {
        "alive_seats": list(decision.get("alive_seats") or []),
        "dead_with_board_seats": list(decision.get("dead_with_board_seats") or []),
        "n_alive": decision.get("n_alive"),
        "odd_alive": decision.get("odd_alive"),
        "ghost_eligible": decision.get("ghost_eligible"),
        "bye_eligible": decision.get("bye_eligible"),
        "history_filters_applied": bool(decision.get("history_filters_applied")),
        "rng_state_digest_pre": decision.get("rng_state_digest_pre"),
        "rng_index_pre": decision.get("rng_index_pre"),
        "shuffled_order": list(decision.get("shuffled_order") or []),
        "pairs": list(decision.get("pairs") or []),
        "legal_candidates": list(view.get("legal_candidates") or []),
        "prior_opponents": list(view.get("prior_opponents") or []),
        "history_constrained_candidates": list(
            view.get("history_constrained_candidates") or []
        ),
        "chosen": view.get("chosen"),
        "choice_in_candidates": bool(view.get("choice_in_candidates")),
        "pairing_index": view.get("pairing_index"),
    }


def iter_pairing_schedule_rows(
    leftover_rows: Sequence[Dict],
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    treatment_punch: Sequence[Dict],
    turns=None,
) -> List[Dict]:
    """Reproduce 3I pairing-schedule leftover punch rows (exclusive)."""
    window = tuple(turns or LATE_TURNS)
    t_late = _late_t1t3_rows(treatment_punch, window)
    t_punch_by_key: Dict[Tuple[int, int, int], int] = Counter()
    for row in t_late:
        key = _punch_key(row)
        if key is not None:
            t_punch_by_key[key] += 1
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    grouped: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    for row in leftover_rows:
        key = _punch_key(row)
        if key is None:
            grouped[(-1, -1, -1)].append(row)
            continue
        grouped[key].append(row)

    out: List[Dict] = []
    for key, rows in grouped.items():
        if key == (-1, -1, -1):
            continue
        seed_i, seat_i, turn_i = key
        c_fight = c_fights.get(key)
        t_fight = t_fights.get(key)
        same = same_pairing(c_fight, t_fight, seat_i)
        t_wins = treatment_won(t_fight, seat_i)
        t_n = int(t_punch_by_key.get(key, 0))
        n_cover = t_n if (same and t_wins) else 0
        for i, row in enumerate(rows):
            uncovered = i >= n_cover
            cls = classify_pairing_gap(
                same_pairing=same,
                treatment_wins=t_wins,
                treatment_tie_or_loss=same and not t_wins,
                treatment_t1t3_punches=t_n,
                uncovered=uncovered,
            )
            if cls != "pairing_schedule":
                continue
            subtype = "different_opponent"
            if c_fight is None or t_fight is None:
                subtype = "missing_fight"
            elif _kind_of(c_fight) != "live" or _kind_of(t_fight) != "live":
                subtype = "kind_mismatch"
            out.append({
                "seed": seed_i,
                "turn": turn_i,
                "winner_seat": seat_i,
                "class": cls,
                "subtype": subtype,
                "control_kind": _kind_of(c_fight),
                "treatment_kind": _kind_of(t_fight),
                "control_opponent_seat": _opponent_of(c_fight, seat_i),
                "treatment_opponent_seat": _opponent_of(t_fight, seat_i),
                "row": row,
            })
    return out


def _compare_one_row(row: Dict, c_dec: Optional[Dict], t_dec: Optional[Dict]) -> Dict:
    seat = row.get("winner_seat")
    c_view = _seat_view(c_dec, seat)
    t_view = _seat_view(t_dec, seat)
    c_alive = list((c_dec or {}).get("alive_seats") or [])
    t_alive = list((t_dec or {}).get("alive_seats") or [])
    leftover_c = False
    leftover_t = False
    try:
        leftover_c = int(seat) in {int(s) for s in c_alive}
        leftover_t = int(seat) in {int(s) for s in t_alive}
    except (TypeError, ValueError):
        pass
    c_in = bool((c_view or {}).get("choice_in_candidates"))
    t_in = bool((t_view or {}).get("choice_in_candidates"))
    alive_eq = _canon_set(c_alive) == _canon_set(t_alive)
    ghost_eq = (
        bool((c_dec or {}).get("ghost_eligible"))
        == bool((t_dec or {}).get("ghost_eligible"))
        and bool((c_dec or {}).get("bye_eligible"))
        == bool((t_dec or {}).get("bye_eligible"))
    )
    c_legal = list((c_view or {}).get("legal_candidates") or [])
    t_legal = list((t_view or {}).get("legal_candidates") or [])
    legal_eq = _canon_set(c_legal) == _canon_set(t_legal)
    c_chosen = None if c_view is None else c_view.get("chosen")
    t_chosen = None if t_view is None else t_view.get("chosen")
    chosen_eq = c_chosen == t_chosen
    cls = classify_matchmaking_gap(
        control_present=c_dec is not None,
        treatment_present=t_dec is not None,
        leftover_alive_control=leftover_c,
        leftover_alive_treatment=leftover_t,
        choice_in_candidates_control=c_in,
        choice_in_candidates_treatment=t_in,
        alive_sets_equal=alive_eq,
        ghost_bye_eligible_equal=ghost_eq,
        legal_candidates_equal=legal_eq,
        chosen_equal=chosen_eq,
    )
    return {
        "seed": row.get("seed"),
        "turn": row.get("turn"),
        "leftover_seat": seat,
        "class": cls,
        "pairing_schedule_subtype": row.get("subtype"),
        "control_kind": row.get("control_kind"),
        "treatment_kind": row.get("treatment_kind"),
        "control_fight_opponent": row.get("control_opponent_seat"),
        "treatment_fight_opponent": row.get("treatment_opponent_seat"),
        "control": _slim_decision(c_dec, seat),
        "treatment": _slim_decision(t_dec, seat),
        "alive_sets_equal": alive_eq,
        "ghost_bye_eligible_equal": ghost_eq,
        "legal_candidates_equal": legal_eq,
        "candidate_sets_identical": legal_eq,
        "chosen_equal": chosen_eq,
        "choice_in_candidates_control": c_in,
        "choice_in_candidates_treatment": t_in,
        "choice_reconciled": c_in and t_in,
    }


def attribute_matchmaking(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Dict:
    """Split 3I pairing-schedule leftover rows into matchmaking classes."""
    window = tuple(turns or LATE_TURNS)
    schedule_rows = iter_pairing_schedule_rows(
        leftover_rows, control_raw, treatment_raw,
        treatment_punch=treatment_punch, turns=window,
    )
    c_dec = _index_decisions(control_raw.get("pairing_decisions") or [])
    t_dec = _index_decisions(treatment_raw.get("pairing_decisions") or [])

    counts = Counter()
    subtype_x_class = Counter()
    n_choice_ok = 0
    n_choice_checked = 0
    examples: Dict[str, List[Dict]] = {name: [] for name in MATCHMAKING_COMPONENTS}
    compared: List[Dict] = []

    for row in schedule_rows:
        try:
            key = (int(row["seed"]), int(row["turn"]))
        except (KeyError, TypeError, ValueError):
            rec = _compare_one_row(row, None, None)
            counts["unreconciled"] += 1
            compared.append(rec)
            continue
        rec = _compare_one_row(row, c_dec.get(key), t_dec.get(key))
        cls = rec["class"]
        counts[cls] += 1
        subtype_x_class[f"{row.get('subtype')}:{cls}"] += 1
        n_choice_checked += 1
        if rec["choice_reconciled"]:
            n_choice_ok += 1
        if len(examples[cls]) < _N_EXAMPLES:
            examples[cls].append(rec)
        if len(compared) < 32 or cls != "eligibility":
            if len(compared) < 64:
                compared.append(rec)

    schedule_n = float(len(schedule_rows))
    attributed = {name: float(counts.get(name, 0)) for name in MATCHMAKING_COMPONENTS}
    reconstructed = sum(attributed.values())
    shares = {
        name: share_of_schedule(attributed[name], denom=schedule_n)
        for name in MATCHMAKING_COMPONENTS
    }
    return {
        "turns": list(window),
        "n_pairing_schedule": int(schedule_n),
        "n_leftover_input": len(leftover_rows),
        "published_pairing_schedule": PHASE_3I_PAIRING_SCHEDULE,
        "pairing_schedule_reproduced": int(schedule_n) == PHASE_3I_PAIRING_SCHEDULE,
        "counts": dict(counts),
        "subtype_x_class": dict(subtype_x_class),
        "attributed": attributed,
        "reconstructed_schedule_rows": reconstructed,
        "reconciliation_gap": schedule_n - reconstructed,
        "reconciliation_ok": abs(schedule_n - reconstructed) <= max(
            1.0, 1e-9 * (1 + schedule_n)
        ),
        "candidate_choice_n_checked": n_choice_checked,
        "candidate_choice_n_ok": n_choice_ok,
        "candidate_choice_ok": (
            n_choice_checked == 0 or n_choice_ok == n_choice_checked
        ),
        **{f"share_{k}": v for k, v in shares.items()},
        "examples": examples,
        "n_compared_kept": len(compared),
    }


def compare_matchmaking(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    selection: Optional[Dict] = None,
    retention: Optional[Dict] = None,
    pairing: Optional[Dict] = None,
) -> Dict:
    """3I pairing-schedule lock + matchmaking divergence split."""
    if lifecycle_cmp is None and control_raw.get("arm") is not None:
        greedy_c = summarize_lifecycle_arm(control_raw)
        greedy_t = summarize_lifecycle_arm(treatment_raw)
        lifecycle_cmp = compare_lifecycle(greedy_c, greedy_t)
    if divergence is None:
        divergence = compare_divergence(
            control_raw, treatment_raw, lifecycle_cmp=lifecycle_cmp,
        )
    if selection is None:
        selection = compare_selection(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
        )
    if retention is None:
        retention = compare_retention(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence, selection=selection,
        )
    if pairing is None:
        pairing = attribute_leftover_pairing_bundle(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, retention=retention,
        )

    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch, turns=LATE_TURNS,
        still_fields_t1t3=False,
    )
    leftover_still = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch, turns=LATE_TURNS,
        still_fields_t1t3=True,
    )
    late = attribute_late_t1t3_collapse(
        control_raw, treatment_raw,
        control_punch=c_punch, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch, turns=VERY_LATE_TURNS,
        still_fields_t1t3=False,
    )
    late_pair = pairing.get("attribution") if pairing else None
    if late_pair is None:
        late_pair = attribute_leftover_pairing(
            control_raw, treatment_raw,
            leftover_rows=leftover_rows, treatment_punch=t_punch, turns=LATE_TURNS,
        )
    very_late_pair = pairing.get("very_late_attribution") if pairing else None
    if very_late_pair is None:
        very_late_pair = attribute_leftover_pairing(
            control_raw, treatment_raw,
            leftover_rows=very_late_rows, treatment_punch=t_punch,
            turns=VERY_LATE_TURNS,
        )
    late_mm = attribute_matchmaking(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late_mm = attribute_matchmaking(
        control_raw, treatment_raw,
        leftover_rows=very_late_rows, treatment_punch=t_punch,
        turns=VERY_LATE_TURNS,
    )
    hist_c = reconcile_history_links(
        control_raw.get("fights") or [], control_raw.get("turn_rows") or [],
    )
    hist_t = reconcile_history_links(
        treatment_raw.get("fights") or [], treatment_raw.get("turn_rows") or [],
    )
    decomp = selection.get("decomposition") or {}
    rec_3h = retention.get("reconciliation") or {}
    pair_rec = (pairing or {}).get("reconciliation") or {}
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
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "leftover_reconciliation_ok": late_pair.get("reconciliation_ok"),
        "leftover_n": late_pair.get("n_leftover"),
        "leftover_still_fields_n": len(leftover_still),
        "pairing_schedule_n": late_pair.get("attributed", {}).get("pairing_schedule"),
        "pairing_schedule_reproduced": (
            late_pair.get("attributed", {}).get("pairing_schedule")
            == float(PHASE_3I_PAIRING_SCHEDULE)
        ),
        "matchmaking_reconciliation_ok": late_mm.get("reconciliation_ok"),
        "matchmaking_n": late_mm.get("n_pairing_schedule"),
        "candidate_choice_ok": late_mm.get("candidate_choice_ok"),
        "phase_3g_mixture_reproduced": decomp.get("mixture_turn_winner_tier"),
        "phase_3g_mixture_share_reproduced": decomp.get("share_mixture_turn_winner_tier"),
        "phase_3g_within_share_reproduced": decomp.get("share_within_cell_opponent_carry"),
        "phase_3g_n_control": decomp.get("n_control"),
        "phase_3g_n_treatment": decomp.get("n_treatment"),
        "late_collapse_reconciliation_ok": late.get("reconciliation_ok"),
        "flow_abs_tol": FLOW_ABS_TOL,
        "reweight_abs_tol": REWEIGHT_ABS_TOL,
        "lineage_control": rec_3h.get("lineage_control") or pair_rec.get("lineage_control"),
        "lineage_treatment": rec_3h.get("lineage_treatment") or pair_rec.get("lineage_treatment"),
        "paired": rec_3h.get("paired") or pair_rec.get("paired"),
    }
    if lifecycle_cmp:
        rec["reproduced_3d_board_pool_magnitude"] = (
            (lifecycle_cmp.get("reconciliation") or {}).get(
                "reproduced_3d_board_pool_magnitude"
            )
        )
        rec["reproduced_3e_carry_share"] = (
            (lifecycle_cmp.get("reweighting") or {}).get("share_of_a1_inherited_carry_pool")
        )
        rec["flow_mismatches_control"] = (
            (lifecycle_cmp.get("reconciliation") or {}).get("flow_mismatches_control")
        )
        rec["flow_mismatches_treatment"] = (
            (lifecycle_cmp.get("reconciliation") or {}).get("flow_mismatches_treatment")
        )

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": late_mm,
        "very_late_attribution": very_late_mm,
        "pairing_3i": late_pair,
        "very_late_pairing_3i": very_late_pair,
        "leftover_3h": {
            "leftover": late.get("leftover"),
            "published_leftover": PHASE_3H_LEFTOVER,
            "leftover_reproduced": (
                late.get("leftover") is not None
                and abs(float(late["leftover"]) - float(PHASE_3H_LEFTOVER)) < 1e-9
            ),
            "late_n_reproduced": (
                late.get("n_control_late_t1t3_punch") == PHASE_3H_LATE_CONTROL
                and late.get("n_treatment_late_t1t3_punch") == PHASE_3H_LATE_TREATMENT
            ),
            "leftover_still_fields_n": len(leftover_still),
            "leftover_decomposed_n": late_pair.get("n_leftover"),
        },
        "attribution_3h": late,
        "reconciliation": rec,
        "decomposition_3g": decomp,
        "paired_seats": retention.get("paired_seats"),
        "selection": {
            "decomposition": decomp,
            "reconciliation": selection.get("reconciliation"),
        },
        "timing_3f": None if divergence is None else divergence.get("timing"),
        "lifecycle": {
            "reweighting": None if lifecycle_cmp is None else lifecycle_cmp.get("reweighting"),
            "additive_flow": None if lifecycle_cmp is None else lifecycle_cmp.get("additive_flow"),
        },
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
    }


def attribute_leftover_pairing_bundle(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp=None,
    divergence=None,
    selection=None,
    retention=None,
) -> Dict:
    """Thin wrapper so compare_matchmaking can reuse compare_pairing."""
    from ml.pairing_who_wins_diagnostic import compare_pairing
    return compare_pairing(
        control_raw, treatment_raw,
        lifecycle_cmp=lifecycle_cmp, divergence=divergence,
        selection=selection, retention=retention,
    )
