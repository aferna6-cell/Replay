"""Phase 3T — observational earliest T5 incumbent-synth divergence.

Reuses the 3S T5/T6 class-(3) walk on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For each 3S-affected paired body, walks T5 from recruit-turn start to the
pre-last-play incumbent board and splits the published 3S
replacement_lifecycle / 3R membership_allocation exclusively:

  replacement_lifecycle = carry_in + earlier_t5_membership
                        + paint_repaint + scale_sync + residual

The first T5 synth-state difference (identity+tier+raw+synth) takes the
full term. Membership events are sell/buy/play/triple/transform;
paint is the 2S realloc after a membership change; scale-sync is the
end-of-T5 residual/ratio apply when it falls inside the walk window.
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
from ml.allocation_input_diagnostic import (
    _ensure_paint_rows,
    _pair_bodies,
    classify_membership_event,
    decode_recruit_action,
)
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.matched_state_damage_diagnostic import iter_class3_events
from ml.open_slot_formation_diagnostic import (
    OpenSlotFormationTracer,
    _index_ops,
    _sum_parts,
    _vacancy_stream,
    board_membership_key,
    board_state_key,
    compare_open_slot_formation,
    stamp_play_opening,
)
from ml.pairing_who_wins_diagnostic import _index_seat_fights
from ml.phase_3t_prereg import (
    BODY_EVENT_POOL_FLOW_IDENTITY,
    DIVERGENCE_COMPONENTS,
    EXCLUSIVE_T5_FIRST_DIFF_IDENTITY,
    LIFECYCLE_PROPAGATION_IDENTITY,
    MEMBERSHIP_EVENT_KINDS,
    NESTED_DIVERGENCE_IDENTITY,
    PHASE_3N_CLASS3,
    PHASE_3N_WITHIN_TIER_B,
    PHASE_3O_PRIMARY_N,
    PHASE_3O_SHARE_START_STATS,
    PHASE_3O_SHARE_SYNTH,
    PHASE_3O_T5T6_B,
    PHASE_3P_PRIMARY_N_FIGHTS,
    PHASE_3P_PRIMARY_N_PAIRS,
    PHASE_3P_SHARE_POOL,
    PHASE_3P_SHARE_ROUNDING,
    PHASE_3P_SHARE_TIMING,
    PHASE_3P_SHARE_WEIGHT,
    PHASE_3Q_PRIMARY_N_FIGHTS,
    PHASE_3Q_PRIMARY_N_PAIRS,
    PHASE_3Q_SHARE_LIFECYCLE,
    PHASE_3Q_SHARE_RESIDUAL,
    PHASE_3Q_SHARE_SAME_STATE,
    PHASE_3Q_SHARE_SCALING,
    PHASE_3Q_T1_SYNTH_CONTROL,
    PHASE_3Q_T1_SYNTH_TREATMENT,
    PHASE_3Q_T3_SYNTH_CONTROL,
    PHASE_3Q_T3_SYNTH_TREATMENT,
    PHASE_3R_PRIMARY_N_FIGHTS,
    PHASE_3R_PRIMARY_N_PAIRS,
    PHASE_3R_SHARE_INPUT,
    PHASE_3R_SHARE_MEMBERSHIP,
    PHASE_3R_SHARE_RESIDUAL,
    PHASE_3R_SHARE_ROUNDING,
    PHASE_3R_SHARE_TIMING,
    PHASE_3R_T1_SYNTH_CONTROL,
    PHASE_3R_T1_SYNTH_TREATMENT,
    PHASE_3R_T3_SYNTH_CONTROL,
    PHASE_3R_T3_SYNTH_TREATMENT,
    PHASE_3S_LIFECYCLE_ABS_MASS,
    PHASE_3S_MEMBERSHIP_PROP_ABS_MASS,
    PHASE_3S_MODAL_EARLIEST,
    PHASE_3S_PRIMARY_N_FIGHTS,
    PHASE_3S_PRIMARY_N_PAIRS,
    PHASE_3S_SAME_PRE_PLAY_IDENTITY_RATE,
    PHASE_3S_SAME_PRE_PLAY_STATE_RATE,
    PHASE_3S_SHARE_INCOMING,
    PHASE_3S_SHARE_MEMBERSHIP_PROP,
    PHASE_3S_SHARE_OPENING,
    PHASE_3S_SHARE_ORDER,
    PHASE_3S_SHARE_PRE_PLAY,
    PHASE_3S_SHARE_RESIDUAL,
    WALK_TURN,
    diagnose_phase_3t,
)
from ml.play_lifecycle_diagnostic import (
    _index_plays,
    _last_play,
    _minion_snap,
    _safe_int,
    reconstruct_post_play_sticky,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.scale_sync_diagnostic import (
    _index_syncs,
    _post_play_syncs,
    compare_scale_sync,
    decompose_scale_pair,
)
from ml.play_lifecycle_diagnostic import decompose_play_pair
from ml.survivor_composition_diagnostic import TIERS, clamp_tier
from ml.survivor_mechanic_diagnostic import (
    _fight_for_event,
    _primary_turn,
    collect_class3_minions,
)
from ml.synthetic_allocation_diagnostic import _safe_div

METHODOLOGY_VERSION = "3t_v1"
_N_EXAMPLES = 8
_SYNTH_REPRO_TOL = 0.15
_SHARE_REPRO_TOL = 1e-9
_FLOW_ABS_TOL = 1e-6
_SCALE_SEQ = 10 ** 9


def _as_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _copy_slots(slots: Optional[Sequence[Dict]]) -> List[Dict]:
    return [dict(s) for s in slots or []]


def incumbent_snapshot(slots: Optional[Sequence[Dict]]) -> List[Dict]:
    """Identity / tier / recruit raw / synth for every occupant."""
    rows = []
    for s in slots or []:
        rows.append({
            "slot": _safe_int(s.get("slot"), s.get("board_slot")),
            "card_id": str(s.get("card_id") or s.get("name") or ""),
            "name": str(s.get("name") or ""),
            "tier": int(s.get("tier") or 1),
            "recruit_raw": int(s.get("recruit_raw") or 0),
            "synthetic_share": int(s.get("synthetic_share") or 0),
            "obj_id": s.get("obj_id"),
        })
    return rows


def _board_synth_sum(slots: Optional[Sequence[Dict]]) -> int:
    return int(sum(int(s.get("synthetic_share") or 0) for s in slots or []))


def sticky_after_membership(
    pre_slots: Sequence[Dict],
    post_slots: Sequence[Dict],
    incoming: Optional[Dict] = None,
) -> List[Dict]:
    """Post-membership / pre-paint board: survivors keep sticky synth."""
    pre = list(pre_slots or [])
    post = list(post_slots or [])
    ids_before = [s.get("obj_id") for s in pre]
    ids_after = [s.get("obj_id") for s in post]
    if incoming is not None or any(i not in ids_before for i in ids_after if i is not None):
        return reconstruct_post_play_sticky(pre, incoming, ids_before, ids_after)
    kept = [dict(s) for s in pre if s.get("obj_id") in set(ids_after)]
    for i, row in enumerate(kept):
        row["slot"] = i
    return kept


def _kind_to_component(kind: Optional[str]) -> str:
    k = str(kind or "")
    if k in ("turn_start", "carry_in"):
        return "carry_in"
    if k in MEMBERSHIP_EVENT_KINDS or k in ("membership", "prior_sell"):
        return "earlier_t5_membership"
    if k in ("paint", "repaint", "paint_repaint"):
        return "paint_repaint"
    if k in ("scale_sync", "scale-sync"):
        return "scale_sync"
    return "residual"


def _event_order(ev: Dict) -> Tuple[int, int, int]:
    turn = int(ev.get("turn") or 0)
    seq = ev.get("seq")
    if seq is None:
        seq = ev.get("action_seq")
    if seq is None:
        seq = ev.get("sync_index")
    kind = str(ev.get("kind") or ev.get("event") or "")
    sub = 0
    if kind == "paint":
        sub = 1
    elif kind == "scale_sync":
        seq = _SCALE_SEQ if seq is None else int(seq)
        sub = 2
    elif kind == "turn_start":
        seq = -2
        sub = 0
    elif kind == "death_cleanup":
        seq = -1
        sub = 0
    return (turn, int(seq if seq is not None else 0), sub)


def in_t5_walk_window(ev: Dict, last_play: Optional[Dict]) -> bool:
    """T5 events strictly before the last play; all of T5 if last play is T6+."""
    turn = _safe_int(ev.get("turn"), 0) or 0
    if turn != WALK_TURN:
        return False
    if not last_play:
        return True
    last_turn = _safe_int(last_play.get("turn"), 0) or 0
    if last_turn > WALK_TURN:
        return True
    if last_turn < WALK_TURN:
        return False
    last_seq = last_play.get("action_seq")
    if last_seq is None:
        last_seq = last_play.get("seq")
    last_seq = -1 if last_seq is None else int(last_seq)
    ev_seq = ev.get("seq")
    if ev_seq is None:
        ev_seq = ev.get("action_seq")
    kind = str(ev.get("kind") or ev.get("event") or "")
    if kind == "scale_sync":
        return False
    if kind == "turn_start":
        return True
    if ev_seq is None:
        return True
    return int(ev_seq) < last_seq


def first_synth_component(
    control_events: Sequence[Dict],
    treatment_events: Sequence[Dict],
) -> str:
    """Exclusive first-difference on the T5 incumbent-synth walk."""
    c_list = list(control_events or [])
    t_list = list(treatment_events or [])
    n = max(len(c_list), len(t_list))
    if n == 0:
        return "residual"
    for i in range(n):
        c = c_list[i] if i < len(c_list) else None
        t = t_list[i] if i < len(t_list) else None
        if c is None or t is None:
            return _kind_to_component((t or c).get("kind"))
        c_slots = c.get("slots") or []
        t_slots = t.get("slots") or []
        if board_state_key(c_slots) != board_state_key(t_slots):
            kind = str(c.get("kind") or t.get("kind") or "")
            return _kind_to_component(kind)
        if board_membership_key(c_slots) != board_membership_key(t_slots):
            kind = str(c.get("kind") or t.get("kind") or "")
            return _kind_to_component(kind)
    return "residual"


def decompose_t5_synth_pair(
    control_start: Dict,
    treatment_start: Dict,
    control_play: Optional[Dict],
    treatment_play: Optional[Dict],
    control_events: Sequence[Dict],
    treatment_events: Sequence[Dict],
    control_syncs: Optional[Sequence[Dict]] = None,
    treatment_syncs: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Exclusive five-way split of the 3S replacement_lifecycle term."""
    q = decompose_play_pair(
        control_start, treatment_start, control_play, treatment_play,
    )
    scale = decompose_scale_pair(
        control_start, treatment_start, control_play, treatment_play,
        control_syncs or [], treatment_syncs or [],
    )
    lifecycle = float(q.get("replacement_lifecycle") or 0.0)
    membership_inc = float(scale.get("membership_allocation") or 0.0)
    parts = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
    memb_parts = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
    complete = bool(q.get("snapshots_complete"))
    if not complete:
        component = "residual"
        parts["residual"] = lifecycle
        memb_parts["residual"] = membership_inc
    else:
        component = first_synth_component(control_events, treatment_events)
        parts[component] = lifecycle
        memb_parts[component] = membership_inc
    explained = sum(parts[n] for n in DIVERGENCE_COMPONENTS)
    first_ev = None
    for i, (c, t) in enumerate(zip(control_events or [], treatment_events or [])):
        if board_state_key(c.get("slots")) != board_state_key(t.get("slots")):
            first_ev = c
            break
        if board_membership_key(c.get("slots")) != board_membership_key(t.get("slots")):
            first_ev = c
            break
    return {
        "replacement_lifecycle": lifecycle,
        "membership_allocation": membership_inc,
        "subsequent_scaling": float(q.get("subsequent_scaling") or 0.0),
        "same_state_repaint": float(q.get("same_state_repaint") or 0.0),
        "delta_synth": float(q.get("delta_synth") or 0.0),
        **parts,
        **{f"membership_{name}": memb_parts[name] for name in DIVERGENCE_COMPONENTS},
        "divergence_component": component,
        "first_event_kind": None if first_ev is None else first_ev.get("kind"),
        "first_event_subtype": None if first_ev is None else first_ev.get("subtype"),
        "n_control_events": len(list(control_events or [])),
        "n_treatment_events": len(list(treatment_events or [])),
        "snapshots_complete": complete,
        "explained_lifecycle": explained,
        "residual_vs_lifecycle": lifecycle - explained,
        "same_t5_start_identity": (
            False if not control_events or not treatment_events
            else board_membership_key((control_events[0] or {}).get("slots"))
            == board_membership_key((treatment_events[0] or {}).get("slots"))
        ),
        "same_t5_start_state": (
            False if not control_events or not treatment_events
            else board_state_key((control_events[0] or {}).get("slots"))
            == board_state_key((treatment_events[0] or {}).get("slots"))
        ),
        "s_c_sticky": q.get("s_c_sticky"),
        "s_t_sticky_cf": q.get("s_t_sticky_cf"),
        "s_c_start": q.get("s_c_start"),
        "s_t_start": q.get("s_t_start"),
        "flow_gap_control": scale.get("flow_gap_control"),
        "flow_gap_treatment": scale.get("flow_gap_treatment"),
    }


class T5IncumbentSynthTracer(OpenSlotFormationTracer):
    """3S formation rows plus T5 turn-start / sticky / paint snapshots."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.turn_starts: List[Dict] = []
        self.paint_events: List[Dict] = []

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        super().begin_seat_recruit(seat, turn, player)
        board = list(getattr(player, "board", None) or [])
        slots = [_minion_snap(m, i) for i, m in enumerate(board)]
        self.turn_starts.append({
            "seed": int(self.seed),
            "lobby": int(self.lobby_id),
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(turn),
            "seq": -2,
            "kind": "turn_start",
            "event": "turn_start",
            "slots": [dict(s) for s in slots],
            "incumbents": incumbent_snapshot(slots),
            "board_len": len(slots),
            "synth_sum": _board_synth_sum(slots),
        })

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int,
        ended: bool, player=None,
    ) -> None:
        pre_slots = list(self._pre_slots.get(int(seat), []))
        ids_before = list(self._pre_ids.get(int(seat), []))
        super().after_action(seat, turn, shop_generation, action, ended, player)
        p = player if player is not None else self._live_player.get(int(seat))
        board = list(getattr(p, "board", None) or []) if p is not None else []
        post_slots = [_minion_snap(m, i) for i, m in enumerate(board)]
        ids_after = [id(m) for m in board]
        if not self.recruit_ops:
            return
        op = self.recruit_ops[-1]
        if (
            _safe_int(op.get("seed")) != int(self.seed)
            or _safe_int(op.get("seat")) != int(seat)
            or _safe_int(op.get("turn")) != int(turn)
        ):
            return
        incoming = None
        if self.play_events:
            last = self.play_events[-1]
            if (
                _safe_int(last.get("seed")) == int(self.seed)
                and _safe_int(last.get("seat")) == int(seat)
                and _safe_int(last.get("turn")) == int(turn)
                and last.get("incoming")
            ):
                incoming = dict(last.get("incoming") or {})
        sticky = sticky_after_membership(pre_slots, post_slots, incoming)
        op["pre_slots"] = [dict(s) for s in pre_slots]
        op["post_slots"] = [dict(s) for s in post_slots]
        op["sticky_slots"] = [dict(s) for s in sticky]
        op["incumbents_pre"] = incumbent_snapshot(pre_slots)
        op["incumbents_sticky"] = incumbent_snapshot(sticky)
        op["incumbents_post"] = incumbent_snapshot(post_slots)
        op["synth_sum_pre"] = _board_synth_sum(pre_slots)
        op["synth_sum_sticky"] = _board_synth_sum(sticky)
        op["synth_sum_post"] = _board_synth_sum(post_slots)
        event_kind = classify_membership_event(action, ids_before, ids_after)
        painted = board_state_key(sticky) != board_state_key(post_slots)
        if event_kind is not None and painted:
            self.paint_events.append({
                "seed": int(self.seed),
                "lobby": int(self.lobby_id),
                "arm": self.arm,
                "seat": int(seat),
                "turn": int(turn),
                "seq": op.get("seq"),
                "kind": "paint",
                "event": "paint",
                "subtype": event_kind,
                "trigger": decode_recruit_action(action),
                "pre_slots": [dict(s) for s in sticky],
                "post_slots": [dict(s) for s in post_slots],
                "slots": [dict(s) for s in post_slots],
                "incumbents": incumbent_snapshot(post_slots),
                "board_len_before": len(sticky),
                "board_len_after": len(post_slots),
                "synth_sum_before": _board_synth_sum(sticky),
                "synth_sum_after": _board_synth_sum(post_slots),
            })


def run_t5_incumbent_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    from ml.phase_3t_prereg import assert_seed_range_allowed
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
    play_events: List[Dict] = []
    scale_syncs: List[Dict] = []
    recruit_ops: List[Dict] = []
    combat_shrinks: List[Dict] = []
    turn_starts: List[Dict] = []
    paint_events: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = T5IncumbentSynthTracer(i, seed + i, arm)
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
                play_events.extend(tracer.play_events)
                scale_syncs.extend(tracer.scale_syncs)
                recruit_ops.extend(tracer.recruit_ops)
                combat_shrinks.extend(tracer.combat_shrinks)
                turn_starts.extend(tracer.turn_starts)
                paint_events.extend(tracer.paint_events)
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
        "play_events": play_events,
        "scale_syncs": scale_syncs,
        "recruit_ops": recruit_ops,
        "combat_shrinks": combat_shrinks,
        "turn_starts": turn_starts,
        "paint_events": paint_events,
    }


def run_greedy_control_t5_incumbent(lobbies: int, seed: int) -> Dict:
    return run_t5_incumbent_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_t5_incumbent(lobbies: int, seed: int) -> Dict:
    return run_t5_incumbent_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _index_starts(starts: Sequence[Dict]) -> Dict[Tuple[int, int, int], Dict]:
    out: Dict[Tuple[int, int, int], Dict] = {}
    for ev in starts or []:
        seed = _safe_int(ev.get("seed"))
        seat = _safe_int(ev.get("seat"))
        turn = _safe_int(ev.get("turn"))
        if seed is None or seat is None or turn is None:
            continue
        out[(seed, seat, turn)] = ev
    return out


def _index_paints(paints: Sequence[Dict]) -> Dict[Tuple[int, int], List[Dict]]:
    return _index_ops(paints)


def build_t5_event_stream(
    *,
    turn_starts: Sequence[Dict],
    recruit_ops: Sequence[Dict],
    play_events: Sequence[Dict],
    paint_events: Sequence[Dict],
    scale_syncs: Sequence[Dict],
    combat_shrinks: Sequence[Dict],
    last_play: Optional[Dict],
    seed,
    seat,
) -> List[Dict]:
    """Chronological T5 checkpoints from recruit start to pre-last-play."""
    seed_i = _safe_int(seed)
    seat_i = _safe_int(seat)
    if seed_i is None or seat_i is None:
        return []
    raw: List[Dict] = []
    start = None
    for ev in turn_starts or []:
        if (
            _safe_int(ev.get("seed")) == seed_i
            and _safe_int(ev.get("seat")) == seat_i
            and _safe_int(ev.get("turn")) == WALK_TURN
        ):
            start = ev
            break
    if start is not None:
        raw.append({
            "kind": "turn_start",
            "subtype": "carry_in",
            "turn": WALK_TURN,
            "seq": -2,
            "slots": _copy_slots(start.get("slots") or start.get("incumbents")),
            "incumbents": incumbent_snapshot(
                start.get("slots") or start.get("incumbents")
            ),
            "synth_sum": _board_synth_sum(start.get("slots")),
        })
    for ev in combat_shrinks or []:
        if (
            _safe_int(ev.get("seed")) != seed_i
            or _safe_int(ev.get("seat")) != seat_i
            or _safe_int(ev.get("turn")) != WALK_TURN
        ):
            continue
        row = {
            "kind": "death_cleanup",
            "subtype": "death_cleanup",
            "turn": WALK_TURN,
            "seq": -1,
            "slots": _copy_slots(ev.get("post_slots")),
            "incumbents": incumbent_snapshot(ev.get("post_slots")),
            "synth_sum": _board_synth_sum(ev.get("post_slots")),
            "board_len_before": ev.get("board_len_before"),
            "board_len_after": ev.get("board_len_after"),
        }
        if in_t5_walk_window(row, last_play):
            raw.append(row)
    plays_by_seq = {}
    for ev in play_events or []:
        if (
            _safe_int(ev.get("seed")) != seed_i
            or _safe_int(ev.get("seat")) != seat_i
            or _safe_int(ev.get("turn")) != WALK_TURN
        ):
            continue
        seq = ev.get("action_seq")
        if seq is None:
            seq = ev.get("seq")
        plays_by_seq[int(seq if seq is not None else -1)] = ev
    for op in recruit_ops or []:
        if (
            _safe_int(op.get("seed")) != seed_i
            or _safe_int(op.get("seat")) != seat_i
            or _safe_int(op.get("turn")) != WALK_TURN
        ):
            continue
        seq = _safe_int(op.get("seq"), 0) or 0
        play = plays_by_seq.get(seq)
        event_kind = str(op.get("event") or op.get("kind") or "")
        if event_kind in ("buy", "other", "") and not play:
            if event_kind != "triple":
                continue
        sticky = op.get("sticky_slots")
        if play is not None:
            sticky = play.get("post_play_pre_realloc") or sticky
        if sticky is None:
            sticky = op.get("post_slots") or op.get("pre_slots")
        membership_kind = event_kind
        if play is not None:
            membership_kind = str(play.get("event") or play.get("play_subtype") or event_kind)
        if membership_kind in ("sell_buy_play", "sell_play", "open_slot"):
            membership_kind = "play" if membership_kind != "triple" else "triple"
        if membership_kind == "prior_sell":
            membership_kind = "sell"
        mem = {
            "kind": membership_kind if membership_kind in MEMBERSHIP_EVENT_KINDS else "play",
            "subtype": membership_kind,
            "turn": WALK_TURN,
            "seq": seq,
            "slots": _copy_slots(sticky),
            "incumbents": incumbent_snapshot(sticky),
            "synth_sum": _board_synth_sum(sticky),
            "board_len_before": op.get("board_len_before"),
            "board_len_after": (
                len(sticky) if sticky is not None else op.get("board_len_after")
            ),
        }
        if in_t5_walk_window(mem, last_play):
            raw.append(mem)
        post = None
        if play is not None:
            post = play.get("post_realloc")
        if post is None:
            post = op.get("post_slots")
        if post is not None and in_t5_walk_window(
            {"kind": "paint", "turn": WALK_TURN, "seq": seq}, last_play
        ):
            raw.append({
                "kind": "paint",
                "subtype": membership_kind,
                "turn": WALK_TURN,
                "seq": seq,
                "slots": _copy_slots(post),
                "incumbents": incumbent_snapshot(post),
                "synth_sum": _board_synth_sum(post),
                "board_len_before": len(sticky or []),
                "board_len_after": len(post),
            })
    for ev in scale_syncs or []:
        if (
            _safe_int(ev.get("seed")) != seed_i
            or _safe_int(ev.get("seat")) != seat_i
            or _safe_int(ev.get("turn")) != WALK_TURN
        ):
            continue
        row = {
            "kind": "scale_sync",
            "subtype": "scale_sync",
            "turn": WALK_TURN,
            "seq": _SCALE_SEQ,
            "slots": _copy_slots(ev.get("post_slots") or ev.get("pre_slots")),
            "incumbents": incumbent_snapshot(ev.get("post_slots") or ev.get("pre_slots")),
            "synth_sum": _board_synth_sum(ev.get("post_slots") or ev.get("pre_slots")),
            "pre_slots": _copy_slots(ev.get("pre_slots")),
            "board_flow_gap": ev.get("board_flow_gap"),
        }
        if in_t5_walk_window(row, last_play):
            raw.append(row)
    raw.sort(key=_event_order)
    return raw


def _walk_flow_gap(events: Sequence[Dict], last_play: Optional[Dict]) -> float:
    """|last walk snapshot − pre-last-play incumbents| when last play is T5."""
    if not last_play:
        return 0.0
    last_turn = _safe_int(last_play.get("turn"), 0) or 0
    if last_turn != WALK_TURN or not events:
        return 0.0
    pre = last_play.get("pre_play") or []
    last_slots = (events[-1] or {}).get("slots") or []
    if board_state_key(pre) == board_state_key(last_slots):
        return 0.0
    if board_membership_key(pre) == board_membership_key(last_slots):
        return float(abs(_board_synth_sum(pre) - _board_synth_sum(last_slots)))
    return float(abs(len(pre) - len(last_slots)))


def _event_chain_gaps(events: Sequence[Dict]) -> int:
    """Count consecutive board-len mismatches in the walk."""
    n = 0
    prev_len = None
    for ev in events or []:
        after = ev.get("board_len_after")
        if after is None:
            after = len(ev.get("slots") or [])
        before = ev.get("board_len_before")
        if prev_len is not None and before is not None and int(before) != int(prev_len):
            kind = str(ev.get("kind") or "")
            if kind not in ("turn_start", "paint", "scale_sync"):
                n += 1
        prev_len = after
    return n


def tier_mass_divergence(per_tier: Dict[str, Dict]) -> Dict:
    """Decision shares from within-tier |parts| so T1↓ / T3↑ cannot cancel."""
    mass = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
    n_used = 0
    abs_delta = 0.0
    for cell in (per_tier or {}).values():
        n = int(cell.get("n_pairs") or 0)
        if n <= 0:
            continue
        n_used += n
        abs_delta += n * abs(float(cell.get("replacement_lifecycle") or 0.0))
        for name in DIVERGENCE_COMPONENTS:
            mass[name] += n * abs(float(cell.get(name) or 0.0))
    total_mass = sum(mass.values())

    def _share(part: float) -> Optional[float]:
        if total_mass < 1e-12:
            return None
        return float(part) / total_mass

    return {
        "method": "within_tier_abs_mass_t5_incumbent_synth_identity",
        "n_pairs": n_used,
        "abs_replacement_lifecycle": (
            None if n_used <= 0 else float(abs_delta) / float(n_used)
        ),
        "component_abs_mass": mass,
        "total_abs_mass": total_mass,
        **{name: (
            None if n_used <= 0 else float(mass[name]) / float(n_used)
        ) for name in DIVERGENCE_COMPONENTS},
        **{f"share_of_delta_{name}": _share(mass[name])
           for name in DIVERGENCE_COMPONENTS},
        "divergence_components": list(DIVERGENCE_COMPONENTS),
    }


def tier_mass_membership_prop(per_tier: Dict[str, Dict]) -> Dict:
    """How T5 first-diff tags propagate into 3R membership |mass|."""
    mass = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
    n_used = 0
    abs_delta = 0.0
    for cell in (per_tier or {}).values():
        n = int(cell.get("n_pairs") or 0)
        if n <= 0:
            continue
        n_used += n
        abs_delta += n * abs(float(cell.get("membership_allocation") or 0.0))
        for name in DIVERGENCE_COMPONENTS:
            mass[name] += n * abs(float(cell.get(f"membership_{name}") or 0.0))
    total_mass = sum(mass.values())

    def _share(part: float) -> Optional[float]:
        if total_mass < 1e-12:
            return None
        return float(part) / total_mass

    return {
        "method": "within_tier_abs_mass_t5_membership_propagation",
        "n_pairs": n_used,
        "abs_membership_allocation": (
            None if n_used <= 0 else float(abs_delta) / float(n_used)
        ),
        "component_abs_mass": mass,
        "total_abs_mass": total_mass,
        **{f"share_of_membership_{name}": _share(mass[name])
           for name in DIVERGENCE_COMPONENTS},
        "divergence_components": list(DIVERGENCE_COMPONENTS),
    }


def attribute_t5_incumbent_synth(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    matched: Optional[Dict] = None,
    mechanics: Optional[Dict] = None,
    allocation: Optional[Dict] = None,
    lifecycle: Optional[Dict] = None,
    scale: Optional[Dict] = None,
    formation: Optional[Dict] = None,
) -> Dict:
    """3S locks plus T5 incumbent-synth first-diff split."""
    if formation is None:
        formation = compare_open_slot_formation(
            control_raw, treatment_raw,
            matched=matched, mechanics=mechanics, allocation=allocation,
            lifecycle=lifecycle, scale=scale,
        )
    if scale is None:
        scale = {
            "attribution": (formation or {}).get("scale_3r"),
            "primary": (formation or {}).get("scale_primary_3r"),
        }
    if lifecycle is None:
        lifecycle = {
            "attribution": (formation or {}).get("lifecycle_3q"),
            "primary": (formation or {}).get("lifecycle_primary_3q"),
        }
    events = list(iter_class3_events(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
    ))
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    rows_c, rows_t, n_c, n_t = collect_class3_minions(
        events, c_fights, t_fights, primary_only=True,
    )
    rows_c = _ensure_paint_rows(rows_c, [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(c_fights, ev),) if f
    ])
    rows_t = _ensure_paint_rows(rows_t, [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(t_fights, ev),) if f
    ])
    alloc_pairs, unpaired_c, unpaired_t, n_fights = _pair_bodies(
        events, c_fights, t_fights,
    )
    c_plays = _index_plays(control_raw.get("play_events") or [])
    t_plays = _index_plays(treatment_raw.get("play_events") or [])
    c_syncs = _index_syncs(control_raw.get("scale_syncs") or [])
    t_syncs = _index_syncs(treatment_raw.get("scale_syncs") or [])
    c_ops = _index_ops(control_raw.get("recruit_ops") or [])
    t_ops = _index_ops(treatment_raw.get("recruit_ops") or [])
    c_shrink = _index_ops(control_raw.get("combat_shrinks") or [])
    t_shrink = _index_ops(treatment_raw.get("combat_shrinks") or [])

    pairs: List[Dict] = []
    n_complete = 0
    n_missing_play = 0
    n_flow_mismatch = 0
    first_kinds: List[str] = []
    for p in alloc_pairs:
        ev_seed = p.get("seed")
        ev_seat = p.get("causal_seat")
        ev_turn = p.get("first_divergence_turn")
        c_row = p.get("control") or {}
        t_row = p.get("treatment") or {}
        c_fight = _fight_for_event(c_fights, {
            "seed": ev_seed, "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
        })
        t_fight = _fight_for_event(t_fights, {
            "seed": ev_seed, "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
        })
        c_winner = _safe_int((c_fight or {}).get("winner_seat"), ev_seat)
        t_winner = _safe_int((t_fight or {}).get("winner_seat"), ev_seat)
        c_play = _last_play(c_plays, ev_seed, c_winner, ev_turn)
        t_play = _last_play(t_plays, ev_seed, t_winner, ev_turn)
        if c_play is not None:
            c_play = stamp_play_opening(
                c_play, _vacancy_stream(c_ops, c_shrink, ev_seed, c_winner),
            )
        if t_play is not None:
            t_play = stamp_play_opening(
                t_play, _vacancy_stream(t_ops, t_shrink, ev_seed, t_winner),
            )
        c_post = _post_play_syncs(
            c_syncs, ev_seed, c_winner,
            None if c_play is None else c_play.get("turn"),
            ev_turn,
        )
        t_post = _post_play_syncs(
            t_syncs, ev_seed, t_winner,
            None if t_play is None else t_play.get("turn"),
            ev_turn,
        )
        c_stream = build_t5_event_stream(
            turn_starts=control_raw.get("turn_starts") or [],
            recruit_ops=control_raw.get("recruit_ops") or [],
            play_events=control_raw.get("play_events") or [],
            paint_events=control_raw.get("paint_events") or [],
            scale_syncs=control_raw.get("scale_syncs") or [],
            combat_shrinks=control_raw.get("combat_shrinks") or [],
            last_play=c_play, seed=ev_seed, seat=c_winner,
        )
        t_stream = build_t5_event_stream(
            turn_starts=treatment_raw.get("turn_starts") or [],
            recruit_ops=treatment_raw.get("recruit_ops") or [],
            play_events=treatment_raw.get("play_events") or [],
            paint_events=treatment_raw.get("paint_events") or [],
            scale_syncs=treatment_raw.get("scale_syncs") or [],
            combat_shrinks=treatment_raw.get("combat_shrinks") or [],
            last_play=t_play, seed=ev_seed, seat=t_winner,
        )
        parts = decompose_t5_synth_pair(
            c_row, t_row, c_play, t_play, c_stream, t_stream, c_post, t_post,
        )
        gap_c = _walk_flow_gap(c_stream, c_play)
        gap_t = _walk_flow_gap(t_stream, t_play)
        chain_c = _event_chain_gaps(c_stream)
        chain_t = _event_chain_gaps(t_stream)
        sync_gap_c = parts.get("flow_gap_control")
        sync_gap_t = parts.get("flow_gap_treatment")
        if (
            abs(gap_c) > _FLOW_ABS_TOL
            or abs(gap_t) > _FLOW_ABS_TOL
            or chain_c > 0
            or chain_t > 0
            or (
                sync_gap_c is not None and abs(float(sync_gap_c)) > _FLOW_ABS_TOL
            )
            or (
                sync_gap_t is not None and abs(float(sync_gap_t)) > _FLOW_ABS_TOL
            )
        ):
            n_flow_mismatch += 1
        parts.update({
            "seed": ev_seed,
            "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
            "tier": clamp_tier(p.get("tier") or c_row.get("tier") or t_row.get("tier")),
            "board_slot": p.get("board_slot"),
            "walk_flow_gap_control": gap_c,
            "walk_flow_gap_treatment": gap_t,
            "n_t5_events_control": len(c_stream),
            "n_t5_events_treatment": len(t_stream),
        })
        if parts.get("first_event_kind"):
            first_kinds.append(str(parts["first_event_kind"]))
        if parts.get("snapshots_complete"):
            n_complete += 1
        else:
            n_missing_play += 1
        pairs.append(parts)

    for r in unpaired_t:
        s = float(r.get("synthetic_share") or 0)
        bag = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
        bag["carry_in"] = s
        memb = {f"membership_{name}": 0.0 for name in DIVERGENCE_COMPONENTS}
        pairs.append({
            "replacement_lifecycle": s,
            "membership_allocation": 0.0,
            **bag,
            **memb,
            "divergence_component": "carry_in",
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "treatment",
            "snapshots_complete": False,
        })
    for r in unpaired_c:
        s = float(r.get("synthetic_share") or 0)
        bag = {name: 0.0 for name in DIVERGENCE_COMPONENTS}
        bag["carry_in"] = -s
        memb = {f"membership_{name}": 0.0 for name in DIVERGENCE_COMPONENTS}
        pairs.append({
            "replacement_lifecycle": -s,
            "membership_allocation": 0.0,
            **bag,
            **memb,
            "divergence_component": "carry_in",
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "control",
            "snapshots_complete": False,
        })

    totals = _sum_parts(pairs, DIVERGENCE_COMPONENTS, "replacement_lifecycle")
    n_pairs = max(1, len(pairs))
    means = {k: float(v) / float(n_pairs) for k, v in totals.items()}
    obs_delta = means["replacement_lifecycle"]

    def _share(part: float) -> Optional[float]:
        if abs(obs_delta) < 1e-12:
            return None
        return float(part) / obs_delta

    pooled_signed = {
        "method": "exact_t5_incumbent_synth_paired_slot_identity",
        "n_pairs": len(pairs),
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "n_snapshots_complete": n_complete,
        "n_missing_play": n_missing_play,
        "replacement_lifecycle": means["replacement_lifecycle"],
        **{name: means[name] for name in DIVERGENCE_COMPONENTS},
        "explained_all_parts": sum(means[n] for n in DIVERGENCE_COMPONENTS),
        "residual_vs_delta": (
            means["replacement_lifecycle"]
            - sum(means[n] for n in DIVERGENCE_COMPONENTS)
        ),
        **{f"share_of_delta_{name}": _share(means[name])
           for name in DIVERGENCE_COMPONENTS},
        "divergence_components": list(DIVERGENCE_COMPONENTS),
    }
    per_tier = {}
    for tier in TIERS:
        cell = [p for p in pairs if int(p.get("tier") or 1) == tier]
        if not cell:
            per_tier[str(tier)] = {"n_pairs": 0}
            continue
        ct = _sum_parts(cell, DIVERGENCE_COMPONENTS, "replacement_lifecycle")
        mt = _sum_parts(
            [
                {
                    **{name: p.get(f"membership_{name}") for name in DIVERGENCE_COMPONENTS},
                    "membership_allocation": p.get("membership_allocation"),
                }
                for p in cell
            ],
            DIVERGENCE_COMPONENTS,
            "membership_allocation",
        )
        n_cell = max(1, len(cell))
        cm = {k: float(v) / float(n_cell) for k, v in ct.items()}
        mm = {k: float(v) / float(n_cell) for k, v in mt.items()}
        d = cm["replacement_lifecycle"]

        def _cshare(part: float, den: float = d) -> Optional[float]:
            if abs(den) < 1e-12:
                return None
            return float(part) / den

        per_tier[str(tier)] = {
            "n_pairs": len(cell),
            "replacement_lifecycle": cm["replacement_lifecycle"],
            "membership_allocation": mm["membership_allocation"],
            **{name: cm[name] for name in DIVERGENCE_COMPONENTS},
            **{f"membership_{name}": mm[name] for name in DIVERGENCE_COMPONENTS},
            **{f"share_of_delta_{name}": _cshare(cm[name])
               for name in DIVERGENCE_COMPONENTS},
        }

    primary = tier_mass_divergence(per_tier)
    membership_prop = tier_mass_membership_prop(per_tier)
    kind_counts = Counter(first_kinds)
    modal_kind = None
    if kind_counts:
        modal_kind = kind_counts.most_common(1)[0][0]
    complete = [p for p in pairs if p.get("snapshots_complete")]
    primary.update({
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "n_snapshots_complete": n_complete,
        "n_missing_play": n_missing_play,
        "pooled_signed_replacement_lifecycle": pooled_signed["replacement_lifecycle"],
        "pooled_signed": pooled_signed,
        "residual_vs_delta": pooled_signed["residual_vs_delta"],
        "modal_first_event_kind": modal_kind,
        "first_event_kind_counts": dict(kind_counts),
        "n_event_board_flow_mismatch": n_flow_mismatch,
        "same_t5_start_identity_rate": _safe_div(
            float(sum(1 for p in complete if p.get("same_t5_start_identity"))),
            float(len(complete)),
        ),
        "same_t5_start_state_rate": _safe_div(
            float(sum(1 for p in complete if p.get("same_t5_start_state"))),
            float(len(complete)),
        ),
    })

    form_attr = (formation or {}).get("attribution") or formation or {}
    form_primary = (formation or {}).get("primary") or form_attr.get("primary") or {}
    life_attr = (lifecycle or {}).get("attribution") or lifecycle or {}
    life_primary = (lifecycle or {}).get("primary") or life_attr.get("primary") or {}
    scale_attr = (scale or {}).get("attribution") or scale or {}
    scale_primary = (scale or {}).get("primary") or scale_attr.get("primary") or {}

    def _close(got, published, tol=_SYNTH_REPRO_TOL) -> bool:
        if got is None or published is None:
            return False
        return abs(float(got) - float(published)) <= tol

    t1_c = (
        form_attr.get("t1_synth_control") or life_attr.get("t1_synth_control")
        or scale_attr.get("t1_synth_control")
    )
    t1_t = (
        form_attr.get("t1_synth_treatment") or life_attr.get("t1_synth_treatment")
        or scale_attr.get("t1_synth_treatment")
    )
    t3_c = (
        form_attr.get("t3_synth_control") or life_attr.get("t3_synth_control")
        or scale_attr.get("t3_synth_control")
    )
    t3_t = (
        form_attr.get("t3_synth_treatment") or life_attr.get("t3_synth_treatment")
        or scale_attr.get("t3_synth_treatment")
    )
    share_lifecycle = (
        form_primary.get("share_of_delta_pre_play_membership")
        or life_primary.get("share_of_delta_replacement_lifecycle")
    )
    share_membership = (
        (form_attr.get("membership_propagation") or {}).get(
            "share_of_membership_pre_play_membership"
        )
        or scale_primary.get("share_of_delta_membership_allocation")
    )
    if share_lifecycle is None:
        share_lifecycle = form_primary.get("share_of_delta_pre_play_membership")
    examples = []
    for p in pairs:
        if len(examples) >= _N_EXAMPLES:
            break
        if p.get("unpaired"):
            continue
        examples.append({
            "seed": p.get("seed"),
            "causal_seat": p.get("causal_seat"),
            "first_divergence_turn": p.get("first_divergence_turn"),
            "tier": p.get("tier"),
            "replacement_lifecycle": p.get("replacement_lifecycle"),
            "divergence_component": p.get("divergence_component"),
            "first_event_kind": p.get("first_event_kind"),
            "carry_in": p.get("carry_in"),
            "earlier_t5_membership": p.get("earlier_t5_membership"),
            "paint_repaint": p.get("paint_repaint"),
            "scale_sync": p.get("scale_sync"),
            "residual": p.get("residual"),
            "membership_allocation": p.get("membership_allocation"),
            "n_t5_events_control": p.get("n_t5_events_control"),
            "n_t5_events_treatment": p.get("n_t5_events_treatment"),
        })

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_same_outcome_damage": len(events),
        "n_primary_class3": n_c,
        "n_primary_class3_treatment": n_t,
        "published_same_outcome_damage": PHASE_3N_CLASS3,
        "published_primary_n": PHASE_3O_PRIMARY_N,
        "published_3p_n_pairs": PHASE_3P_PRIMARY_N_PAIRS,
        "published_3p_n_fights": PHASE_3P_PRIMARY_N_FIGHTS,
        "published_3q_n_pairs": PHASE_3Q_PRIMARY_N_PAIRS,
        "published_3q_n_fights": PHASE_3Q_PRIMARY_N_FIGHTS,
        "published_3q_share_lifecycle": PHASE_3Q_SHARE_LIFECYCLE,
        "published_3r_n_pairs": PHASE_3R_PRIMARY_N_PAIRS,
        "published_3r_n_fights": PHASE_3R_PRIMARY_N_FIGHTS,
        "published_3r_share_membership": PHASE_3R_SHARE_MEMBERSHIP,
        "published_3s_n_pairs": PHASE_3S_PRIMARY_N_PAIRS,
        "published_3s_n_fights": PHASE_3S_PRIMARY_N_FIGHTS,
        "published_3s_share_pre_play": PHASE_3S_SHARE_PRE_PLAY,
        "same_outcome_damage_reproduced": len(events) == PHASE_3N_CLASS3,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_B_reproduced": form_attr.get("phase_3n_B_reproduced")
        or life_attr.get("phase_3n_B_reproduced")
        or scale_attr.get("phase_3n_B_reproduced"),
        "phase_3o_t5t6_B": PHASE_3O_T5T6_B,
        "phase_3o_B_reproduced": form_attr.get("phase_3o_B_reproduced")
        or life_attr.get("phase_3o_B_reproduced")
        or scale_attr.get("phase_3o_B_reproduced"),
        "phase_3o_share_start_stats": PHASE_3O_SHARE_START_STATS,
        "phase_3o_share_synth": PHASE_3O_SHARE_SYNTH,
        "phase_3p_share_timing": PHASE_3P_SHARE_TIMING,
        "phase_3p_share_pool": PHASE_3P_SHARE_POOL,
        "phase_3p_share_weight": PHASE_3P_SHARE_WEIGHT,
        "phase_3p_share_rounding": PHASE_3P_SHARE_ROUNDING,
        "phase_3q_share_same_state": life_primary.get(
            "share_of_delta_same_state_repaint"
        ),
        "phase_3q_share_lifecycle": share_lifecycle if share_lifecycle is not None
        else life_primary.get("share_of_delta_replacement_lifecycle"),
        "phase_3q_share_scaling": life_primary.get(
            "share_of_delta_subsequent_scaling"
        ),
        "phase_3q_share_residual": life_primary.get("share_of_delta_residual"),
        "phase_3q_lifecycle_reproduced": _close(
            life_primary.get("share_of_delta_replacement_lifecycle")
            or form_attr.get("phase_3q_share_lifecycle"),
            PHASE_3Q_SHARE_LIFECYCLE, _SHARE_REPRO_TOL,
        ),
        "phase_3r_share_membership": (
            scale_primary.get("share_of_delta_membership_allocation")
            or form_attr.get("phase_3r_share_membership")
        ),
        "phase_3r_membership_reproduced": _close(
            scale_primary.get("share_of_delta_membership_allocation")
            or form_attr.get("phase_3r_share_membership"),
            PHASE_3R_SHARE_MEMBERSHIP, _SHARE_REPRO_TOL,
        ),
        "phase_3s_share_pre_play": form_primary.get(
            "share_of_delta_pre_play_membership"
        ),
        "phase_3s_pre_play_reproduced": _close(
            form_primary.get("share_of_delta_pre_play_membership"),
            PHASE_3S_SHARE_PRE_PLAY, _SHARE_REPRO_TOL,
        ),
        "t1_synth_control": t1_c,
        "t1_synth_treatment": t1_t,
        "t3_synth_control": t3_c,
        "t3_synth_treatment": t3_t,
        "t1_synth_reproduced": (
            _close(t1_c, PHASE_3R_T1_SYNTH_CONTROL)
            and _close(t1_t, PHASE_3R_T1_SYNTH_TREATMENT)
        ),
        "t3_synth_reproduced": (
            _close(t3_c, PHASE_3R_T3_SYNTH_CONTROL)
            and _close(t3_t, PHASE_3R_T3_SYNTH_TREATMENT)
        ),
        "primary": primary,
        "membership_propagation": membership_prop,
        "per_tier": per_tier,
        "divergence_reconciliation": {
            "nested_ok": abs(float(primary.get("residual_vs_delta") or 0.0)) <= 1e-6,
            "event_board_flow_ok": n_flow_mismatch == 0,
            "n_event_board_flow_mismatch": n_flow_mismatch,
            "identity": NESTED_DIVERGENCE_IDENTITY,
            "flow_identity": BODY_EVENT_POOL_FLOW_IDENTITY,
            "exclusive_identity": EXCLUSIVE_T5_FIRST_DIFF_IDENTITY,
            "lifecycle_identity": LIFECYCLE_PROPAGATION_IDENTITY,
        },
        "examples": examples,
        "modal_first_event_kind": modal_kind,
        "lifecycle_3q": {
            "n_pairs": life_primary.get("n_pairs"),
            "share_same_state": life_primary.get(
                "share_of_delta_same_state_repaint"
            ),
            "share_lifecycle": life_primary.get(
                "share_of_delta_replacement_lifecycle"
            ),
            "share_scaling": life_primary.get(
                "share_of_delta_subsequent_scaling"
            ),
            "share_residual": life_primary.get("share_of_delta_residual"),
            "n_snapshots_complete": life_primary.get("n_snapshots_complete"),
        },
        "scale_3r": {
            "n_pairs": scale_primary.get("n_pairs"),
            "share_input": scale_primary.get("share_of_delta_pre_sync_input_state"),
            "share_timing": scale_primary.get("share_of_delta_sync_timing_count"),
            "share_membership": scale_primary.get(
                "share_of_delta_membership_allocation"
            ),
            "share_rounding": scale_primary.get("share_of_delta_rounding_residue"),
            "n_snapshots_complete": scale_primary.get("n_snapshots_complete"),
        },
        "formation_3s": {
            "n_pairs": form_primary.get("n_pairs"),
            "share_pre_play": form_primary.get(
                "share_of_delta_pre_play_membership"
            ),
            "share_incoming": form_primary.get("share_of_delta_incoming_identity"),
            "share_opening": form_primary.get("share_of_delta_slot_opening_cause"),
            "share_order": form_primary.get("share_of_delta_buy_play_order"),
            "modal_earliest": form_attr.get("modal_earliest_membership_diverge_turn")
            or form_primary.get("modal_earliest_membership_diverge_turn"),
            "n_snapshots_complete": form_primary.get("n_snapshots_complete"),
        },
        "published_3s_locks": {
            "n_pairs": PHASE_3S_PRIMARY_N_PAIRS,
            "n_fights": PHASE_3S_PRIMARY_N_FIGHTS,
            "share_pre_play": PHASE_3S_SHARE_PRE_PLAY,
            "share_incoming": PHASE_3S_SHARE_INCOMING,
            "share_opening": PHASE_3S_SHARE_OPENING,
            "share_order": PHASE_3S_SHARE_ORDER,
            "share_residual": PHASE_3S_SHARE_RESIDUAL,
            "share_membership_prop": PHASE_3S_SHARE_MEMBERSHIP_PROP,
            "lifecycle_abs_mass": PHASE_3S_LIFECYCLE_ABS_MASS,
            "membership_abs_mass": PHASE_3S_MEMBERSHIP_PROP_ABS_MASS,
            "same_identity_rate": PHASE_3S_SAME_PRE_PLAY_IDENTITY_RATE,
            "same_state_rate": PHASE_3S_SAME_PRE_PLAY_STATE_RATE,
            "modal_earliest": PHASE_3S_MODAL_EARLIEST,
        },
    }


def compare_t5_incumbent_synth(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    selection: Optional[Dict] = None,
    pairing: Optional[Dict] = None,
    timing: Optional[Dict] = None,
    chain: Optional[Dict] = None,
    first: Optional[Dict] = None,
    matched: Optional[Dict] = None,
    mechanics: Optional[Dict] = None,
    allocation: Optional[Dict] = None,
    lifecycle: Optional[Dict] = None,
    scale: Optional[Dict] = None,
    formation: Optional[Dict] = None,
) -> Dict:
    """3S lock + T5 incumbent-synth first-diff split."""
    if formation is None:
        formation = compare_open_slot_formation(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain, first=first, matched=matched,
            mechanics=mechanics, allocation=allocation,
            lifecycle=lifecycle, scale=scale,
        )
    if scale is None:
        scale = {
            "attribution": formation.get("scale_3r"),
            "primary": formation.get("scale_primary_3r"),
            "lifecycle_3q": formation.get("lifecycle_3q"),
            "lifecycle_primary_3q": formation.get("lifecycle_primary_3q"),
            "matched_state": formation.get("matched_state"),
            "mechanics_3o": formation.get("mechanics_3o"),
            "allocation_3p": formation.get("allocation_3p"),
            "source": formation.get("source"),
            "first_divergence_3m": formation.get("first_divergence_3m"),
            "decomposition_3g": formation.get("decomposition_3g"),
            "reconciliation": formation.get("reconciliation"),
        }
    if lifecycle is None:
        lifecycle = {
            "attribution": formation.get("lifecycle_3q"),
            "primary": formation.get("lifecycle_primary_3q"),
            "matched_state": formation.get("matched_state"),
            "mechanics_3o": formation.get("mechanics_3o"),
            "allocation_3p": formation.get("allocation_3p"),
            "source": formation.get("source"),
            "first_divergence_3m": formation.get("first_divergence_3m"),
            "decomposition_3g": formation.get("decomposition_3g"),
            "reconciliation": formation.get("reconciliation"),
        }
    if mechanics is None:
        mechanics = {
            "attribution": formation.get("matched_state"),
            "primary": formation.get("mechanics_3o"),
            "source": formation.get("source"),
            "matched_state": formation.get("matched_state"),
            "first_divergence_3m": formation.get("first_divergence_3m"),
            "decomposition_3g": formation.get("decomposition_3g"),
        }
    if matched is None:
        matched = {
            "attribution": formation.get("matched_state"),
            "reconciliation": formation.get("reconciliation"),
        }
    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch,
        still_fields_t1t3=False,
    )
    attr = attribute_t5_incumbent_synth(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch,
        matched=matched, mechanics=mechanics, allocation=allocation,
        lifecycle=lifecycle, scale=scale, formation=formation,
    )
    rec = dict((formation or {}).get("reconciliation") or {})
    rec.update({
        "nested_divergence_identity": NESTED_DIVERGENCE_IDENTITY,
        "body_event_pool_flow_identity": BODY_EVENT_POOL_FLOW_IDENTITY,
        "exclusive_t5_first_diff_identity": EXCLUSIVE_T5_FIRST_DIFF_IDENTITY,
        "lifecycle_propagation_identity": LIFECYCLE_PROPAGATION_IDENTITY,
        "phase_3q_lifecycle_reproduced": attr.get("phase_3q_lifecycle_reproduced"),
        "phase_3r_membership_reproduced": attr.get("phase_3r_membership_reproduced"),
        "phase_3s_pre_play_reproduced": attr.get("phase_3s_pre_play_reproduced"),
        "t1_synth_reproduced": attr.get("t1_synth_reproduced"),
        "t3_synth_reproduced": attr.get("t3_synth_reproduced"),
        "divergence_nested_ok": (attr.get("divergence_reconciliation") or {}).get(
            "nested_ok"
        ),
        "event_board_flow_ok": (attr.get("divergence_reconciliation") or {}).get(
            "event_board_flow_ok"
        ),
    })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": attr,
        "primary": attr.get("primary"),
        "per_tier": attr.get("per_tier"),
        "membership_propagation": attr.get("membership_propagation"),
        "lifecycle_3q": attr.get("lifecycle_3q"),
        "scale_3r": attr.get("scale_3r"),
        "formation_3s": attr.get("formation_3s"),
        "source": formation.get("source") if formation else None,
        "matched_state": formation.get("matched_state") if formation else None,
        "mechanics_3o": formation.get("mechanics_3o") if formation else None,
        "allocation_3p": formation.get("allocation_3p") if formation else None,
        "lifecycle_primary_3q": (
            formation.get("lifecycle_primary_3q") if formation else None
        ),
        "scale_primary_3r": formation.get("scale_primary_3r") if formation else None,
        "formation_primary_3s": formation.get("primary") if formation else None,
        "first_divergence_3m": (
            formation.get("first_divergence_3m") if formation else None
        ),
        "reconciliation": rec,
        "decomposition_3g": formation.get("decomposition_3g") if formation else None,
        "published_3s_locks": attr.get("published_3s_locks"),
        "published_3r_locks": formation.get("published_3r_locks") if formation else None,
        "published_3q_locks": formation.get("published_3q_locks") if formation else None,
        "published_3p_locks": formation.get("published_3p_locks") if formation else None,
        "published_3o_locks": formation.get("published_3o_locks") if formation else None,
        "published_3n_locks": formation.get("published_3n_locks") if formation else None,
    }


__all__ = [
    "T5IncumbentSynthTracer",
    "attribute_t5_incumbent_synth",
    "build_t5_event_stream",
    "compare_t5_incumbent_synth",
    "decompose_t5_synth_pair",
    "diagnose_phase_3t",
    "first_synth_component",
    "in_t5_walk_window",
    "incumbent_snapshot",
    "run_greedy_2s_treatment_t5_incumbent",
    "run_greedy_control_t5_incumbent",
    "sticky_after_membership",
    "tier_mass_divergence",
    "tier_mass_membership_prop",
]
