"""Phase 3S — observational open-slot board-formation attribution.

Reuses the 3R T5/T6 class-(3) walk on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For each arm, traces backward from the last open-slot play to the event
that created that open slot, then splits the published 3Q
replacement_lifecycle term exclusively:

  replacement_lifecycle = pre_play_membership + incoming_identity
                        + slot_opening_cause + buy_play_order + residual

The same exclusive tag is applied to the 3R membership_allocation
increment on that paired body.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    A_BUY0,
    BUY_COST,
    MAX_BOARD,
    N_BUY,
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
from ml.pairing_who_wins_diagnostic import _index_seat_fights
from ml.phase_3s_prereg import (
    EXCLUSIVE_FIRST_DIFF_IDENTITY,
    FORMATION_COMPONENTS,
    FORMATION_FLOW_RECONCILE_IDENTITY,
    MEMBERSHIP_PROPAGATION_IDENTITY,
    NESTED_FORMATION_IDENTITY,
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
    SLOT_OPENING_CAUSES,
    diagnose_phase_3s,
)
from ml.play_lifecycle_diagnostic import (
    _index_plays,
    _last_play,
    _minion_snap,
    _safe_int,
    compare_play_lifecycle,
    decompose_play_pair,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.scale_sync_diagnostic import (
    ScaleSyncTracer,
    _index_syncs,
    _post_play_syncs,
    compare_scale_sync,
    decompose_scale_pair,
)
from ml.survivor_composition_diagnostic import TIERS, clamp_tier
from ml.survivor_mechanic_diagnostic import (
    _fight_for_event,
    _primary_turn,
    collect_class3_minions,
)
from ml.synthetic_allocation_diagnostic import _safe_div

METHODOLOGY_VERSION = "3s_v1"
_N_EXAMPLES = 8
_SYNTH_REPRO_TOL = 0.15
_SHARE_REPRO_TOL = 1e-9
_FLOW_ABS_TOL = 1e-6


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _as_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def board_membership_key(slots: Optional[Sequence[Dict]]) -> Tuple:
    """Sorted (card_id, tier, recruit_raw) multiset; ignores synth/slot."""
    rows = []
    for s in slots or []:
        rows.append((
            str(s.get("card_id") or s.get("name") or ""),
            int(s.get("tier") or 1),
            int(s.get("recruit_raw") or 0),
        ))
    return tuple(sorted(rows))


def board_composition_key(slots: Optional[Sequence[Dict]]) -> Tuple:
    """Slot-ordered identity; placement-sensitive composition."""
    ordered = sorted(
        list(slots or []),
        key=lambda r: int(r.get("slot") if r.get("slot") is not None
                          else r.get("board_slot") or 0),
    )
    return tuple(
        (
            str(s.get("card_id") or s.get("name") or ""),
            int(s.get("tier") or 1),
            int(s.get("recruit_raw") or 0),
        )
        for s in ordered
    )


def incoming_identity_key(incoming: Optional[Dict]) -> Tuple:
    if not incoming:
        return ("", 0, 0)
    return (
        str(incoming.get("card_id") or incoming.get("name") or ""),
        int(incoming.get("tier") or 0),
        int(incoming.get("recruit_raw") or incoming.get("incoming_recruit_raw") or 0),
    )


def slot_opening_key(play: Optional[Dict]) -> Tuple:
    play = play or {}
    return (
        str(play.get("slot_opening_cause") or ""),
        _safe_int(play.get("slot_opening_turn"), -1),
        _safe_int(play.get("turn"), -1),
        _safe_int(play.get("turns_open"), -1),
    )


def buy_play_order_key(play: Optional[Dict]) -> Tuple:
    play = play or {}
    order = play.get("buy_play_order") or []
    return (
        tuple(str(k) for k in order),
        _safe_int(play.get("gold"), -1),
        bool(play.get("could_afford_buy")),
    )


def shop_offer_key(offers: Optional[Sequence[Dict]]) -> Tuple:
    rows = []
    for s in offers or []:
        rows.append((
            str(s.get("card_id") or s.get("name") or ""),
            int(s.get("tier") or 1),
            int(s.get("recruit_raw") or 0),
        ))
    return tuple(sorted(rows))


def classify_slot_opening_cause(
    play: Dict,
    vacancy_events: Sequence[Dict],
) -> Dict:
    """Most recent unrefilled vacancy before this play.

    Vacancy kinds: prior_sell, death_cleanup, triple_transform.
    If none remain unrefilled, the slot is normal under-fill.
    """
    play_turn = _safe_int(play.get("turn"), 0) or 0
    play_seq = _safe_int(play.get("seq"), play.get("action_seq"))
    play_seq = 0 if play_seq is None else int(play_seq)
    board_len_before = _safe_int(play.get("board_len_before"), 0) or 0

    prior = []
    for ev in vacancy_events or []:
        turn = _safe_int(ev.get("turn"), 0) or 0
        seq = _safe_int(ev.get("seq"), ev.get("action_seq"))
        seq = -1 if seq is None else int(seq)
        if turn < play_turn or (turn == play_turn and seq < play_seq):
            prior.append((turn, seq, ev))
    prior.sort(key=lambda r: (r[0], r[1]))

    last_vacancy = None
    last_target = None
    for turn, seq, ev in prior:
        kind = str(ev.get("vacancy_kind") or ev.get("kind") or "")
        before = _safe_int(ev.get("board_len_before"), 0) or 0
        after = _safe_int(ev.get("board_len_after"), 0) or 0
        if kind in ("prior_sell", "death_cleanup", "triple_transform") or after < before:
            last_vacancy = ev
            last_target = before
        elif kind == "play" or after > before:
            if last_vacancy is not None and last_target is not None:
                if after >= last_target:
                    last_vacancy = None
                    last_target = None

    if last_vacancy is None:
        cause = "normal_underfill"
        open_turn = None
        open_kind = "normal_underfill"
        open_seq = None
    else:
        open_kind = str(
            last_vacancy.get("vacancy_kind") or last_vacancy.get("kind") or ""
        )
        if open_kind not in SLOT_OPENING_CAUSES:
            if open_kind in ("sell", "prior_sell"):
                open_kind = "prior_sell"
            elif open_kind in ("triple", "triple_transform", "transform"):
                open_kind = "triple_transform"
            elif open_kind in ("death", "death_cleanup", "generated_cleanup"):
                open_kind = "death_cleanup"
            else:
                open_kind = "normal_underfill"
        cause = open_kind
        open_turn = _safe_int(last_vacancy.get("turn"))
        open_seq = _safe_int(
            last_vacancy.get("seq"), last_vacancy.get("action_seq")
        )

    turns_open = None
    if open_turn is not None:
        turns_open = int(play_turn) - int(open_turn)
    flow_gap = None
    if last_vacancy is not None:
        after = _safe_int(last_vacancy.get("board_len_after"), 0) or 0
        flow_gap = int(board_len_before) - int(after)
    return {
        "slot_opening_cause": cause,
        "slot_opening_turn": open_turn,
        "slot_opening_seq": open_seq,
        "slot_opening_kind": open_kind,
        "turns_open": turns_open,
        "vacancy_board_len_after": (
            None if last_vacancy is None
            else _safe_int(last_vacancy.get("board_len_after"))
        ),
        "opening_flow_gap": flow_gap,
        "underfill": board_len_before < MAX_BOARD,
    }


def first_formation_component(
    control_play: Optional[Dict],
    treatment_play: Optional[Dict],
) -> str:
    """Exclusive first-difference rank on last-play snapshots."""
    if not control_play or not treatment_play:
        return "residual"
    c_pre = control_play.get("pre_play") or []
    t_pre = treatment_play.get("pre_play") or []
    if (
        board_membership_key(c_pre) != board_membership_key(t_pre)
        or board_composition_key(c_pre) != board_composition_key(t_pre)
        or _safe_int(control_play.get("board_len_before"), 0)
        != _safe_int(treatment_play.get("board_len_before"), 0)
    ):
        return "pre_play_membership"
    c_in = control_play.get("incoming") or {
        "card_id": None,
        "tier": control_play.get("incoming_tier"),
        "recruit_raw": control_play.get("incoming_recruit_raw"),
    }
    t_in = treatment_play.get("incoming") or {
        "card_id": None,
        "tier": treatment_play.get("incoming_tier"),
        "recruit_raw": treatment_play.get("incoming_recruit_raw"),
    }
    if incoming_identity_key(c_in) != incoming_identity_key(t_in):
        return "incoming_identity"
    if slot_opening_key(control_play) != slot_opening_key(treatment_play):
        return "slot_opening_cause"
    if buy_play_order_key(control_play) != buy_play_order_key(treatment_play):
        return "buy_play_order"
    return "residual"


def earliest_membership_diverge_turn(
    control_plays: Sequence[Dict],
    treatment_plays: Sequence[Dict],
    fight_turn: Optional[int] = None,
) -> Optional[int]:
    """First turn where last-play pre-play membership keys differ."""
    by_turn_c: Dict[int, Dict] = {}
    by_turn_t: Dict[int, Dict] = {}
    hi = _safe_int(fight_turn)
    for ev in control_plays or []:
        turn = _safe_int(ev.get("turn"))
        if turn is None:
            continue
        if hi is not None and turn > hi:
            continue
        by_turn_c[turn] = ev
    for ev in treatment_plays or []:
        turn = _safe_int(ev.get("turn"))
        if turn is None:
            continue
        if hi is not None and turn > hi:
            continue
        by_turn_t[turn] = ev
    for turn in sorted(set(by_turn_c) | set(by_turn_t)):
        c = by_turn_c.get(turn) or {}
        t = by_turn_t.get(turn) or {}
        if board_membership_key(c.get("pre_play")) != board_membership_key(
            t.get("pre_play")
        ):
            return int(turn)
        if incoming_identity_key(c.get("incoming")) != incoming_identity_key(
            t.get("incoming")
        ):
            return int(turn)
    return None


def decompose_formation_pair(
    control_start: Dict,
    treatment_start: Dict,
    control_play: Optional[Dict],
    treatment_play: Optional[Dict],
    control_syncs: Optional[Sequence[Dict]] = None,
    treatment_syncs: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Exclusive five-way split of 3Q replacement_lifecycle.

    Identity (residual 0 when snapshots exist and first-diff is assigned):
      replacement_lifecycle = pre_play_membership + incoming_identity
                            + slot_opening_cause + buy_play_order + residual
    The tagged 3R membership_allocation rides along on the same component.
    """
    q = decompose_play_pair(
        control_start, treatment_start, control_play, treatment_play,
    )
    scale = decompose_scale_pair(
        control_start, treatment_start, control_play, treatment_play,
        control_syncs or [], treatment_syncs or [],
    )
    lifecycle = float(q.get("replacement_lifecycle") or 0.0)
    membership_inc = float(scale.get("membership_allocation") or 0.0)
    parts = {name: 0.0 for name in FORMATION_COMPONENTS}
    memb_parts = {name: 0.0 for name in FORMATION_COMPONENTS}
    complete = bool(q.get("snapshots_complete"))
    if not complete:
        component = "residual"
        parts["residual"] = lifecycle
        memb_parts["residual"] = membership_inc
    else:
        component = first_formation_component(control_play, treatment_play)
        parts[component] = lifecycle
        memb_parts[component] = membership_inc
    explained = sum(parts[n] for n in FORMATION_COMPONENTS)
    return {
        "replacement_lifecycle": lifecycle,
        "membership_allocation": membership_inc,
        "subsequent_scaling": float(q.get("subsequent_scaling") or 0.0),
        "same_state_repaint": float(q.get("same_state_repaint") or 0.0),
        "delta_synth": float(q.get("delta_synth") or 0.0),
        **parts,
        **{f"membership_{name}": memb_parts[name] for name in FORMATION_COMPONENTS},
        "formation_component": component,
        "snapshots_complete": complete,
        "explained_lifecycle": explained,
        "residual_vs_lifecycle": lifecycle - explained,
        "control_subtype": q.get("control_subtype"),
        "treatment_subtype": q.get("treatment_subtype"),
        "subtype_mismatch": q.get("subtype_mismatch"),
        "control_opening": None if not control_play else control_play.get(
            "slot_opening_cause"
        ),
        "treatment_opening": None if not treatment_play else treatment_play.get(
            "slot_opening_cause"
        ),
        "opening_mismatch": (
            None if not control_play or not treatment_play
            else control_play.get("slot_opening_cause")
            != treatment_play.get("slot_opening_cause")
        ),
        "s_c_sticky": q.get("s_c_sticky"),
        "s_t_sticky_cf": q.get("s_t_sticky_cf"),
        "s_c_start": q.get("s_c_start"),
        "s_t_start": q.get("s_t_start"),
        "flow_gap_control": scale.get("flow_gap_control"),
        "flow_gap_treatment": scale.get("flow_gap_treatment"),
    }


class OpenSlotFormationTracer(ScaleSyncTracer):
    """3R scale-sync rows plus recruit-op / vacancy / shop-gold snapshots."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.recruit_ops: List[Dict] = []
        self.combat_shrinks: List[Dict] = []
        self._pre_econ: Dict[int, Dict] = {}
        self._action_seq: Dict[int, int] = {}
        self._end_recruit_board: Dict[int, List[Dict]] = {}

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._pre_econ.clear()
        self._action_seq.clear()
        self._end_recruit_board.clear()

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        super().begin_seat_recruit(seat, turn, player)
        self._action_seq[int(seat)] = 0
        board = list(getattr(player, "board", None) or [])
        snap = [_minion_snap(m, i) for i, m in enumerate(board)]
        prev = self._end_recruit_board.get(int(seat))
        if prev is not None and len(snap) < len(prev):
            self.combat_shrinks.append({
                "seed": int(self.seed),
                "lobby": int(self.lobby_id),
                "arm": self.arm,
                "seat": int(seat),
                "turn": int(turn),
                "seq": -1,
                "kind": "death_cleanup",
                "vacancy_kind": "death_cleanup",
                "board_len_before": len(prev),
                "board_len_after": len(snap),
                "pre_slots": [dict(s) for s in prev],
                "post_slots": [dict(s) for s in snap],
            })
        self._end_recruit_board[int(seat)] = [dict(s) for s in snap]

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask,
    ) -> None:
        super().before_action(seat, turn, shop_generation, obs, mask)
        p = self._live_player.get(int(seat))
        shop_src = []
        if p is not None:
            shop_src = list(getattr(p, "shop", None) or [])
        elif obs:
            shop_src = list(obs.get("shop") or [])
        offers = []
        for i, m in enumerate(shop_src):
            if hasattr(m, "card_id") or hasattr(m, "name"):
                offers.append(_minion_snap(m, i, zone="shop"))
            elif isinstance(m, dict):
                offers.append({
                    "slot": i,
                    "zone": "shop",
                    "name": str(m.get("name") or ""),
                    "card_id": str(m.get("card_id") or m.get("name") or ""),
                    "tier": int(m.get("tier") or 1),
                    "recruit_raw": int(
                        m.get("recruit_raw")
                        or (int(m.get("attack") or 0) + int(m.get("health") or 0))
                    ),
                })
        gold = None
        if obs is not None and obs.get("gold") is not None:
            gold = int(obs.get("gold") or 0)
        elif p is not None:
            gold = int(getattr(p, "gold", 0) or 0)
        can_buy = False
        if mask is not None and gold is not None and gold >= BUY_COST:
            for i in range(N_BUY):
                try:
                    if mask[A_BUY0 + i]:
                        can_buy = True
                        break
                except (TypeError, IndexError, KeyError):
                    break
        self._pre_econ[int(seat)] = {
            "gold": gold,
            "shop": offers,
            "shop_generation": int(shop_generation),
            "could_afford_buy": bool(can_buy and offers),
        }

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int,
        ended: bool, player=None,
    ) -> None:
        econ = dict(self._pre_econ.get(int(seat)) or {})
        ids_before = list(self._pre_ids.get(int(seat), []))
        pre_slots = list(self._pre_slots.get(int(seat), []))
        kind = decode_recruit_action(action)
        seq = int(self._action_seq.get(int(seat), 0))
        super().after_action(seat, turn, shop_generation, action, ended, player)
        p = player if player is not None else self._live_player.get(int(seat))
        board = list(getattr(p, "board", None) or []) if p is not None else []
        ids_after = [id(m) for m in board]
        event_kind = classify_membership_event(action, ids_before, ids_after)
        vacancy_kind = None
        if kind == "sell" and len(board) < len(pre_slots):
            vacancy_kind = "prior_sell"
        if event_kind == "triple":
            vacancy_kind = "triple_transform"
        op = {
            "seed": int(self.seed),
            "lobby": int(self.lobby_id),
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(turn),
            "seq": seq,
            "kind": kind,
            "event": event_kind,
            "vacancy_kind": vacancy_kind,
            "board_len_before": len(pre_slots),
            "board_len_after": len(board),
            "gold": econ.get("gold"),
            "shop": list(econ.get("shop") or []),
            "could_afford_buy": bool(econ.get("could_afford_buy")),
            "shop_generation": econ.get("shop_generation"),
        }
        self.recruit_ops.append(op)
        self._action_seq[int(seat)] = seq + 1
        if p is not None and (ended or kind in ("other",)):
            self._end_recruit_board[int(seat)] = [
                _minion_snap(m, i) for i, m in enumerate(board)
            ]
        if kind != "play" or not self.play_events:
            return
        last = self.play_events[-1]
        if (
            _safe_int(last.get("seed")) != int(self.seed)
            or _safe_int(last.get("seat")) != int(seat)
            or _safe_int(last.get("turn")) != int(turn)
        ):
            return
        turn_ops = [
            o for o in self.recruit_ops
            if _safe_int(o.get("seat")) == int(seat)
            and _safe_int(o.get("turn")) == int(turn)
            and _safe_int(o.get("seed")) == int(self.seed)
        ]
        last.update({
            "action_seq": seq,
            "gold": econ.get("gold"),
            "shop": list(econ.get("shop") or []),
            "shop_offer_set": shop_offer_key(econ.get("shop") or []),
            "could_afford_buy": bool(econ.get("could_afford_buy")),
            "buy_play_order": [str(o.get("kind") or "") for o in turn_ops],
            "n_buys_this_turn": sum(1 for o in turn_ops if o.get("kind") == "buy"),
            "n_sells_this_turn": sum(1 for o in turn_ops if o.get("kind") == "sell"),
            "n_plays_this_turn": sum(1 for o in turn_ops if o.get("kind") == "play"),
        })

    def after_scale_all(self, env: BGEnv) -> None:
        super().after_scale_all(env)
        for p in list(getattr(env, "players", None) or []):
            seat = int(getattr(p, "idx", -1))
            board = [_minion_snap(m, i) for i, m in enumerate(list(p.board or []))]
            self._end_recruit_board[seat] = board


def run_open_slot_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    from ml.phase_3s_prereg import assert_seed_range_allowed
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

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = OpenSlotFormationTracer(i, seed + i, arm)
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
    }


def run_greedy_control_open_slot(lobbies: int, seed: int) -> Dict:
    return run_open_slot_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_open_slot(lobbies: int, seed: int) -> Dict:
    return run_open_slot_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _index_ops(ops: Sequence[Dict]) -> Dict[Tuple[int, int], List[Dict]]:
    out: Dict[Tuple[int, int], List[Dict]] = {}
    for ev in ops or []:
        seed = _safe_int(ev.get("seed"))
        seat = _safe_int(ev.get("seat"))
        if seed is None or seat is None:
            continue
        out.setdefault((seed, seat), []).append(ev)
    for key, rows in out.items():
        rows.sort(key=lambda r: (
            int(r.get("turn") or 0),
            int(r.get("seq") if r.get("seq") is not None else r.get("action_seq") or 0),
        ))
        out[key] = rows
    return out


def _vacancy_stream(
    ops_index: Dict[Tuple[int, int], List[Dict]],
    shrink_index: Dict[Tuple[int, int], List[Dict]],
    seed, seat,
) -> List[Dict]:
    seed_i = _safe_int(seed)
    seat_i = _safe_int(seat)
    if seed_i is None or seat_i is None:
        return []
    rows = []
    for ev in ops_index.get((seed_i, seat_i), []) or []:
        rows.append(ev)
    for ev in shrink_index.get((seed_i, seat_i), []) or []:
        rows.append(ev)
    rows.sort(key=lambda r: (
        int(r.get("turn") or 0),
        int(r.get("seq") if r.get("seq") is not None else r.get("action_seq") or -1),
    ))
    return rows


def stamp_play_opening(play: Dict, vacancy_events: Sequence[Dict]) -> Dict:
    stamped = dict(play)
    opening = classify_slot_opening_cause(stamped, vacancy_events)
    stamped.update(opening)
    return stamped


def _sum_parts(pairs: Sequence[Dict], names: Sequence[str], extra: str) -> Dict[str, float]:
    totals = {name: 0.0 for name in names}
    totals[extra] = 0.0
    for p in pairs:
        for name in names:
            totals[name] += float(p.get(name) or 0.0)
        totals[extra] += float(p.get(extra) or 0.0)
    return totals


def tier_mass_formation(per_tier: Dict[str, Dict]) -> Dict:
    """Decision shares from within-tier |parts| so T1↓ / T3↑ cannot cancel."""
    mass = {name: 0.0 for name in FORMATION_COMPONENTS}
    n_used = 0
    abs_delta = 0.0
    for cell in (per_tier or {}).values():
        n = int(cell.get("n_pairs") or 0)
        if n <= 0:
            continue
        n_used += n
        abs_delta += n * abs(float(cell.get("replacement_lifecycle") or 0.0))
        for name in FORMATION_COMPONENTS:
            mass[name] += n * abs(float(cell.get(name) or 0.0))
    total_mass = sum(mass.values())

    def _share(part: float) -> Optional[float]:
        if total_mass < 1e-12:
            return None
        return float(part) / total_mass

    return {
        "method": "within_tier_abs_mass_open_slot_formation_identity",
        "n_pairs": n_used,
        "abs_replacement_lifecycle": (
            None if n_used <= 0 else float(abs_delta) / float(n_used)
        ),
        "component_abs_mass": mass,
        "total_abs_mass": total_mass,
        **{name: (
            None if n_used <= 0 else float(mass[name]) / float(n_used)
        ) for name in FORMATION_COMPONENTS},
        **{f"share_of_delta_{name}": _share(mass[name])
           for name in FORMATION_COMPONENTS},
        "formation_components": list(FORMATION_COMPONENTS),
    }


def tier_mass_membership_prop(per_tier: Dict[str, Dict]) -> Dict:
    """How formation tags propagate into 3R membership |mass|."""
    mass = {name: 0.0 for name in FORMATION_COMPONENTS}
    n_used = 0
    abs_delta = 0.0
    for cell in (per_tier or {}).values():
        n = int(cell.get("n_pairs") or 0)
        if n <= 0:
            continue
        n_used += n
        abs_delta += n * abs(float(cell.get("membership_allocation") or 0.0))
        for name in FORMATION_COMPONENTS:
            mass[name] += n * abs(float(cell.get(f"membership_{name}") or 0.0))
    total_mass = sum(mass.values())

    def _share(part: float) -> Optional[float]:
        if total_mass < 1e-12:
            return None
        return float(part) / total_mass

    return {
        "method": "within_tier_abs_mass_membership_propagation",
        "n_pairs": n_used,
        "abs_membership_allocation": (
            None if n_used <= 0 else float(abs_delta) / float(n_used)
        ),
        "component_abs_mass": mass,
        "total_abs_mass": total_mass,
        **{f"share_of_membership_{name}": _share(mass[name])
           for name in FORMATION_COMPONENTS},
        "formation_components": list(FORMATION_COMPONENTS),
    }


def attribute_open_slot_formation(
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
) -> Dict:
    """3R locks plus last-play open-slot formation split."""
    if scale is None:
        scale = compare_scale_sync(
            control_raw, treatment_raw,
            matched=matched, mechanics=mechanics, allocation=allocation,
            lifecycle=lifecycle,
        )
    if lifecycle is None:
        lifecycle = {
            "attribution": (scale or {}).get("lifecycle_3q"),
            "primary": (scale or {}).get("lifecycle_primary_3q"),
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
    earliest_turns: List[int] = []
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
        parts = decompose_formation_pair(
            c_row, t_row, c_play, t_play, c_post, t_post,
        )
        c_hist = c_plays.get((_safe_int(ev_seed), _safe_int(c_winner))) or []
        t_hist = t_plays.get((_safe_int(ev_seed), _safe_int(t_winner))) or []
        earliest = earliest_membership_diverge_turn(c_hist, t_hist, ev_turn)
        if earliest is not None:
            earliest_turns.append(int(earliest))
        sticky_flow_c = None
        sticky_flow_t = None
        if c_play is not None:
            implicit = int(c_play.get("implicit_sticky_pool") or 0)
            sticky_sum = int(c_play.get("sticky_shares_sum") or 0)
            sticky_flow_c = sticky_sum - implicit
        if t_play is not None:
            implicit = int(t_play.get("implicit_sticky_pool") or 0)
            sticky_sum = int(t_play.get("sticky_shares_sum") or 0)
            sticky_flow_t = sticky_sum - implicit
        if (
            (sticky_flow_c is not None and abs(sticky_flow_c) > _FLOW_ABS_TOL)
            or (sticky_flow_t is not None and abs(sticky_flow_t) > _FLOW_ABS_TOL)
        ):
            n_flow_mismatch += 1
        opening_gap_c = None if c_play is None else c_play.get("opening_flow_gap")
        opening_gap_t = None if t_play is None else t_play.get("opening_flow_gap")
        parts.update({
            "seed": ev_seed,
            "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
            "tier": clamp_tier(p.get("tier") or c_row.get("tier") or t_row.get("tier")),
            "board_slot": p.get("board_slot"),
            "earliest_membership_diverge_turn": earliest,
            "sticky_flow_gap_control": sticky_flow_c,
            "sticky_flow_gap_treatment": sticky_flow_t,
            "opening_flow_gap_control": opening_gap_c,
            "opening_flow_gap_treatment": opening_gap_t,
        })
        if parts.get("snapshots_complete"):
            n_complete += 1
        else:
            n_missing_play += 1
        pairs.append(parts)

    for r in unpaired_t:
        s = float(r.get("synthetic_share") or 0)
        bag = {name: 0.0 for name in FORMATION_COMPONENTS}
        bag["pre_play_membership"] = s
        memb = {f"membership_{name}": 0.0 for name in FORMATION_COMPONENTS}
        pairs.append({
            "replacement_lifecycle": s,
            "membership_allocation": 0.0,
            **bag,
            **memb,
            "formation_component": "pre_play_membership",
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "treatment",
            "snapshots_complete": False,
        })
    for r in unpaired_c:
        s = float(r.get("synthetic_share") or 0)
        bag = {name: 0.0 for name in FORMATION_COMPONENTS}
        bag["pre_play_membership"] = -s
        memb = {f"membership_{name}": 0.0 for name in FORMATION_COMPONENTS}
        pairs.append({
            "replacement_lifecycle": -s,
            "membership_allocation": 0.0,
            **bag,
            **memb,
            "formation_component": "pre_play_membership",
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "control",
            "snapshots_complete": False,
        })

    totals = _sum_parts(pairs, FORMATION_COMPONENTS, "replacement_lifecycle")
    memb_totals = _sum_parts(
        [
            {
                **{name: p.get(f"membership_{name}") for name in FORMATION_COMPONENTS},
                "membership_allocation": p.get("membership_allocation"),
            }
            for p in pairs
        ],
        FORMATION_COMPONENTS,
        "membership_allocation",
    )
    n_pairs = max(1, len(pairs))
    means = {k: float(v) / float(n_pairs) for k, v in totals.items()}
    obs_delta = means["replacement_lifecycle"]

    def _share(part: float) -> Optional[float]:
        if abs(obs_delta) < 1e-12:
            return None
        return float(part) / obs_delta

    pooled_signed = {
        "method": "exact_open_slot_formation_paired_slot_identity",
        "n_pairs": len(pairs),
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "n_snapshots_complete": n_complete,
        "n_missing_play": n_missing_play,
        "replacement_lifecycle": means["replacement_lifecycle"],
        **{name: means[name] for name in FORMATION_COMPONENTS},
        "explained_all_parts": sum(means[n] for n in FORMATION_COMPONENTS),
        "residual_vs_delta": (
            means["replacement_lifecycle"]
            - sum(means[n] for n in FORMATION_COMPONENTS)
        ),
        **{f"share_of_delta_{name}": _share(means[name])
           for name in FORMATION_COMPONENTS},
        "formation_components": list(FORMATION_COMPONENTS),
    }
    per_tier = {}
    for tier in TIERS:
        cell = [p for p in pairs if int(p.get("tier") or 1) == tier]
        if not cell:
            per_tier[str(tier)] = {"n_pairs": 0}
            continue
        ct = _sum_parts(cell, FORMATION_COMPONENTS, "replacement_lifecycle")
        mt = _sum_parts(
            [
                {
                    **{name: p.get(f"membership_{name}") for name in FORMATION_COMPONENTS},
                    "membership_allocation": p.get("membership_allocation"),
                }
                for p in cell
            ],
            FORMATION_COMPONENTS,
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
            **{name: cm[name] for name in FORMATION_COMPONENTS},
            **{f"membership_{name}": mm[name] for name in FORMATION_COMPONENTS},
            **{f"share_of_delta_{name}": _cshare(cm[name])
               for name in FORMATION_COMPONENTS},
            "opening_mismatch_rate": _safe_div(
                float(sum(1 for p in cell if p.get("opening_mismatch"))),
                float(len(cell)),
            ),
        }

    primary = tier_mass_formation(per_tier)
    membership_prop = tier_mass_membership_prop(per_tier)
    cause_counts = Counter(
        (p.get("treatment_opening") or p.get("control_opening"))
        for p in pairs
        if (p.get("treatment_opening") or p.get("control_opening"))
    )
    modal_cause = None
    if cause_counts:
        modal_cause = cause_counts.most_common(1)[0][0]
    modal_earliest = None
    earliest_counts = Counter(earliest_turns)
    if earliest_counts:
        modal_earliest = earliest_counts.most_common(1)[0][0]
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
        "modal_slot_opening_cause": modal_cause,
        "slot_opening_cause_counts": dict(cause_counts),
        "modal_earliest_membership_diverge_turn": modal_earliest,
        "earliest_membership_diverge_counts": {
            str(k): int(v) for k, v in earliest_counts.items()
        },
        "n_event_board_flow_mismatch": n_flow_mismatch,
        "open_slot_rate_control": _safe_div(
            float(sum(1 for p in complete if p.get("control_subtype") == "open_slot")),
            float(len(complete)),
        ),
        "open_slot_rate_treatment": _safe_div(
            float(sum(1 for p in complete if p.get("treatment_subtype") == "open_slot")),
            float(len(complete)),
        ),
    })

    life_attr = (lifecycle or {}).get("attribution") or lifecycle or {}
    life_primary = (lifecycle or {}).get("primary") or life_attr.get("primary") or {}
    scale_attr = (scale or {}).get("attribution") or scale or {}
    scale_primary = (scale or {}).get("primary") or scale_attr.get("primary") or {}

    def _close(got, published, tol=_SYNTH_REPRO_TOL) -> bool:
        if got is None or published is None:
            return False
        return abs(float(got) - float(published)) <= tol

    t1_c = life_attr.get("t1_synth_control") or scale_attr.get("t1_synth_control")
    t1_t = life_attr.get("t1_synth_treatment") or scale_attr.get("t1_synth_treatment")
    t3_c = life_attr.get("t3_synth_control") or scale_attr.get("t3_synth_control")
    t3_t = life_attr.get("t3_synth_treatment") or scale_attr.get("t3_synth_treatment")
    share_lifecycle = life_primary.get("share_of_delta_replacement_lifecycle")
    share_membership = scale_primary.get("share_of_delta_membership_allocation")

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
            "formation_component": p.get("formation_component"),
            "pre_play_membership": p.get("pre_play_membership"),
            "incoming_identity": p.get("incoming_identity"),
            "slot_opening_cause": p.get("slot_opening_cause"),
            "buy_play_order": p.get("buy_play_order"),
            "residual": p.get("residual"),
            "membership_allocation": p.get("membership_allocation"),
            "control_opening": p.get("control_opening"),
            "treatment_opening": p.get("treatment_opening"),
            "earliest_membership_diverge_turn": p.get(
                "earliest_membership_diverge_turn"
            ),
        })

    opening_summary = {
        "control": dict(Counter(
            p.get("control_opening") for p in complete if p.get("control_opening")
        )),
        "treatment": dict(Counter(
            p.get("treatment_opening") for p in complete if p.get("treatment_opening")
        )),
    }
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
        "same_outcome_damage_reproduced": len(events) == PHASE_3N_CLASS3,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_B_reproduced": life_attr.get("phase_3n_B_reproduced")
        or scale_attr.get("phase_3n_B_reproduced"),
        "phase_3o_t5t6_B": PHASE_3O_T5T6_B,
        "phase_3o_B_reproduced": life_attr.get("phase_3o_B_reproduced")
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
        "phase_3q_share_lifecycle": share_lifecycle,
        "phase_3q_share_scaling": life_primary.get(
            "share_of_delta_subsequent_scaling"
        ),
        "phase_3q_share_residual": life_primary.get("share_of_delta_residual"),
        "phase_3q_lifecycle_reproduced": _close(
            share_lifecycle, PHASE_3Q_SHARE_LIFECYCLE, _SHARE_REPRO_TOL,
        ),
        "phase_3r_share_membership": share_membership,
        "phase_3r_membership_reproduced": _close(
            share_membership, PHASE_3R_SHARE_MEMBERSHIP, _SHARE_REPRO_TOL,
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
        "opening_causes": opening_summary,
        "formation_reconciliation": {
            "nested_ok": abs(float(primary.get("residual_vs_delta") or 0.0)) <= 1e-6,
            "event_board_flow_ok": n_flow_mismatch == 0,
            "n_event_board_flow_mismatch": n_flow_mismatch,
            "identity": NESTED_FORMATION_IDENTITY,
            "flow_identity": FORMATION_FLOW_RECONCILE_IDENTITY,
            "exclusive_identity": EXCLUSIVE_FIRST_DIFF_IDENTITY,
            "membership_identity": MEMBERSHIP_PROPAGATION_IDENTITY,
        },
        "examples": examples,
        "modal_slot_opening_cause": modal_cause,
        "modal_earliest_membership_diverge_turn": modal_earliest,
        "lifecycle_3q": {
            "n_pairs": life_primary.get("n_pairs"),
            "share_same_state": life_primary.get(
                "share_of_delta_same_state_repaint"
            ),
            "share_lifecycle": share_lifecycle,
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
            "share_membership": share_membership,
            "share_rounding": scale_primary.get("share_of_delta_rounding_residue"),
            "n_snapshots_complete": scale_primary.get("n_snapshots_complete"),
        },
    }


def compare_open_slot_formation(
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
) -> Dict:
    """3R lock + T5/T6 open-slot formation split."""
    if scale is None:
        scale = compare_scale_sync(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain, first=first, matched=matched,
            mechanics=mechanics, allocation=allocation, lifecycle=lifecycle,
        )
    if lifecycle is None:
        lifecycle = {
            "attribution": scale.get("lifecycle_3q"),
            "primary": scale.get("lifecycle_primary_3q"),
            "matched_state": scale.get("matched_state"),
            "mechanics_3o": scale.get("mechanics_3o"),
            "allocation_3p": scale.get("allocation_3p"),
            "source": scale.get("source"),
            "first_divergence_3m": scale.get("first_divergence_3m"),
            "decomposition_3g": scale.get("decomposition_3g"),
            "reconciliation": scale.get("reconciliation"),
            "published_3p_locks": scale.get("published_3p_locks"),
            "published_3o_locks": scale.get("published_3o_locks"),
            "published_3n_locks": scale.get("published_3n_locks"),
        }
    if mechanics is None:
        mechanics = {
            "attribution": scale.get("matched_state"),
            "primary": scale.get("mechanics_3o"),
            "source": scale.get("source"),
            "matched_state": scale.get("matched_state"),
            "first_divergence_3m": scale.get("first_divergence_3m"),
            "decomposition_3g": scale.get("decomposition_3g"),
        }
    if matched is None:
        matched = {
            "attribution": scale.get("matched_state"),
            "reconciliation": scale.get("reconciliation"),
        }
    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch,
        still_fields_t1t3=False,
    )
    attr = attribute_open_slot_formation(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch,
        matched=matched, mechanics=mechanics, allocation=allocation,
        lifecycle=lifecycle, scale=scale,
    )
    rec = dict((scale or {}).get("reconciliation") or {})
    rec.update({
        "nested_formation_identity": NESTED_FORMATION_IDENTITY,
        "formation_flow_reconcile_identity": FORMATION_FLOW_RECONCILE_IDENTITY,
        "exclusive_first_diff_identity": EXCLUSIVE_FIRST_DIFF_IDENTITY,
        "membership_propagation_identity": MEMBERSHIP_PROPAGATION_IDENTITY,
        "phase_3q_lifecycle_reproduced": attr.get("phase_3q_lifecycle_reproduced"),
        "phase_3r_membership_reproduced": attr.get("phase_3r_membership_reproduced"),
        "t1_synth_reproduced": attr.get("t1_synth_reproduced"),
        "t3_synth_reproduced": attr.get("t3_synth_reproduced"),
        "formation_nested_ok": (attr.get("formation_reconciliation") or {}).get(
            "nested_ok"
        ),
        "event_board_flow_ok": (attr.get("formation_reconciliation") or {}).get(
            "event_board_flow_ok"
        ),
    })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": attr,
        "primary": attr.get("primary"),
        "per_tier": attr.get("per_tier"),
        "membership_propagation": attr.get("membership_propagation"),
        "opening_causes": attr.get("opening_causes"),
        "lifecycle_3q": attr.get("lifecycle_3q"),
        "scale_3r": attr.get("scale_3r"),
        "source": scale.get("source") if scale else None,
        "matched_state": scale.get("matched_state") if scale else None,
        "mechanics_3o": scale.get("mechanics_3o") if scale else None,
        "allocation_3p": scale.get("allocation_3p") if scale else None,
        "lifecycle_primary_3q": (
            scale.get("lifecycle_primary_3q") if scale else None
        ),
        "scale_primary_3r": scale.get("primary") if scale else None,
        "first_divergence_3m": (
            scale.get("first_divergence_3m") if scale else None
        ),
        "reconciliation": rec,
        "decomposition_3g": scale.get("decomposition_3g") if scale else None,
        "published_3r_locks": {
            "n_pairs": PHASE_3R_PRIMARY_N_PAIRS,
            "n_fights": PHASE_3R_PRIMARY_N_FIGHTS,
            "share_input": PHASE_3R_SHARE_INPUT,
            "share_timing": PHASE_3R_SHARE_TIMING,
            "share_membership": PHASE_3R_SHARE_MEMBERSHIP,
            "share_rounding": PHASE_3R_SHARE_ROUNDING,
            "share_residual": PHASE_3R_SHARE_RESIDUAL,
            "t1_control": PHASE_3R_T1_SYNTH_CONTROL,
            "t1_treatment": PHASE_3R_T1_SYNTH_TREATMENT,
            "t3_control": PHASE_3R_T3_SYNTH_CONTROL,
            "t3_treatment": PHASE_3R_T3_SYNTH_TREATMENT,
        },
        "published_3q_locks": {
            "n_pairs": PHASE_3Q_PRIMARY_N_PAIRS,
            "n_fights": PHASE_3Q_PRIMARY_N_FIGHTS,
            "share_same_state": PHASE_3Q_SHARE_SAME_STATE,
            "share_lifecycle": PHASE_3Q_SHARE_LIFECYCLE,
            "share_scaling": PHASE_3Q_SHARE_SCALING,
            "share_residual": PHASE_3Q_SHARE_RESIDUAL,
            "t1_control": PHASE_3Q_T1_SYNTH_CONTROL,
            "t1_treatment": PHASE_3Q_T1_SYNTH_TREATMENT,
            "t3_control": PHASE_3Q_T3_SYNTH_CONTROL,
            "t3_treatment": PHASE_3Q_T3_SYNTH_TREATMENT,
        },
        "published_3p_locks": scale.get("published_3p_locks") if scale else None,
        "published_3o_locks": scale.get("published_3o_locks") if scale else None,
        "published_3n_locks": scale.get("published_3n_locks") if scale else None,
    }


__all__ = [
    "OpenSlotFormationTracer",
    "attribute_open_slot_formation",
    "board_composition_key",
    "board_membership_key",
    "buy_play_order_key",
    "classify_slot_opening_cause",
    "compare_open_slot_formation",
    "decompose_formation_pair",
    "diagnose_phase_3s",
    "earliest_membership_diverge_turn",
    "first_formation_component",
    "incoming_identity_key",
    "run_greedy_2s_treatment_open_slot",
    "run_greedy_control_open_slot",
    "slot_opening_key",
    "stamp_play_opening",
    "tier_mass_formation",
    "tier_mass_membership_prop",
]
