"""Phase 3Q — observational play-lifecycle sticky-vs-repaint causal audit.

Reuses the 3P T5/T6 class-(3) walk on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For every play on a class-(3) T5/T6 trajectory, snapshots pre-play
board/body synth, incoming recruit raw/tier, open-slot vs
sell→buy→play, post-play pre-reallocation, post-reallocation, and
post-scale combat-start state. Offline same-state counterfactuals:

  (A) control board under would-be 2S recruit-raw-proportional repaint
  (B) treatment board under sticky incumbent synth / no repaint

Then splits treatment−control combat-start body synth into same-state
repaint, replacement/open-slot lifecycle, subsequent scaling, and residual.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    A_PLAY0,
    MAX_BOARD,
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    minion_combat_stats,
    minion_recruit_stats,
    minion_synthetic_delta,
    recruit_value_stats_enabled,
)
from ml.allocation_input_diagnostic import (
    AllocationInputTracer,
    _ensure_paint_rows,
    _pair_bodies,
    _slot_snap,
    classify_membership_event,
    compare_allocation_inputs,
    decode_recruit_action,
    reconstruct_board_paint,
)
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.matched_state_damage_diagnostic import iter_class3_events
from ml.pairing_who_wins_diagnostic import _index_seat_fights
from ml.phase_3q_prereg import (
    LIFECYCLE_COMPONENTS,
    NESTED_LIFECYCLE_IDENTITY,
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
    PHASE_3P_T1_SYNTH_CONTROL,
    PHASE_3P_T1_SYNTH_TREATMENT,
    PHASE_3P_T3_SYNTH_CONTROL,
    PHASE_3P_T3_SYNTH_TREATMENT,
    PLAY_POOL_RECONCILE_IDENTITY,
    PLAY_SUBTYPES,
    PRIMARY_TURNS,
    SAME_STATE_IDENTITY,
    diagnose_phase_3q,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.survivor_composition_diagnostic import TIERS, clamp_tier
from ml.survivor_mechanic_diagnostic import (
    _fight_for_event,
    _primary_turn,
    collect_class3_minions,
    compare_survivor_mechanics,
)
from ml.synthetic_allocation_diagnostic import _safe_div

METHODOLOGY_VERSION = "3q_v1"
_N_EXAMPLES = 8
_SYNTH_REPRO_TOL = 0.15


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _safe_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _minion_snap(m, slot: Optional[int] = None, zone: str = "board") -> Dict:
    row = _slot_snap(m, 0 if slot is None else int(slot))
    row["zone"] = zone
    row["combat_raw"] = int(minion_combat_stats(m))
    row["attack"] = int(getattr(m, "attack", 0) or 0)
    row["health"] = int(getattr(m, "health", 0) or 0)
    row["recruit_attack"] = int(getattr(m, "recruit_attack", 0) or 0)
    row["recruit_health"] = int(getattr(m, "recruit_health", 0) or 0)
    return row


def classify_play_subtype(
    *,
    sold: bool,
    bought: bool,
    triple: bool = False,
    board_full_at_sell: bool = False,
) -> str:
    """open_slot vs sell→buy→play vs sell→play vs triple."""
    del board_full_at_sell
    if triple:
        return "triple"
    if sold and bought:
        return "sell_buy_play"
    if sold:
        return "sell_play"
    return "open_slot"


def reconstruct_post_play_sticky(
    pre_slots: Sequence[Dict],
    incoming: Optional[Dict],
    ids_before: Sequence[int],
    ids_after: Sequence[int],
) -> List[Dict]:
    """Offline post-play / pre-reallocation board: incumbents keep synth."""
    before = list(ids_before)
    after = list(ids_after)
    removed = set(i for i in before if i not in after)
    kept = [dict(s) for s in pre_slots if s.get("obj_id") not in removed]
    incoming_id = None if incoming is None else incoming.get("obj_id")
    already = incoming_id is not None and any(
        s.get("obj_id") == incoming_id for s in kept
    )
    if incoming is not None and incoming_id in after and not already:
        row = dict(incoming)
        row["slot"] = len(kept)
        row["zone"] = "board"
        kept.append(row)
    for i, row in enumerate(kept):
        row["slot"] = i
    return kept


def paint_same_state(rows: Sequence[Dict], abstract_pool=None) -> Dict:
    """CF (A): 2S recruit-raw-proportional repaint of this exact board."""
    stamped = [dict(r) for r in rows]
    if abstract_pool is None:
        implicit = float(sum(int(r.get("synthetic_share") or 0) for r in stamped))
        abstract_pool = implicit
    return reconstruct_board_paint(stamped, abstract_pool)


def sticky_same_state(rows: Sequence[Dict]) -> Dict:
    """CF (B): keep incumbent / incoming synth; do not repaint."""
    stamped = [dict(r) for r in rows]
    share_sum = int(sum(int(r.get("synthetic_share") or 0) for r in stamped))
    for r in stamped:
        r["painted_pool"] = share_sum
        r["pool_source"] = "sticky_incumbent"
    return {
        "rows": stamped,
        "painted_pool": share_sum,
        "synthetic_shares_sum": share_sum,
        "shares_sum_to_painted_pool": True,
        "pool_source": "sticky_incumbent",
    }


def _board_synth_sum(rows: Sequence[Dict]) -> int:
    return int(sum(int(r.get("synthetic_share") or 0) for r in rows or []))


def _match_slot(rows: Sequence[Dict], slot: int, row: Optional[Dict] = None) -> Optional[Dict]:
    for s in rows or []:
        if _safe_int(s.get("slot"), s.get("board_slot")) == int(slot):
            return s
    if not row:
        return None
    key = (
        str(row.get("name") or ""),
        str(row.get("card_id") or ""),
        int(row.get("recruit_raw") or 0),
    )
    for s in rows or []:
        if (
            str(s.get("name") or ""),
            str(s.get("card_id") or ""),
            int(s.get("recruit_raw") or 0),
        ) == key:
            return s
    return None


def _synth_at(
    rows: Sequence[Dict], slot: int, row: Optional[Dict] = None,
) -> Optional[int]:
    match = _match_slot(rows, slot, row)
    if match is None:
        return None
    return int(match.get("synthetic_share") or 0)


def decompose_play_pair(
    control_start: Dict,
    treatment_start: Dict,
    control_play: Optional[Dict],
    treatment_play: Optional[Dict],
) -> Dict:
    """Exact four-way split of treatment−control combat-start body synth.

    Identity (residual ~ 0 when last-play snapshots exist):
      ΔS = same_state_repaint + replacement_lifecycle
         + subsequent_scaling + residual
      same_state_repaint     = S_t_paint − S_t_sticky_cf
      replacement_lifecycle  = S_t_sticky_cf − S_c_sticky
      subsequent_scaling     = (S_t_start − S_t_paint) − (S_c_start − S_c_sticky)
    """
    s_c_start = float(control_start.get("synthetic_share") or 0)
    s_t_start = float(treatment_start.get("synthetic_share") or 0)
    slot = _safe_int(
        control_start.get("board_slot"),
        _safe_int(treatment_start.get("board_slot"), 0),
    )
    c_sticky = _synth_at(
        (control_play or {}).get("post_play_pre_realloc") or [],
        slot, control_start,
    )
    t_paint = _synth_at(
        (treatment_play or {}).get("post_realloc") or [],
        slot, treatment_start,
    )
    t_sticky = _synth_at(
        (treatment_play or {}).get("cf_b_sticky") or [],
        slot, treatment_start,
    )
    c_paint = _synth_at(
        (control_play or {}).get("cf_a_paint") or [],
        slot, control_start,
    )
    missing = any(v is None for v in (c_sticky, t_paint, t_sticky))
    if missing:
        delta = s_t_start - s_c_start
        return {
            "delta_synth": delta,
            "same_state_repaint": 0.0,
            "replacement_lifecycle": 0.0,
            "subsequent_scaling": 0.0,
            "residual": delta,
            "same_state_control_cf": None,
            "snapshots_complete": False,
            "control_subtype": (control_play or {}).get("play_subtype"),
            "treatment_subtype": (treatment_play or {}).get("play_subtype"),
            "subtype_mismatch": (
                (control_play or {}).get("play_subtype")
                != (treatment_play or {}).get("play_subtype")
            ),
        }
    c_sticky_f = float(c_sticky)
    t_paint_f = float(t_paint)
    t_sticky_f = float(t_sticky)
    same_state = t_paint_f - t_sticky_f
    lifecycle = t_sticky_f - c_sticky_f
    scale = (s_t_start - t_paint_f) - (s_c_start - c_sticky_f)
    delta = s_t_start - s_c_start
    residual = delta - (same_state + lifecycle + scale)
    same_state_c = None if c_paint is None else float(c_paint) - c_sticky_f
    return {
        "delta_synth": delta,
        "same_state_repaint": same_state,
        "replacement_lifecycle": lifecycle,
        "subsequent_scaling": scale,
        "residual": residual,
        "same_state_control_cf": same_state_c,
        "snapshots_complete": True,
        "control_subtype": (control_play or {}).get("play_subtype"),
        "treatment_subtype": (treatment_play or {}).get("play_subtype"),
        "subtype_mismatch": (
            (control_play or {}).get("play_subtype")
            != (treatment_play or {}).get("play_subtype")
        ),
        "s_c_sticky": c_sticky_f,
        "s_t_sticky_cf": t_sticky_f,
        "s_t_paint": t_paint_f,
        "s_c_paint_cf": None if c_paint is None else float(c_paint),
        "s_c_start": s_c_start,
        "s_t_start": s_t_start,
    }


class PlayLifecycleTracer(AllocationInputTracer):
    """3P paint rows plus every play-event lifecycle snapshot."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.play_events: List[Dict] = []
        self._hand_slots: Dict[int, List[Dict]] = {}
        self._turn_ops: Dict[int, Dict] = {}
        self._play_seq: Dict[int, int] = {}

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._hand_slots.clear()
        self._turn_ops.clear()
        self._play_seq.clear()

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        super().begin_seat_recruit(seat, turn, player)
        self._turn_ops[int(seat)] = {
            "sold": False,
            "bought": False,
            "board_full_at_sell": False,
            "sell_slot": None,
        }
        self._play_seq[int(seat)] = 0

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask,
    ) -> None:
        super().before_action(seat, turn, shop_generation, obs, mask)
        p = self._live_player.get(int(seat))
        if p is None:
            return
        hand = list(getattr(p, "hand", None) or [])
        self._hand_slots[int(seat)] = [
            _minion_snap(m, i, zone="hand") for i, m in enumerate(hand)
        ]

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int,
        ended: bool, player=None,
    ) -> None:
        ids_before = list(self._pre_ids.get(int(seat), []))
        pre_slots = list(self._pre_slots.get(int(seat), []))
        hand_before = list(self._hand_slots.get(int(seat), []))
        ops = self._turn_ops.setdefault(int(seat), {
            "sold": False, "bought": False, "board_full_at_sell": False,
            "sell_slot": None,
        })
        kind = decode_recruit_action(action)
        if kind == "sell":
            ops["sold"] = True
            ops["sell_slot"] = int(action - A_SELL0)
            ops["board_full_at_sell"] = len(pre_slots) >= MAX_BOARD
        elif kind == "buy":
            ops["bought"] = True
        super().after_action(seat, turn, shop_generation, action, ended, player)
        p = player if player is not None else self._live_player.get(int(seat))
        if p is None or kind != "play":
            return
        board = list(getattr(p, "board", None) or [])
        ids_after = [id(m) for m in board]
        event_kind = classify_membership_event(action, ids_before, ids_after)
        if event_kind is None:
            return
        hand_idx = int(action - A_PLAY0)
        incoming = None
        if 0 <= hand_idx < len(hand_before):
            incoming = dict(hand_before[hand_idx])
        triple = event_kind == "triple"
        subtype = classify_play_subtype(
            sold=bool(ops.get("sold")),
            bought=bool(ops.get("bought")),
            triple=triple,
            board_full_at_sell=bool(ops.get("board_full_at_sell")),
        )
        sticky_rows = reconstruct_post_play_sticky(
            pre_slots, incoming, ids_before, ids_after,
        )
        post_realloc = [_minion_snap(m, i) for i, m in enumerate(board)]
        abstract_pool = float(getattr(p, "abstract_pool", 0.0) or 0.0)
        implicit_pool = float(_board_synth_sum(sticky_rows))
        cf_a = paint_same_state(sticky_rows, implicit_pool)
        cf_b = sticky_same_state(sticky_rows)
        paint_actual = reconstruct_board_paint(post_realloc, abstract_pool)
        seq = int(self._play_seq.get(int(seat), 0))
        self._play_seq[int(seat)] = seq + 1
        self.play_events.append({
            "seed": int(self.seed),
            "lobby": int(self.lobby_id),
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(turn),
            "seq": seq,
            "event": event_kind,
            "play_subtype": subtype,
            "sold": bool(ops.get("sold")),
            "bought": bool(ops.get("bought")),
            "board_full_at_sell": bool(ops.get("board_full_at_sell")),
            "board_len_before": len(pre_slots),
            "board_len_after": len(board),
            "incoming": incoming,
            "incoming_recruit_raw": (
                None if incoming is None else int(incoming.get("recruit_raw") or 0)
            ),
            "incoming_tier": (
                None if incoming is None else int(incoming.get("tier") or 1)
            ),
            "pre_play": [dict(s) for s in pre_slots],
            "post_play_pre_realloc": sticky_rows,
            "post_realloc": post_realloc,
            "cf_a_paint": cf_a["rows"],
            "cf_b_sticky": cf_b["rows"],
            "abstract_pool": abstract_pool,
            "implicit_sticky_pool": implicit_pool,
            "cf_a_painted_pool": cf_a["painted_pool"],
            "actual_painted_pool": paint_actual["painted_pool"],
            "cf_a_shares_sum_ok": bool(cf_a["shares_sum_to_painted_pool"]),
            "actual_shares_sum_ok": bool(paint_actual["shares_sum_to_painted_pool"]),
            "sticky_shares_sum": _board_synth_sum(sticky_rows),
            "post_realloc_shares_sum": _board_synth_sum(post_realloc),
            "post_scale_slots": None,
            "combat_start_slots": None,
        })
        # A completed play consumes the pending sell/buy chain.
        ops["sold"] = False
        ops["bought"] = False
        ops["board_full_at_sell"] = False
        ops["sell_slot"] = None

    def after_scale_all(self, env: BGEnv) -> None:
        parent = getattr(super(), "after_scale_all", None)
        if callable(parent):
            parent(env)
        turn = int(getattr(env, "turn", 0) or 0)
        by_seat = {}
        for p in list(getattr(env, "players", None) or []):
            snap = [_minion_snap(m, i) for i, m in enumerate(list(p.board or []))]
            by_seat[int(p.idx)] = {
                "slots": snap,
                "abstract_pool": float(getattr(p, "abstract_pool", 0.0) or 0.0),
            }
        for ev in self.play_events:
            if ev.get("turn") != turn:
                continue
            post = by_seat.get(int(ev["seat"]))
            if post is None:
                continue
            ev["post_scale_slots"] = post["slots"]
            ev["post_scale_pool"] = post["abstract_pool"]

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        winner = rec.get("winner_seat")
        if winner is None:
            return
        turn = _safe_int(rec.get("turn"), 0)
        last = None
        for ev in reversed(self.play_events):
            if int(ev["seat"]) == int(winner) and int(ev["turn"]) <= int(turn):
                last = ev
                break
        if last is None:
            return
        last["combat_start_slots"] = [
            {
                "slot": _safe_int(r.get("board_slot"), i),
                "name": r.get("name"),
                "card_id": r.get("card_id"),
                "tier": r.get("tier"),
                "recruit_raw": r.get("recruit_raw"),
                "synthetic_share": r.get("synthetic_share"),
            }
            for i, r in enumerate(rec.get("start_minions") or [])
        ]
        last["combat_start_pool"] = rec.get("painted_pool")


def run_lifecycle_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    from ml.phase_3q_prereg import assert_seed_range_allowed
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

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = PlayLifecycleTracer(i, seed + i, arm)
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
    }


def run_greedy_control_lifecycle(lobbies: int, seed: int) -> Dict:
    return run_lifecycle_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_lifecycle(lobbies: int, seed: int) -> Dict:
    return run_lifecycle_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _index_plays(play_events: Sequence[Dict]) -> Dict[Tuple[int, int], List[Dict]]:
    out: Dict[Tuple[int, int], List[Dict]] = {}
    for ev in play_events or []:
        seed = _safe_int(ev.get("seed"))
        seat = _safe_int(ev.get("seat"))
        if seed is None or seat is None:
            continue
        out.setdefault((seed, seat), []).append(ev)
    for key, rows in out.items():
        rows.sort(key=lambda r: (int(r.get("turn") or 0), int(r.get("seq") or 0)))
        out[key] = rows
    return out


def _last_play(
    index: Dict[Tuple[int, int], List[Dict]],
    seed, seat, turn,
) -> Optional[Dict]:
    seed_i = _safe_int(seed)
    seat_i = _safe_int(seat)
    turn_i = _safe_int(turn)
    if seed_i is None or seat_i is None or turn_i is None:
        return None
    rows = index.get((seed_i, seat_i)) or []
    last = None
    for ev in rows:
        if int(ev.get("turn") or 0) <= turn_i:
            last = ev
    return last


def _sum_parts(pairs: Sequence[Dict]) -> Dict[str, float]:
    totals = {name: 0.0 for name in LIFECYCLE_COMPONENTS}
    totals["delta_synth"] = 0.0
    for p in pairs:
        for name in LIFECYCLE_COMPONENTS:
            totals[name] += float(p.get(name) or 0.0)
        totals["delta_synth"] += float(p.get("delta_synth") or 0.0)
    return totals


def tier_mass_lifecycle(per_tier: Dict[str, Dict]) -> Dict:
    """Decision shares from within-tier |parts| so T1↓ / T3↑ cannot cancel."""
    mass = {name: 0.0 for name in LIFECYCLE_COMPONENTS}
    n_used = 0
    abs_delta = 0.0
    for cell in (per_tier or {}).values():
        n = int(cell.get("n_pairs") or 0)
        if n <= 0:
            continue
        n_used += n
        abs_delta += n * abs(float(cell.get("delta_synth") or 0.0))
        for name in LIFECYCLE_COMPONENTS:
            mass[name] += n * abs(float(cell.get(name) or 0.0))
    total_mass = sum(mass.values())

    def _share(part: float) -> Optional[float]:
        if total_mass < 1e-12:
            return None
        return float(part) / total_mass

    return {
        "method": "within_tier_abs_mass_play_lifecycle_identity",
        "n_pairs": n_used,
        "abs_delta_synth": (
            None if n_used <= 0 else float(abs_delta) / float(n_used)
        ),
        "component_abs_mass": mass,
        "total_abs_mass": total_mass,
        **{name: (
            None if n_used <= 0 else float(mass[name]) / float(n_used)
        ) for name in LIFECYCLE_COMPONENTS},
        **{f"share_of_delta_{name}": _share(mass[name])
           for name in LIFECYCLE_COMPONENTS},
        "lifecycle_components": list(LIFECYCLE_COMPONENTS),
    }


def _play_sample_keys(events: Sequence[Dict], fights_c, fights_t) -> set:
    keys = set()
    for ev in events:
        if not _primary_turn(ev):
            continue
        seed = _safe_int(ev.get("seed"))
        for fight in (
            _fight_for_event(fights_c, ev),
            _fight_for_event(fights_t, ev),
        ):
            if fight is None:
                continue
            winner = _safe_int(fight.get("winner_seat"), ev.get("causal_seat"))
            if seed is None or winner is None:
                continue
            keys.add((seed, int(winner)))
        seat = _safe_int(ev.get("causal_seat"))
        if seed is not None and seat is not None:
            keys.add((seed, seat))
    return keys


def attribute_play_lifecycle(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    matched: Optional[Dict] = None,
    mechanics: Optional[Dict] = None,
    allocation: Optional[Dict] = None,
) -> Dict:
    """3P locks plus last-play sticky-vs-repaint / lifecycle / scaling split."""
    if mechanics is None:
        mechanics = compare_survivor_mechanics(
            control_raw, treatment_raw, matched=matched,
        )
    if allocation is None:
        allocation = compare_allocation_inputs(
            control_raw, treatment_raw, matched=matched, mechanics=mechanics,
        )
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
    sample_keys = _play_sample_keys(events, c_fights, t_fights)
    sample_plays_c = [
        ev for ev in (control_raw.get("play_events") or [])
        if (_safe_int(ev.get("seed")), _safe_int(ev.get("seat"))) in sample_keys
    ]
    sample_plays_t = [
        ev for ev in (treatment_raw.get("play_events") or [])
        if (_safe_int(ev.get("seed")), _safe_int(ev.get("seat"))) in sample_keys
    ]

    pairs: List[Dict] = []
    n_complete = 0
    n_missing_play = 0
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
        parts = decompose_play_pair(c_row, t_row, c_play, t_play)
        parts.update({
            "seed": ev_seed,
            "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
            "tier": clamp_tier(p.get("tier") or c_row.get("tier") or t_row.get("tier")),
            "board_slot": p.get("board_slot"),
        })
        if parts.get("snapshots_complete"):
            n_complete += 1
        else:
            n_missing_play += 1
        pairs.append(parts)

    for r in unpaired_t:
        s = float(r.get("synthetic_share") or 0)
        pairs.append({
            "delta_synth": s,
            "same_state_repaint": 0.0,
            "replacement_lifecycle": s,
            "subsequent_scaling": 0.0,
            "residual": 0.0,
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "treatment",
            "snapshots_complete": False,
        })
    for r in unpaired_c:
        s = float(r.get("synthetic_share") or 0)
        pairs.append({
            "delta_synth": -s,
            "same_state_repaint": 0.0,
            "replacement_lifecycle": -s,
            "subsequent_scaling": 0.0,
            "residual": 0.0,
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "control",
            "snapshots_complete": False,
        })

    totals = _sum_parts(pairs)
    n_pairs = max(1, len(pairs))
    means = {k: float(v) / float(n_pairs) for k, v in totals.items()}
    obs_delta = means["delta_synth"]

    def _share(part: float) -> Optional[float]:
        if abs(obs_delta) < 1e-12:
            return None
        return float(part) / obs_delta

    pooled_signed = {
        "method": "exact_play_lifecycle_paired_slot_identity",
        "n_pairs": len(pairs),
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "n_snapshots_complete": n_complete,
        "n_missing_play": n_missing_play,
        "delta_synth": means["delta_synth"],
        **{name: means[name] for name in LIFECYCLE_COMPONENTS},
        "explained_all_parts": sum(means[n] for n in LIFECYCLE_COMPONENTS),
        "residual_vs_delta": (
            means["delta_synth"] - sum(means[n] for n in LIFECYCLE_COMPONENTS)
        ),
        **{f"share_of_delta_{name}": _share(means[name])
           for name in LIFECYCLE_COMPONENTS},
        "lifecycle_components": list(LIFECYCLE_COMPONENTS),
    }
    per_tier = {}
    for tier in TIERS:
        cell = [p for p in pairs if int(p.get("tier") or 1) == tier]
        if not cell:
            per_tier[str(tier)] = {"n_pairs": 0}
            continue
        ct = _sum_parts(cell)
        n_cell = max(1, len(cell))
        cm = {k: float(v) / float(n_cell) for k, v in ct.items()}
        d = cm["delta_synth"]

        def _cshare(part: float, den: float = d) -> Optional[float]:
            if abs(den) < 1e-12:
                return None
            return float(part) / den

        per_tier[str(tier)] = {
            "n_pairs": len(cell),
            "delta_synth": cm["delta_synth"],
            **{name: cm[name] for name in LIFECYCLE_COMPONENTS},
            **{f"share_of_delta_{name}": _cshare(cm[name])
               for name in LIFECYCLE_COMPONENTS},
            "subtype_mismatch_rate": _safe_div(
                float(sum(1 for p in cell if p.get("subtype_mismatch"))),
                float(len(cell)),
            ),
        }

    primary = tier_mass_lifecycle(per_tier)
    subtype_mass = {name: 0.0 for name in PLAY_SUBTYPES}
    for p in pairs:
        sub = p.get("treatment_subtype") or p.get("control_subtype")
        if sub not in subtype_mass:
            continue
        subtype_mass[sub] += abs(float(p.get("replacement_lifecycle") or 0.0))
    subtype_total = sum(subtype_mass.values())
    primary.update({
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "n_snapshots_complete": n_complete,
        "n_missing_play": n_missing_play,
        "pooled_signed_delta_synth": pooled_signed["delta_synth"],
        "pooled_signed": pooled_signed,
        "residual_vs_delta": pooled_signed["residual_vs_delta"],
        "subtype_abs_mass": subtype_mass,
        **{f"share_of_lifecycle_{name}": (
            None if subtype_total < 1e-12 else float(subtype_mass[name]) / subtype_total
        ) for name in PLAY_SUBTYPES},
    })

    def _play_summary(plays: Sequence[Dict]) -> Dict:
        n = len(plays)
        counts = Counter(str(p.get("play_subtype") or "") for p in plays)
        incoming_raw = [
            float(p.get("incoming_recruit_raw"))
            for p in plays if p.get("incoming_recruit_raw") is not None
        ]
        incoming_tier = [
            float(p.get("incoming_tier"))
            for p in plays if p.get("incoming_tier") is not None
        ]
        return {
            "n_plays": n,
            "n_open_slot": int(counts.get("open_slot") or 0),
            "n_sell_buy_play": int(counts.get("sell_buy_play") or 0),
            "n_sell_play": int(counts.get("sell_play") or 0),
            "n_triple": int(counts.get("triple") or 0),
            "p_open_slot": _safe_div(float(counts.get("open_slot") or 0), float(n)),
            "p_sell_buy_play": _safe_div(
                float(counts.get("sell_buy_play") or 0), float(n)
            ),
            "p_sell_play": _safe_div(float(counts.get("sell_play") or 0), float(n)),
            "p_triple": _safe_div(float(counts.get("triple") or 0), float(n)),
            "mean_incoming_recruit_raw": _mean(incoming_raw),
            "mean_incoming_tier": _mean(incoming_tier),
            "cf_a_share_mismatches": sum(
                1 for p in plays if p.get("cf_a_shares_sum_ok") is False
            ),
            "actual_share_mismatches": sum(
                1 for p in plays if p.get("actual_shares_sum_ok") is False
            ),
        }

    plays_c = _play_summary(sample_plays_c)
    plays_t = _play_summary(sample_plays_t)

    def _close(got, published) -> bool:
        if got is None or published is None:
            return False
        return abs(float(got) - float(published)) <= _SYNTH_REPRO_TOL

    attr_3p = (allocation or {}).get("attribution") or allocation or {}
    t1_c = attr_3p.get("t1_synth_control")
    t1_t = attr_3p.get("t1_synth_treatment")
    t3_c = attr_3p.get("t3_synth_control")
    t3_t = attr_3p.get("t3_synth_treatment")
    if t1_c is None:
        t1_c = _mean([
            float(r.get("synthetic_share") or 0)
            for r in rows_c if int(r.get("tier") or 1) == 1
        ])
        t1_t = _mean([
            float(r.get("synthetic_share") or 0)
            for r in rows_t if int(r.get("tier") or 1) == 1
        ])
        t3_c = _mean([
            float(r.get("synthetic_share") or 0)
            for r in rows_c if int(r.get("tier") or 1) == 3
        ])
        t3_t = _mean([
            float(r.get("synthetic_share") or 0)
            for r in rows_t if int(r.get("tier") or 1) == 3
        ])

    n_cf_a_bad = plays_c["cf_a_share_mismatches"] + plays_t["cf_a_share_mismatches"]
    n_actual_bad = (
        plays_c["actual_share_mismatches"] + plays_t["actual_share_mismatches"]
    )
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
            "delta_synth": p.get("delta_synth"),
            "same_state_repaint": p.get("same_state_repaint"),
            "replacement_lifecycle": p.get("replacement_lifecycle"),
            "subsequent_scaling": p.get("subsequent_scaling"),
            "residual": p.get("residual"),
            "control_subtype": p.get("control_subtype"),
            "treatment_subtype": p.get("treatment_subtype"),
            "same_state_control_cf": p.get("same_state_control_cf"),
        })
    mech_attr = (mechanics or {}).get("attribution") or {}
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_same_outcome_damage": len(events),
        "n_primary_class3": n_c,
        "n_primary_class3_treatment": n_t,
        "published_same_outcome_damage": PHASE_3N_CLASS3,
        "published_primary_n": PHASE_3O_PRIMARY_N,
        "published_3p_n_pairs": PHASE_3P_PRIMARY_N_PAIRS,
        "published_3p_n_fights": PHASE_3P_PRIMARY_N_FIGHTS,
        "same_outcome_damage_reproduced": len(events) == PHASE_3N_CLASS3,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_B_reproduced": mech_attr.get("phase_3n_B_reproduced"),
        "phase_3o_t5t6_B": PHASE_3O_T5T6_B,
        "phase_3o_B_reproduced": attr_3p.get("phase_3o_B_reproduced"),
        "phase_3o_share_start_stats": PHASE_3O_SHARE_START_STATS,
        "phase_3o_share_synth": PHASE_3O_SHARE_SYNTH,
        "phase_3p_share_timing": PHASE_3P_SHARE_TIMING,
        "phase_3p_share_pool": PHASE_3P_SHARE_POOL,
        "phase_3p_share_weight": PHASE_3P_SHARE_WEIGHT,
        "phase_3p_share_rounding": PHASE_3P_SHARE_ROUNDING,
        "t1_synth_control": t1_c,
        "t1_synth_treatment": t1_t,
        "t3_synth_control": t3_c,
        "t3_synth_treatment": t3_t,
        "t1_synth_reproduced": (
            _close(t1_c, PHASE_3P_T1_SYNTH_CONTROL)
            and _close(t1_t, PHASE_3P_T1_SYNTH_TREATMENT)
        ),
        "t3_synth_reproduced": (
            _close(t3_c, PHASE_3P_T3_SYNTH_CONTROL)
            and _close(t3_t, PHASE_3P_T3_SYNTH_TREATMENT)
        ),
        "primary": primary,
        "per_tier": per_tier,
        "plays_control": plays_c,
        "plays_treatment": plays_t,
        "paint_reconciliation": {
            "cf_a_n_share_mismatch": n_cf_a_bad,
            "actual_n_share_mismatch": n_actual_bad,
            "paint_ok": n_cf_a_bad == 0 and n_actual_bad == 0,
            "nested_ok": abs(float(primary.get("residual_vs_delta") or 0.0)) <= 1e-6,
            "identity": PLAY_POOL_RECONCILE_IDENTITY,
            "same_state_identity": SAME_STATE_IDENTITY,
            "nested_identity": NESTED_LIFECYCLE_IDENTITY,
        },
        "examples": examples,
        "allocation_3p": {
            "n_pairs": (attr_3p.get("primary") or {}).get("n_pairs"),
            "share_timing": (attr_3p.get("primary") or {}).get(
                "share_of_delta_timing_membership"
            ),
            "t1_synth_reproduced": attr_3p.get("t1_synth_reproduced"),
            "t3_synth_reproduced": attr_3p.get("t3_synth_reproduced"),
        },
        "mechanics_3o": {
            "n_primary_class3": mech_attr.get("n_primary_class3"),
            "phase_3n_B_reproduced": mech_attr.get("phase_3n_B_reproduced"),
            "primary_share_start_stats": (
                (mech_attr.get("primary") or {}).get("share_of_B_start_stats")
            ),
            "primary_share_synth": (
                (mech_attr.get("primary") or {}).get("share_of_B_synthetic_allocation")
            ),
        },
    }


def compare_play_lifecycle(
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
) -> Dict:
    """3P lock + T5/T6 play-lifecycle sticky-vs-repaint split."""
    if mechanics is None:
        mechanics = compare_survivor_mechanics(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain, first=first, matched=matched,
        )
    if matched is None:
        matched = {
            "attribution": mechanics.get("matched_state"),
            "reconciliation": mechanics.get("reconciliation"),
        }
    if allocation is None:
        allocation = compare_allocation_inputs(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain, first=first, matched=matched, mechanics=mechanics,
        )
    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch,
        still_fields_t1t3=False,
    )
    attr = attribute_play_lifecycle(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch,
        matched=matched, mechanics=mechanics, allocation=allocation,
    )
    rec = dict((allocation or {}).get("reconciliation") or {})
    rec.update({
        "same_state_identity": SAME_STATE_IDENTITY,
        "nested_lifecycle_identity": NESTED_LIFECYCLE_IDENTITY,
        "play_pool_reconcile_identity": PLAY_POOL_RECONCILE_IDENTITY,
        "phase_3n_B_reproduced": attr.get("phase_3n_B_reproduced"),
        "phase_3o_B_reproduced": attr.get("phase_3o_B_reproduced"),
        "t1_synth_reproduced": attr.get("t1_synth_reproduced"),
        "t3_synth_reproduced": attr.get("t3_synth_reproduced"),
        "paint_ok": (attr.get("paint_reconciliation") or {}).get("paint_ok"),
        "primary_nested_ok": (attr.get("paint_reconciliation") or {}).get("nested_ok"),
    })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": attr,
        "primary": attr.get("primary"),
        "per_tier": attr.get("per_tier"),
        "plays_control": attr.get("plays_control"),
        "plays_treatment": attr.get("plays_treatment"),
        "source": mechanics.get("source"),
        "matched_state": mechanics.get("matched_state"),
        "mechanics_3o": mechanics.get("primary"),
        "allocation_3p": allocation.get("primary") if allocation else None,
        "first_divergence_3m": mechanics.get("first_divergence_3m"),
        "reconciliation": rec,
        "decomposition_3g": mechanics.get("decomposition_3g"),
        "published_3p_locks": {
            "n_pairs": PHASE_3P_PRIMARY_N_PAIRS,
            "n_fights": PHASE_3P_PRIMARY_N_FIGHTS,
            "share_timing": PHASE_3P_SHARE_TIMING,
            "share_pool": PHASE_3P_SHARE_POOL,
            "share_weight": PHASE_3P_SHARE_WEIGHT,
            "share_rounding": PHASE_3P_SHARE_ROUNDING,
            "t1_control": PHASE_3P_T1_SYNTH_CONTROL,
            "t1_treatment": PHASE_3P_T1_SYNTH_TREATMENT,
            "t3_control": PHASE_3P_T3_SYNTH_CONTROL,
            "t3_treatment": PHASE_3P_T3_SYNTH_TREATMENT,
        },
        "published_3o_locks": (allocation or {}).get("published_3o_locks"),
        "published_3n_locks": mechanics.get("published_3n_locks"),
    }


__all__ = [
    "PlayLifecycleTracer",
    "attribute_play_lifecycle",
    "classify_play_subtype",
    "compare_play_lifecycle",
    "decompose_play_pair",
    "diagnose_phase_3q",
    "paint_same_state",
    "reconstruct_post_play_sticky",
    "run_greedy_2s_treatment_lifecycle",
    "run_greedy_control_lifecycle",
    "sticky_same_state",
    "tier_mass_lifecycle",
]
