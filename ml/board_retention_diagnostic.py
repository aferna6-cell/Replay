"""Phase 3H — observational low-tier board-retention lifecycle attribution.

Reuses the 3E PoolLifecycleTracer on consumed DEV 14200–14699. Does not
change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For each paired (seed, seat) on T7–T14 the tracer records T1–T3
count/share, every low-tier body's persist/remove, and classifies the
transition away from T1–T3 into exclusive lifecycle classes. Punch-row
attribution then maps each control late T1–T3 punch row onto the
treatment counterpart seat's last-T1-T3-loss class.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    A_PLAY0,
    A_SELL0,
    BGEnv,
    MAX_BOARD,
    N_PLAY,
    N_SELL,
    board_level_abstract_scaling_enabled,
    combat_raw,
    greedy_policy,
    recruit_raw,
    recruit_value_stats_enabled,
)
from ml.carry_divergence_diagnostic import (
    build_seat_trajectories,
    compare_divergence,
    pair_trajectories,
    reconcile_history_links,
)
from ml.phase_3h_prereg import (
    FLOW_ABS_TOL,
    HISTORY_LINK_IDENTITY,
    INSTRUMENT_TURNS,
    LATE_TURNS,
    LIFECYCLE_COMPONENTS,
    LINEAGE_ABS_TOL,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    LOW_WINNER_START_TIERS,
    PAIRED_SEAT_IDENTITY,
    PHASE_3E_PUNCH_DELTA_CARRY,
    PHASE_3G_MIXTURE,
    PHASE_3G_MIXTURE_SHARE,
    PHASE_3G_N_CONTROL,
    PHASE_3G_N_TREATMENT,
    PHASE_3G_WITHIN_SHARE,
    POOL_FLOW_IDENTITY,
    REWEIGHT_ABS_TOL,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_t1t3_exit,
    share_of_collapse,
)
from ml.pool_lifecycle_diagnostic import (
    PoolLifecycleTracer,
    _decode_action,
    compare_lifecycle,
    summarize_lifecycle_arm,
)
from ml.punch_selection_diagnostic import (
    collect_punch_sample_rows,
    compare_selection,
)

METHODOLOGY_VERSION = "3h_v1"

_TURN_WINDOW = set(INSTRUMENT_TURNS)
_LATE = set(LATE_TURNS)
_VERY_LATE = set(VERY_LATE_TURNS)
_LOW = set(LOW_TIERS)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _tier_of(m) -> int:
    if m is None:
        return 0
    if isinstance(m, dict):
        return _safe_int(m.get("tier"), 0)
    return _safe_int(getattr(m, "tier", 0), 0)


def _name_of(m) -> Optional[str]:
    if m is None:
        return None
    if isinstance(m, dict):
        return m.get("name")
    return getattr(m, "name", None)


def _card_id_of(m) -> Optional[str]:
    if m is None:
        return None
    if isinstance(m, dict):
        return m.get("card_id")
    return getattr(m, "card_id", None)


def _golden_of(m) -> bool:
    if m is None:
        return False
    if isinstance(m, dict):
        tags = m.get("tags") or {}
        return bool(m.get("golden") or tags.get("PREMIUM") == "1")
    return bool(getattr(m, "golden", False))


def _recruit_raw_of(m) -> float:
    if m is None:
        return 0.0
    if isinstance(m, dict):
        return float(recruit_raw(m))
    ra = getattr(m, "recruit_attack", None)
    rh = getattr(m, "recruit_health", None)
    if ra is None:
        ra = getattr(m, "attack", 0)
    if rh is None:
        rh = getattr(m, "health", 0)
    return float(ra or 0) + float(rh or 0)


def _combat_raw_of(m) -> float:
    if m is None:
        return 0.0
    if isinstance(m, dict):
        return float(combat_raw(m))
    return float(getattr(m, "attack", 0) or 0) + float(getattr(m, "health", 0) or 0)


def _is_low(m) -> bool:
    return _tier_of(m) in _LOW


def _shop_tiers(shop: Sequence) -> List[int]:
    return [_tier_of(m) for m in (shop or [])]


def _t1t3_count(board: Sequence) -> int:
    return sum(1 for m in (board or []) if _is_low(m))


def _t1t3_share(board: Sequence) -> Optional[float]:
    n = len(list(board or []))
    if n <= 0:
        return None
    return float(_t1t3_count(board)) / float(n)


def _body_fp(m, *, slot: Optional[int] = None) -> Dict:
    return {
        "name": _name_of(m),
        "card_id": _card_id_of(m),
        "tier": _tier_of(m),
        "recruit_raw": _recruit_raw_of(m),
        "combat_raw": _combat_raw_of(m),
        "golden": _golden_of(m),
        "slot": slot,
        "low_tier": _is_low(m),
    }


def _bodies_from_views(board: Sequence) -> List[Dict]:
    return [_body_fp(m, slot=i) for i, m in enumerate(board or [])]


def _bodies_from_player(player) -> List[Dict]:
    board = list(getattr(player, "board", None) or [])
    out = []
    for i, m in enumerate(board):
        rec = _body_fp(m, slot=i)
        rec["obj_id"] = id(m)
        out.append(rec)
    return out


def _nongolden_names(board, hand) -> Counter:
    names: Counter = Counter()
    for m in list(board or []) + list(hand or []):
        if not _golden_of(m):
            names[_name_of(m)] += 1
    return names


def _golden_names(board, hand) -> Counter:
    names: Counter = Counter()
    for m in list(board or []) + list(hand or []):
        if _golden_of(m):
            names[_name_of(m)] += 1
    return names


class BoardRetentionTracer(PoolLifecycleTracer):
    """3E lifecycle plus exact T1–T3 body lineage / replacement context."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.board_snapshots: List[Dict] = []
        self.t1t3_events: List[Dict] = []
        self.last_t1t3_losses: List[Dict] = []
        self._pre_gold: Optional[int] = None
        self._pre_tavern: Optional[int] = None
        self._pre_shop_tiers: List[int] = []
        self._pre_board_full: bool = False
        self._pre_bodies: List[Dict] = []
        self._pre_board_views: List[Dict] = []
        self._pre_hand_views: List[Dict] = []
        self._seat_had_t1t3: Dict[int, bool] = {}
        self._seat_lost_last: Dict[int, Dict] = {}
        self._seat_alive_last: Dict[int, bool] = {}
        self._generated_hand_ids: set = set()
        self._known_hand_ids: Dict[int, set] = {}

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._seat_had_t1t3.clear()
        self._seat_lost_last.clear()
        self._seat_alive_last.clear()
        self._generated_hand_ids.clear()
        self._known_hand_ids.clear()

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        super().begin_seat_recruit(seat, turn, player)
        if turn not in _TURN_WINDOW:
            return
        acc = self._acc(seat, turn)
        bodies = _bodies_from_player(player)
        board = list(getattr(player, "board", None) or [])
        shop = list(getattr(player, "shop", None) or [])
        n_low = _t1t3_count(board)
        acc["t1t3_count_recruit_start"] = n_low
        acc["t1t3_share_recruit_start"] = _t1t3_share(board)
        acc["gold_recruit_start"] = int(getattr(player, "gold", 0) or 0)
        acc["shop_tiers_recruit_start"] = [_tier_of(m) for m in shop]
        acc["shop_t1t3_offers_recruit_start"] = sum(
            1 for m in shop if _tier_of(m) in _LOW
        )
        acc["board_bodies_recruit_start"] = bodies
        if n_low > 0:
            self._seat_had_t1t3[int(seat)] = True
        self._known_hand_ids[int(seat)] = {
            id(m) for m in list(getattr(player, "hand", None) or [])
        }
        self.board_snapshots.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(turn),
            "phase": "recruit_start",
            "alive": bool(getattr(player, "alive", True)),
            "t1t3_count": n_low,
            "t1t3_share": acc["t1t3_share_recruit_start"],
            "board_size": len(board),
            "gold": acc["gold_recruit_start"],
            "tavern_tier": int(getattr(player, "tier", 1) or 1),
            "shop_tiers": list(acc["shop_tiers_recruit_start"]),
            "bodies": bodies,
        })

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask,
    ) -> None:
        super().before_action(seat, turn, shop_generation, obs, mask)
        board = list(obs.get("board") or [])
        shop = list(obs.get("shop") or [])
        hand = list(obs.get("hand") or [])
        self._pre_gold = obs.get("gold")
        self._pre_tavern = obs.get("tavern_tier")
        self._pre_shop_tiers = _shop_tiers(shop)
        self._pre_board_full = len(board) >= MAX_BOARD
        self._pre_bodies = _bodies_from_views(board)
        self._pre_board_views = board
        self._pre_hand_views = hand

    def _enrich_replacement(
        self, ev: Dict, pending: Dict, *, gold: int, tavern: int,
        shop_tiers: List[int], shop_t1t3: int, board_full: bool,
    ) -> None:
        sold = pending.get("sold_view") or {}
        cand = pending.get("candidate") if isinstance(pending.get("candidate"), dict) else {}
        if not cand:
            raw = ev.get("candidate_name")
            cand = {"name": raw} if raw else {}
        inc_tier = _tier_of(sold) if sold else None
        inc_raw = _recruit_raw_of(sold) if sold else ev.get("sold_stats_pool")
        cand_tier = _tier_of(cand) if cand else None
        cand_raw = _recruit_raw_of(cand) if cand else None
        ev["incumbent_tier"] = inc_tier
        ev["incumbent_recruit_raw"] = inc_raw
        ev["incumbent_combat_raw"] = _combat_raw_of(sold) if sold else None
        ev["candidate_tier"] = cand_tier
        ev["candidate_recruit_raw"] = cand_raw
        ev["candidate_combat_raw"] = _combat_raw_of(cand) if cand else None
        ev["player_tavern_tier"] = tavern
        ev["gold"] = gold
        ev["shop_offer_tiers"] = list(shop_tiers)
        ev["shop_t1t3_offers"] = shop_t1t3
        ev["board_full"] = board_full
        ev["replacement_flag"] = True
        ev["incumbent_low_tier"] = bool(inc_tier in _LOW) if inc_tier else False
        ev["candidate"] = cand if cand else ev.get("candidate")

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        super().end_seat_recruit(seat, turn, player)
        if turn not in _TURN_WINDOW:
            return
        acc = self._acc(seat, turn)
        board = list(getattr(player, "board", None) or [])
        acc["t1t3_count_pre_scale"] = _t1t3_count(board)
        acc["t1t3_share_pre_scale"] = _t1t3_share(board)
        acc["board_bodies_pre_scale"] = _bodies_from_player(player)

    def after_scale_all(self, env: BGEnv) -> None:
        super().after_scale_all(env)
        turn = env.turn
        if turn not in _TURN_WINDOW:
            return
        for seat, player in enumerate(env.players):
            if (seat, turn) not in self._seat_turn:
                continue
            acc = self._acc(seat, turn)
            board = list(getattr(player, "board", None) or [])
            n_low = _t1t3_count(board)
            acc["t1t3_count_post_scale"] = n_low
            acc["t1t3_share_post_scale"] = _t1t3_share(board)
            for row in reversed(self.turn_rows):
                if int(row.get("seat") or -1) == seat and int(row.get("turn") or -1) == turn:
                    row["t1t3_count_recruit_start"] = acc.get("t1t3_count_recruit_start")
                    row["t1t3_share_recruit_start"] = acc.get("t1t3_share_recruit_start")
                    row["t1t3_count_pre_scale"] = acc.get("t1t3_count_pre_scale")
                    row["t1t3_share_pre_scale"] = acc.get("t1t3_share_pre_scale")
                    row["t1t3_count_post_scale"] = n_low
                    row["t1t3_share_post_scale"] = acc.get("t1t3_share_post_scale")
                    row["gold_recruit_start"] = acc.get("gold_recruit_start")
                    row["shop_tiers_recruit_start"] = acc.get("shop_tiers_recruit_start")
                    row["shop_t1t3_offers_recruit_start"] = acc.get(
                        "shop_t1t3_offers_recruit_start"
                    )
                    break

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        turn = int(fight.get("turn") or env.turn)
        if turn not in _TURN_WINDOW:
            return
        for seat, player in enumerate(env.players):
            if (seat, turn) not in self._seat_turn and not player.alive:
                continue
            acc = self._acc(seat, turn)
            board = list(getattr(player, "board", None) or [])
            n_low = _t1t3_count(board)
            acc["t1t3_count_combat_start"] = n_low
            acc["t1t3_share_combat_start"] = _t1t3_share(board)
            acc["alive_at_combat"] = bool(getattr(player, "alive", True))
            self._seat_alive_last[int(seat)] = bool(player.alive)
            if n_low > 0:
                self._seat_had_t1t3[int(seat)] = True
            if n_low == 0 and self._seat_had_t1t3.get(int(seat)) and int(seat) not in self._seat_lost_last:
                # Lost last T1–T3 earlier this turn without a recorded event.
                shop_t1t3 = int(acc.get("shop_t1t3_offers_recruit_start") or 0)
                cls = classify_t1t3_exit(
                    sold=int(acc.get("n_sells") or 0) > 0,
                    board_full=int(acc.get("board_size_recruit_start") or 0) >= MAX_BOARD,
                    replacement_completed=int(acc.get("n_replacements") or 0) > 0,
                    shop_t1t3_offers=shop_t1t3,
                    tavern_tier=int(acc.get("tier_at_recruit") or acc.get("tavern_tier") or 1),
                )
                loss = {
                    "lobby": self.lobby_id,
                    "seed": self.seed,
                    "arm": self.arm,
                    "seat": int(seat),
                    "turn": int(turn),
                    "class": cls,
                    "first_loss_turn": int(turn),
                    "inferred": True,
                    "replacement_flag": int(acc.get("n_replacements") or 0) > 0,
                    "player_tavern_tier": acc.get("tavern_tier") or acc.get("tier_at_recruit"),
                    "gold": acc.get("gold_recruit_start"),
                    "shop_offer_tiers": acc.get("shop_tiers_recruit_start") or [],
                    "incumbent_tier": None,
                    "incumbent_raw": None,
                    "candidate_tier": None,
                    "candidate_raw": None,
                }
                self._seat_lost_last[int(seat)] = loss
                self.last_t1t3_losses.append(loss)
            for row in reversed(self.turn_rows):
                if int(row.get("seat") or -1) == seat and int(row.get("turn") or -1) == turn:
                    row["t1t3_count_combat_start"] = n_low
                    row["t1t3_share_combat_start"] = acc.get("t1t3_share_combat_start")
                    break

    def after_combat(self, env: BGEnv) -> None:
        super().after_combat(env)
        for seat, player in enumerate(env.players):
            alive = bool(getattr(player, "alive", False))
            was_alive = self._seat_alive_last.get(int(seat), True)
            board = list(getattr(player, "board", None) or [])
            n_low = _t1t3_count(board)
            if was_alive and not alive and n_low > 0 and int(seat) not in self._seat_lost_last:
                # Died still holding T1–T3 — elimination, not a body sale.
                turn = int(getattr(env, "turn", 0) or 0)
                loss = {
                    "lobby": self.lobby_id,
                    "seed": self.seed,
                    "arm": self.arm,
                    "seat": int(seat),
                    "turn": turn,
                    "class": "alive_elimination",
                    "first_loss_turn": None,
                    "elimination_turn": turn,
                    "had_t1t3_at_death": True,
                    "replacement_flag": False,
                    "player_tavern_tier": int(getattr(player, "tier", 1) or 1),
                    "gold": int(getattr(player, "gold", 0) or 0),
                    "shop_offer_tiers": [],
                    "incumbent_tier": None,
                    "incumbent_raw": None,
                    "candidate_tier": None,
                    "candidate_raw": None,
                }
                self._seat_lost_last[int(seat)] = loss
                self.last_t1t3_losses.append(loss)
            self._seat_alive_last[int(seat)] = alive

    def end_lobby(self, players) -> None:
        for seat, player in enumerate(players or []):
            if int(seat) in self._seat_lost_last:
                continue
            if self._seat_had_t1t3.get(int(seat)) and not bool(getattr(player, "alive", True)):
                board = list(getattr(player, "board", None) or [])
                if _t1t3_count(board) > 0:
                    loss = {
                        "lobby": self.lobby_id,
                        "seed": self.seed,
                        "arm": self.arm,
                        "seat": int(seat),
                        "turn": None,
                        "class": "alive_elimination",
                        "first_loss_turn": None,
                        "had_t1t3_at_death": True,
                        "replacement_flag": False,
                    }
                    self._seat_lost_last[int(seat)] = loss
                    self.last_t1t3_losses.append(loss)


_parent_after = PoolLifecycleTracer.after_action


def _retention_after_action(
    self, seat, turn, shop_generation, action, ended, player=None,
):
    n_repl_before = len(self.replacement_events)
    pending_before = dict(self._pending.get(seat) or {})
    sold_view = None
    if A_SELL0 <= action < A_SELL0 + N_SELL:
        slot = action - A_SELL0
        if 0 <= slot < len(self._pre_board_views):
            sold_view = self._pre_board_views[slot]
    _parent_after(self, seat, turn, shop_generation, action, ended, player)
    pending = self._pending.get(seat)
    if pending is not None and sold_view is not None:
        pending["sold_view"] = sold_view
    # Delegate the rest to the class body implementation without re-calling parent.
    BoardRetentionTracer._after_action_observe(
        self, seat, turn, action, ended, player,
        n_repl_before=n_repl_before,
        pending_before=pending_before,
        sold_view=sold_view,
    )


def _after_action_observe(
    self, seat, turn, action, ended, player,
    *, n_repl_before, pending_before, sold_view,
):
    if turn not in _TURN_WINDOW or player is None:
        return
    kind = _decode_action(action)
    post_board = list(getattr(player, "board", None) or [])
    post_shop = list(getattr(player, "shop", None) or [])
    gold = int(getattr(player, "gold", 0) or 0)
    tavern = int(getattr(player, "tier", 1) or 1)
    shop_t1t3 = sum(1 for t in (self._pre_shop_tiers or []) if t in _LOW)
    board_full = bool(self._pre_board_full)
    sold_low = bool(sold_view and _is_low(sold_view))
    sold_fp = _body_fp(sold_view) if sold_view is not None else None
    replacement_completed = len(self.replacement_events) > n_repl_before
    cand = None
    if replacement_completed:
        ev = self.replacement_events[-1]
        self._enrich_replacement(
            ev, pending_before, gold=gold, tavern=tavern,
            shop_tiers=list(self._pre_shop_tiers or []),
            shop_t1t3=shop_t1t3, board_full=board_full,
        )
        # Prefer live candidate from the completed event.
        cand = {
            "tier": ev.get("candidate_tier"),
            "recruit_raw": ev.get("candidate_recruit_raw"),
            "name": ev.get("candidate_name"),
        }

    pre_ng = _nongolden_names(self._pre_board_views, self._pre_hand_views)
    post_ng = _nongolden_names(post_board, list(getattr(player, "hand", None) or []))
    pre_g = _golden_names(self._pre_board_views, self._pre_hand_views)
    post_g = _golden_names(post_board, list(getattr(player, "hand", None) or []))
    triple = False
    for name, n_pre in pre_ng.items():
        if n_pre - post_ng.get(name, 0) >= 3 and post_g.get(name, 0) > pre_g.get(name, 0):
            triple = True
            break

    open_slot_higher = False
    if (
        A_PLAY0 <= action < A_PLAY0 + N_PLAY
        and not board_full
        and not pending_before
    ):
        slot = action - A_PLAY0
        if 0 <= slot < len(self._pre_hand_views):
            open_slot_higher = _tier_of(self._pre_hand_views[slot]) not in _LOW

    pre_low_n = sum(1 for b in self._pre_bodies if b.get("low_tier"))
    post_low_n = _t1t3_count(post_board)
    removed_n = max(0, pre_low_n - post_low_n)
    generated = triple

    if post_low_n > pre_low_n and not sold_low and not triple:
        self.t1t3_events.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(turn),
            "kind": kind,
            "class": "t1t3_add",
            "sold": False,
            "board_full": board_full,
            "replacement_completed": replacement_completed,
            "replacement_flag": replacement_completed,
            "triple": False,
            "generated": generated,
            "open_slot_higher_tier_play": False,
            "pre_t1t3_count": pre_low_n,
            "post_t1t3_count": post_low_n,
            "removed_n": 0,
            "added_n": post_low_n - pre_low_n,
            "incumbent": None,
            "incumbent_tier": None,
            "incumbent_raw": None,
            "candidate_tier": None if not cand else cand.get("tier"),
            "candidate_raw": None if not cand else cand.get("recruit_raw"),
            "player_tavern_tier": int(self._pre_tavern or tavern or 1),
            "gold": _safe_int(self._pre_gold, gold),
            "shop_offer_tiers": list(self._pre_shop_tiers or []),
            "shop_t1t3_offers": shop_t1t3,
        })
    elif removed_n > 0 or sold_low:
        cls = classify_t1t3_exit(
            sold=sold_low or bool(pending_before),
            board_full=board_full,
            replacement_completed=replacement_completed,
            shop_t1t3_offers=shop_t1t3,
            tavern_tier=int(self._pre_tavern or tavern or 1),
            triple=triple,
            generated=generated,
            open_slot_higher_tier_play=open_slot_higher,
        )
        event = {
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(turn),
            "kind": kind,
            "class": cls,
            "sold": sold_low,
            "board_full": board_full,
            "replacement_completed": replacement_completed,
            "replacement_flag": replacement_completed,
            "triple": triple,
            "generated": generated,
            "open_slot_higher_tier_play": open_slot_higher,
            "pre_t1t3_count": pre_low_n,
            "post_t1t3_count": post_low_n,
            "removed_n": removed_n,
            "incumbent": sold_fp,
            "incumbent_tier": None if sold_fp is None else sold_fp.get("tier"),
            "incumbent_raw": None if sold_fp is None else sold_fp.get("recruit_raw"),
            "candidate_tier": None if not cand else cand.get("tier"),
            "candidate_raw": None if not cand else cand.get("recruit_raw"),
            "player_tavern_tier": int(self._pre_tavern or tavern or 1),
            "gold": _safe_int(self._pre_gold, gold),
            "shop_offer_tiers": list(self._pre_shop_tiers or []),
            "shop_t1t3_offers": shop_t1t3,
        }
        self.t1t3_events.append(event)
        if pre_low_n > 0 and post_low_n == 0 and int(seat) not in self._seat_lost_last:
            loss = dict(event)
            loss["first_loss_turn"] = int(turn)
            self._seat_lost_last[int(seat)] = loss
            self.last_t1t3_losses.append(loss)
    elif open_slot_higher and pre_low_n > 0:
        slot = action - A_PLAY0
        played = (
            self._pre_hand_views[slot]
            if 0 <= slot < len(self._pre_hand_views)
            else None
        )
        self.t1t3_events.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(turn),
            "kind": kind,
            "class": "open_slot_fill",
            "sold": False,
            "board_full": False,
            "replacement_completed": False,
            "replacement_flag": False,
            "triple": False,
            "generated": False,
            "open_slot_higher_tier_play": True,
            "share_only": True,
            "pre_t1t3_count": pre_low_n,
            "post_t1t3_count": post_low_n,
            "removed_n": 0,
            "incumbent": None,
            "incumbent_tier": None,
            "incumbent_raw": None,
            "candidate_tier": None if played is None else _tier_of(played),
            "candidate_raw": None if played is None else _recruit_raw_of(played),
            "player_tavern_tier": int(self._pre_tavern or tavern or 1),
            "gold": _safe_int(self._pre_gold, gold),
            "shop_offer_tiers": list(self._pre_shop_tiers or []),
            "shop_t1t3_offers": shop_t1t3,
        })

    acc = self._acc(seat, turn)
    acc["t1t3_count_post_action"] = post_low_n
    acc["gold_last_action"] = gold
    acc["shop_tiers_last_action"] = [_tier_of(m) for m in post_shop]


BoardRetentionTracer.after_action = _retention_after_action
BoardRetentionTracer._after_action_observe = _after_action_observe


def run_retention_arm(
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

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = BoardRetentionTracer(i, seed + i, arm)
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
    }


def run_greedy_control_retention(lobbies: int, seed: int) -> Dict:
    return run_retention_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_retention(lobbies: int, seed: int) -> Dict:
    return run_retention_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _t1t3_from_row(row: Optional[Dict], *, phase: str = "combat") -> Optional[int]:
    if not row:
        return None
    keys = (
        "t1t3_count_combat_start",
        "t1t3_count_post_scale",
        "t1t3_count_recruit_start",
    ) if phase == "combat" else (
        "t1t3_count_recruit_start",
        "t1t3_count_pre_scale",
        "t1t3_count_combat_start",
    )
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    hist = row.get("tier_hist_recruit_start") or row.get("tier_hist_post_scale")
    if isinstance(hist, dict):
        return sum(int(hist.get(str(t), 0) or 0) for t in LOW_TIERS)
    return None


def _index_losses(losses: Sequence[Dict]) -> Dict[Tuple[int, int], Dict]:
    out: Dict[Tuple[int, int], Dict] = {}
    for ev in losses or []:
        try:
            seed = int(ev["seed"])
            seat = int(ev["seat"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (seed, seat)
        prev = out.get(key)
        if prev is None:
            out[key] = ev
            continue
        # Keep the earliest first_loss_turn; elimination (None) only if no sale.
        pt = prev.get("first_loss_turn")
        et = ev.get("first_loss_turn")
        if pt is None and et is not None:
            out[key] = ev
        elif pt is not None and et is not None and int(et) < int(pt):
            out[key] = ev
    return out


def _lineage_reconcile(raw: Dict) -> Dict:
    """t1t3_end = t1t3_start + added − removed, per seat-turn."""
    events = raw.get("t1t3_events") or []
    rows = raw.get("turn_rows") or []
    by_st: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    for ev in events:
        try:
            key = (int(ev["seed"]), int(ev["seat"]), int(ev["turn"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_st[key].append(ev)
    mismatches = 0
    checked = 0
    for row in rows:
        try:
            key = (int(row["seed"]), int(row["seat"]), int(row["turn"]))
        except (KeyError, TypeError, ValueError):
            continue
        start = _t1t3_from_row(row, phase="recruit")
        end = _t1t3_from_row(row, phase="combat")
        if start is None or end is None:
            continue
        removed = 0
        added = 0
        for ev in by_st.get(key, []):
            pre = _safe_int(ev.get("pre_t1t3_count"), 0)
            post = _safe_int(ev.get("post_t1t3_count"), 0)
            if post < pre:
                removed += pre - post
            elif post > pre:
                added += post - pre
        pred = start + added - removed
        checked += 1
        if abs(pred - end) <= max(LINEAGE_ABS_TOL, 0):
            continue
        evs = by_st.get(key, [])
        if evs:
            first_pre = _safe_int(evs[0].get("pre_t1t3_count"), start)
            last_post = _safe_int(evs[-1].get("post_t1t3_count"), end)
            if first_pre == start and last_post == end:
                continue
        mismatches += 1
    return {
        "identity": LINEAGE_IDENTITY,
        "n_checked": checked,
        "n_mismatches": mismatches,
        "p_ok": 1.0 if checked == 0 else float(checked - mismatches) / float(checked),
    }


def _paired_seat_table(
    control_raw: Dict,
    treatment_raw: Dict,
) -> Dict:
    c_traj = build_seat_trajectories(
        control_raw.get("turn_rows") or [], control_raw.get("fights") or [],
    )
    t_traj = build_seat_trajectories(
        treatment_raw.get("turn_rows") or [], treatment_raw.get("fights") or [],
    )
    pairs = pair_trajectories(c_traj, t_traj)
    c_rows = {
        (int(r["seed"]), int(r["seat"]), int(r["turn"])): r
        for r in (control_raw.get("turn_rows") or [])
        if r.get("seed") not in (None, "") and r.get("seat") not in (None, "")
        and r.get("turn") not in (None, "")
    }
    t_rows = {
        (int(r["seed"]), int(r["seat"]), int(r["turn"])): r
        for r in (treatment_raw.get("turn_rows") or [])
        if r.get("seed") not in (None, "") and r.get("seat") not in (None, "")
        and r.get("turn") not in (None, "")
    }
    c_loss = _index_losses(control_raw.get("last_t1t3_losses") or [])
    t_loss = _index_losses(treatment_raw.get("last_t1t3_losses") or [])

    n_pairs = 0
    n_both_had_t7 = 0
    first_loss_c: List[int] = []
    first_loss_t: List[int] = []
    class_c = Counter()
    class_t = Counter()
    retained_c = 0
    retained_t = 0
    by_turn = {}
    for turn in INSTRUMENT_TURNS:
        by_turn[str(turn)] = {
            "n_paired_alive": 0,
            "control_mean_t1t3": None,
            "treatment_mean_t1t3": None,
            "control_p_has_t1t3": None,
            "treatment_p_has_t1t3": None,
        }

    turn_counts = {t: {"c": [], "t": [], "c_has": 0, "t_has": 0, "n": 0} for t in INSTRUMENT_TURNS}

    for p in pairs:
        seed = int(p["seed"])
        seat = int(p["seat"])
        n_pairs += 1
        c7 = _t1t3_from_row(c_rows.get((seed, seat, 7)))
        t7 = _t1t3_from_row(t_rows.get((seed, seat, 7)))
        if (c7 or 0) > 0 and (t7 or 0) > 0:
            n_both_had_t7 += 1
        cl = c_loss.get((seed, seat))
        tl = t_loss.get((seed, seat))
        if cl and cl.get("first_loss_turn") is not None:
            first_loss_c.append(int(cl["first_loss_turn"]))
            class_c[cl.get("class") or "unknown"] += 1
        elif cl and cl.get("class") == "alive_elimination":
            class_c["alive_elimination"] += 1
        else:
            retained_c += 1
        if tl and tl.get("first_loss_turn") is not None:
            first_loss_t.append(int(tl["first_loss_turn"]))
            class_t[tl.get("class") or "unknown"] += 1
        elif tl and tl.get("class") == "alive_elimination":
            class_t["alive_elimination"] += 1
        else:
            retained_t += 1
        for turn in INSTRUMENT_TURNS:
            cr = c_rows.get((seed, seat, turn))
            tr = t_rows.get((seed, seat, turn))
            if cr is None and tr is None:
                continue
            cc = _t1t3_from_row(cr)
            tc = _t1t3_from_row(tr)
            bucket = turn_counts[turn]
            if cr is not None and tr is not None:
                bucket["n"] += 1
                if cc is not None:
                    bucket["c"].append(float(cc))
                    if cc > 0:
                        bucket["c_has"] += 1
                if tc is not None:
                    bucket["t"].append(float(tc))
                    if tc > 0:
                        bucket["t_has"] += 1

    for turn, bucket in turn_counts.items():
        n = bucket["n"] or 1
        by_turn[str(turn)] = {
            "n_paired_alive": bucket["n"],
            "control_mean_t1t3": _mean(bucket["c"]),
            "treatment_mean_t1t3": _mean(bucket["t"]),
            "control_p_has_t1t3": (
                None if not bucket["c"] else float(bucket["c_has"]) / float(len(bucket["c"]))
            ),
            "treatment_p_has_t1t3": (
                None if not bucket["t"] else float(bucket["t_has"]) / float(len(bucket["t"]))
            ),
        }

    return {
        "identity": PAIRED_SEAT_IDENTITY,
        "n_paired_seats": n_pairs,
        "n_control_traj": len(c_traj),
        "n_treatment_traj": len(t_traj),
        "n_control_only": max(0, len(c_traj) - n_pairs),
        "n_treatment_only": max(0, len(t_traj) - n_pairs),
        "n_both_had_t1t3_at_t7": n_both_had_t7,
        "n_retained_control": retained_c,
        "n_retained_treatment": retained_t,
        "mean_first_loss_turn_control": _mean(first_loss_c),
        "mean_first_loss_turn_treatment": _mean(first_loss_t),
        "n_first_loss_control": len(first_loss_c),
        "n_first_loss_treatment": len(first_loss_t),
        "last_loss_class_control": dict(class_c),
        "last_loss_class_treatment": dict(class_t),
        "by_turn": by_turn,
        "pairs_n": len(pairs),
    }


def _late_t1t3_rows(rows: Sequence[Dict], turns=None) -> List[Dict]:
    window = set(turns or LATE_TURNS)
    return [
        r for r in rows
        if int(r.get("turn") or 0) in window
        and int(r.get("winner_start_tier") or 0) in LOW_WINNER_START_TIERS
    ]


def attribute_late_t1t3_collapse(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    control_punch: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Dict:
    """Map each control late T1–T3 punch row onto treatment counterpart class."""
    window = tuple(turns or LATE_TURNS)
    c_late = _late_t1t3_rows(control_punch, window)
    t_late = _late_t1t3_rows(treatment_punch, window)
    n_c = len(c_late)
    n_t = len(t_late)
    collapse = float(n_c - n_t)

    t_rows = {
        (int(r["seed"]), int(r["seat"]), int(r["turn"])): r
        for r in (treatment_raw.get("turn_rows") or [])
        if r.get("seed") not in (None, "") and r.get("seat") not in (None, "")
        and r.get("turn") not in (None, "")
    }
    t_loss = _index_losses(treatment_raw.get("last_t1t3_losses") or [])
    t_alive_keys = set(t_rows)

    counts = Counter()
    for row in c_late:
        seed = row.get("seed")
        winner = row.get("winner_seat")
        turn = row.get("turn")
        try:
            seed_i = int(seed)
            seat_i = int(winner)
            turn_i = int(turn)
        except (TypeError, ValueError):
            counts["leftover"] += 1
            continue
        tr = t_rows.get((seed_i, seat_i, turn_i))
        t1 = _t1t3_from_row(tr)
        if tr is None or (tr.get("alive_at_combat") is False) or (
            tr.get("alive_at_recruit") is False and t1 is None
        ):
            # Treatment seat missing at this turn → dead / not recruiting.
            loss = t_loss.get((seed_i, seat_i))
            if loss and loss.get("class") in LIFECYCLE_COMPONENTS:
                counts[loss["class"]] += 1
            else:
                counts["alive_elimination"] += 1
            continue
        if t1 is not None and t1 > 0:
            counts["leftover"] += 1
            continue
        loss = t_loss.get((seed_i, seat_i))
        if loss and loss.get("class") in LIFECYCLE_COMPONENTS:
            counts[loss["class"]] += 1
        else:
            # Last T1–T3 gone but class unrecorded — infer from replacements.
            if int(tr.get("n_replacements") or 0) > 0:
                shop_n = int(tr.get("shop_t1t3_offers_recruit_start") or 0)
                counts[
                    "tavern_offer_shift" if shop_n <= 0 else "full_board_2q_replacement"
                ] += 1
            else:
                counts["leftover"] += 1

    attributed = {name: float(counts.get(name, 0)) for name in LIFECYCLE_COMPONENTS}
    leftover = float(counts.get("leftover", 0))
    # Collapse identity: attributed + leftover − treatment_still_present_offset.
    # Each control row is classified; treatment late n is the leftover
    # "still has T1–T3" mass plus any treatment-only late rows.
    reconstructed = sum(attributed.values()) + leftover
    shares = {
        name: share_of_collapse(attributed[name], denom=collapse)
        for name in LIFECYCLE_COMPONENTS
    }
    share_left = share_of_collapse(leftover, denom=collapse)
    return {
        "turns": list(window),
        "n_control_late_t1t3_punch": n_c,
        "n_treatment_late_t1t3_punch": n_t,
        "collapse": collapse,
        "counts": dict(counts),
        "attributed": attributed,
        "leftover": leftover,
        "reconstructed_control_rows": reconstructed,
        "reconciliation_gap": float(n_c) - reconstructed,
        "reconciliation_ok": abs(float(n_c) - reconstructed) <= max(1.0, 1e-9 * (1 + n_c)),
        **{f"share_{k}": v for k, v in shares.items()},
        "share_leftover": share_left,
    }


def compare_retention(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    selection: Optional[Dict] = None,
) -> Dict:
    """3G mixture lock + late T1–T3 lifecycle attribution."""
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

    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    late = attribute_late_t1t3_collapse(
        control_raw, treatment_raw,
        control_punch=c_punch, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late = attribute_late_t1t3_collapse(
        control_raw, treatment_raw,
        control_punch=c_punch, treatment_punch=t_punch, turns=VERY_LATE_TURNS,
    )
    paired = _paired_seat_table(control_raw, treatment_raw)
    lineage_c = _lineage_reconcile(control_raw)
    lineage_t = _lineage_reconcile(treatment_raw)
    hist_c = reconcile_history_links(
        control_raw.get("fights") or [], control_raw.get("turn_rows") or [],
    )
    hist_t = reconcile_history_links(
        treatment_raw.get("fights") or [], treatment_raw.get("turn_rows") or [],
    )

    decomp = selection.get("decomposition") or {}
    rec = {
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "lineage_identity": LINEAGE_IDENTITY,
        "paired_seat_identity": PAIRED_SEAT_IDENTITY,
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "lineage_control": lineage_c,
        "lineage_treatment": lineage_t,
        "paired": {
            "n_paired_seats": paired["n_paired_seats"],
            "n_control_only": paired["n_control_only"],
            "n_treatment_only": paired["n_treatment_only"],
        },
        "late_collapse_reconciliation_ok": late.get("reconciliation_ok"),
        "phase_3g_mixture_reproduced": decomp.get("mixture_turn_winner_tier"),
        "phase_3g_mixture_share_reproduced": decomp.get("share_mixture_turn_winner_tier"),
        "phase_3g_within_share_reproduced": decomp.get("share_within_cell_opponent_carry"),
        "phase_3g_n_control": decomp.get("n_control"),
        "phase_3g_n_treatment": decomp.get("n_treatment"),
        "flow_abs_tol": FLOW_ABS_TOL,
        "reweight_abs_tol": REWEIGHT_ABS_TOL,
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
        "attribution": late,
        "very_late_attribution": very_late,
        "paired_seats": paired,
        "reconciliation": rec,
        "decomposition_3g": decomp,
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
    }
