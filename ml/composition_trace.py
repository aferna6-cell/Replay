"""Event-level recruit traces for Phase 2C composition diagnosis.

Uses ``BGEnv.play_scripted`` with an observational ``RecruitTracer`` so traced
rollouts are behavior-equivalent to ordinary Simulator v1.1 rollouts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Dict, List, Optional, Sequence

from hsbg_coach.bg_env import (
    A_BUY0,
    A_END,
    A_FREEZE,
    A_LEVEL,
    A_PLAY0,
    A_ROLL,
    A_SELL0,
    BGEnv,
    N_BUY,
    N_PLAY,
    N_SELL,
    greedy_policy,
)
from hsbg_coach.build_path import infer_target, load_archetypes
from hsbg_coach.pace import board_stats


def _card_view(m) -> Dict:
    return {
        "name": m.name,
        "card_id": m.card_id,
        "tier": m.tier,
        "attack": m.attack,
        "health": m.health,
        "tribes": list(m.tribes),
        "golden": m.golden,
    }


def _cards_view(minions) -> List[Dict]:
    return [_card_view(m) for m in minions]


def _cards_view_from_dicts(minions) -> List[Dict]:
    return list(minions or [])


def _target_summary(board) -> Optional[Dict]:
    fit = infer_target(board)
    if not fit:
        return None
    return {
        "archetype_key": fit.arch.key,
        "archetype_name": fit.arch.name,
        "tribe": fit.arch.tribe,
        "coverage": fit.coverage,
        "core_have": fit.have,
        "core_total": fit.core_total,
        "core_cards": list(fit.arch.core.keys()),
    }


def _decode_action(action: int, pre_shop: List[Dict],
                   pre_hand: List[Dict], pre_board: List[Dict]) -> Dict:
    if A_BUY0 <= action < A_BUY0 + N_BUY:
        idx = action - A_BUY0
        card = pre_shop[idx] if idx < len(pre_shop) else None
        return {"kind": "buy", "slot": idx, "card": card}
    if A_PLAY0 <= action < A_PLAY0 + N_PLAY:
        idx = action - A_PLAY0
        card = pre_hand[idx] if idx < len(pre_hand) else None
        return {"kind": "play", "slot": idx, "card": card}
    if A_SELL0 <= action < A_SELL0 + N_SELL:
        idx = action - A_SELL0
        card = pre_board[idx] if idx < len(pre_board) else None
        return {"kind": "sell", "slot": idx, "card": card}
    if action == A_ROLL:
        return {"kind": "roll", "slot": None, "card": None}
    if action == A_LEVEL:
        return {"kind": "level", "slot": None, "card": None}
    if action == A_FREEZE:
        return {"kind": "freeze", "slot": None, "card": None}
    if action == A_END:
        return {"kind": "end", "slot": None, "card": None}
    return {"kind": "unknown", "slot": None, "card": None}


def board_fingerprint(board: List[Dict]) -> str:
    """Stable hash of a board for rollout equivalence checks."""
    payload = sorted(
        (c.get("name"), c.get("attack"), c.get("health"), c.get("golden"))
        for c in (board or []) if c.get("name"))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class RecruitTracer:
    """Observational callbacks for ``play_scripted`` (must not mutate env)."""

    def __init__(self, lobby_id: int, seed: int):
        self.lobby_id = lobby_id
        self.seed = seed
        self.lobby_tribes: List[str] = []
        self.game_length = 0
        self.events: List[Dict] = []
        self.turn_summaries: List[Dict] = []
        self.player_finals: List[Dict] = []
        self._pre: Optional[Dict] = None
        self._board_before_recruit: List[Dict] = []

    def begin_lobby(self, lobby_id: int, _rng_seed: int, lobby_tribes: List[str]) -> None:
        self.lobby_id = lobby_id
        self.lobby_tribes = list(lobby_tribes)

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        self._board_before_recruit = _cards_view(player.board)

    def before_action(self, seat: int, turn: int, shop_generation: int,
                      obs: Dict, mask: List[bool]) -> None:
        pre_shop = _cards_view_from_dicts(obs.get("shop"))
        legal_buy_slots = [
            i for i in range(len(pre_shop))
            if i < len(mask) and mask[A_BUY0 + i]]
        self._pre = {
            "seat": seat,
            "turn": turn,
            "shop_generation": shop_generation,
            "pre_shop": pre_shop,
            "pre_hand": _cards_view_from_dicts(obs.get("hand")),
            "pre_board": _cards_view_from_dicts(obs.get("board")),
            "gold_before": obs.get("gold"),
            "tavern_tier": obs.get("tavern_tier"),
            "legal_buy_slots": legal_buy_slots,
            "target_before": _target_summary(obs.get("board")),
        }

    def after_action(self, seat: int, turn: int, shop_generation: int,
                     action: int, ended: bool, player=None) -> None:
        if self._pre is None:
            return
        pre = self._pre
        decoded = _decode_action(
            action, pre["pre_shop"], pre["pre_hand"], pre["pre_board"])
        post_board = _cards_view(player.board) if player is not None else pre["pre_board"]
        post_hand = _cards_view(player.hand) if player is not None else pre["pre_hand"]

        self.events.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "seat": seat,
            "turn": turn,
            "shop_generation": pre["shop_generation"],
            "action": decoded["kind"],
            "slot": decoded["slot"],
            "card": decoded["card"],
            "gold_before": pre["gold_before"],
            "tavern_tier": pre["tavern_tier"],
            "pre_shop": pre["pre_shop"],
            "legal_buy_slots": pre["legal_buy_slots"],
            "board_before": pre["pre_board"],
            "hand_before": pre["pre_hand"],
            "board_after": post_board,
            "hand_after": post_hand,
            "target_before": pre["target_before"],
            "lobby_tribes": list(self.lobby_tribes),
            "ended_recruit": ended,
        })
        self._pre = None

    def end_seat_recruit(self, seat: int, turn: int, player) -> None:
        self.turn_summaries.append({
            "lobby": self.lobby_id,
            "seed": self.seed,
            "seat": seat,
            "turn": turn,
            "placement": None,
            "tavern_tier": player.tier,
            "gold": player.gold,
            "board_before_recruit": self._board_before_recruit,
            "board_after_recruit": _cards_view(player.board),
            "target": _target_summary(player.board),
            "lobby_tribes": list(self.lobby_tribes),
            "board_stats": board_stats({"board": player.board}),
        })

    def end_lobby(self, players) -> None:
        for seat, pl in enumerate(players):
            final_board = _cards_view(pl.board) if pl.board else _cards_view(pl.last_board)
            self.player_finals.append({
                "lobby": self.lobby_id,
                "seed": self.seed,
                "seat": seat,
                "placement": pl.placement,
                "game_length": self.game_length,
                "final_board": final_board,
                "final_board_fingerprint": board_fingerprint(final_board),
                "target": _target_summary(
                    pl.board if pl.board else pl.last_board),
                "lobby_tribes": list(self.lobby_tribes),
            })
        for ts in self.turn_summaries:
            ts["placement"] = players[ts["seat"]].placement


def run_traced_rollouts(lobbies: int, seed: int = 0,
                        policy: Callable = greedy_policy,
                        scaling_mode: str = "residual") -> Dict:
    """Play greedy lobbies via ``play_scripted`` and record recruit events."""
    all_events: List[Dict] = []
    all_turn_summaries: List[Dict] = []
    all_player_finals: List[Dict] = []
    lobby_meta: List[Dict] = []

    for lobby_i in range(lobbies):
        lobby_seed = seed + lobby_i
        env = BGEnv(seed=lobby_seed, scaling_mode=scaling_mode)
        tracer = RecruitTracer(lobby_id=lobby_i, seed=lobby_seed)
        env.play_scripted([policy] * env.n_players, recruit_tracer=tracer)
        game_length = env.turn
        for pf in tracer.player_finals:
            pf["game_length"] = game_length
        all_events.extend(tracer.events)
        all_turn_summaries.extend(tracer.turn_summaries)
        all_player_finals.extend(tracer.player_finals)
        lobby_meta.append({
            "lobby": lobby_i,
            "seed": lobby_seed,
            "lobby_tribes": list(env.lobby_tribes),
            "game_length": env.turn,
        })
        del env

    return {
        "lobbies": lobbies,
        "seed": seed,
        "scaling_mode": scaling_mode,
        "events": all_events,
        "turn_summaries": all_turn_summaries,
        "player_finals": all_player_finals,
        "lobby_meta": lobby_meta,
        "archetypes": [
            {"key": a.key, "name": a.name, "tribe": a.tribe,
             "core_cards": list(a.core.keys()), "board_count": a.board_count}
            for a in load_archetypes()
        ],
    }


def run_plain_rollouts(lobbies: int, seed: int = 0,
                       policy: Callable = greedy_policy,
                       scaling_mode: str = "residual") -> List[Dict]:
    """Ordinary ``play_scripted`` finals for equivalence checks."""
    finals: List[Dict] = []
    for lobby_i in range(lobbies):
        lobby_seed = seed + lobby_i
        env = BGEnv(seed=lobby_seed, scaling_mode=scaling_mode)
        env.play_scripted([policy] * env.n_players)
        for seat, pl in enumerate(env.players):
            final_board = _cards_view(pl.board) if pl.board else _cards_view(pl.last_board)
            finals.append({
                "lobby": lobby_i,
                "seed": lobby_seed,
                "seat": seat,
                "placement": pl.placement,
                "final_board_fingerprint": board_fingerprint(final_board),
            })
        del env
    return finals
