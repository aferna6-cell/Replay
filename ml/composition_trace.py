"""Event-level recruit traces for Phase 2C composition diagnosis."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from hsbg_coach.bg_env import (
    A_BUY0, A_END, A_FREEZE, A_LEVEL, A_PLAY0, A_ROLL, A_SELL0,
    BGEnv, BUY_COST, MAX_TURNS, N_BUY, N_PLAY, N_SELL, greedy_policy,
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


def _decode_action(action: int, obs: Dict, pre_shop: List[Dict],
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


def _detect_triple_discover(pre_board, pre_hand, post_board, post_hand,
                            acted_card_name: Optional[str]) -> Dict:
    pre_golden = sum(1 for m in pre_board + pre_hand if m.get("golden"))
    post_golden = sum(1 for m in post_board + post_hand if m.get("golden"))
    triple = post_golden > pre_golden
    discover = None
    if triple:
        pre_names = {(m["name"], m.get("golden")) for m in pre_board + pre_hand}
        for m in post_hand + post_board:
            key = (m["name"], m.get("golden"))
            if key not in pre_names and m["name"] != acted_card_name:
                discover = m
                break
            if m.get("golden") and not any(
                    x.get("golden") and x["name"] == m["name"] for x in pre_board + pre_hand):
                pass
        if discover is None and len(post_hand) > len(pre_hand):
            for m in post_hand:
                if all(m["name"] != p["name"] or p.get("golden") != m.get("golden")
                       for p in pre_hand):
                    discover = m
                    break
    return {"triple": triple, "discover_reward": discover}


def run_traced_rollouts(lobbies: int, seed: int = 0,
                        policy: Callable = greedy_policy,
                        scaling_mode: str = "residual") -> Dict:
    """Play greedy lobbies and record recruit-phase events (measurement only)."""
    events: List[Dict] = []
    turn_summaries: List[Dict] = []
    player_finals: List[Dict] = []

    for lobby_i in range(lobbies):
        env = BGEnv(seed=seed + lobby_i, scaling_mode=scaling_mode)
        env.reset()
        policies: Sequence[Callable] = [policy] * env.n_players
        game_length = 0

        while not env._done:
            for seat in range(env.n_players):
                p = env.players[seat]
                if not p.alive:
                    continue
                pol = (policies[seat] if seat < len(policies) else None) or greedy_policy
                board_before_recruit = _cards_view(p.board)
                shop_offered = _cards_view(p.shop)

                for _ in range(40):
                    obs = env.observe(seat)
                    mask = env.legal_mask(seat)
                    pre_shop = _cards_view(p.shop)
                    pre_hand = _cards_view(p.hand)
                    pre_board = _cards_view(p.board)
                    pre_gold = p.gold
                    pre_tier = p.tier

                    action = pol(obs, mask, env.rng)
                    decoded = _decode_action(action, obs, pre_shop, pre_hand, pre_board)
                    env._apply(seat, action)

                    post_board = _cards_view(p.board)
                    post_hand = _cards_view(p.hand)
                    triple_info = _detect_triple_discover(
                        pre_board, pre_hand, post_board, post_hand,
                        (decoded.get("card") or {}).get("name"))

                    events.append({
                        "lobby": lobby_i,
                        "seed": seed + lobby_i,
                        "seat": seat,
                        "turn": env.turn,
                        "action": decoded["kind"],
                        "slot": decoded["slot"],
                        "card": decoded["card"],
                        "gold_before": pre_gold,
                        "gold_after": p.gold,
                        "tavern_tier": pre_tier,
                        "shop_offered": shop_offered,
                        "board_before": pre_board,
                        "board_after": post_board,
                        "hand_after": post_hand,
                        "triple": triple_info["triple"],
                        "discover_reward": triple_info["discover_reward"],
                        "target": _target_summary(p.board),
                    })

                    if decoded["kind"] == "end":
                        break

                turn_summaries.append({
                    "lobby": lobby_i,
                    "seed": seed + lobby_i,
                    "seat": seat,
                    "turn": env.turn,
                    "placement": None,
                    "tavern_tier": p.tier,
                    "gold": p.gold,
                    "board_before_recruit": board_before_recruit,
                    "board_after_recruit": _cards_view(p.board),
                    "shop_offered": shop_offered,
                    "target": _target_summary(p.board),
                    "board_stats": board_stats({"board": p.board}),
                })

            game_length = env.turn
            env._scale_all()
            env._run_combat()
            alive = [pl for pl in env.players if pl.alive]
            if len(alive) <= 1 or env.turn >= MAX_TURNS:
                env._finalize()
            else:
                env._start_turn()

        for r in turn_summaries:
            if r["lobby"] == lobby_i:
                r["placement"] = env.players[r["seat"]].placement

        for seat, pl in enumerate(env.players):
            final_board = _cards_view(pl.board) if pl.board else _cards_view(pl.last_board)
            player_finals.append({
                "lobby": lobby_i,
                "seed": seed + lobby_i,
                "seat": seat,
                "placement": pl.placement,
                "game_length": game_length,
                "final_board": final_board,
                "target": _target_summary(
                    pl.board if pl.board else pl.last_board),
            })

        del env

    return {
        "lobbies": lobbies,
        "seed": seed,
        "scaling_mode": scaling_mode,
        "events": events,
        "turn_summaries": turn_summaries,
        "player_finals": player_finals,
        "archetypes": [
            {"key": a.key, "name": a.name, "tribe": a.tribe,
             "core_cards": list(a.core.keys()), "board_count": a.board_count}
            for a in load_archetypes()
        ],
    }
