"""Whole-game value: score a state by expected FINAL placement, not just the
current board.

Every move the recommender suggests should account for the rest of the game. So
instead of ranking actions by immediate board strength, we rank them by
`expected_placement(state_after_move)` — a blend of three signals:

  1. **Learned board value** — the eval net (trained on real final boards →
     placement) reads the board composition. This is already a whole-game signal:
     "how well does this kind of board *finish*."
  2. **Trajectory** — the multi-turn planner projects where this state is heading
     (tier / board-vs-curve / HP). Being ahead of the pace curve lowers expected
     placement; a line that dies raises it.
  3. **Survival** — low HP pushes expected placement up (closer to going out).

Because each action's resulting state is valued this way, leveling, rolling and
buying are finally compared on ONE axis — expected finish — with the future of
the game baked in. It's an honest blend (the board term is learned; trajectory +
HP are the heuristic economy model), not a perfect oracle.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import multiturn
from .actions import (
    BUY, BUY_SPELL, SELL, LEVEL, ROLL, REPOSITION, HERO_POWER, BUY_COST, SELL_VALUE,
    MAX_BOARD, tavern_up_cost,
)
from .advisor import advise_actions, _as_state, Action
from .board_value import get_scorer, _val, _name
from .pace import load_pace, _at as _curve_at

_LOW_HP = 15.0
_K_TRAJ = 1.5          # being ahead/behind the pace curve, in placement units
_K_HP = 1.5           # HP risk, in placement units
_DEATH_PEN = 2.0      # a projected death is a big placement hit
_K_POS = 3.0          # combat win% gain from repositioning -> placement units

_ECON = "unloaded"     # cached economy value model (or None if unavailable)


def _econ_model():
    """The learned economy trajectory value (ml/econ_value.pt), if trained."""
    global _ECON
    if _ECON == "unloaded":
        _ECON = None
        try:
            import os
            from ml.econ_value import EconValue
            p = os.path.join(os.path.dirname(__file__), "..", "ml", "econ_value.pt")
            if os.path.isfile(p):
                _ECON = EconValue.load(p)
        except Exception:
            _ECON = None
    return _ECON


_SELL_TEMPO_PEN = 0.6         # a naked sell loses a body — you want 7 minions
_SELL_FULL_BOARD = 7          # full Battlegrounds board


def _get(s, k, d=None):
    return s.get(k, d) if isinstance(s, dict) else getattr(s, k, d)


_K_SIM_TECH = 2.2          # combat win% delta -> placement units for sim-grounded tech


def _tech_adjust(action, opponent_board, player_board=None):
    """(placement_adjustment, reason|None) for a tech-card BUY, matchup-aware.
    Negative promotes the card; positive demotes it. (0, None) for non-tech.

    Preferred path: simulate the player's board with vs without the tech against
    the last opponent and value it by the real combat-win delta — this 'recognizes
    the situation' for ANY opponent board (the sim now models Tunnel Blaster's AOE
    deathrattle and Venomous). Falls back to the keyword heuristic with no opponent."""
    from .card_roles import is_tech, tech_assessment
    minion = action.detail.get("minion")
    cid = (minion.get("card_id") if isinstance(minion, dict)
           else getattr(minion, "card_id", None))
    if not is_tech(cid, action.target):
        return 0.0, None
    if opponent_board and player_board and minion is not None:
        simmed = _sim_tech(player_board, minion, opponent_board)
        if simmed is not None:
            return simmed
    return tech_assessment(cid, action.target, opponent_board) or (0.0, None)


def _sim_tech(board, candidate, opponent):
    """Placement adjustment from the simulated combat-win delta of adding `candidate`
    to `board` against `opponent`. None if the sim can't run."""
    try:
        from .sim import simulate, Combatant
        opp = [Combatant.from_minion(m) for m in opponent]
        if not opp:
            return None
        base = simulate([Combatant.from_minion(m) for m in board], opp, runs=200).win_pct
        with_tech = simulate([Combatant.from_minion(m) for m in board]
                             + [Combatant.from_minion(candidate)], opp, runs=200).win_pct
        delta = with_tech - base
        adj = -max(-_MAX_TECH, min(_MAX_TECH, delta * _K_SIM_TECH))
        if delta > 0.03:
            return adj, f"+{delta:.0%} combat win vs their board (sim)"
        return adj, "no combat swing vs their board (sim) — situational"
    except Exception:
        return None


_MAX_TECH = 0.8


def _build_path_adjust(action, snapshot):
    """(placement_adjustment, reason) for how a BUY advances a reachable winning
    archetype. Negative = advances the build; positive = scatters it mid-game."""
    from .build_path import path_value
    minion = action.detail.get("minion")
    tribe = None
    if isinstance(minion, dict):
        tribe = minion.get("tribe") or (minion.get("tags") or {}).get("tribe")
    try:
        return path_value(_get(snapshot, "board", []) or [], action.target,
                          _get(snapshot, "tavern_tier"), candidate_tribe=tribe)
    except Exception:
        return 0.0, None


_K_SYN = 0.18              # effect-synergy points -> placement units
_MAX_SYN = 0.6


def _effect_synergy_adjust(action, snapshot, kb):
    """(placement_adjustment, reason) from how a BUY's effects mesh with the board
    — generalizes combos from card text (produces/wants), not co-occurrence data."""
    if kb is None:
        return 0.0, None
    try:
        from .effect_synergy import board_synergy
        from .cards import by_name
        idx = by_name(kb)
        cand = idx.get(action.target)
        if cand is None:
            return 0.0, None
        board = [idx.get(_name(m)) for m in (_get(snapshot, "board", []) or [])]
        board = [c for c in board if c]
        score, reasons = board_synergy(cand, board)
        if score <= 0:
            return 0.0, None
        return -min(_MAX_SYN, score * _K_SYN), ("; ".join(reasons) if reasons else None)
    except Exception:
        return 0.0, None


def _sell_penalty(state) -> float:
    """A standalone sell shrinks your board; you almost always want a full 7.
    Penalize it (scaled up when you have few minions) so the recommender won't
    suggest selling unless the board genuinely improves enough to overcome it, or
    it's a buy making room (handled inside the buy action's 'sell X for room')."""
    n = len(state.get("board", []) or [])
    short = max(0, _SELL_FULL_BOARD - n)            # how far below a full board
    return _SELL_TEMPO_PEN + short * 0.08


def expected_placement(snapshot, scorer=None, pace=None, horizon: int = 4) -> float:
    """Expected final placement (1=1st … 8=last; lower is better) for a state."""
    scorer = scorer or get_scorer()
    pace = pace if pace is not None else load_pace()
    board = list(_get(snapshot, "board", []) or [])
    hero_id = _get(snapshot, "hero", None) or "UNKNOWN"

    equity = scorer.equity(board, hero_id)              # 0..1, higher = better
    board_placement = 8.0 - equity * 7.0                 # composition value (learned)

    turn = _get(snapshot, "turn", None) or 8
    tier = _get(snapshot, "tavern_tier", None) or 1
    hp = _get(snapshot, "hero_health", None)

    econ = _econ_model()
    if econ is not None:
        # Learned trajectory value (trained on self-play lobbies): blend the
        # board-composition read with the economy/tempo/HP outlook.
        from ml.econ_env import alive_at
        strength = sum(_val(m) for m in board)
        curve = _curve_at(pace.get("scaling", {}), turn) or max(1.0, strength)
        ratio = strength / curve if curve else 1.0
        econ_pl = econ.predict(turn, tier, strength, ratio,
                               hp if hp is not None else 30.0,
                               players_left=alive_at(turn))
        return max(1.0, min(8.0, 0.5 * board_placement + 0.5 * econ_pl))

    # Heuristic fallback (no econ model trained yet).
    placement = board_placement
    plan = multiturn.best_plan(snapshot, pace, horizon)
    if plan and plan.projection:
        term = plan.projection[-1]
        placement += -(term.ratio - 1.0) * _K_TRAJ       # ahead of curve -> better
        if plan.died:
            placement += _DEATH_PEN
    if hp is not None and hp < _LOW_HP:
        placement += (_LOW_HP - hp) / _LOW_HP * _K_HP
    return max(1.0, min(8.0, placement))


def _apply(state: dict, action: Action) -> dict:
    """Resulting state after an action (buy/sell/level/roll). Roll can't see the
    next shop, so it's modelled as 'spend 1 gold, board unchanged' — its value
    comes from the trajectory/HP terms, not a simulated draw."""
    s = copy.deepcopy(state)
    board, shop = s["board"], s["shop"]
    if action.kind == BUY:
        if len(board) >= MAX_BOARD:
            board.remove(min(board, key=_val))
        for i, m in enumerate(shop):
            if _name(m) == action.target:
                board.append(shop.pop(i))
                break
        else:
            board.append(action.detail.get("minion") or {"name": action.target})
        s["gold"] -= BUY_COST
    elif action.kind == SELL:
        for i, m in enumerate(board):
            if _name(m) == action.target:
                board.pop(i)
                break
        s["gold"] += SELL_VALUE
    elif action.kind == LEVEL:
        cost = tavern_up_cost(s["tavern_tier"]) or 0
        s["tavern_tier"] = min(6, s["tavern_tier"] + 1)
        s["gold"] -= cost
    elif action.kind == ROLL:
        s["gold"] -= 1
    return s


@dataclass
class WholeGameRec:
    action: Action
    placement: float        # expected final placement after this move (lower better)
    reason: str
    gain: float             # base_placement - placement (positive = improves finish)

    def line(self) -> str:
        return (f"  finish {self.placement:.1f} ({self.gain:+.2f})  "
                f"{self.action.describe()} — {self.reason}")


def rank_actions(snapshot, kb=None, scorer=None, pace=None, hero_ctx=None,
                 horizon: int = 4) -> Tuple[List[WholeGameRec], float]:
    """Rank every legal action by the expected final placement of its result.

    Reuses `advise_actions` for the action set + synergy reasons, then re-scores
    each by whole-game value so leveling/rolling/buying are directly comparable."""
    scorer = scorer or get_scorer()
    pace = pace if pace is not None else load_pace()
    enemy_boards = _get(snapshot, "opponents_seen", None) or None
    plan = advise_actions(snapshot, kb=kb, hero_ctx=hero_ctx, scorer=scorer,
                          enemy_boards=enemy_boards)
    base = expected_placement(snapshot, scorer, pace, horizon)
    state = _as_state(snapshot)

    enemy0 = enemy_boards[0] if enemy_boards else None
    recs: List[WholeGameRec] = []
    for sa in plan.ranked:
        a = sa.action
        reason = sa.reason
        if a.kind in (BUY, SELL, LEVEL, ROLL):
            v = expected_placement(_apply(state, a), scorer, pace, horizon)
            if a.kind == BUY:                            # matchup-aware tech read
                adj, tech_reason = _tech_adjust(a, enemy0,
                                                _get(snapshot, "board", []) or [])
                v = max(1.0, min(8.0, v + adj))
                if tech_reason:
                    reason = tech_reason
                # Build-path: does this buy advance a reachable winning comp? This
                # is the mid-game navigation signal — value the move by where the
                # board is *heading*, not just how it looks now.
                padj, preason = _build_path_adjust(a, snapshot)
                if padj:
                    v = max(1.0, min(8.0, v + padj))
                    if preason and not tech_reason:
                        reason = preason
                # Effect-text synergy: does this card's mechanics combo with the
                # board (produces what they pay off, or pays off what they make)?
                sadj, sreason = _effect_synergy_adjust(a, snapshot, kb)
                if sadj:
                    v = max(1.0, min(8.0, v + sadj))
                    if sreason and not tech_reason and not preason:
                        reason = sreason
            elif a.kind == SELL:                         # you want a full board of 7
                v = min(8.0, v + _sell_penalty(state))
        elif a.kind == BUY_SPELL:
            # Spells don't change the board composition the eval net reads, so we
            # value them off base via spell_roles' placement bonus + the reason.
            from .spell_roles import spell_value
            spell = a.detail.get("spell") or {}
            cid = (spell.get("card_id") if isinstance(spell, dict)
                   else getattr(spell, "card_id", None))
            bonus, sreason = spell_value(cid, a.target, a.cost,
                                         _get(snapshot, "gold") or 0)
            v = max(1.0, base + bonus)
            reason = sreason
        elif a.kind == HERO_POWER:
            v = max(1.0, base - 0.15)        # using the hero power is generally +EV
        elif a.kind == REPOSITION and sa.delta:
            # Reposition doesn't change board composition, so placement is flat —
            # but a better attack order raises combat win%. Convert that win-rate
            # gain (carried in sa.delta) into a small placement improvement so good
            # positioning can surface among the ranked moves.
            v = max(1.0, base - sa.delta * _K_POS)
        else:
            v = base                                     # freeze/end: neutral here
        recs.append(WholeGameRec(a, round(v, 2), reason, round(base - v, 2)))
    recs.sort(key=lambda r: r.placement)
    return recs, base
