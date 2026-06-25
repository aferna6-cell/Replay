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
    BUY, SELL, LEVEL, ROLL, BUY_COST, SELL_VALUE, MAX_BOARD, tavern_up_cost,
)
from .advisor import advise_actions, _as_state, Action
from .board_value import get_scorer, _val, _name
from .pace import load_pace

_LOW_HP = 15.0
_K_TRAJ = 1.5          # being ahead/behind the pace curve, in placement units
_K_HP = 1.5           # HP risk, in placement units
_DEATH_PEN = 2.0      # a projected death is a big placement hit


def _get(s, k, d=None):
    return s.get(k, d) if isinstance(s, dict) else getattr(s, k, d)


def expected_placement(snapshot, scorer=None, pace=None, horizon: int = 4) -> float:
    """Expected final placement (1=1st … 8=last; lower is better) for a state."""
    scorer = scorer or get_scorer()
    pace = pace if pace is not None else load_pace()
    board = list(_get(snapshot, "board", []) or [])
    hero_id = _get(snapshot, "hero", None) or "UNKNOWN"

    equity = scorer.equity(board, hero_id)              # 0..1, higher = better
    placement = 8.0 - equity * 7.0                       # 1..8, lower = better

    plan = multiturn.best_plan(snapshot, pace, horizon)
    if plan and plan.projection:
        term = plan.projection[-1]
        placement += -(term.ratio - 1.0) * _K_TRAJ       # ahead of curve -> better
        if plan.died:
            placement += _DEATH_PEN

    hp = _get(snapshot, "hero_health", None)
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
    plan = advise_actions(snapshot, kb=kb, hero_ctx=hero_ctx, scorer=scorer)
    base = expected_placement(snapshot, scorer, pace, horizon)
    state = _as_state(snapshot)

    recs: List[WholeGameRec] = []
    for sa in plan.ranked:
        a = sa.action
        if a.kind in (BUY, SELL, LEVEL, ROLL):
            v = expected_placement(_apply(state, a), scorer, pace, horizon)
        else:
            v = base                                     # reposition/freeze/end: neutral here
        recs.append(WholeGameRec(a, round(v, 2), sa.reason, round(base - v, 2)))
    recs.sort(key=lambda r: r.placement)
    return recs, base
