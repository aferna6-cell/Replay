"""Turn-level beam search — the whole turn planned globally, not greedily.

`advisor.plan_turn` applies the locally-best move and repeats: it can never
find sequences whose first move looks bad alone (sell two pieces → buy the
pair → triple; level → buy from the stronger pool). With ≤10 gold a turn's
deterministic action space (buys / sells / one level) is small, so we search
it: beam over action *sequences*, score each terminal state by whole-game
expected placement (`game_value.expected_placement` — the eval net + trajectory
+ HP blend), and keep the best line.

Non-deterministic actions can't be searched (a roll's outcome is hidden), so
roll/freeze stay as scored decorations on the chosen line — same honesty as
the advisor, but now on top of a globally-chosen buy/sell/level sequence.

Stdlib only; works with any scorer (heuristic or the learned set net).
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .actions import (
    Action, BUY, BUY_COST, SELL_VALUE, ROLL_COST, MAX_BOARD, MAX_TIER,
    tavern_up_cost,
)
from .board_value import get_scorer, _val, _name
from .game_value import (
    expected_placement, _keep_value, _build_path_adjust,
    _effect_synergy_adjust, _quality_buy_adjust, _filler_penalty,
    _low_tier_penalty, _off_comp_penalty,
)
from .pace import load_pace

_TRIPLE_BONUS = 1.2        # matches game_value's triple promotion
_SELL_CANDIDATES = 2       # weakest-by-keep-value sells considered per node
_EPS = 1e-6


def _get(s, k, d=None):
    return s.get(k, d) if isinstance(s, dict) else getattr(s, k, d)


def _as_state(snapshot) -> Dict:
    def mdict(m):
        if isinstance(m, dict):
            return dict(m)
        return {"name": getattr(m, "name", None),
                "card_id": getattr(m, "card_id", None),
                "attack": getattr(m, "attack", None),
                "health": getattr(m, "health", None),
                "tags": dict(getattr(m, "tags", {}) or {})}
    return {
        "board": [mdict(m) for m in (_get(snapshot, "board", []) or [])],
        "hand": [mdict(m) for m in (_get(snapshot, "hand", []) or [])],
        "shop": [mdict(m) for m in (_get(snapshot, "shop", []) or [])],
        "gold": int(_get(snapshot, "gold") or 0),
        "tavern_tier": int(_get(snapshot, "tavern_tier") or 1),
        "hero_health": _get(snapshot, "hero_health"),
        "turn": _get(snapshot, "turn"),
        "level_cost": _get(snapshot, "level_cost"),
    }


@dataclass
class _Node:
    state: Dict
    steps: List[str] = field(default_factory=list)
    actions: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    bonus: float = 0.0          # triple bonuses collected along the line
    leveled: bool = False
    score: float = 8.0          # expected placement (lower better), incl. bonus


@dataclass
class TurnPlan:
    steps: List[str]            # human-followable, in order
    expected: float             # expected final placement after the line
    base: float                 # expected placement doing nothing
    searched: int               # nodes evaluated (observability)
    # The chosen line as structured (kind, target) pairs — "buy"/"sell" carry
    # the card name, "level"/"roll" a None target. What programmatic consumers
    # (the RL distillation teacher) execute; `steps` is the human view of the
    # same line plus the advisory roll/reposition/end decorations.
    actions: List[Tuple[str, Optional[str]]] = field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.base - self.expected


def _completes_triple(state: Dict, name: str) -> bool:
    owned = state["board"] + state.get("hand", [])
    return sum(1 for m in owned if _name(m) == name) >= 2


def _buy_adjust(state: Dict, minion: Dict, kb) -> float:
    """The same card-knowledge read `rank_actions` applies to a BUY — meta
    quality, build-path fit, effect synergy, and the filler/low-tier/off-comp
    guards — so the searched line and the displayed ranking agree on which
    buys matter. Negative = better (placement units)."""
    a = Action(BUY, _name(minion), BUY_COST, {"minion": minion})
    padj, _ = _build_path_adjust(a, state)
    sadj, _ = _effect_synergy_adjust(a, state, kb)
    qadj, _, q_strong = _quality_buy_adjust(a)
    adj = padj + sadj + qadj
    if not q_strong and not (sadj <= -0.3 or padj <= -0.3):
        adj += max(_filler_penalty(a, state["board"]),
                   _low_tier_penalty(a, state["tavern_tier"], kb),
                   _off_comp_penalty(a, state, kb))
    return adj


def _score(state: Dict, scorer, pace, bonus: float,
           stat_tiebreak: float = 0.0) -> float:
    v = expected_placement(state, scorer, pace)
    if stat_tiebreak:
        # Monotone board-strength term under the learned value. A calibrated
        # value net saturates on clearly-winning boards (everything reads
        # "1st"), which blinds the search exactly where discrimination is
        # needed. Curve-normalized linear stats break those ties toward the
        # bigger board and make selling compounded stats read as the real
        # loss it is (log-scaled stats hid it). Env-mode teacher only.
        from .pace import _at as _curve_at
        curve = _curve_at(pace.get("scaling", {}), state.get("turn") or 8) or 10.0
        stats = sum(_val(m) for m in state["board"])
        v -= stat_tiebreak * min(4.0, stats / max(curve, 1.0))
    return v - bonus


def _key(state: Dict) -> Tuple:
    return (tuple(sorted(_name(m) for m in state["board"])),
            tuple(sorted(_name(m) for m in state["shop"])),
            state["gold"], state["tavern_tier"])


def _expand(node: _Node, kb, knowledge: bool = True) -> List[_Node]:
    """Deterministic successor states: buys, room-making sells, one level.

    knowledge=False searches on pure state value (scorer + trajectory) with no
    real-meta buy adjustments — the right mode inside the Phase 0 env, whose
    dynamics don't contain the real game's comp/meta layer."""
    out: List[_Node] = []
    s = node.state

    # Buys — each shop minion, straight to the board when there's room.
    if s["gold"] >= BUY_COST and len(s["board"]) < MAX_BOARD:
        for i, m in enumerate(s["shop"]):
            ns = copy.deepcopy(s)
            bought = ns["shop"].pop(i)
            ns["board"].append(bought)
            ns["gold"] -= BUY_COST
            bonus = _TRIPLE_BONUS if _completes_triple(s, _name(m)) else 0.0
            if knowledge:
                bonus -= _buy_adjust(s, m, kb)    # negative adjust = better buy
            label = f"Buy {_name(m)}" + (
                " — completes a TRIPLE" if bonus > _TRIPLE_BONUS - _EPS else "")
            out.append(_Node(ns, node.steps + [label],
                             node.actions + [("buy", _name(m))],
                             node.bonus + bonus, node.leveled))

    # Sells — only the weakest few by keep-value (stats + comp synergy), and
    # only when selling can enable something (room for a buy, or gold for one).
    board = s["board"]
    sell_useful = (len(board) >= MAX_BOARD and s["gold"] >= BUY_COST) or (
        s["shop"] and s["gold"] + SELL_VALUE >= BUY_COST and s["gold"] < BUY_COST)
    if board and sell_useful:
        ranked = sorted(board, key=lambda m: _keep_value(m, board, kb))
        for m in ranked[:_SELL_CANDIDATES]:
            ns = copy.deepcopy(s)
            for j, x in enumerate(ns["board"]):
                if _name(x) == _name(m):
                    ns["board"].pop(j)
                    break
            ns["gold"] += SELL_VALUE
            out.append(_Node(ns, node.steps + [f"Sell {_name(m)}"],
                             node.actions + [("sell", _name(m))],
                             node.bonus, node.leveled))

    # Level — once per turn; opens the stronger pool (valued via trajectory).
    if not node.leveled and s["tavern_tier"] < MAX_TIER:
        cost = s.get("level_cost")
        if cost is None:
            cost = tavern_up_cost(s["tavern_tier"])
        if cost is not None and s["gold"] >= cost:
            ns = copy.deepcopy(s)
            ns["gold"] -= cost
            ns["tavern_tier"] += 1
            ns["level_cost"] = None
            out.append(_Node(ns, node.steps + [
                f"Tier up to {ns['tavern_tier']} ({cost}g)"],
                node.actions + [("level", None)],
                node.bonus, True))
    return out


def plan_turn_search(snapshot, kb=None, scorer=None, pace=None,
                     beam: int = 8, depth: int = 8,
                     knowledge: bool = True,
                     stat_tiebreak: float = 0.0) -> TurnPlan:
    """Beam-search the turn. Returns the best action line plus trailing
    roll/reposition/end decorations (non-deterministic, so advisory)."""
    scorer = scorer or get_scorer()
    pace = pace if pace is not None else load_pace()

    root = _Node(_as_state(snapshot))
    root.score = _score(root.state, scorer, pace, 0.0, stat_tiebreak)
    base = root.score
    frontier = [root]
    best = root
    seen = {_key(root.state)}
    evaluated = 1

    for _ in range(depth):
        nxt: List[_Node] = []
        for node in frontier:
            for child in _expand(node, kb, knowledge):
                k = _key(child.state)
                if k in seen:
                    continue
                seen.add(k)
                child.score = _score(child.state, scorer, pace, child.bonus,
                                     stat_tiebreak)
                evaluated += 1
                nxt.append(child)
        if not nxt:
            break
        nxt.sort(key=lambda n: n.score)
        frontier = nxt[:beam]
        if frontier[0].score < best.score - _EPS:
            best = frontier[0]

    steps = list(best.steps)
    actions = list(best.actions)
    st = best.state
    if st["gold"] >= BUY_COST + ROLL_COST and st["shop"]:
        steps.append("Roll — gold left and no improving buys in this shop")
        actions.append(("roll", None))
    elif st["gold"] >= ROLL_COST and st["shop"] and len(st["board"]) < MAX_BOARD:
        steps.append("Roll — spend the last gold looking for a piece")
        actions.append(("roll", None))
    if len(st["board"]) >= 2:
        steps.append("Reposition for combat (see positioning)")
    steps.append("End turn")
    return TurnPlan(steps=steps, expected=round(max(1.0, best.score), 2),
                    base=round(max(1.0, base), 2), searched=evaluated,
                    actions=actions)
