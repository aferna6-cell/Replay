"""Heuristic economy advisor — when to buy / level / roll / freeze / sell.

Encodes well-accepted Battlegrounds fundamentals as transparent rules over a
Snapshot. No ML, no opponent data needed — this is the tempo-vs-greed layer that
complements the sim-based positioning advice. Every suggestion carries a
rationale so the advice is auditable, and thresholds live in ``EconomyConfig`` so
they're easy to tune (and later, to let the learned policy override).

These are deliberately conservative defaults reflecting common consensus
(fill the board early, level when healthy with spare gold, freeze multiple wanted
minions you can't all afford, sell filler for strict upgrades on a full board).
Hero-specific and comp-specific timing is the job of later layers (population
stats + learned policy); this is the sane baseline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bg import ActionType

MINION_COST = 3   # cost to buy a minion from the shop
ROLL_COST = 1     # cost to refresh the shop


@dataclass
class EconomyConfig:
    low_health: int = 15        # below this, bias to tempo (buy/stay alive)
    safe_health: int = 28       # at/above this, greed (level/economy) is fine
    level_min_gold: int = 5     # don't suggest leveling under this much gold
    full_board: int = 7
    early_turns: int = 3        # turns where filling the board dominates
    max_tier: int = 6


@dataclass
class Suggestion:
    action: str                 # ActionType value
    rationale: str
    priority: float             # 0..1; higher = stronger
    detail: Dict = field(default_factory=dict)


def _val(m) -> int:
    if isinstance(m, dict):
        a, h = m.get("attack") or 0, m.get("health") or 0
    else:
        a, h = getattr(m, "attack", 0) or 0, getattr(m, "health", 0) or 0
    return int(a) + int(h)


def _name(m) -> str:
    if isinstance(m, dict):
        return m.get("name") or m.get("card_id") or "?"
    return getattr(m, "name", None) or getattr(m, "card_id", None) or "?"


def _get(snapshot, key, default=None):
    if isinstance(snapshot, dict):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def advise(snapshot, config: Optional[EconomyConfig] = None) -> List[Suggestion]:
    """Return economy suggestions for a Snapshot, ranked by priority (desc)."""
    cfg = config or EconomyConfig()
    turn = _get(snapshot, "turn") or 0
    tier = _get(snapshot, "tavern_tier") or 1
    gold = _get(snapshot, "gold")
    health = _get(snapshot, "hero_health")
    board = list(_get(snapshot, "board", []) or [])
    shop = list(_get(snapshot, "shop", []) or [])

    out: List[Suggestion] = []
    if gold is None:
        # Without gold we can't reason about spend; surface that, don't guess.
        out.append(Suggestion(ActionType.END_TURN.value,
                              "Gold unknown (not parsed yet) — can't advise spend.",
                              0.1))
        return out

    board_count = len(board)
    low_hp = health is not None and health < cfg.low_health
    safe_hp = health is None or health >= cfg.safe_health

    best_shop = max(shop, key=_val) if shop else None
    weakest_board = min(board, key=_val) if board else None

    # --- BUY: fill the board, especially early -----------------------------
    if best_shop and board_count < cfg.full_board and gold >= MINION_COST:
        early = turn <= cfg.early_turns
        pri = 0.85 if early else 0.6
        if low_hp:
            pri += 0.1  # tempo matters more when low
        out.append(Suggestion(
            ActionType.BUY.value,
            f"Buy {_name(best_shop)} to develop board ({board_count}/{cfg.full_board}).",
            min(pri, 0.99), {"name": _name(best_shop)}))

    # --- TIER UP: level when healthy with spare gold -----------------------
    if tier < cfg.max_tier and gold >= cfg.level_min_gold:
        # Greedier when healthy and holding gold; weaker when low health.
        pri = 0.7 if safe_hp else 0.35
        if low_hp:
            pri = 0.2
        if board_count == 0 and turn <= 1:
            pri = max(pri, 0.5)  # turn 1 you often can't do much else
        out.append(Suggestion(
            ActionType.TIER_UP.value,
            (f"Level to tier {tier + 1}"
             + (" — healthy with spare gold." if safe_hp
                else " only if you can afford the tempo hit.")),
            pri, {"to_tier": tier + 1}))

    # --- SELL: full board + a strict upgrade available ---------------------
    if (board_count >= cfg.full_board and best_shop and weakest_board
            and _val(best_shop) > _val(weakest_board) and gold >= MINION_COST):
        out.append(Suggestion(
            ActionType.SELL.value,
            f"Board full — sell {_name(weakest_board)} for {_name(best_shop)} "
            f"({_val(best_shop)} > {_val(weakest_board)} stats).",
            0.65, {"sell": _name(weakest_board), "buy": _name(best_shop)}))

    # --- FREEZE: multiple wanted minions you can't all afford --------------
    if shop and weakest_board is not None:
        wanted = [m for m in shop if _val(m) >= _val(weakest_board)]
        affordable = gold // MINION_COST
        if len(wanted) >= 2 and affordable < len(wanted):
            out.append(Suggestion(
                ActionType.FREEZE.value,
                f"Freeze — {len(wanted)} wanted minions, can only afford "
                f"{affordable} now.",
                0.55, {"wanted": [_name(m) for m in wanted]}))

    # --- ROLL: spare gold, nothing better to do ----------------------------
    nothing_to_buy = (not best_shop) or (
        weakest_board is not None and best_shop is not None
        and _val(best_shop) <= _val(weakest_board) and board_count >= cfg.full_board)
    if gold >= ROLL_COST and (board_count >= cfg.full_board or nothing_to_buy):
        # Don't roll away gold you need to level.
        pri = 0.4 if not (tier < cfg.max_tier and gold >= cfg.level_min_gold) else 0.25
        out.append(Suggestion(
            ActionType.ROLL.value,
            "Roll to find upgrades — board set and gold to spare.",
            pri, {}))

    out.sort(key=lambda s: s.priority, reverse=True)
    return out


def top_advice(snapshot, config: Optional[EconomyConfig] = None) -> Optional[str]:
    """Single best economy suggestion as a one-liner, or None."""
    sugg = advise(snapshot, config)
    if not sugg:
        return None
    s = sugg[0]
    return f"{s.action}: {s.rationale}"
