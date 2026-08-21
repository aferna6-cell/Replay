"""Enumerate every legal action at a Battlegrounds decision point.

The recommender's job is to rank "what should I do now?", so first we need the
full menu of what's *possible*: buy each shop minion, sell each board minion,
roll, tier up, reposition, freeze, end. This module knows the rules (costs, board
cap, tier cap) and nothing about which action is good — that's the advisor.

Stdlib only.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# BG economy constants.
BUY_COST = 3
SELL_VALUE = 1
ROLL_COST = 1
MAX_BOARD = 7
MAX_TIER = 6
# Base tavern-up cost from tier T -> T+1. Real cost drops by 1 per turn you wait;
# we don't see that discount, so this is the affordability *upper bound*.
UPGRADE_COST = {1: 5, 2: 7, 3: 8, 4: 9, 5: 10}

BUY = "buy"
BUY_SPELL = "buy_spell"
SELL = "sell"
ROLL = "roll"
LEVEL = "level"
REPOSITION = "reposition"
FREEZE = "freeze"
HERO_POWER = "hero_power"
END = "end"
DARK_GIFT = "dark_gift"


def tavern_up_cost(tier: Optional[int]) -> Optional[int]:
    return UPGRADE_COST.get(tier or 0)


@dataclass
class Action:
    kind: str
    target: Optional[str] = None       # card name for buy/sell
    cost: int = 0                      # gold spent (negative = gold gained)
    detail: Dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == BUY:
            return f"Buy {self.target}"
        if self.kind == BUY_SPELL:
            return f"Buy spell: {self.target} ({self.cost}g)"
        if self.kind == HERO_POWER:
            tail = f" ({self.cost}g)" if self.cost else ""
            return f"Use hero power: {self.target}{tail}"
        if self.kind == SELL:
            return f"Sell {self.target}"
        if self.kind == LEVEL:
            return f"Tier up to {self.detail.get('to_tier', '?')} ({self.cost}g)"
        if self.kind == ROLL:
            return "Roll the shop"
        if self.kind == REPOSITION:
            return "Reposition the board"
        if self.kind == FREEZE:
            return "Freeze the shop"
        if self.kind == DARK_GIFT:
            return (f"Use dark gift: {self.target}" if self.target
                    else "Use dark gift")
        return "End turn"


def _get(snap, key, default=None):
    return snap.get(key, default) if isinstance(snap, dict) else getattr(snap, key, default)


def _name(m) -> str:
    if isinstance(m, dict):
        return m.get("name") or m.get("card_id") or "?"
    return getattr(m, "name", None) or getattr(m, "card_id", None) or "?"


def legal_actions(snapshot, kb=None) -> List[Action]:
    """Every action that is legal given current gold, tier, board and shop."""
    gold = _get(snapshot, "gold")
    tier = _get(snapshot, "tavern_tier") or 1
    board = list(_get(snapshot, "board", []) or [])
    shop = list(_get(snapshot, "shop", []) or [])
    gold = 0 if gold is None else int(gold)

    actions: List[Action] = []

    # Buy — need 3 gold; if the board is full it requires a sell first (the
    # advisor models that as buy-with-sell-for-room).
    if gold >= BUY_COST:
        for m in shop:
            actions.append(Action(BUY, _name(m), BUY_COST, {"minion": m}))

    # Use a hero power — one action per usable BUTTON (heroes can hold
    # several: Marin's treasures, gift-style powers — calibrated vs a real
    # log 2026-08-20).
    powers = _get(snapshot, "hero_powers", None)
    if not powers:
        hp = _get(snapshot, "hero_power", None)
        powers = [hp] if hp else []
    for hp in powers:
        if not (hp and hp.get("usable")):
            continue
        hp_cost = int(hp.get("cost") or 0)
        if gold >= hp_cost:
            actions.append(Action(HERO_POWER, hp.get("name") or "Hero Power",
                                  hp_cost, {"hero_power": hp}))

    # Buy a tavern spell — variable cost (its own COST), affordability checked.
    for sp in (_get(snapshot, "shop_spells", []) or []):
        cost = sp.get("cost") if isinstance(sp, dict) else getattr(sp, "cost", None)
        cost = BUY_COST if cost is None else int(cost)
        if gold >= cost:
            actions.append(Action(BUY_SPELL, _name(sp), cost, {"spell": sp}))

    # Sell — always legal, refunds 1 gold.
    for m in board:
        actions.append(Action(SELL, _name(m), -SELL_VALUE, {"minion": m}))

    # Roll — costs 1 gold, needs a shop to refresh.
    if gold >= ROLL_COST and shop:
        actions.append(Action(ROLL, cost=ROLL_COST))

    # Tier up — use the live discounted cost when we have it (the tavern lowers the
    # cost by 1 per turn on a tier), falling back to the base. This is what makes
    # early aggressive leveling (e.g. tier 2 on turn 2) legal.
    if tier < MAX_TIER:
        cost = _get(snapshot, "level_cost", None)
        if cost is None:
            cost = tavern_up_cost(tier)
        if cost is not None and gold >= cost:
            actions.append(Action(LEVEL, cost=cost, detail={"to_tier": tier + 1}))

    # Reposition — free, needs at least two minions to matter.
    if len(board) >= 2:
        actions.append(Action(REPOSITION))

    # Freeze — free, needs a shop.
    if shop:
        actions.append(Action(FREEZE))

    # Press a dark gift — free, timing is the decision (spec req 10). The
    # Director weighs WHEN; we only surface that the button exists.
    for g in (_get(snapshot, "dark_gifts", []) or []):
        actions.append(Action(DARK_GIFT, g.get("name"), 0, {"gift": g}))

    actions.append(Action(END))
    return actions
