"""Enumerate every legal action at a Battlegrounds decision point.

The recommender's job is to rank "what should I do now?", so first we need the
full menu of what's *possible*: buy each shop minion, sell each board minion,
roll, tier up, reposition, freeze, activate board minions, end. This module
knows the rules (costs, board cap, tier cap) and nothing about which action is
good — that's the advisor.

Stdlib only.
"""

import re
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
# Semantic subtype for Season-14 clickable minions. The current engine's generic
# activatable/economy scorer is the HERO_POWER path, so legal_actions routes these
# through that path while preserving this subtype in Action.detail. That makes the
# existing advisor + whole-game fallback score it immediately without pretending
# it is a buy/sell, while the description/log remains explicitly "Activate X".
MINION_ACTIVATE = "minion_activate"
END = "end"
DARK_GIFT = "dark_gift"


def tavern_up_cost(tier: Optional[int]) -> Optional[int]:
    return UPGRADE_COST.get(tier or 0)


@dataclass
class Action:
    kind: str
    target: Optional[str] = None       # card name for buy/sell/activate
    cost: int = 0                      # gold spent (negative = gold gained)
    detail: Dict = field(default_factory=dict)

    @property
    def semantic_kind(self) -> str:
        return self.detail.get("semantic_kind") or self.kind

    def describe(self) -> str:
        if self.semantic_kind == MINION_ACTIVATE:
            tail = f" ({self.cost}g)" if self.cost else ""
            return f"Activate {self.target}{tail}"
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


def _card_id(m) -> Optional[str]:
    return (m.get("card_id") if isinstance(m, dict)
            else getattr(m, "card_id", None))


def _tags(m) -> Dict:
    tags = m.get("tags", {}) if isinstance(m, dict) else getattr(m, "tags", {})
    return tags or {}


def _kb_card(m, kb):
    """Resolve live minion -> CardKnowledge without requiring a name index."""
    if not kb:
        return None
    cid = _card_id(m)
    if cid and cid in kb:
        return kb[cid]
    nm = _name(m)
    for ck in kb.values():
        if getattr(ck, "name", None) == nm:
            return ck
    return None


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _activate_info(m, kb=None) -> Optional[Dict]:
    """Return grounded Season-14 Activate metadata for one board minion.

    Blizzard's Activate minions are INTERACTABLE_OBJECTs. Their live gold price
    is INTERACTABLE_OBJECT_COST and the BG-specific tooltip marker is
    BACON_ACTIVATE_TOOLTIP. We prefer those Power.log tags, with card text as a
    fallback so a refreshed HearthstoneJSON KB still works if a client omits one
    static tag. EXHAUSTED is the standard live once-per-turn gate; possible
    explicit used-state tags are honored too if Blizzard emits them.
    """
    tags = _tags(m)
    ck = _kb_card(m, kb)
    text = getattr(ck, "text", "") if ck is not None else ""
    plain = re.sub(r"<[^>]+>", "", text or "")

    marker = (
        str(tags.get("INTERACTABLE_OBJECT", "")) == "1"
        or str(tags.get("BACON_ACTIVATE_TOOLTIP", "")) == "1"
        or bool(re.search(r"\bActivate\s*\(", plain, re.IGNORECASE))
    )
    if not marker:
        return None

    # If the live entity explicitly says interaction is disabled, trust it.
    if "INTERACTABLE_OBJECT" in tags and str(tags.get("INTERACTABLE_OBJECT")) == "0":
        return None
    used_tags = (
        "EXHAUSTED", "INTERACTABLE_OBJECT_EXHAUSTED",
        "INTERACTABLE_OBJECT_USED", "INTERACTABLE_OBJECT_USED_THIS_TURN",
    )
    if any(str(tags.get(tag, "0")) == "1" for tag in used_tags):
        return None

    cost = _as_int(tags.get("INTERACTABLE_OBJECT_COST"))
    if cost is None:
        match = re.search(r"\bActivate\s*\((\d+)\)", plain, re.IGNORECASE)
        cost = int(match.group(1)) if match else 0

    return {
        "name": _name(m),
        "card_id": _card_id(m),
        "entity_id": (m.get("entity_id") if isinstance(m, dict)
                      else getattr(m, "entity_id", None)),
        "cost": cost,
        "text": plain.strip(),
        "tags": dict(tags),
    }


def legal_actions(snapshot, kb=None) -> List[Action]:
    """Every action that is legal given current gold, tier, board and shop."""
    gold = _get(snapshot, "gold")
    tier = _get(snapshot, "tavern_tier") or 1
    board = list(_get(snapshot, "board", []) or [])
    shop = list(_get(snapshot, "shop", []) or [])
    phase = str(_get(snapshot, "phase", "") or "").lower()
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

    # Season 14 Activate — a board minion is a clickable Recruit-phase action,
    # with its own gold cost and once-per-turn availability. Route through the
    # already-live activatable scorer but preserve semantic_kind + effect details
    # so the Director/log knows this is a MINION activation, never a hero power.
    if phase in ("", "recruit"):
        for m in board:
            info = _activate_info(m, kb)
            if info is not None and gold >= int(info["cost"] or 0):
                actions.append(Action(
                    HERO_POWER, info["name"], int(info["cost"] or 0),
                    {"semantic_kind": MINION_ACTIVATE,
                     "activate": info, "minion": m},
                ))

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