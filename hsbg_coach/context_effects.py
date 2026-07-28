"""Context effects — turn hero-power / trinket / anomaly text into env knobs.

The recruit-phase simulator already tracks gold, tavern tier, shop contents, roll
and upgrade costs, health and copy counts. A large share of what anomalies, hero
powers and trinkets actually *do* in the recruit phase is move exactly those
knobs: "Start at 10 Gold", "Minions cost 2 Gold", "Only Tavern Tiers 2, 4 and 6
exist", "You only need 2 copies to make minions Golden".

So we parse the rules text into a small typed hook set the env can execute
directly. Everything that needs machinery the env doesn't have — Discover, a
second hero power, Buddies, Quests, start-of-combat effects — is reported as
UNSUPPORTED rather than silently approximated, and reaches the model as a text
feature instead of as dynamics.

`coverage()` is the honest scoreboard: how much of the real card set the env can
mechanically execute today.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .context_cards import ContextCard

# --- hook names -------------------------------------------------------------
# Each maps onto state the recruit env already owns (or a one-field addition).
START_GOLD = "start_gold"                  # int: opening gold
START_TIER = "start_tier"                  # int: opening tavern tier
START_HEALTH = "start_health"              # int: opening hero health
START_ARMOR = "start_armor"                # int: extra armor
MINION_COST = "minion_cost"                # int: gold per shop minion
REFRESH_COST = "refresh_cost"              # int: gold per roll
REFRESH_DISABLED = "refresh_disabled"      # flag: rolling unavailable
UPGRADE_COST_DELTA = "upgrade_cost_delta"  # int: +/- on tavern-up cost
FREE_BUYS_PER_TURN = "free_buys_per_turn"  # int: first N buys cost 0
EXTRA_COPY_ON_BUY = "extra_copy_on_buy"    # int: bonus copies on first buy
ALLOWED_TIERS = "allowed_tiers"            # list[int]: tiers that exist
SHOP_TIER_ONLY = "shop_tier_only"          # flag: shop offers your tier only
SHOP_SLOTS_DELTA = "shop_slots_delta"      # int: +/- shop size
TRIPLE_COPIES = "triple_copies"            # int: copies needed to golden
GOLD_PER_TURN = "gold_per_turn"            # int: bonus gold each turn
END_OF_TURN_BUFF = "end_of_turn_buff"      # (atk, hp) applied to the board

UNSUPPORTED = "unsupported"

# Mechanics the recruit env has no machinery for. Listed so an unparsed card is
# classified as "needs feature X", not as an anonymous miss.
_NEEDS = [
    (re.compile(r"\bdiscover\b", re.I), "discover"),
    (re.compile(r"second hero power", re.I), "second_hero_power"),
    (re.compile(r"\bbudd(y|ies)\b", re.I), "buddies"),
    (re.compile(r"\bquest\b", re.I), "quests discover"),
    (re.compile(r"start of combat", re.I), "combat"),
    (re.compile(r"\bdeathrattle|battlecry\b", re.I), "minion effects"),
]


@dataclass
class Hook:
    name: str
    value: object = None
    note: str = ""


@dataclass
class ContextEffect:
    card_id: str
    name: str
    kind: str
    hooks: List[Hook] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return bool(self.hooks)


_NUM = r"(\d+)"


def _first(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, re.I)
    return int(m.group(1)) if m else None


def parse_text(text: str) -> List[Hook]:
    """Rules text -> the env knobs it moves. Conservative: no match, no hook."""
    hooks: List[Hook] = []
    t = text or ""

    def add(name: str, value: object) -> None:
        hooks.append(Hook(name, value))

    v = _first(rf"start (?:at|with) {_NUM} gold", t)
    if v is not None:
        add(START_GOLD, v)
    v = _first(rf"start at tavern tier {_NUM}", t)
    if v is not None:
        add(START_TIER, v)
    v = _first(rf"start at {_NUM} health", t)
    if v is not None:
        add(START_HEALTH, v)
    v = _first(rf"start with {_NUM} extra armor", t)
    if v is not None:
        add(START_ARMOR, v)
    v = _first(rf"minions cost {_NUM} gold", t)
    if v is not None:
        add(MINION_COST, v)
    v = _first(rf"refresh(?:ing)? (?:the tavern )?costs? {_NUM}", t)
    if v is not None:
        add(REFRESH_COST, v)
    if re.search(r"cannot refresh", t, re.I):
        add(REFRESH_DISABLED, True)
    if re.search(r"first minion you buy each turn is free", t, re.I):
        add(FREE_BUYS_PER_TURN, 1)
    if re.search(r"first time you buy a card each turn, get an extra copy", t, re.I):
        add(EXTRA_COPY_ON_BUY, 1)
    if re.search(r"tavern only offers cards of your tier", t, re.I):
        add(SHOP_TIER_ONLY, True)
    v = _first(rf"only need {_NUM} copies", t)
    if v is not None:
        add(TRIPLE_COPIES, v)

    m = re.search(r"only tavern tiers ([\d,\s and]+) exist", t, re.I)
    if m:
        tiers = [int(n) for n in re.findall(r"\d", m.group(1))]
        if tiers:
            add(ALLOWED_TIERS, tiers)

    m = re.search(rf"at the (?:start|end) of (?:each|your) turn.*?\+{_NUM}/\+{_NUM}",
                  t, re.I)
    if m:
        add(END_OF_TURN_BUFF, (int(m.group(1)), int(m.group(2))))

    m = re.search(rf"(?:gain|get) {_NUM} (?:extra )?gold", t, re.I)
    if m and re.search(r"each turn|every turn", t, re.I):
        add(GOLD_PER_TURN, int(m.group(1)))

    return hooks


def needs_for(text: str) -> List[str]:
    """Which missing machinery this text depends on (for unsupported cards)."""
    return sorted({label for pat, label in _NEEDS if pat.search(text or "")})


def classify(card: ContextCard) -> ContextEffect:
    hooks = parse_text(card.text)
    return ContextEffect(card_id=card.card_id, name=card.name, kind=card.kind,
                         hooks=hooks, needs=[] if hooks else needs_for(card.text))


def coverage(cards: Sequence[ContextCard]) -> Dict[str, dict]:
    """Per-kind mechanical coverage: what the env can execute vs. what it can't."""
    out: Dict[str, dict] = {}
    for c in cards:
        eff = classify(c)
        row = out.setdefault(c.kind, {"total": 0, "supported": 0, "needs": {}})
        row["total"] += 1
        if eff.supported:
            row["supported"] += 1
        else:
            for n in eff.needs or ["other"]:
                row["needs"][n] = row["needs"].get(n, 0) + 1
    for row in out.values():
        row["pct"] = round(100.0 * row["supported"] / max(1, row["total"]), 1)
    return out
