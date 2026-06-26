"""Concrete advice for Choose-One battlecry minions.

A Choose-One minion (e.g. Intrepid Botanist) makes you pick one of two halves
when you play it. 'Pick the half that fits' is useless — the coach should name
the half. This is a small, curated table keyed by card id / name; each entry
gives a default pick with reasoning, and an optional board-state override (e.g.
take the defensive half when you're low). Unknown Choose-One minions fall back to
the generic hint, so this only ever adds specificity, never blocks.
"""

from typing import Optional

_LOW_HP = 15

# card_id -> advice. `low` fires when you're at low HP (survival > scaling).
_TABLE = {
    # Intrepid Botanist (BG32_237): +1 Attack (Pristine Lilies) vs +1 Health
    # (Giant Dewdrop) to your Tavern spells this game.
    "BG32_237": {
        "default": "Choose One → take +Attack (Pristine Lilies): attack on every "
                   "tavern spell trades up and pushes damage",
        "low": "Choose One → take +Health (Giant Dewdrop): you're low, the extra "
               "health keeps your buffed minions alive",
    },
}
_BY_NAME = {"Intrepid Botanist": "BG32_237"}


def choose_one_advice(minion, snapshot=None) -> Optional[str]:
    """Concrete 'take the +Attack/+Health half' line for a known Choose-One minion,
    or None if we don't have a curated pick for it (caller uses a generic hint)."""
    cid = minion.get("card_id") if isinstance(minion, dict) else getattr(minion, "card_id", None)
    name = minion.get("name") if isinstance(minion, dict) else getattr(minion, "name", None)
    entry = _TABLE.get(cid) or _TABLE.get(_BY_NAME.get(name or "", ""))
    if not entry:
        return None
    hp = (snapshot or {}).get("hero_health") if isinstance(snapshot, dict) else None
    if hp is not None and hp < _LOW_HP and entry.get("low"):
        return entry["low"]
    return entry["default"]
