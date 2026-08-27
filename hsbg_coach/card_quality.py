"""Meta card quality — which minions/spells are actually good, from real top-MMR
stats (data/stats/firestone_card_stats.json: averagePlacement per card, lower is
better).

The eval net judges a board's composition, but it under-rates individual standout
cards, so the coach would roll past genuinely strong minions. This gives every
shop card a placement nudge from how it actually performs in high-MMR lobbies, so
good cards get bought and rolling only wins when the shop is truly weak.
"""

import json
import os
from functools import lru_cache
from typing import Optional, Tuple

_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stats",
                     "firestone_card_stats.json")
# Median averagePlacement across the dataset — the "neutral" card. Below = strong.
_NEUTRAL = 3.4
_STRONG = 3.05            # clearly a top pick
_WEAK = 3.75             # below-average for this stage


@lru_cache(maxsize=1)
def _index():
    by_id, by_name = {}, {}
    try:
        cards = json.load(open(_PATH, encoding="utf-8")).get("cards", [])
    except Exception:
        return by_id, by_name
    for c in cards:
        ap = c.get("averagePlacement")
        if ap is None:
            continue
        if c.get("cardId"):
            by_id[c["cardId"]] = ap
        if c.get("name"):
            by_name[c["name"].lower()] = ap
    return by_id, by_name


def placement(card_id: Optional[str] = None, name: Optional[str] = None) -> Optional[float]:
    """Real top-MMR average placement for a card (lower = better), or None.
    Name lookups go through the blended Firestone+HSReplay prior
    (card_meta_stats — HSReplay weighted higher); card-id lookups fall back
    to the Firestone-only index (the only source keyed by id)."""
    if name:
        try:
            from .card_meta_stats import prior
            p = prior(name)
            if p is not None:
                return p[0]
        except Exception:
            pass
    by_id, by_name = _index()
    if card_id and card_id in by_id:
        return by_id[card_id]
    if name and name.lower() in by_name:
        return by_name[name.lower()]
    return None


def buy_adjust(card_id: Optional[str], name: Optional[str]
               ) -> Tuple[float, Optional[str], bool]:
    """(placement_adjustment, reason, is_strong) for buying this card, from its meta
    performance. Negative = promote. is_strong flags a top pick (exempt from the
    filler/off-comp penalties — a known-great card is worth buying)."""
    ap = placement(card_id, name)
    if ap is None:
        return 0.0, None, False
    # Weighted heavily: a card's real meta placement reflects how good its EFFECT
    # is (not its stat line), and that's what should drive buys.
    adj = max(-1.3, min(0.5, (ap - _NEUTRAL) * 1.0))
    if ap <= _STRONG:
        return adj, f"strong meta pick (avg {ap:.1f}) — buy it, don't roll past it", True
    if ap >= _WEAK:
        return adj, None, False
    return adj, None, False
