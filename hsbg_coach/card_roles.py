"""Card roles the board-value model doesn't capture on its own.

The eval net scores a minion mostly by stats + keywords, so *situational tech*
cards — whose value is entirely "they answer a specific enemy board" — get rated
as if their effect always applies. Tunnel Blaster (deathrattle: 3 to all) is only
strong into wide/low-health boards; Deadly Spore (Venomous) only trades up into a
single big threat. Bought blindly they're mediocre stat-sticks.

We don't have full opponent context at buy time, so we apply a modest discount:
enough to stop tech cards from being the *default* top recommendation, not enough
to bury them when nothing else is better. The reason string tells the player it's
a read-dependent pick, which is the actual nuance.

Keyed by cardId (stable) with name fallbacks. Extend as more get flagged.
"""

from typing import Optional

# cardId -> short note on when the card is actually good.
_TECH = {
    "BG_DAL_775": "board-clear tech — best into wide, low-health enemy boards",
    "BGS_131": "Venomous tech — best as a one-shot answer to a single big threat",
}
# name fallbacks (cardId is preferred; names cover re-pooled/renamed ids).
_TECH_NAMES = {
    "Tunnel Blaster": _TECH["BG_DAL_775"],
    "Deadly Spore": _TECH["BGS_131"],
}

# How much to shave off a tech card's buy priority/equity-delta. Tuned to demote
# from "the pick" to "an option", not to forbid.
TECH_BUY_DISCOUNT = 0.18


def tech_note(card_id: Optional[str], name: Optional[str]) -> Optional[str]:
    """Return the situational note if this card is read-dependent tech, else None."""
    if card_id and card_id in _TECH:
        return _TECH[card_id]
    if name and name in _TECH_NAMES:
        return _TECH_NAMES[name]
    return None


def is_tech(card_id: Optional[str], name: Optional[str]) -> bool:
    return tech_note(card_id, name) is not None
