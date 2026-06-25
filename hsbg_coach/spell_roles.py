"""Tavern (Battlegrounds) spell knowledge + value.

Tavern spells sit in the shop next to minions and cost a variable amount of gold
(a COST tag, unlike minions' flat 3). To recommend one we need to know what it
does — our minion card KB doesn't carry spells — so this module holds a small,
extensible table keyed by cardId, plus a generic fallback so an unknown spell is
still surfaced (named + costed) rather than ignored.

Value is expressed as a *placement adjustment* (negative = better finish) so it
slots straight into the whole-game ranker, same as minions/tech. Known strong
effects (gold, triples, free minions) get a real bonus; unknown spells get a
small, honest "it's an option — read the effect" nudge scaled by how cheap and
affordable they are.

Seed the table as real cardIds are confirmed from live logs (entityName is in the
Power.log, so names resolve even before an effect is curated here).
"""

from typing import Dict, Optional, Tuple

# cardId -> (placement_bonus, note). Bonus is negative = recommend more.
# Seeded conservatively; extend as spells are confirmed from real games.
_KNOWN: Dict[str, Tuple[float, str]] = {
    # "Pointy Arrow" token seen in a real log — minor combat trick, situational.
    "EBG_Spell_014": (-0.1, "tavern spell — small combat trick"),
}

# Generic value when we don't know the spell yet: a cheap, affordable tavern spell
# is usually fine tempo, but we won't over-rank an unknown effect.
_GENERIC_BONUS = -0.15
_GENERIC_NOTE = "tavern spell — usually worth it if cheap; check the effect"


def spell_value(card_id: Optional[str], name: Optional[str], cost: int,
                gold: int) -> Tuple[float, str]:
    """(placement_adjustment, reason) for buying a tavern spell.

    Negative adjustment = better finish. Unknown spells get a modest, affordability
    -scaled nudge so they surface as a real option without pretending to know the
    effect."""
    if card_id and card_id in _KNOWN:
        return _KNOWN[card_id]
    # Unknown: only nudge it up if you can comfortably afford it (cheap relative to
    # your gold), else it's neutral so it doesn't crowd out a real board buy.
    if cost <= 2 and gold >= cost:
        return _GENERIC_BONUS, _GENERIC_NOTE
    return 0.0, _GENERIC_NOTE


def spell_name(card_id: Optional[str], name: Optional[str]) -> str:
    return name or card_id or "Tavern spell"
