"""Anomaly awareness — let the active Battlegrounds anomaly change what to buy.

Anomalies reshape a lobby (e.g. Timewarped supercharges certain minions' tavern
effects). The recommender must factor the live anomaly in, not ignore it.

Grounded signals (from real Power.logs) come first; name-based guidance notes are
a fallback for anomalies we can describe but don't yet have a mechanical hook for.
Add new anomalies here — keep each entry small and cite the log signal.
"""

from typing import Optional, Tuple

# Tag the client sets on a minion whose tavern text is rewritten (supercharged) by
# the Timewarped anomaly — i.e. exactly the minions you want to buy under it.
_TIMEWARP_TAG = "HAS_TIMEWARPED_TAVERN_ALT_TEXT"


def timewarped_boost(tags: dict) -> Tuple[float, Optional[str]]:
    """(placement_adjustment, reason) for buying a shop minion under Timewarped.
    Negative = promote. Fires only on minions the anomaly actually buffs."""
    if tags and str(tags.get(_TIMEWARP_TAG, "")) == "1":
        return (-0.8, "Timewarped — the anomaly supercharges this minion's effect; "
                      "prioritize buying it")
    return 0.0, None


def buy_adjust(minion_tags: dict) -> Tuple[float, Optional[str]]:
    """Anomaly-driven adjustment for buying this shop minion (tag-grounded)."""
    return timewarped_boost(minion_tags or {})


# Name-keyed strategic notes for anomalies we can describe but don't yet hook
# mechanically. Shown as guidance; extend as we confirm each from real logs.
_NOTES = {
    "timewarped": "Timewarped: buy minions whose tavern text is rewritten — their "
                  "effect is doubled-up. They're the lobby's best value.",
    "the golden arena": "Golden Arena: every minion is golden — value triples; "
                        "prioritize tripling and tier-relevant bodies.",
    "secrets of norgannon": "Secrets of Norgannon: you start higher-tier — push "
                            "tempo and spend, the economy is loosened.",
}


def anomaly_note(name: Optional[str]) -> Optional[str]:
    """A one-line strategic note for the active anomaly, or None if we don't have
    curated guidance for it (the overlay still shows the anomaly name)."""
    if not name:
        return None
    key = name.strip().lower()
    if key in _NOTES:
        return _NOTES[key]
    for k, note in _NOTES.items():            # loose match (e.g. 'Timewarped Minions')
        if k in key or key in k:
            return note
    return None
