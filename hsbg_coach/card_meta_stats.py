"""Blended per-card population priors — the anti-survivorship-bias signal.

card2vec (and the final-board corpora it trains on) only see boards that made
it to the end of a game, so cards that *look* common on winning boards get
over-credited and cards that quietly lose games leave no trace. The population
card stats are computed across ALL games — every time a card was played,
whatever the placement — so feeding them to the eval net as explicit features
counteracts that bias.

Two sources, same schema (``{"cards": [{name, cardId, averagePlacement,
impact, ...}]}``), blended per card when both know it:

  * ``data/stats/firestone_card_stats.json`` — auto-refreshed from Firestone's
    public CDN (``hsbg_coach refresh-stats``); always available.
  * ``data/stats/hsreplay_card_stats.json``  — optional, imported from an
    HSReplay.net minions-page export via ``hsbg_coach import-hsreplay``
    (HSReplay has no public API and blocks non-browser traffic, so this file
    only exists if you export it from your own logged-in browser session).

``averagePlacement``: mean final placement across all games where the card was
played (lower = better; ~3.4 is neutral). ``impact``: placement-without minus
placement-with (positive = you place better when you have it).
"""

import json
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

_STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stats")
# Source blend weights. HSReplay outweighs Firestone by design: its BG data
# comes from the larger HDT-uploading population and (via the capture tool)
# is filtered to top-10% lobbies, so where the two disagree we lean HSReplay.
SOURCES = {"firestone_card_stats.json": 1.0, "hsreplay_card_stats.json": 2.0}

# Neutral averagePlacement (kept in sync with card_quality._NEUTRAL).
NEUTRAL_PLACEMENT = 3.4


def _load(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("cards", [])
    except (OSError, ValueError):
        return []


@lru_cache(maxsize=1)
def _index() -> Dict[str, Tuple[float, float]]:
    """lowercase name -> (averagePlacement, impact), weighted-mean-blended
    across the sources that know the card (weights in SOURCES). Missing
    impact contributes 0."""
    acc: Dict[str, list] = {}
    for fname, w in SOURCES.items():
        for c in _load(os.path.join(_STATS_DIR, fname)):
            name = (c.get("name") or "").lower()
            ap = c.get("averagePlacement")
            if not name or ap is None:
                continue
            acc.setdefault(name, []).append(
                (w, float(ap), float(c.get("impact") or 0.0)))
    return {name: (sum(w * a for w, a, _ in rows) / sum(w for w, _, _ in rows),
                   sum(w * i for w, _, i in rows) / sum(w for w, _, _ in rows))
            for name, rows in acc.items()}


def prior(name: str) -> Optional[Tuple[float, float]]:
    """(averagePlacement, impact) for a card name, or None if no source has it.
    Golden minions share their base card's stats."""
    idx = _index()
    key = (name or "").lower()
    hit = idx.get(key)
    if hit is None and key.startswith("golden "):
        hit = idx.get(key[len("golden "):])
    return hit


def reload() -> None:
    """Drop the cache (after refresh-stats / import-hsreplay)."""
    _index.cache_clear()
