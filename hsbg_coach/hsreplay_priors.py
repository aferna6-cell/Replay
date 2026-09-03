"""Accessors for the imported HSReplay stats beyond per-card placement.

card_meta_stats handles the minion-placement blend. This module serves the
rest of the capture — per-turn purchase rates, the leveling curve, hero and
trinket placements — to the advisor layer. Everything degrades to "no data"
silently when the files haven't been imported yet.

Files (written by ``hsbg_coach import-hsreplay``):
  hsreplay_purchase_rates_by_turn_stats.json  what top players buy each turn
  hsreplay_tavern_up_stats.json               pct of lobbies at tier per round
  hsreplay_heroes_stats.json                  hero avg placement (adjusted)
  hsreplay_trinkets_stats.json                trinket avg placement + pick rate
"""

import json
import os
from typing import Dict, List, Optional, Tuple

_STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stats")
# Weight of HSReplay vs Firestone when blending (matches card_meta_stats).
HSREPLAY_WEIGHT = 2.0

_cache: Dict[str, Optional[dict]] = {}


def _load(fname: str) -> Optional[dict]:
    if fname not in _cache:
        try:
            with open(os.path.join(_STATS_DIR, fname), encoding="utf-8") as fh:
                _cache[fname] = json.load(fh)
        except (OSError, ValueError):
            _cache[fname] = None
    return _cache[fname]


def reload() -> None:
    _cache.clear()


def blend(firestone: Optional[float], hsreplay: Optional[float],
          w: float = HSREPLAY_WEIGHT) -> Optional[float]:
    """Weighted mean of the two sources; whichever exists when only one does."""
    if firestone is None:
        return hsreplay
    if hsreplay is None:
        return firestone
    return (firestone + w * hsreplay) / (1.0 + w)


# --- per-turn purchase rates ----------------------------------------------

def pick_rate(name: str, turn: Optional[int]) -> Optional[float]:
    """How often top players buy this card when offered on this turn."""
    data = _load("hsreplay_purchase_rates_by_turn_stats.json")
    if not data or turn is None:
        return None
    for row in data.get("turns", {}).get(str(turn), []):
        if row.get("name", "").lower() == (name or "").lower():
            return row.get("pickRate")
    return None


def turn_buy_adjust(name: str, turn: Optional[int]
                    ) -> Tuple[float, Optional[str]]:
    """(placement_adjustment, reason) from real turn-specific buy behavior.
    A tiebreaker on top of the placement-based quality nudge, never the
    driver — pick rates fold in positioning/comp context we can't see."""
    pr = pick_rate(name, turn)
    if pr is None:
        return 0.0, None
    if pr >= 0.35:
        return -0.25, (f"top players take this {pr:.0%} of the time "
                       f"it's offered on turn {turn}")
    if pr <= 0.02:
        return 0.15, None          # near-universally skipped at this stage
    return 0.0, None


# --- leveling curve --------------------------------------------------------

def tavern_tier_curve() -> Dict[int, float]:
    """Expected tavern tier per recruit round from the real distribution."""
    data = _load("hsreplay_tavern_up_stats.json")
    out: Dict[int, float] = {}
    for rnd, tiers in (data or {}).get("rounds", {}).items():
        total = expected = 0.0
        for tier, cell in tiers.items():
            pct = cell.get("pctAtTier") or 0.0
            total += pct
            expected += int(tier) * pct
        if total > 0:
            out[int(rnd)] = round(expected / total, 3)
    return out


# --- hero / trinket placement priors --------------------------------------

def _placement_index(fname: str, prefer: List[str]) -> Dict[str, float]:
    data = _load(fname)
    idx: Dict[str, float] = {}
    for item in (data or {}).get("items", []):
        name = (item.get("name") or "").lower()
        for key in prefer:
            ap = item.get(key)
            if name and isinstance(ap, (int, float)) and 1.0 <= ap <= 8.0:
                idx[name] = float(ap)
                break
    return idx


def hero_prior(name: str) -> Optional[float]:
    """Hero average placement (anomaly-adjusted when available)."""
    return _placement_index(
        "hsreplay_heroes_stats.json",
        ["adjustedAveragePlacement", "averagePlacement"]).get(
            (name or "").lower())


def trinket_prior(name: str) -> Optional[float]:
    return _placement_index("hsreplay_trinkets_stats.json",
                            ["averagePlacement"]).get((name or "").lower())


def trinket_items() -> List[dict]:
    """Full trinket rows (for entries Firestone doesn't know)."""
    data = _load("hsreplay_trinkets_stats.json")
    return (data or {}).get("items", [])
