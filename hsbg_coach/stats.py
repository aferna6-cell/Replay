"""Hero & comp population stats — priors that teach the model how to play.

These are the population-data half of the design (specs §5): "what's objectively
strong" — which heroes want which comps, which comps place best, what their core
minions and power turns are. They feed the recommender *now* (via ``HeroContext``)
and are the prior features the eventual ML learns deltas from. Your own games (the
trajectory recorder) provide the personalization half.

Source is configurable. The data ships as a small sample so everything works
offline; point ``load_*`` at a real export (Firestone's bgs-global-stats /
firestone-data, or an HSReplay export) or a live URL once a fetchable endpoint is
confirmed. We deliberately do NOT hardcode an undocumented vendor URL — see ADR
``decisions/2026-06-24-firestone-bridge.md`` for the source decision.

Normalized schema (vendor-agnostic), JSON:

  hero_stats.json   {"heroes": [
      {"cardId","name","averagePosition","pickRate","bestTribes":[...],
       "playstyle":"economy"|"tempo"|"flexible"}]}
  comp_stats.json   {"comps": [
      {"name","tribe","averagePosition","popularity","coreCards":[...],
       "powerTurns":[int,...],"tier":"S".."D"}]}
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .economy import HeroContext

_STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stats")
SAMPLE_HEROES = os.path.join(_STATS_DIR, "sample_hero_stats.json")
SAMPLE_COMPS = os.path.join(_STATS_DIR, "sample_comp_stats.json")
# Real Firestone snapshot (written by `refresh-stats`); preferred when present.
FIRESTONE_HEROES = os.path.join(_STATS_DIR, "firestone_hero_stats.json")
FIRESTONE_COMPS = os.path.join(_STATS_DIR, "firestone_comp_stats.json")
FIRESTONE_TRINKETS = os.path.join(_STATS_DIR, "firestone_trinket_stats.json")
FIRESTONE_BOARDS = os.path.join(_STATS_DIR, "firestone_final_boards.json")


def default_hero_source() -> str:
    return FIRESTONE_HEROES if os.path.isfile(FIRESTONE_HEROES) else SAMPLE_HEROES


def default_comp_source() -> str:
    return FIRESTONE_COMPS if os.path.isfile(FIRESTONE_COMPS) else SAMPLE_COMPS


def default_trinket_source() -> Optional[str]:
    return FIRESTONE_TRINKETS if os.path.isfile(FIRESTONE_TRINKETS) else None


@dataclass
class HeroStats:
    card_id: str
    name: str
    average_position: float = 4.5
    pick_rate: float = 0.0
    best_tribes: List[str] = field(default_factory=list)
    playstyle: str = "flexible"          # economy | tempo | flexible


@dataclass
class CompStats:
    name: str
    tribe: Optional[str]
    average_position: float = 4.5
    popularity: float = 0.0
    core_cards: List[str] = field(default_factory=list)
    power_turns: List[int] = field(default_factory=list)
    tier: str = "?"


@dataclass
class TrinketStats:
    name: str
    card_id: str
    average_position: float = 4.5
    pick_rate: float = 0.0
    tier: str = "?"


def _read_json(source: str) -> dict:
    """Load JSON from a local path or an http(s) URL (urllib honors proxy env)."""
    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(source, timeout=20) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    with open(source, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_hero_stats(source: str = SAMPLE_HEROES) -> List[HeroStats]:
    data = _read_json(source)
    return [
        HeroStats(
            card_id=h.get("cardId", ""),
            name=h.get("name", ""),
            average_position=float(h.get("averagePosition", 4.5)),
            pick_rate=float(h.get("pickRate", 0.0)),
            best_tribes=list(h.get("bestTribes", [])),
            playstyle=h.get("playstyle", "flexible"),
        )
        for h in data.get("heroes", [])
    ]


def load_comp_stats(source: str = SAMPLE_COMPS) -> List[CompStats]:
    data = _read_json(source)
    return [
        CompStats(
            name=c.get("name", ""),
            tribe=c.get("tribe"),
            average_position=float(c.get("averagePosition", 4.5)),
            popularity=float(c.get("popularity", 0.0)),
            core_cards=list(c.get("coreCards", [])),
            power_turns=list(c.get("powerTurns", [])),
            tier=c.get("tier", "?"),
        )
        for c in data.get("comps", [])
    ]


def load_trinket_stats(source: Optional[str]) -> List[TrinketStats]:
    if not source:
        return []
    data = _read_json(source)
    return [
        TrinketStats(
            name=t.get("name", ""),
            card_id=t.get("cardId", ""),
            average_position=float(t.get("averagePosition", 4.5)),
            pick_rate=float(t.get("pickRate", 0.0)),
            tier=t.get("tier", "?"),
        )
        for t in data.get("trinkets", [])
    ]


class StatsDB:
    """Loaded hero + comp stats with lookups for advice."""

    def __init__(self, heroes: List[HeroStats], comps: List[CompStats],
                 trinkets: Optional[List[TrinketStats]] = None) -> None:
        self.heroes = heroes
        self.comps = comps
        self.trinkets = trinkets or []

    @classmethod
    def load(cls, hero_source: Optional[str] = None,
             comp_source: Optional[str] = None,
             trinket_source: Optional[str] = None) -> "StatsDB":
        """Load stats. Defaults to the real Firestone snapshot if present, else
        the bundled sample data."""
        return cls(
            load_hero_stats(hero_source or default_hero_source()),
            load_comp_stats(comp_source or default_comp_source()),
            load_trinket_stats(trinket_source or default_trinket_source()),
        )

    def best_trinkets(self, limit: int = 5) -> List[TrinketStats]:
        return sorted(self.trinkets, key=lambda t: t.average_position)[:limit]

    def hero(self, name_or_id: str) -> Optional[HeroStats]:
        key = (name_or_id or "").lower()
        for h in self.heroes:
            if h.name.lower() == key or h.card_id.lower() == key:
                return h
        return None

    def best_comps(self, tribes: Optional[List[str]] = None,
                   limit: int = 3) -> List[CompStats]:
        """Comps ranked by average placement (lower is better), optionally
        restricted to tribes available this lobby."""
        pool = self.comps
        if tribes is not None:
            allowed = {t.lower() for t in tribes}
            pool = [c for c in pool if c.tribe is None or c.tribe.lower() in allowed]
        return sorted(pool, key=lambda c: c.average_position)[:limit]

    def best_comp_for_hero(self, hero: str,
                           available_tribes: Optional[List[str]] = None
                           ) -> Optional[CompStats]:
        """The strongest comp for a hero: prefers the hero's best tribes, falls
        back to the overall best available comp."""
        h = self.hero(hero)
        prefer = [t.lower() for t in (h.best_tribes if h else [])]
        ranked = self.best_comps(available_tribes, limit=len(self.comps) or 1)
        if not ranked:
            return None
        on_hero = [c for c in ranked if c.tribe and c.tribe.lower() in prefer]
        return (on_hero or ranked)[0]


_PLAYSTYLE_AGGRESSION = {"economy": 0.2, "greedy": 0.25, "tempo": -0.15,
                         "aggressive": -0.2, "flexible": 0.0}


def build_hero_context(hero: str, db: StatsDB,
                       available_tribes: Optional[List[str]] = None) -> HeroContext:
    """Turn population stats into a HeroContext the recommender consumes.

    Picks the statistically best comp for this hero (given the lobby's tribes),
    targets its tribe, recommends its core minions, and biases leveling by the
    hero's playstyle. This is how the advice becomes hero/comp-specific.
    """
    h = db.hero(hero)
    comp = db.best_comp_for_hero(hero, available_tribes)
    aggression = _PLAYSTYLE_AGGRESSION.get(h.playstyle if h else "flexible", 0.0)
    return HeroContext(
        hero=h.name if h else hero,
        target_tribe=comp.tribe if comp else None,
        recommended_minions=list(comp.core_cards) if comp else [],
        level_aggression=aggression,
    )


def load_final_boards(source: Optional[str] = None) -> List[dict]:
    """Real winning-board sets per comp (core-card frequencies + example boards).
    The richest population signal — 'what the winning board looks like.'"""
    src = source or (FIRESTONE_BOARDS if os.path.isfile(FIRESTONE_BOARDS) else None)
    if not src:
        return []
    return _read_json(src).get("boards", [])


def example_boards(archetype: str, source: Optional[str] = None) -> List[dict]:
    """Example winning boards (minions + positions + keywords) for a comp."""
    for b in load_final_boards(source):
        if b.get("archetype") == archetype:
            return b.get("examples", [])
    return []
