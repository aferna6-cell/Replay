"""Lobby setup — 8 heroes and one anomaly, the way a real game starts.

A Battlegrounds game does not begin with eight identical seats. Eight heroes get
dealt, one anomaly is rolled for the whole lobby, and everyone shops out of the
same finite pool for the rest of the night. Training without that means training
on a game nobody plays.

This module rolls that opening and hands the env the knobs it needs:

  * **Heroes** — sampled from the *active* pool (`context_cards`), which is the
    114 heroes live play data has actually measured, not every hero ever printed.
  * **Hero powers** — resolved from each hero via `heroPowerDbfId`.
  * **Anomaly** — one for the lobby, from the 29 in rotation.
  * **Start hooks** — the anomaly's parsed effect ("Start at 10 Gold", "Start at
    Tavern Tier 2") as env knobs, via `context_effects`.

Appearance rates are uniform here. Real BG weights some anomalies more than
others, but no public per-anomaly rate exists, so uniform is the honest default
rather than an invented weighting.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import context_effects as fx
from .context_cards import (ANOMALY, ContextCard, HERO, load_context_kb,
                            of_kind)


@dataclass
class Seat:
    """One player's identity for the game."""
    idx: int
    hero: Optional[ContextCard] = None
    hero_power: Optional[ContextCard] = None

    @property
    def hero_name(self) -> str:
        return self.hero.name if self.hero else "No Hero"


@dataclass
class LobbySetup:
    seats: List[Seat] = field(default_factory=list)
    anomaly: Optional[ContextCard] = None
    hooks: Dict[str, object] = field(default_factory=dict)

    @property
    def anomaly_name(self) -> str:
        return self.anomaly.name if self.anomaly else "None"

    def hook(self, name: str, default=None):
        return self.hooks.get(name, default)

    def describe(self) -> str:
        who = ", ".join(s.hero_name for s in self.seats)
        applied = ", ".join(f"{k}={v}" for k, v in sorted(self.hooks.items()))
        line = f"Anomaly: {self.anomaly_name}"
        if applied:
            line += f"  [{applied}]"
        return f"{line}\nHeroes: {who}"


def roll_lobby(rng: random.Random, n_players: int = 8,
               kb: Optional[Dict[str, ContextCard]] = None,
               anomalies: bool = True) -> LobbySetup:
    """Deal `n_players` distinct heroes and one lobby-wide anomaly.

    `anomalies=False` deals the heroes but leaves the base economy alone.
    """
    kb = kb if kb is not None else load_context_kb()
    heroes = of_kind(kb, HERO, active_only=True)
    pool = of_kind(kb, ANOMALY, active_only=True)

    seats: List[Seat] = []
    if heroes:
        # Sample without replacement — two seats can't be the same hero.
        picked = rng.sample(heroes, min(n_players, len(heroes)))
        while len(picked) < n_players:                # tiny pool: allow repeats
            picked.append(rng.choice(heroes))
        for i, h in enumerate(picked):
            seats.append(Seat(idx=i, hero=h,
                              hero_power=kb.get(h.hero_power_id or "")))
    else:
        seats = [Seat(idx=i) for i in range(n_players)]

    anomaly = rng.choice(pool) if (anomalies and pool) else None
    hooks: Dict[str, object] = {}
    if anomaly:
        for h in fx.parse_text(anomaly.text):
            hooks[h.name] = h.value
    return LobbySetup(seats=seats, anomaly=anomaly, hooks=hooks)


def start_gold(setup: LobbySetup, default: int) -> int:
    return int(setup.hook(fx.START_GOLD, default))


def start_tier(setup: LobbySetup, default: int) -> int:
    return int(setup.hook(fx.START_TIER, default))


def start_health(setup: LobbySetup, default: int) -> int:
    hp = int(setup.hook(fx.START_HEALTH, default))
    return hp + int(setup.hook(fx.START_ARMOR, 0) or 0)


def minion_cost(setup: LobbySetup, default: int) -> int:
    return int(setup.hook(fx.MINION_COST, default))


def refresh_cost(setup: LobbySetup, default: int) -> int:
    if setup.hook(fx.REFRESH_DISABLED):
        return 10 ** 6                     # priced out of reach == unavailable
    return int(setup.hook(fx.REFRESH_COST, default))


def triple_copies(setup: LobbySetup, default: int = 3) -> int:
    return int(setup.hook(fx.TRIPLE_COPIES, default))


def allowed_tiers(setup: LobbySetup, max_tier: int) -> List[int]:
    tiers = setup.hook(fx.ALLOWED_TIERS)
    if isinstance(tiers, (list, tuple)) and tiers:
        return sorted(int(t) for t in tiers)
    return list(range(1, max_tier + 1))
