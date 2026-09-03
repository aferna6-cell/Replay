"""Pace benchmarks — am I keeping up with how top players level and scale?

Three curves, from ``firestone_pace.json``:

- **tavern_tier**: the tavern tier strong players are ON at each turn — the real
  "when to level" benchmark, reaching tier 5-6 in the late game.
- **leveling**: average tier of minions *played* each turn. NOTE: this is a board-
  *composition* number (winning boards keep low-tier staples all game, so it
  plateaus ~3.6) — it is NOT tavern tier, and must not be used for leveling
  advice. Kept for reference only.
- **scaling**: average total board stats each turn.

``pace_advice`` compares your current turn's tier (vs ``tavern_tier``) and board
(vs ``scaling``), so the advisor can say "you're behind the leveling pace" with
real numbers behind it.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stats")
FIRESTONE_PACE = os.path.join(_STATS_DIR, "firestone_pace.json")

# How far behind/ahead before we say something.
TIER_BEHIND = 0.5
TIER_AHEAD = 0.5
STATS_BEHIND = 0.7   # board below 70% of benchmark = behind
STATS_AHEAD = 1.3

# High-level tavern-tier-by-turn baseline. The old "leveling" curve measured avg
# tier of cards *played* (plateaus ~3.6, NOT tavern tier). This is the real pace
# strong players hold: ~tier 4 by t7, tier 5 by t9, tier 6 by ~t12. It "varies
# game to game" (push faster when ahead/healthy) — refine per-hero from real
# Power.logs, which log your actual tavern tier every turn.
STANDARD_TAVERN_TIER = {1: 1.0, 2: 1.6, 3: 2.2, 4: 2.8, 5: 3.3, 6: 3.8, 7: 4.3,
                        8: 4.8, 9: 5.2, 10: 5.5, 11: 5.8, 12: 6.0, 13: 6.0, 14: 6.0}


def load_pace(source: Optional[str] = None) -> Dict[str, Dict[int, float]]:
    src = source or (FIRESTONE_PACE if os.path.isfile(FIRESTONE_PACE) else None)
    d = json.load(open(src, encoding="utf-8")) if src else {}
    tavern = {int(k): v for k, v in d.get("tavern_tier", {}).items()} or dict(STANDARD_TAVERN_TIER)
    return {
        "tavern_tier": tavern,
        "leveling": {int(k): v for k, v in d.get("leveling", {}).items()},
        "scaling": {int(k): v for k, v in d.get("scaling", {}).items()},
    }


def _at(curve: Dict[int, float], turn: int) -> Optional[float]:
    if not curve:
        return None
    if turn in curve:
        return curve[turn]
    below = [t for t in curve if t <= turn]
    return curve[max(below)] if below else curve[min(curve)]


def _get(snapshot, key, default=None):
    if isinstance(snapshot, dict):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def board_stats(snapshot) -> int:
    total = 0
    for m in _get(snapshot, "board", []) or []:
        if isinstance(m, dict):
            total += (m.get("attack") or 0) + (m.get("health") or 0)
        else:
            total += (getattr(m, "attack", 0) or 0) + (getattr(m, "health", 0) or 0)
    return total


@dataclass
class PaceVerdict:
    turn: Optional[int]
    your_tier: Optional[int]
    bench_tier: Optional[float]
    your_stats: Optional[int]
    bench_stats: Optional[float]
    behind_leveling: bool = False
    ahead_leveling: bool = False
    behind_scaling: bool = False
    notes: List[str] = field(default_factory=list)


def pace_advice(snapshot, pace: Dict[str, Dict[int, float]]) -> PaceVerdict:
    turn = _get(snapshot, "turn")
    tier = _get(snapshot, "tavern_tier")
    stats = board_stats(snapshot)
    # Use the real tavern-tier curve for leveling (fall back to leveling only if
    # tavern_tier is somehow absent).
    bt = _at(pace.get("tavern_tier") or pace.get("leveling", {}), turn) if turn else None
    bs = _at(pace.get("scaling", {}), turn) if turn else None
    v = PaceVerdict(turn=turn, your_tier=tier, bench_tier=bt,
                    your_stats=stats, bench_stats=bs)

    if tier is not None and bt is not None:
        if tier < bt - TIER_BEHIND:
            v.behind_leveling = True
            v.notes.append(
                f"Behind top-10% leveling: tier {tier} on turn {turn}, "
                f"top players ~tier {bt:.1f}. Consider leveling.")
        elif tier > bt + TIER_AHEAD:
            v.ahead_leveling = True
            v.notes.append(
                f"Ahead of leveling pace (tier {tier} vs ~{bt:.1f}); "
                f"you can play for tempo/board.")
    if bs and stats < bs * STATS_BEHIND:
        v.behind_scaling = True
        v.notes.append(
            f"Board behind the top-10% stat curve ({stats} vs ~{bs:.0f} expected) "
            f"— prioritize stats/upgrades.")
    return v
