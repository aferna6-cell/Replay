"""Recommendation facade — one ranked move list for a Snapshot.

Combines the two log-independent advice layers built so far:
- economy heuristics (buy/level/roll/freeze/sell) — always available
- sim-based positioning — when an enemy board (or a field of likely enemies) is
  known

This is the pre-ML recommendation surface. The learned policy (milestone 6) will
eventually re-rank or replace these, but the structure stays: every layer emits
``Recommendation`` objects with a priority, and the facade merges them. Combat
odds are surfaced as an info line for the overlay.

Why a field of enemies: in the recruit phase you don't know who you'll fight, so
positioning/odds are averaged over plausible opponents (e.g. recent lobby
boards) rather than overfit to one. At combat, pass the single revealed board.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .bg import ActionType
from .economy import advise, HeroContext
from .position import positioning_advice
from .sim import simulate


@dataclass
class Recommendation:
    action: str
    rationale: str
    priority: float
    source: str                       # "economy" | "positioning"
    detail: Dict = field(default_factory=dict)


def _get(snapshot, key, default=None):
    if isinstance(snapshot, dict):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def combat_odds(
    snapshot,
    enemy_boards: Sequence[Sequence],
    runs: int = 200,
    seed: int = 0,
) -> Optional[str]:
    """Win/tie/loss for the current board order, averaged over the enemy field."""
    board = list(_get(snapshot, "board", []) or [])
    if not enemy_boards:
        return None
    wins = ties = losses = 0
    for j, enemy in enumerate(enemy_boards):
        r = simulate(board, list(enemy), runs=runs, seed=seed + j)
        wins += r.wins
        ties += r.ties
        losses += r.losses
    total = wins + ties + losses
    if not total:
        return None
    return (f"win {wins/total:.0%} / tie {ties/total:.0%} / loss {losses/total:.0%}")


def recommend(
    snapshot,
    enemy_boards: Optional[Sequence[Sequence]] = None,
    runs: int = 150,
    seed: int = 0,
    hero_ctx: Optional[HeroContext] = None,
) -> List[Recommendation]:
    """Merged, ranked recommendations across economy + positioning.

    hero_ctx (optional) makes the economy advice hero/comp-specific.
    """
    recs: List[Recommendation] = []

    for s in advise(snapshot, hero_ctx=hero_ctx):
        recs.append(Recommendation(s.action, s.rationale, s.priority,
                                   "economy", s.detail))

    board = list(_get(snapshot, "board", []) or [])
    if board and enemy_boards:
        advice = positioning_advice(board, enemy_boards, runs=runs, seed=seed)
        if advice:
            # A concrete reposition gain is high value; "fine as is" is low.
            pri = 0.72 if advice.lower().startswith("reposition") else 0.3
            recs.append(Recommendation(ActionType.POSITION.value, advice, pri,
                                       "positioning", {}))

    recs.sort(key=lambda r: r.priority, reverse=True)
    return recs


def overlay_payload(
    snapshot,
    enemy_boards: Optional[Sequence[Sequence]] = None,
    runs: int = 150,
    seed: int = 0,
) -> Dict:
    """Build (snapshot_dict, odds_string) for the overlay, plus a recs list.

    Returns a dict the overlay/poll-provider can consume directly.
    """
    snap_dict = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
    odds = combat_odds(snapshot, enemy_boards, runs=runs, seed=seed) if enemy_boards else None
    recs = recommend(snapshot, enemy_boards, runs=runs, seed=seed)
    snap_dict = dict(snap_dict)
    snap_dict["notes"] = list(snap_dict.get("notes") or []) + [
        f"{r.action}: {r.rationale}" for r in recs[:3]
    ]
    return {"snapshot": snap_dict, "odds": odds, "recommendations": recs}
