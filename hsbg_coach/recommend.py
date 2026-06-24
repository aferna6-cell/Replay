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
from . import firestone_bridge


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
    tier = _get(snapshot, "tavern_tier") or 1
    hp = _get(snapshot, "hero_health") or 30
    wins = ties = losses = 0
    for j, enemy in enumerate(enemy_boards):
        # Uses the Firestone backend when installed, else the pure sim.
        r = firestone_bridge.simulate(board, list(enemy), runs=runs, seed=seed + j,
                                      my_tier=tier, my_hp=hp)
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
    kb=None,
    pace=None,
) -> List[Recommendation]:
    """Merged, ranked recommendations across economy + positioning + synergy + pace.

    hero_ctx (optional) makes the economy advice hero/comp-specific.
    kb (optional CardKnowledge dict from cards.load_kb) enables synergy-aware buys.
    pace (optional, from pace.load_pace()) nudges leveling vs the top-10% curve.
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

    if kb is not None:
        recs.extend(_synergy_buys(snapshot, kb, hero_ctx))

    if pace:
        recs.extend(_pace_recs(snapshot, pace))

    recs.sort(key=lambda r: r.priority, reverse=True)
    return recs


def _pace_recs(snapshot, pace) -> List[Recommendation]:
    """Turn pace-vs-benchmark deltas into recommendations."""
    from .pace import pace_advice
    v = pace_advice(snapshot, pace)
    out = []
    if v.behind_leveling:
        # Bigger tier gap = stronger nudge.
        gap = (v.bench_tier or 0) - (v.your_tier or 0)
        pri = min(0.85, 0.6 + gap * 0.15)
        out.append(Recommendation(ActionType.TIER_UP.value, v.notes[0],
                                  round(pri, 2), "pace", {"tier_gap": round(gap, 2)}))
    elif v.behind_scaling:
        out.append(Recommendation(ActionType.BUY.value,
                                  next(n for n in v.notes if "stat curve" in n),
                                  0.55, "pace", {}))
    return out


def _synergy_buys(snapshot, kb, hero_ctx) -> List[Recommendation]:
    """Synergy-aware buy advice: rank the shop against your board + comp."""
    from .synergy import rank_shop, resolve
    from .cards import by_name
    shop_names = [_name(m) for m in (_get(snapshot, "shop", []) or [])]
    board_names = [_name(m) for m in (_get(snapshot, "board", []) or [])]
    if not shop_names:
        return []
    idx = by_name(kb)
    shop = resolve(shop_names, kb, idx)
    board = resolve(board_names, kb, idx)
    if not shop:
        return []
    target = hero_ctx.target_tribe if hero_ctx else None
    ranked = rank_shop(shop, board, target_tribe=target)
    top, verdict = ranked[0]
    if verdict.score <= 0:
        return []
    # Map synergy score (~0-5) into a buy priority band.
    pri = min(0.9, 0.55 + verdict.score * 0.07)
    reason = f"Buy {top.name} (synergy {verdict.score}): " + "; ".join(verdict.reasons[:2])
    return [Recommendation(ActionType.BUY.value, reason, round(pri, 2),
                           "synergy", {"name": top.name, "score": verdict.score})]


def _name(m):
    if isinstance(m, dict):
        return m.get("name") or m.get("card_id") or ""
    return getattr(m, "name", None) or getattr(m, "card_id", None) or ""


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
