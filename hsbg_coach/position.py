"""Positioning optimizer — find the best board ordering via the combat sim.

Positioning is one of the highest-skill BG decisions (attack order, where taunts
and deathrattles sit). This searches orderings of your minions and scores each
with ``sim.simulate``, returning the arrangement with the best win rate.

Two modes:
- ``optimize_vs(my, enemy)`` — when the opponent board is known (right before
  combat). Exact for the matchup.
- ``optimize_vs_field(my, [enemy, ...])`` — the realistic *recruit-phase* case:
  you don't know who you'll fight, so score each ordering averaged over a field
  of plausible opponents (e.g. recent lobby boards). Robust, not overfit to one
  enemy.

For <= MAX_EXACT minions we try every permutation; beyond that we sample, always
including the current order so "leave it as is" is on the table.
"""

import itertools
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .sim import Combatant, SimResult, simulate

MAX_EXACT = 6  # 6! = 720 orderings; above this we sample


@dataclass
class PositioningResult:
    ordering: List[int]          # indices into the original board, best first-to-act order
    result: SimResult
    is_current: bool             # True if this equals the board's current order

    @property
    def win_pct(self) -> float:
        return self.result.win_pct


def _as_combatants(board: Sequence) -> List[Combatant]:
    return [m if isinstance(m, Combatant) else Combatant.from_minion(m) for m in board]


def _candidate_orderings(n: int, sample: int, rng: random.Random) -> List[List[int]]:
    base = list(range(n))
    if n <= 1:
        return [base]
    if n <= MAX_EXACT:
        return [list(p) for p in itertools.permutations(base)]
    # Too many permutations: sample, but always include the current order.
    seen = {tuple(base)}
    out = [base]
    while len(out) < sample:
        perm = base[:]
        rng.shuffle(perm)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            out.append(perm)
    return out


def _score(ordering, my, enemies, runs, seed) -> SimResult:
    arranged = [my[i] for i in ordering]
    # Average over the enemy field by concatenating runs across enemies.
    agg = SimResult(0, 0, 0, 0, 0.0, 0.0)
    dealt = taken = 0
    for j, enemy in enumerate(enemies):
        r = simulate([m.copy() for m in arranged], [m.copy() for m in enemy],
                     runs=runs, seed=seed + j)
        agg.runs += r.runs
        agg.wins += r.wins
        agg.ties += r.ties
        agg.losses += r.losses
        dealt += r.avg_damage_dealt * r.wins
        taken += r.avg_damage_taken * r.losses
    agg.avg_damage_dealt = (dealt / agg.wins) if agg.wins else 0.0
    agg.avg_damage_taken = (taken / agg.losses) if agg.losses else 0.0
    return agg


def optimize_vs_field(
    my_board: Sequence,
    enemy_boards: Sequence[Sequence],
    runs: int = 200,
    seed: int = 0,
    sample: int = 120,
    top: int = 3,
) -> List[PositioningResult]:
    """Rank orderings of my_board by average win% over a field of enemy boards."""
    my = _as_combatants(my_board)
    enemies = [_as_combatants(e) for e in enemy_boards] or [[]]
    n = len(my)
    rng = random.Random(seed)

    current = list(range(n))
    results = []
    for ordering in _candidate_orderings(n, sample, rng):
        res = _score(ordering, my, enemies, runs, seed)
        results.append(PositioningResult(
            ordering=ordering, result=res, is_current=(ordering == current)))

    # Best win%, then fewest losses as tiebreak.
    results.sort(key=lambda r: (r.win_pct, -r.result.loss_pct), reverse=True)
    return results[:top]


def optimize_vs(
    my_board: Sequence,
    enemy_board: Sequence,
    runs: int = 300,
    seed: int = 0,
    sample: int = 120,
    top: int = 3,
) -> List[PositioningResult]:
    """Rank orderings against a single known opponent board."""
    return optimize_vs_field(my_board, [enemy_board], runs=runs, seed=seed,
                             sample=sample, top=top)


def positioning_advice(
    my_board: Sequence,
    enemy_boards: Sequence[Sequence],
    runs: int = 200,
    seed: int = 0,
) -> Optional[str]:
    """One-line advice: does reordering beat the current arrangement, and by how much?"""
    ranked = optimize_vs_field(my_board, enemy_boards, runs=runs, seed=seed, top=len(my_board) or 1)
    if not ranked:
        return None
    best = ranked[0]
    current = next((r for r in ranked if r.is_current), None)
    if best.is_current or current is None:
        return f"Current positioning is best ({best.win_pct:.0%} win)."
    gain = best.win_pct - current.win_pct
    if gain < 0.03:
        return f"Positioning is fine ({current.win_pct:.0%} win; best alt only +{gain:.0%})."
    order = " ".join(str(i + 1) for i in best.ordering)
    return (f"Reposition to [{order}] for {best.win_pct:.0%} win "
            f"(+{gain:.0%} vs current {current.win_pct:.0%}).")
