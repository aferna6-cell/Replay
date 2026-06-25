"""Abstract 8-player Battlegrounds *economy* environment for self-play.

We can't faithfully simulate every card's text (that's re-implementing the game),
but the economy/tempo layer IS regular enough to simulate: gold income, tavern
tiers, board strength growing vs the pace curve, and — crucially — real combat /
damage / elimination across an 8-player lobby that produces honest *placement*
labels.

So the board is abstracted to a single `strength` scalar (≈ total stats), grown
multiplicatively per turn (buffs compound, matching the exponential pace curve).
The value the simulation adds over the hand-tuned heuristic is the **emergent
placement distribution** from playing out full lobbies with combat variance — not
hand-weights. Each player's per-turn state is labeled with the finish it led to,
which trains the learned trajectory value (`econ_value.py`).

Grounded in `data/stats/firestone_pace.json`; an approximation of the economy, not
the card game.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from hsbg_coach.pace import load_pace, _at as _curve_at

MAX_TIER = 6
_UPGRADE = {1: 5, 2: 7, 3: 8, 4: 9, 5: 10}
_POLICIES = ["standard", "tempo", "greedy", "random"]


def gold_at(turn: int) -> int:
    return min(10, turn + 2)


def _curve(sca, turn) -> float:
    v = _curve_at(sca, turn) if sca else None
    return v if v else max(4.0, 6.0 * (turn - 1))


def _exp_tier(lev, turn) -> float:
    v = _curve_at(lev, turn) if lev else None
    return v if v else min(MAX_TIER, 1 + (turn - 1) // 2)


@dataclass
class Player:
    strength: float
    policy: str
    tier: int = 1
    hp: float = 30.0
    alive: bool = True
    placement: Optional[int] = None
    # per-turn (turn, tier, strength, ratio, hp, players_left)
    traj: List[Tuple] = field(default_factory=list)


def _intent(policy, tier, exp_tier, rng) -> str:
    if policy == "tempo":
        return "level" if tier < exp_tier - 0.7 else "tempo"
    if policy == "greedy":
        return "level" if tier < min(MAX_TIER, exp_tier + 1) else "tempo"
    if policy == "random":
        return rng.choice(["level", "tempo", "tempo"])
    return "level" if (tier < exp_tier - 0.2 and tier < MAX_TIER) else "tempo"


def _grow(strength, tier, intent, exp_tier, c_growth, rng):
    if intent == "level" and tier < MAX_TIER:
        tier += 1
        g = c_growth * 0.7                       # board lags the turn you tier up
    else:
        if tier + 0.5 >= exp_tier:
            g = c_growth * (1.05 + 0.05 * max(0.0, tier - exp_tier))
        else:
            g = c_growth * 0.85                  # under-tiered can't keep pace
    g *= rng.uniform(0.9, 1.1)                    # execution variance
    return max(strength, 1.0) * g, tier


def simulate_lobby(pace, seed: int = 0, n: int = 8, max_turns: int = 14) -> List[Player]:
    rng = random.Random(seed)
    sca = pace.get("scaling", {})
    lev = pace.get("leveling", {})
    start = _curve(sca, 1)
    players = [Player(strength=start * rng.uniform(0.7, 1.3), policy=_POLICIES[i % 4])
               for i in range(n)]

    for turn in range(2, max_turns + 1):
        alive = [p for p in players if p.alive]
        if len(alive) <= 1:
            break
        exp_tier = _exp_tier(lev, turn)
        prev, cur = _curve(sca, turn - 1), _curve(sca, turn)
        c = cur / prev if prev else 1.0
        for p in alive:
            intent = _intent(p.policy, p.tier, exp_tier, rng)
            p.strength, p.tier = _grow(p.strength, p.tier, intent, exp_tier, c, rng)
            p.traj.append((turn, p.tier, p.strength, p.strength / cur, p.hp, len(alive)))

        # Combat: random pairings; loser takes damage scaled by winner's tier.
        order = alive[:]
        rng.shuffle(order)
        for i in range(0, len(order) - 1, 2):
            a, b = order[i], order[i + 1]
            if abs(a.strength - b.strength) < 0.02 * cur:
                continue                          # ~tie, no damage
            pa = 1.0 / (1.0 + math.exp(-(a.strength - b.strength) / (0.25 * cur + 1)))
            win, lose = (a, b) if rng.random() < pa else (b, a)
            lose.hp -= win.tier + 2
        if len(order) % 2 == 1:                   # odd player fights the "field"
            p = order[-1]
            if p.strength < cur * rng.uniform(0.8, 1.2):
                p.hp -= p.tier + 2

        dead = [p for p in alive if p.hp <= 0]
        if dead:
            place = len(alive)                    # weakest dead gets the worst slot
            for p in sorted(dead, key=lambda x: x.strength):
                p.alive = False
                p.placement = place
                place -= 1

    # Rank survivors by strength for the remaining top places.
    survivors = sorted((p for p in players if p.alive), key=lambda p: -p.strength)
    for i, p in enumerate(survivors):
        p.placement = i + 1
    return players


def lobby_examples(players: List[Player]) -> List[Tuple[List[float], int]]:
    """(feature_vector, placement) for every decision point in a finished lobby."""
    out = []
    for p in players:
        if p.placement is None:
            continue
        for (turn, tier, strength, ratio, hp, left) in p.traj:
            out.append((features(turn, tier, strength, ratio, hp, left), p.placement))
    return out


def features(turn, tier, strength, ratio, hp, players_left) -> List[float]:
    return [turn / 14.0, tier / 6.0, math.log1p(max(strength, 0)) / 15.0,
            min(ratio, 3.0) / 3.0, max(hp, 0) / 40.0, players_left / 8.0]


def generate(n_lobbies: int, pace=None, seed: int = 0):
    pace = pace if pace is not None else load_pace()
    X, y = [], []
    for i in range(n_lobbies):
        for feat, place in lobby_examples(simulate_lobby(pace, seed=seed + i)):
            X.append(feat)
            y.append(place)
    return X, y
