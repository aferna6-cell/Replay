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


def combat_damage(winner_tier: int, turn: int) -> int:
    """BG damage ≈ winner's tavern tier + surviving minions; the board fills over
    the game, so damage grows — which is what makes late games end fast."""
    minions = min(7, max(1, turn - 2))
    return winner_tier + minions


# Avg players still alive at each turn (from the simulator, post damage-tuning) —
# used to query the value net with a realistic lobby size, not a flat constant.
def alive_at(turn: int) -> int:
    if turn <= 8:
        return 8
    return {9: 7, 10: 5, 11: 4, 12: 3, 13: 2, 14: 2}.get(turn, 1)


def _curve(sca, turn) -> float:
    v = _curve_at(sca, turn) if sca else None
    return v if v else max(4.0, 6.0 * (turn - 1))


def _exp_tier(lev, turn) -> float:
    # `lev` is the tavern-tier curve (what tier strong players are ON each turn),
    # so the sim's "on-tier" target now climbs to 5-6 late game.
    v = _curve_at(lev, turn) if lev else None
    return v if v else min(MAX_TIER, 1 + (turn - 1) // 2)


@dataclass
class Player:
    strength: float
    policy: str
    idx: int = 0
    tier: int = 1
    hp: float = 30.0
    alive: bool = True
    placement: Optional[int] = None
    # per-turn (turn, tier, strength, ratio, hp, players_left)
    traj: List[Tuple] = field(default_factory=list)
    # per-turn (feature_vector, action_idx) when driven by a policy decider
    actions: List[Tuple] = field(default_factory=list)


def _intent(policy, tier, exp_tier, rng) -> str:
    if policy == "tempo":
        return "level" if tier < exp_tier - 0.7 else "tempo"
    if policy == "greedy":
        return "level" if tier < min(MAX_TIER, exp_tier + 1) else "tempo"
    if policy == "random":
        return rng.choice(["level", "tempo", "tempo"])
    return "level" if (tier < exp_tier - 0.2 and tier < MAX_TIER) else "tempo"


def _grow(strength, tier, intent, exp_tier, c_growth, rng):
    # Higher tiers scale HARDER (a tier-6 minion crushes a tier-3 one) — the
    # payoff that makes tiering up worth the tempo it costs. Kept modest so the
    # optimal pace tracks the tavern curve rather than rushing past it.
    tier_bonus = 0.035 * tier
    deficit = max(0.0, exp_tier - tier)          # how many tiers below the curve
    if intent == "level" and tier < MAX_TIER:
        tier += 1
        g = c_growth * 0.6                        # leveling sacrifices the turn…
    else:
        # …but staying under-tiered snowballs: each tier you're behind cuts your
        # growth hard (a tier-1 board late game is nearly dead), floored so it
        # never goes negative. On/above curve you scale slightly past it. Tuned
        # (level cost 0.6, deficit 0.32) so on-curve leveling beats both rushing
        # and never-leveling — verified in the heuristic-policy ranking.
        g = c_growth * max(0.3, 0.98 + tier_bonus - 0.32 * deficit)
    g *= rng.uniform(0.88, 1.14)                 # execution variance
    s = max(strength, 1.0) * g
    # Fat upside: occasionally high-roll a strong board (a bomb / triple) — the
    # comeback fuel that makes a behind/low-HP level a real gamble. A rarer bust
    # balances it. Only gently tier-scaled so it doesn't itself drive rushing.
    roll = rng.random()
    if roll < 0.04 + 0.008 * tier:
        s *= rng.uniform(1.4, 2.2)
    elif roll > 0.94:
        s *= rng.uniform(0.7, 0.9)
    return s, tier


def simulate_lobby(pace, seed: int = 0, n: int = 8, max_turns: int = 14,
                   explore: float = 0.15, deciders=None) -> List[Player]:
    """Play out an 8-player economy lobby.

    `deciders` (optional) is a length-n list; deciders[i](features) -> (intent,
    action_idx) drives player i with a policy (recording its actions for RL); a
    None entry uses the built-in heuristic policy. Default: all heuristic."""
    rng = random.Random(seed)
    sca = pace.get("scaling", {})
    lev = pace.get("tavern_tier") or pace.get("leveling", {})   # tavern-tier target
    start = _curve(sca, 1)
    # Wider spread of starts + policies broadens the state coverage the value net
    # sees, so it's reliable on the off-policy states the recommender evaluates.
    players = [Player(strength=start * rng.uniform(0.55, 1.5), policy=_POLICIES[i % 4],
                      idx=i) for i in range(n)]

    for turn in range(2, max_turns + 1):
        alive = [p for p in players if p.alive]
        if len(alive) <= 1:
            break
        exp_tier = _exp_tier(lev, turn)
        prev, cur = _curve(sca, turn - 1), _curve(sca, turn)
        c = cur / prev if prev else 1.0
        for p in alive:
            dec = deciders[p.idx] if deciders else None
            if dec is not None:
                feat = features(turn, p.tier, p.strength, p.strength / cur, p.hp, len(alive))
                intent, aidx = dec(feat)
                p.actions.append((feat, aidx))
            elif rng.random() < explore:          # exploration -> off-policy states
                intent = rng.choice(["level", "tempo"])
            else:
                intent = _intent(p.policy, p.tier, exp_tier, rng)
            p.strength, p.tier = _grow(p.strength, p.tier, intent, exp_tier, c, rng)
            p.traj.append((turn, p.tier, p.strength, p.strength / cur, p.hp, len(alive)))

        # Combat: random pairings; loser takes damage scaled by tier + board size.
        order = alive[:]
        rng.shuffle(order)
        for i in range(0, len(order) - 1, 2):
            a, b = order[i], order[i + 1]
            if abs(a.strength - b.strength) < 0.02 * cur:
                continue                          # ~tie, no damage
            pa = 1.0 / (1.0 + math.exp(-(a.strength - b.strength) / (0.25 * cur + 1)))
            win, lose = (a, b) if rng.random() < pa else (b, a)
            lose.hp -= combat_damage(win.tier, turn)
        if len(order) % 2 == 1:                   # odd player fights the "field"
            p = order[-1]
            if p.strength < cur * rng.uniform(0.8, 1.2):
                p.hp -= combat_damage(p.tier, turn)

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
