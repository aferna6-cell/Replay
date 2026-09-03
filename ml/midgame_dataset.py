"""Mid-game training data — fixing the eval net's distribution shift.

The old training set was *final* boards (turn 12+, fully developed, winners'
comps) while the advisor queries the net on *mid-game* boards (turn 5-9,
half-built) — states the net had literally never seen. The Phase 0 env closes
that gap: play full scripted lobbies and label EVERY per-turn recruit state
with the final placement it led to. Unlimited, honestly-labeled mid-game
examples in exactly the shape `ml/tokens.py` encodes.

The scripted field is deliberately diverse (on-curve / greedy-leveler / tempo /
one chaotic seat) so the value function sees off-policy states, not just one
playstyle's trajectory.
"""

import random
from typing import Dict, List, Optional

from hsbg_coach import cards
from hsbg_coach.bg_env import (
    BGEnv, greedy_policy, make_greedy_policy, random_policy,
)
from . import seeds
from .board_features import minion_from_snapshot

_STYLES = [greedy_policy, make_greedy_policy(0.7), make_greedy_policy(-0.7)]


def _field(rng: random.Random) -> List:
    """8 seats: mixed pacing styles + one chaotic seat for state coverage."""
    seats = [rng.choice(_STYLES) for _ in range(8)]
    seats[rng.randrange(8)] = random_policy
    return seats


def generate_examples(lobbies: int = 200, seed: int = 0,
                      byname: Optional[Dict] = None,
                      min_minions: int = 1) -> List[Dict]:
    """Env self-play -> dataset rows in board_dataset's example shape:
    {minions (normalized), hero, label, state, group}."""
    byname = byname if byname is not None else cards.by_name(cards.load_kb())
    if lobbies:
        seeds.check_training_range(
            "ml.midgame_dataset", seeds.midgame_lobby_seed(seed, 0),
            seeds.midgame_lobby_seed(seed, lobbies - 1))
    rng = random.Random(seed)
    out: List[Dict] = []
    for i in range(lobbies):
        env = BGEnv(seed=seeds.midgame_lobby_seed(seed, i))
        for rec in env.play_scripted(_field(rng)):
            s = rec["state"]
            minions = [m for m in (minion_from_snapshot(x, byname)
                                   for x in s["board"]) if m]
            if len(minions) < min_minions:
                continue
            out.append({
                "minions": minions,
                "hero": "UNKNOWN",
                "label": float(rec["placement"]),
                "state": {
                    "tavern_tier": s["tavern_tier"],
                    "gold": s["gold"],
                    "hero_health": s["hero_health"],
                    "turn": s["turn"],
                    "opponent_profiles": [
                        {"strength": s.get("max_opp_strength", 0)}],
                    "trinkets": [],
                    "anomaly": None,
                },
                "group": f"lobby{i}",
                "turn": rec["turn"],
            })
    return out
