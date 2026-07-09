"""Shared plumbing for the RL track: episode rollout + policy evaluation."""

import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from hsbg_coach.bg_env import BGEnv, greedy_policy, random_policy
from hsbg_coach import cards
from .env_obs import encode_obs

MAX_DECISIONS = 400          # hard cap on agent decisions per episode


def rollout(policy_step: Callable, seed: int,
            opponents: Optional[Sequence[Callable]] = None,
            emb: Optional[Dict] = None, byname: Optional[Dict] = None,
            shaping: float = 0.0) -> Dict:
    """Play one episode from seat 0.

    `policy_step(arrays, legal_mask) -> (action, logp, value)`.
    Returns the trajectory (arrays per step) + placement. Optional shaping adds
    a small on-pace leveling signal at each end-of-turn, annealed by the caller
    (spec §5: keep shaping weights small, anneal toward pure placement).
    """
    from hsbg_coach.pace import STANDARD_TAVERN_TIER
    env = BGEnv(seed=seed, opponent_policies=list(opponents or []))
    obs = env.reset(seed=seed)
    traj = {"tokens": [], "mask": [], "zones": [], "ctx": [], "legal": [],
            "action": [], "logp": [], "value": [], "reward": []}
    placement = 8
    for _ in range(MAX_DECISIONS):
        legal = env.legal_mask(0)
        arrays = encode_obs(obs, emb or {}, byname)
        a, logp, v = policy_step(arrays, legal)
        prev_turn = obs["turn"]
        prev_tier = obs["tavern_tier"]
        obs, reward, done, info = env.step(a)
        if shaping and (done or obs["turn"] != prev_turn):
            target = STANDARD_TAVERN_TIER.get(prev_turn, 6.0)
            reward += shaping * max(-2.0, min(1.0, prev_tier - target)) * 0.05
        traj["tokens"].append(arrays[0])
        traj["mask"].append(arrays[1])
        traj["zones"].append(arrays[2])
        traj["ctx"].append(arrays[3])
        traj["legal"].append(np.asarray(legal, dtype=np.float32))
        traj["action"].append(a)
        traj["logp"].append(logp)
        traj["value"].append(v)
        traj["reward"].append(reward)
        if done:
            placement = info.get("placement", 8)
            break
    traj["placement"] = placement
    return traj


def mixed_field(rng: random.Random, league: Sequence[Callable]) -> List[Callable]:
    """7 opponent seats: mostly greedy, one chaotic, league checkpoints mixed in."""
    seats: List[Callable] = []
    for _ in range(7):
        r = rng.random()
        if league and r < 0.35:
            seats.append(rng.choice(list(league)))
        elif r < 0.9:
            seats.append(greedy_policy)
        else:
            seats.append(random_policy)
    return seats


def evaluate_policy(env_policy: Callable, episodes: int = 30, seed: int = 9000,
                    field: Optional[Sequence[Callable]] = None) -> float:
    """Average placement of `env_policy(obs, mask, rng)` in seat 0 vs a field
    (default: all-greedy — the baseline the spec says Phase 1 must beat)."""
    total = 0.0
    for i in range(episodes):
        env = BGEnv(seed=seed + i,
                    opponent_policies=list(field or [greedy_policy] * 7))
        obs = env.reset(seed=seed + i)
        rng = random.Random(seed + i)
        for _ in range(MAX_DECISIONS):
            a = env_policy(obs, env.legal_mask(0), rng)
            obs, reward, done, info = env.step(a)
            if done:
                total += info.get("placement", 8)
                break
        else:
            total += 8
    return total / episodes


def kb_byname():
    return cards.by_name(cards.load_kb())
