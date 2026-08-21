"""Search-expert teacher tests: structured actions + env-action mapping."""

import random

import pytest

from hsbg_coach.bg_env import (
    BGEnv, A_BUY0, A_PLAY0, A_SELL0, A_ROLL, A_LEVEL, A_END, N_PLAY,
)
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.turn_search import plan_turn_search


def test_turn_plan_exposes_structured_actions():
    snap = {"board": [], "hand": [],
            "shop": [{"name": "Big", "attack": 8, "health": 8}],
            "gold": 4, "tavern_tier": 3, "hero_health": 25, "turn": 8}
    plan = plan_turn_search(snap, kb=None, scorer=HeuristicScorer({}), pace={})
    assert ("buy", "Big") in plan.actions
    kinds = {k for k, _ in plan.actions}
    assert kinds <= {"buy", "sell", "level", "roll"}


def test_expert_plays_hand_first_and_acts_legally():
    from ml.search_expert import SearchExpert
    expert = SearchExpert(beam=3, depth=3)
    env = BGEnv(seed=21)
    obs = env.reset(seed=21)
    rng = random.Random(0)
    saw_play = False
    for _ in range(80):
        mask = env.legal_mask(0)
        a = expert(obs, mask, rng)
        assert mask[a]                       # never returns an illegal action
        if A_PLAY0 <= a < A_PLAY0 + N_PLAY:
            saw_play = True
        if obs["hand"] and any(mask[A_PLAY0:A_PLAY0 + N_PLAY]):
            assert A_PLAY0 <= a < A_PLAY0 + N_PLAY   # hand plays come first
        obs, _, done, _ = env.step(a)
        if done:
            break
    assert saw_play                          # it bought and played something


def test_evaluate_detailed_reports_board_metrics():
    from hsbg_coach.bg_env import greedy_policy
    from ml.rl_common import evaluate_detailed
    m = evaluate_detailed(greedy_policy, episodes=3, seed=123)
    assert 1.0 <= m["placement"] <= 8.0
    assert m["final_board_stats"] > 0
    assert 1.0 <= m["final_tier"] <= 6.0
    assert 0.0 <= m["top4"] <= 1.0
    assert m["episodes"] == 3
