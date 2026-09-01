"""Tests for Phase 2E seeded-core stress policy and experiment."""

import random

from hsbg_coach.bg_env import (
    A_BUY0, A_END, A_PLAY0, greedy_policy, seeded_core_stress_greedy_policy,
)
from hsbg_coach.seeded_core_stress_policy import (
    PHASE_2E_EVAL_SEED_BASE,
    seeded_core_buy_override,
)


def _obs(shop, board=None, hand=None, turn=8, tier=4, gold=10):
    return {
        "turn": turn,
        "gold": gold,
        "tavern_tier": tier,
        "board": board or [],
        "hand": hand or [],
        "shop": shop,
    }


def _mask(n_shop: int, *, can_play=False):
    mask = [False] * 28
    for i in range(n_shop):
        mask[A_BUY0 + i] = True
    if can_play:
        mask[A_PLAY0] = True
    mask[A_END] = True
    return mask


def test_frozen_eval_seed_base():
    assert PHASE_2E_EVAL_SEED_BASE == 1000


def test_override_skips_when_not_seeded():
    board = []
    shop = [
        {"name": "Titus Rivendare", "attack": 1, "health": 7, "tribes": ["Undead"]},
    ]
    obs = _obs(shop, board=board)
    mask = _mask(1)
    assert seeded_core_buy_override(obs, mask, [0]) is None


def test_override_buys_missing_core_when_seeded():
    board = [{"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]}]
    shop = [
        {"name": "Impulsive Trickster", "attack": 10, "health": 10, "tribes": ["Demon"]},
        {"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Neutral"]},
    ]
    obs = _obs(shop, board=board, tier=4)
    mask = _mask(2)
    pick = seeded_core_buy_override(obs, mask, [0, 1])
    assert pick == 1


def test_stress_policy_play_before_buy():
    rng = random.Random(0)
    obs = _obs(
        [{"name": "Z", "attack": 1, "health": 1, "tribes": []}],
        board=[{"name": "X", "attack": 1, "health": 1, "tribes": []}],
        hand=[{"name": "Y", "attack": 2, "health": 2, "tribes": []}],
    )
    mask = _mask(1, can_play=True)
    assert seeded_core_stress_greedy_policy(obs, mask, rng) == greedy_policy(
        obs, mask, rng)


def test_phase_2e_smoke():
    from ml.fidelity_phase_2e import run_phase_2e
    result = run_phase_2e(
        lobbies=2, seed=PHASE_2E_EVAL_SEED_BASE,
        out_dir="/tmp/sim_fidelity_phase_2e_test",
        require_clean_tree=False)
    assert result["decision"]["decision_branch"]
    assert result["evaluation_seed_base"] == 1000
