"""Tests for Phase 2D build-aware greedy policy and experiment runner."""

import pytest

from hsbg_coach.bg_env import (
    A_BUY0, A_END, A_LEVEL, A_PLAY0, build_aware_greedy_policy, greedy_policy,
)
from hsbg_coach.build_aware_policy import (
    BUILD_PATH_BUY_DIVISOR,
    POLICY_CONFIG_FINGERPRINT,
    build_aware_buy_score,
    raw_stat_buy_score,
)


def _obs(shop, board=None, turn=8, tier=4, gold=10, hand=None):
    return {
        "turn": turn,
        "gold": gold,
        "tavern_tier": tier,
        "board": board or [],
        "hand": hand or [],
        "shop": shop,
    }


def _mask_with_buys(n_shop: int, *, can_level=False, can_play=False):
    mask = [False] * 28
    for i in range(n_shop):
        mask[A_BUY0 + i] = True
    if can_level:
        mask[A_LEVEL] = True
    if can_play:
        mask[A_PLAY0] = True
    mask[A_END] = True
    return mask


def test_policy_config_frozen():
    assert POLICY_CONFIG_FINGERPRINT["build_path_buy_divisor"] == BUILD_PATH_BUY_DIVISOR
    assert BUILD_PATH_BUY_DIVISOR == 5.0


def test_raw_stat_score_matches_attack_health():
    obs = _obs([{"name": "A", "attack": 3, "health": 4, "tribes": []}])
    assert raw_stat_buy_score(obs, 0) == 7.0


def test_build_aware_prefers_core_when_stats_close():
    """Core with modest path bonus can beat slightly larger off-stat minion."""
    board = [{"name": "Titus Rivendare", "attack": 1, "health": 7, "tribes": ["Undead"]}]
    shop = [
        {"name": "Impulsive Trickster", "attack": 3, "health": 3, "tribes": ["Demon"]},
        {"name": "Titus Rivendare", "attack": 1, "health": 7, "tribes": ["Undead"]},
    ]
    obs = _obs(shop, board=board, tier=4)
    core_score = build_aware_buy_score(obs, 1)
    off_score = build_aware_buy_score(obs, 0)
    assert core_score > off_score


def test_build_aware_does_not_always_buy_core():
    """Large raw-stat gap should still favor tempo over weak path signal."""
    board = [{"name": "Titus Rivendare", "attack": 1, "health": 7, "tribes": ["Undead"]}]
    shop = [
        {"name": "Huge Body", "attack": 20, "health": 20, "tribes": ["Beast"]},
        {"name": "Titus Rivendare", "attack": 1, "health": 7, "tribes": ["Undead"]},
    ]
    obs = _obs(shop, board=board, tier=4)
    assert build_aware_buy_score(obs, 0) > build_aware_buy_score(obs, 1)


def test_build_aware_play_and_level_unchanged():
    """Non-buy decisions match greedy ordering (play before buy)."""
    import random
    rng = random.Random(0)
    board = [{"name": "X", "attack": 1, "health": 1, "tribes": []}]
    hand = [{"name": "Y", "attack": 2, "health": 2, "tribes": []}]
    shop = [{"name": "Z", "attack": 10, "health": 10, "tribes": []}]
    obs = _obs(shop, board=board, hand=hand, tier=2, turn=3)
    mask = _mask_with_buys(1, can_play=True)
    assert build_aware_greedy_policy(obs, mask, rng) == greedy_policy(obs, mask, rng)


def test_traced_rollouts_equivalent_control_unchanged():
    from ml.composition_trace import run_plain_rollouts, run_traced_rollouts
    from hsbg_coach.bg_env import greedy_policy
    n, seed = 4, 99
    plain = run_plain_rollouts(n, seed=seed, policy=greedy_policy)
    traces = run_traced_rollouts(n, seed=seed, policy=greedy_policy)
    for p in plain:
        t = next(x for x in traces["player_finals"]
                 if x["lobby"] == p["lobby"] and x["seat"] == p["seat"])
        assert p["placement"] == t["placement"]


def test_phase_2d_smoke():
    from tests.ml_testutil import require_ml
    require_ml()  # contract builder records a Torch/NumPy runtime fingerprint
    from ml.fidelity_phase_2d import run_phase_2d
    result = run_phase_2d(
        lobbies=2, seed=0, out_dir="/tmp/sim_fidelity_phase_2d_test",
        require_clean_tree=False)
    assert result["control"]["label"] == "control"
    assert result["treatment"]["label"] == "treatment"
    assert "acceptance" in result
    assert result["policy_config_hash_sha256"]
