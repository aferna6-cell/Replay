"""Tests for residual end-of-turn scaling (Simulator v1.1)."""

import pytest

from hsbg_coach.bg_env import BGEnv, EnvMinion, VALID_SCALING_MODES, greedy_policy

from ml.fidelity_paired import (freeze_success_thresholds, paired_turn_comparison,
                                per_lobby_turn_means)
from ml.fidelity_metrics import run_fidelity_rollouts


def _player(env, idx=0):
    return env.players[idx]


def test_scaling_mode_must_be_valid():
    with pytest.raises(ValueError, match="scaling_mode"):
        BGEnv(scaling_mode="raito")
    for mode in VALID_SCALING_MODES:
        BGEnv(scaling_mode=mode)


def test_paired_comparison_uses_identical_lobby_set():
    per_v1 = {
        0: {14: 100.0, 15: 110.0},
        1: {14: 200.0},
        2: {14: 300.0, 15: 320.0},
    }
    per_v11 = {
        0: {14: 90.0, 15: 95.0},
        1: {14: 180.0, 16: 400.0},
        2: {15: 300.0},
    }
    paired = paired_turn_comparison(per_v1, per_v11, turns=(14,))
    row = paired["14"]
    assert row["n_paired_lobbies"] == 2
    assert row["v1_lobbies_improved_ratio"] == 2
    assert row["unpaired_aggregate"]["n_v1_lobbies"] == 3
    assert row["unpaired_aggregate"]["n_v1_1_lobbies"] == 2


def test_residual_late_game_skips_when_board_exceeds_pace():
    env = BGEnv(seed=0, scaling_mode="residual")
    env.reset(seed=0)
    env.turn = 14
    p = _player(env)
    p.tier = 6
    p.turns_since_level = 2
    p.board = [EnvMinion("x", "Big", 6, 12000, 12000, [], [])]
    before = p.strength()
    env._end_of_turn_scaling_residual(p)
    assert p.strength() == before


def test_residual_early_turn_matches_ratio_add():
    env_ratio = BGEnv(seed=1, scaling_mode="ratio")
    env_res = BGEnv(seed=1, scaling_mode="residual")
    for env in (env_ratio, env_res):
        env.reset(seed=1)
        env.turn = 6
        p = _player(env)
        p.tier = 4
        p.turns_since_level = 2
        p.board = [EnvMinion("a", "A", 4, 40, 40, [], [])]
    env_ratio._end_of_turn_scaling_ratio(_player(env_ratio))
    env_res._end_of_turn_scaling_residual(_player(env_res))
    assert _player(env_res).strength() == _player(env_ratio).strength()


def test_ratio_mode_preserves_v1_behavior():
    env = BGEnv(seed=2, scaling_mode="ratio")
    env.reset(seed=2)
    env.turn = 8
    p = _player(env)
    p.tier = 5
    p.turns_since_level = 1
    p.board = [EnvMinion("a", "A", 4, 100, 100, [], [])]
    before = p.strength()
    env._end_of_turn_scaling_ratio(p)
    assert p.strength() > before


def test_paired_comparison_improves_late_turns_smoke():
    rows_v1 = run_fidelity_rollouts(4, seed=0, scaling_mode="ratio")
    rows_v11 = run_fidelity_rollouts(4, seed=0, scaling_mode="residual")
    paired = paired_turn_comparison(
        per_lobby_turn_means(rows_v1), per_lobby_turn_means(rows_v11))
    assert paired["14"]["v1_1_mean_ratio"] < paired["14"]["v1_mean_ratio"]


def test_freeze_thresholds_before_treatment():
    rows = run_fidelity_rollouts(8, seed=0, scaling_mode="ratio")
    thresholds = freeze_success_thresholds(per_lobby_turn_means(rows), seed=0)
    assert thresholds["gates"]["turn_14_primary_max_ratio"] > 0
