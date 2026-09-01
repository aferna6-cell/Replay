"""Tests for residual end-of-turn scaling (Simulator v1.1)."""

from hsbg_coach.bg_env import BGEnv, EnvMinion, greedy_policy

from ml.fidelity_paired import (freeze_success_thresholds,
                                paired_turn_comparison, per_lobby_turn_means)
from ml.fidelity_metrics import run_fidelity_rollouts


def _player(env, idx=0):
    return env.players[idx]


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
