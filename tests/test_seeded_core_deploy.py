"""Tests for Phase 2G seeded-core deployment stress policy."""

import random

from hsbg_coach.bg_env import (
    A_PLAY0, A_SELL0, seeded_core_deploy_stress_greedy_policy,
    seeded_core_stress_greedy_policy,
)
from hsbg_coach.seeded_core_deploy_policy import (
    PHASE_2G_EVAL_SEED_BASE,
    seeded_core_deploy_sell_action,
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


def _mask(*, n_shop=0, sell_slots=None, can_play=False):
    mask = [False] * 28
    for i in range(n_shop):
        mask[7 + i] = True
    if sell_slots:
        for i in sell_slots:
            mask[A_SELL0 + i] = True
    if can_play:
        mask[A_PLAY0] = True
    mask[27] = True
    return mask


def test_frozen_eval_seed_base():
    assert PHASE_2G_EVAL_SEED_BASE == 2000


def test_deploy_sell_weakest_non_core_when_hand_core_stuck():
    board = [
        {"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]},
        {"name": "Filler", "attack": 1, "health": 1, "tribes": ["Neutral"]},
        {"name": "Big", "attack": 5, "health": 5, "tribes": ["Neutral"]},
        {"name": "B", "attack": 2, "health": 2, "tribes": []},
        {"name": "C", "attack": 2, "health": 2, "tribes": []},
        {"name": "D", "attack": 2, "health": 2, "tribes": []},
        {"name": "E", "attack": 2, "health": 2, "tribes": []},
    ]
    hand = [{"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Naga"]}]
    obs = _obs([], board=board, hand=hand, tier=4)
    mask = _mask(sell_slots=list(range(7)))
    action = seeded_core_deploy_sell_action(obs, mask)
    assert action == A_SELL0 + 1  # weakest non-core is Filler


def test_deploy_sell_skips_when_only_cores_on_board():
    board = [{"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]}] * 7
    hand = [{"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Naga"]}]
    obs = _obs([], board=board, hand=hand, tier=4)
    mask = _mask(sell_slots=list(range(7)))
    assert seeded_core_deploy_sell_action(obs, mask) is None


def test_deploy_policy_sells_before_play_when_board_full():
    rng = random.Random(0)
    board = [
        {"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]},
        {"name": "Filler", "attack": 1, "health": 1, "tribes": ["Neutral"]},
        {"name": "X", "attack": 2, "health": 2, "tribes": []},
        {"name": "Y", "attack": 2, "health": 2, "tribes": []},
        {"name": "Z", "attack": 2, "health": 2, "tribes": []},
        {"name": "W", "attack": 2, "health": 2, "tribes": []},
        {"name": "V", "attack": 2, "health": 2, "tribes": []},
    ]
    hand = [{"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Naga"]}]
    obs = _obs([], board=board, hand=hand, tier=4)
    mask = _mask(sell_slots=[1], can_play=True)
    action = seeded_core_deploy_stress_greedy_policy(obs, mask, rng)
    assert action == A_SELL0 + 1


def test_deploy_policy_matches_stress_when_no_deploy_needed():
    rng = random.Random(1)
    obs = _obs(
        [{"name": "Z", "attack": 1, "health": 1, "tribes": []}],
        board=[{"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]}],
        hand=[],
        tier=4,
    )
    mask = _mask(n_shop=1)
    assert seeded_core_deploy_stress_greedy_policy(obs, mask, rng) == (
        seeded_core_stress_greedy_policy(obs, mask, rng))


def test_phase_2g_smoke():
    from tests.ml_testutil import require_ml
    require_ml()  # contract builder records a Torch/NumPy runtime fingerprint
    from ml.fidelity_phase_2g import run_phase_2g
    result = run_phase_2g(
        lobbies=2, seed=PHASE_2G_EVAL_SEED_BASE,
        out_dir="/tmp/sim_fidelity_phase_2g_test",
        require_clean_tree=False)
    assert result["decision"]["decision_branch"]
    assert result["evaluation_seed_base"] == 2000


def test_phase_2g_decision_macro_guard():
    from ml.phase_2g_decision import evaluate_phase_2g_decision
    ctrl_m = {"seeded_current_target": {"reached_2_core": 0},
              "sim_final_winner_coverage_mean": 0.01}
    treat_m = {"seeded_current_target": {"reached_2_core": 10},
               "sim_final_winner_coverage_mean": 0.05}
    lc = {"n_fulfilled_purchases": 30, "funnel": {"played": 20}}
    bad_macro = {"turn_14_stats_ratio_delta": 1.0, "game_length_delta": 0.0}
    out = evaluate_phase_2g_decision(ctrl_m, treat_m, lc, lc, bad_macro)
    assert out["decision_branch"] == "macro_regression"
