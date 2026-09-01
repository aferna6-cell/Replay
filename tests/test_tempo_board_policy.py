"""Tests for Phase 2H tempo-aware board management policy."""

import random

from hsbg_coach.bg_env import (
    A_PLAY0,
    A_SELL0,
    greedy_policy,
    tempo_board_greedy_policy,
)
from hsbg_coach.tempo_board_policy import (
    LAMBDA_BUILD_CANDIDATES,
    PHASE_2H_CONFIRM_SEED,
    TempoBoardGreedyPolicy,
    policies_for_lobby,
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


def _mask(*, n_shop=0, sell_slots=None, play_slots=None):
    mask = [False] * 28
    for i in range(n_shop):
        mask[7 + i] = True
    if sell_slots:
        for i in sell_slots:
            mask[A_SELL0 + i] = True
    if play_slots:
        for i in play_slots:
            mask[A_PLAY0 + i] = True
    mask[27] = True
    return mask


def test_lambda_candidates_frozen():
    assert LAMBDA_BUILD_CANDIDATES == (4, 8, 12)


def test_unseeded_matches_greedy():
    rng = random.Random(0)
    obs = _obs(
        [{"name": "Z", "attack": 5, "health": 5, "tribes": []}],
        board=[],
        hand=[],
    )
    mask = _mask(n_shop=1)
    assert tempo_board_greedy_policy(obs, mask, rng) == greedy_policy(obs, mask, rng)


def test_seeded_deploy_sells_non_core_not_first_hand_blindly():
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
    hand = [
        {"name": "Junk", "attack": 1, "health": 1, "tribes": []},
        {"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Naga"]},
    ]
    obs = _obs([], board=board, hand=hand, tier=4)
    mask = _mask(sell_slots=[1], play_slots=[0, 1])
    policy = TempoBoardGreedyPolicy(8.0)
    action = policy(obs, mask, rng)
    assert action != A_PLAY0
    assert action == A_SELL0 + 1 or action == A_PLAY0 + 1


def test_policies_for_lobby_independent_instances():
    ps = policies_for_lobby(8.0, 8)
    assert len(ps) == 8
    assert ps[0] is not ps[1]


def test_phase_2h_smoke():
    from ml.fidelity_phase_2h import _run_single_policy_arm, _run_tempo_arm
    greedy = _run_single_policy_arm(
        2, PHASE_2H_CONFIRM_SEED, greedy_policy, "greedy")
    treatment = _run_tempo_arm(
        2, PHASE_2H_CONFIRM_SEED, 8.0, "tempo",
        greedy_baseline_traces=greedy["traces"])
    assert treatment["mechanism"]["seeded_current_target"] is not None
    assert treatment["lifecycle"]["n_fulfilled_purchases"] >= 0
