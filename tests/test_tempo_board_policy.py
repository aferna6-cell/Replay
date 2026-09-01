"""Tests for Phase 2H tempo-aware board management policy (2h_v3)."""

import random

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
    A_SELL0,
    greedy_policy,
    tempo_board_greedy_policy,
)
from hsbg_coach.tempo_board_policy import (
    LAMBDA_BUILD_CANDIDATES,
    METHODOLOGY_VERSION,
    PHASE_2H_CONFIRM_SEED,
    TempoBoardGreedyPolicy,
    _deploy_build_gain,
    _shop_build_gain,
    policies_for_lobby,
)
from ml.fidelity_phase_2h import (
    assert_trace_lobby_integrity,
    run_fidelity_rollouts_tempo,
    run_traced_rollouts_policy_list,
    run_traced_rollouts_tempo,
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


def _mask(*, n_shop=0, sell_slots=None, play_slots=None, buy_slots=None):
    mask = [False] * 28
    slots = buy_slots if buy_slots is not None else list(range(n_shop))
    for i in slots:
        mask[A_BUY0 + i] = True
    if sell_slots:
        for i in sell_slots:
            mask[A_SELL0 + i] = True
    if play_slots:
        for i in play_slots:
            mask[A_PLAY0 + i] = True
    mask[27] = True
    return mask


def _full_board_fillers():
    return [
        {"name": "Filler", "attack": 1, "health": 1, "tribes": ["Neutral"]},
        {"name": "X", "attack": 2, "health": 2, "tribes": []},
        {"name": "Y", "attack": 2, "health": 2, "tribes": []},
        {"name": "Z", "attack": 2, "health": 2, "tribes": []},
        {"name": "W", "attack": 2, "health": 2, "tribes": []},
        {"name": "V", "attack": 2, "health": 2, "tribes": []},
    ]


def test_methodology_version():
    assert METHODOLOGY_VERSION == "2h_v3"
    assert LAMBDA_BUILD_CANDIDATES == (4, 8, 12)
    assert PHASE_2H_CONFIRM_SEED == 6000


def test_unseeded_matches_greedy():
    rng = random.Random(0)
    obs = _obs([{"name": "Z", "attack": 5, "health": 5, "tribes": []}])
    mask = _mask(n_shop=1)
    assert tempo_board_greedy_policy(obs, mask, rng) == greedy_policy(obs, mask, rng)


def test_hand_core_has_deploy_gain_when_on_board_only():
    from hsbg_coach.build_path import infer_target
    board = [{"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]}]
    fit = infer_target(board)
    on_board = {"Fauna Whisperer"}
    held = on_board | {"Balinda Stonehearth"}
    shop_gain = _shop_build_gain("Balinda Stonehearth", fit, 4, held)
    deploy_gain = _deploy_build_gain("Balinda Stonehearth", fit, 4, on_board)
    assert shop_gain == 0.0
    assert deploy_gain > 0.0


def test_trace_lobby_integrity_multi_lobby():
    traces = run_traced_rollouts_tempo(3, seed=6000, lambda_build=8.0)
    assert_trace_lobby_integrity(traces, 3)
    assert len({p["lobby"] for p in traces["player_finals"]}) == 3


def test_sell_then_deploy_completes_hand_core():
    rng = random.Random(0)
    policy = TempoBoardGreedyPolicy(12.0)
    board = [
        {"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]},
        *_full_board_fillers(),
    ]
    hand = [{"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Naga"]}]
    obs1 = _obs([], board=board, hand=hand, tier=4)
    mask1 = _mask(sell_slots=list(range(7)), play_slots=[0])
    a1 = policy(obs1, mask1, rng)
    assert a1 == A_SELL0 + 1
    assert policy.pending is not None
    assert policy.pending.source == "hand"
    assert policy.pending.stage == "play"
    assert policy.pending.candidate_name == "Balinda Stonehearth"
    assert policy.stats.replacement_sells == 1
    assert policy.stats.target_core_buys == 0

    board2 = [b for i, b in enumerate(board) if i != 1]
    obs2 = _obs([], board=board2, hand=hand, tier=4)
    mask2 = _mask(play_slots=[0])
    a2 = policy(obs2, mask2, rng)
    assert a2 == A_PLAY0
    assert policy.pending is None
    assert policy.stats.target_core_deploys == 1
    assert policy.stats.compound_transitions_completed == 1


def test_shop_full_board_sell_buy_play_completes_intended_card():
    rng = random.Random(0)
    policy = TempoBoardGreedyPolicy(12.0)
    board = [
        {"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]},
        *_full_board_fillers(),
    ]
    shop_core = {"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Naga"]}
    obs1 = _obs([shop_core], board=board, tier=4, gold=10)
    mask1 = _mask(buy_slots=[0], sell_slots=list(range(7)))
    a1 = policy(obs1, mask1, rng)
    assert a1 == A_SELL0 + 1
    assert policy.pending is not None
    assert policy.pending.source == "shop"
    assert policy.pending.stage == "buy"
    assert policy.pending.candidate_name == "Balinda Stonehearth"

    board2 = [b for i, b in enumerate(board) if i != 1]
    obs2 = _obs([shop_core], board=board2, tier=4, gold=10)
    mask2 = _mask(buy_slots=[0])
    a2 = policy(obs2, mask2, rng)
    assert a2 == A_BUY0
    assert policy.pending is not None
    assert policy.pending.stage == "play"
    assert policy.stats.target_core_buys == 1

    hand = [shop_core]
    obs3 = _obs([], board=board2, hand=hand, tier=4)
    mask3 = _mask(play_slots=[0])
    a3 = policy(obs3, mask3, rng)
    assert a3 == A_PLAY0
    assert policy.pending is None
    assert policy.stats.target_core_deploys == 1
    assert policy.stats.compound_transitions_completed == 1
    assert policy.stats.compound_transitions_abandoned == 0


def test_fresh_policy_state_per_lobby():
    _, policies = run_fidelity_rollouts_tempo(3, seed=6000, lambda_build=8.0)
    assert len(policies) == 24
    for p in policies:
        assert p.pending is None


def test_no_pending_survives_into_new_lobby():
    from hsbg_coach.tempo_board_policy import PendingTransition
    stale = TempoBoardGreedyPolicy(12.0)
    stale.pending = PendingTransition(
        source="hand", stage="play", candidate_slot=0,
        candidate_name="Stale", replacement_slot=0,
        net_value=1.0, build_gain=1.0, raw_sacrifice=0.0)
    fresh = policies_for_lobby(12.0, 8)
    assert all(p.pending is None for p in fresh)
    assert stale.pending is not None


def test_non_core_buy_does_not_increment_target_core_buys():
    rng = random.Random(0)
    policy = TempoBoardGreedyPolicy(12.0)
    board = [{"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]}]
    shop = [{"name": "Plain Minion", "attack": 10, "health": 10, "tribes": []}]
    obs = _obs(shop, board=board, tier=4)
    mask = _mask(buy_slots=[0])
    action = policy(obs, mask, rng)
    assert action == A_BUY0
    assert policy.stats.tempo_selected_buys == 1
    assert policy.stats.target_core_buys == 0


def test_non_core_play_does_not_increment_target_core_deploys():
    rng = random.Random(0)
    policy = TempoBoardGreedyPolicy(12.0)
    board = [{"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]}]
    hand = [{"name": "Plain Minion", "attack": 10, "health": 10, "tribes": []}]
    obs = _obs([], board=board, hand=hand, tier=4)
    mask = _mask(play_slots=[0])
    action = policy(obs, mask, rng)
    assert action == A_PLAY0
    assert policy.stats.tempo_selected_deploys == 1
    assert policy.stats.target_core_deploys == 0


def test_transition_accounting_balances():
    rng = random.Random(0)
    policy = TempoBoardGreedyPolicy(12.0)
    board = [
        {"name": "Fauna Whisperer", "attack": 2, "health": 2, "tribes": ["Naga"]},
        *_full_board_fillers(),
    ]
    hand = [{"name": "Balinda Stonehearth", "attack": 4, "health": 4, "tribes": ["Naga"]}]
    obs1 = _obs([], board=board, hand=hand, tier=4)
    policy(obs1, _mask(sell_slots=list(range(7)), play_slots=[0]), rng)
    board2 = [b for i, b in enumerate(board) if i != 1]
    policy(_obs([], board=board2, hand=hand, tier=4), _mask(play_slots=[0]), rng)
    s = policy.stats
    assert s.compound_transitions_planned == 1
    assert s.compound_transitions_completed == 1
    assert s.compound_transitions_abandoned == 0
    assert s.compound_transitions_completed + s.compound_transitions_abandoned <= s.compound_transitions_planned + 1


def test_policy_stats_not_doubled_across_runs():
    from ml.fidelity_phase_2h import _run_tempo_arm
    arm1 = _run_tempo_arm(
        2, PHASE_2H_CONFIRM_SEED, 8.0, "t1", collect_policy_stats=True)
    arm2 = _run_tempo_arm(
        2, PHASE_2H_CONFIRM_SEED, 8.0, "t2", collect_policy_stats=True)
    s1 = arm1["policy_stats"]
    s2 = arm2["policy_stats"]
    assert s1["tempo_selected_buys"] == s2["tempo_selected_buys"]
    assert s1["compound_transitions_planned"] == s2["compound_transitions_planned"]


def test_phase_2h_smoke():
    from ml.fidelity_phase_2h import _run_single_policy_arm, _run_tempo_arm
    greedy = _run_single_policy_arm(
        2, PHASE_2H_CONFIRM_SEED, greedy_policy, "greedy")
    treatment = _run_tempo_arm(
        2, PHASE_2H_CONFIRM_SEED, 8.0, "tempo",
        greedy_baseline_traces=greedy["traces"])
    assert treatment["methodology_version"] == METHODOLOGY_VERSION
    assert_trace_lobby_integrity(treatment["traces"], 2)
    seeded = treatment["mechanism"]["seeded_current_target"]
    assert seeded["legally_buyable_exposures"] >= 0
