"""Self-play policy tests. Env wiring is stdlib; training needs torch and skips
without it."""

import pytest

from hsbg_coach.pace import load_pace
from ml.econ_env import simulate_lobby

PACE = load_pace()


def test_deciders_drive_players_and_record_actions():
    # A trivial decider that always tempos; players using it record actions.
    def always_tempo(feat):
        return "tempo", 1
    deciders = [always_tempo] + [None] * 7
    players = simulate_lobby(PACE, seed=1, deciders=deciders)
    p0 = players[0]
    assert p0.actions and all(a == 1 for _, a in p0.actions)   # recorded the action
    assert all(not p.actions for p in players[1:])             # heuristics record none


def test_reward_monotonic():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from ml.econ_policy import reward
    assert reward(1) > reward(4) > reward(8)
    assert reward(1) == pytest.approx(1.0)


def test_policy_learns_to_beat_random_and_levels_when_behind():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from ml.econ_policy import train, evaluate, recommend_intent
    net, info = train(iters=45, lobbies=48, verbose=False)
    # Beats the heuristic field (4.5 = even); full training reaches ~3.9.
    assert info["final"] < 4.4
    # Sensible behavior: more inclined to level when under-tiered/behind than when
    # already high-tier and ahead.
    behind = recommend_intent(7, 2, 120, 0.5, 25, 6, net)
    ahead = recommend_intent(7, 6, 400, 1.6, 25, 6, net)

    def p_level(turn, tier, strength, ratio, hp, left):
        import torch, torch.nn.functional as F
        from ml.econ_env import features
        from ml.econ_policy import ACTIONS
        with torch.no_grad():
            probs = F.softmax(net(torch.tensor(
                [features(turn, tier, strength, ratio, hp, left)], dtype=torch.float32)), -1)[0]
        return float(probs[ACTIONS.index("level")])
    assert p_level(7, 2, 120, 0.5, 25, 6) > p_level(7, 6, 400, 1.6, 25, 6)
