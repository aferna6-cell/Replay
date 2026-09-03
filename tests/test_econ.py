"""Economy self-play simulator + value net. Env tests are stdlib; training needs
torch and skips without it."""

import pytest

from hsbg_coach.pace import load_pace
from ml.econ_env import simulate_lobby, generate, features, lobby_examples

PACE = load_pace()


def test_lobby_assigns_unique_placements():
    players = simulate_lobby(PACE, seed=3)
    places = sorted(p.placement for p in players)
    assert places == list(range(1, 9))          # 1..8, each once


def test_winner_is_stronger_than_loser():
    players = simulate_lobby(PACE, seed=5)
    byplace = {p.placement: p for p in players}
    assert byplace[1].strength > byplace[8].strength


def test_features_length_and_dataset():
    assert len(features(8, 4, 300, 1.0, 25, 4)) == 6
    X, y = generate(20, PACE)
    assert len(X) == len(y) and X
    assert all(1 <= p <= 8 for p in y)


def test_value_net_learns_and_orders_states():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from ml.econ_value import train, EconValue
    X, y = generate(1500, PACE)
    model, hist = train(X, y, epochs=20, verbose=False)
    assert hist["val_r"] > 0.4                   # learns real signal from sim
    ev = EconValue(model)
    healthy = ev.predict(8, 4, 400, 1.4, 30, 4)
    behind = ev.predict(8, 4, 120, 0.4, 8, 4)
    assert behind > healthy                      # behind+low-HP finishes worse
