"""Deep combat-value-net tests. Skipped entirely when torch/numpy aren't
installed, so the core stdlib-only suite stays green everywhere."""

import random
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ml.encode import board_to_array, MAX_MINIONS, NUM_FEATURES
from ml.data import random_board
from ml.model import CombatValueNet
from ml.train import train
from hsbg_coach.sim import Combatant as C


def test_encode_shapes_and_mask():
    rng = random.Random(0)
    b = random_board(rng)
    arr, mask = board_to_array(b)
    assert arr.shape == (MAX_MINIONS, NUM_FEATURES)
    assert mask.sum() == min(len(b), MAX_MINIONS)


def test_model_output_shape():
    m = CombatValueNet()
    xa = torch.zeros(4, MAX_MINIONS, NUM_FEATURES)
    ma = torch.ones(4, MAX_MINIONS)
    assert m(xa, ma, xa, ma).shape == (4, 3)


def test_predict_probs_are_a_distribution():
    m = CombatValueNet()
    xa = torch.randn(3, MAX_MINIONS, NUM_FEATURES)
    ma = torch.ones(3, MAX_MINIONS)
    p = m.predict_probs(xa, ma, xa, ma)
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)


@pytest.fixture(scope="module")
def trained():
    # One small training run shared across the learning tests.
    model, metrics = train(n_train=1500, n_val=400, runs=40, epochs=12,
                           verbose=False)
    return model, metrics


def test_net_learns_the_simulator(trained):
    _, metrics = trained
    assert metrics["win_mae"] < 0.15        # within 15 percentage points of sim
    assert metrics["outcome_acc"] > 0.80


def test_strong_board_predicts_win(trained):
    model, _ = trained
    a, ma = board_to_array([C(12, 12), C(12, 12)])
    b, mb = board_to_array([])              # empty enemy
    p = model.predict_probs(
        torch.from_numpy(a[None]), torch.from_numpy(ma[None]),
        torch.from_numpy(b[None]), torch.from_numpy(mb[None]))
    assert p[0, 0] > 0.6                     # high win probability
