"""Generate (board_a, board_b) -> win/tie/loss training data from the combat sim.

This is the whole point: the simulator is an *unlimited labeled-data generator*.
We sample random boards, run the sim to get soft labels (win/tie/loss fractions),
and train a net to approximate the simulator — fast, differentiable, and usable
inside the RL agent's lookahead. No human game logs required.
"""

import random
import numpy as np

from hsbg_coach.sim import Combatant, simulate
from .encode import board_to_array, MAX_MINIONS, NUM_FEATURES


def random_board(rng: random.Random):
    """A plausible-ish random board: 1-7 minions, varied stats + occasional
    keywords. Breadth matters more than realism here — we want the net to learn
    the simulator's function across the space."""
    n = rng.randint(1, MAX_MINIONS)
    out = []
    for _ in range(n):
        out.append(Combatant(
            attack=rng.randint(0, 15),
            health=rng.randint(1, 15),
            divine_shield=rng.random() < 0.15,
            taunt=rng.random() < 0.20,
            poisonous=rng.random() < 0.08,
            reborn=rng.random() < 0.10,
            windfury=rng.random() < 0.08,
            cleave=rng.random() < 0.05,
        ))
    return out


def make_dataset(n: int, runs: int = 80, seed: int = 0):
    """Return arrays: XA, MA, XB, MB (features+masks) and Y[n,3] = win/tie/loss."""
    rng = random.Random(seed)
    XA = np.zeros((n, MAX_MINIONS, NUM_FEATURES), dtype=np.float32)
    XB = np.zeros((n, MAX_MINIONS, NUM_FEATURES), dtype=np.float32)
    MA = np.zeros((n, MAX_MINIONS), dtype=np.float32)
    MB = np.zeros((n, MAX_MINIONS), dtype=np.float32)
    Y = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        a, b = random_board(rng), random_board(rng)
        r = simulate(a, b, runs=runs, seed=rng.randint(0, 1 << 30))
        XA[i], MA[i] = board_to_array(a)
        XB[i], MB[i] = board_to_array(b)
        Y[i] = [r.win_pct, r.tie_pct, r.loss_pct]
    return XA, MA, XB, MB, Y
