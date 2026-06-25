"""Train the economy value net on self-play lobbies.

  python -m ml.train_econ --lobbies 4000 --epochs 30

Generates placement-labeled economy trajectories from the 8-player simulator and
fits econ-state -> expected placement. Saves ml/econ_value.pt.
"""

import argparse
import os

from .econ_env import generate
from .econ_value import train, EconValue

_OUT = os.path.join(os.path.dirname(__file__), "econ_value.pt")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the economy value net")
    p.add_argument("--lobbies", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default=_OUT)
    a = p.parse_args(argv)

    print(f"Simulating {a.lobbies} lobbies…")
    X, y = generate(a.lobbies)
    print(f"{len(X)} decision points.")
    model, hist = train(X, y, epochs=a.epochs)
    print(f"\nval MAE {hist['val_mae']:.3f} placements · val Pearson r {hist['val_r']:.3f}")
    EconValue(model).save(a.out)
    print(f"Saved -> {a.out}")

    # Qualitative sanity: same turn/tier, vary HP and board-vs-curve.
    ev = EconValue(model)
    base = dict(turn=8, tier=4, strength=300, ratio=1.0, hp=25, players_left=4)
    print("\nsanity (lower placement = better):")
    print(f"  on-curve, healthy : {ev.predict(**base):.2f}")
    print(f"  ahead of curve    : {ev.predict(**{**base, 'ratio':1.8, 'strength':540}):.2f}")
    print(f"  behind curve      : {ev.predict(**{**base, 'ratio':0.5, 'strength':150}):.2f}")
    print(f"  low HP            : {ev.predict(**{**base, 'hp':5}):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
