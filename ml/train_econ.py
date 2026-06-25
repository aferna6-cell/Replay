"""Train the economy value net on self-play lobbies.

  python -m ml.train_econ --lobbies 4000 --epochs 30

Generates placement-labeled economy trajectories from the 8-player simulator and
fits econ-state -> expected placement. Saves ml/econ_value.pt.
"""

import argparse
import os

from .econ_env import generate
from .econ_value import train, EconValue, reliability

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

    # Calibration: does each prediction bin hit its actual placement?
    Xc, yc = generate(max(200, a.lobbies // 10), seed=90210)
    print("\ncalibration (pred-bin -> actual mean placement; want ≈ bin center):")
    for lo, n, actual in reliability(model, Xc, yc):
        print(f"   {lo}-{lo+1}  n={n:5d}  actual {actual:.2f}")

    # Qualitative sanity at a realistic turn-8 lobby (8 alive).
    ev = EconValue(model)
    base = dict(turn=8, tier=4, strength=300, ratio=1.0, hp=25, players_left=8)
    print("\nsanity (lower placement = better · top4 = P(finish≤4)):")
    for label, kw in [("on-curve, healthy", {}),
                      ("ahead of curve", {"ratio": 1.8, "strength": 540}),
                      ("behind curve", {"ratio": 0.5, "strength": 150}),
                      ("low HP", {"hp": 5})]:
        q = {**base, **kw}
        print(f"  {label:18s}: {ev.predict(**q):.2f}   top4 {ev.top4(**q):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
