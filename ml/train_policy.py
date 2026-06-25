"""Train the self-play economy policy by REINFORCE.

  python -m ml.train_policy --iters 60 --lobbies 64

Saves ml/econ_policy.pt and reports whether the learned policy beats the heuristic
field, plus the strategy it learned (P(level) across game states).
"""

import argparse
import os

import torch
import torch.nn.functional as F

from .econ_policy import train, save, ACTIONS
from .econ_env import features

_OUT = os.path.join(os.path.dirname(__file__), "econ_policy.pt")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the self-play economy policy")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lobbies", type=int, default=64)
    p.add_argument("--out", default=_OUT)
    a = p.parse_args(argv)

    net, info = train(iters=a.iters, lobbies=a.lobbies)
    print(f"\nFinal: vs-heuristic-field avg placement {info['final']:.2f}  "
          f"(4.50 = even; lower = the learned policy wins)")
    save(net, a.out)
    print(f"Saved -> {a.out}")

    # What strategy did it learn? P(level) by tier-vs-curve at a mid-game turn.
    print("\nLearned strategy — P(level up) at turn 7, hp 25, 6 alive:")
    for tier, ratio, label in [(2, 0.5, "under-tier, behind"),
                               (3, 1.0, "on-tier, on-curve"),
                               (5, 1.6, "high-tier, ahead")]:
        with torch.no_grad():
            probs = F.softmax(net(torch.tensor(
                [features(7, tier, 140, ratio, 25, 6)], dtype=torch.float32)), -1)[0]
        plevel = float(probs[ACTIONS.index("level")])
        print(f"  tier {tier} {label:22s}: P(level) {plevel:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
