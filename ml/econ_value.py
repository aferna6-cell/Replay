"""Learned trajectory value: economy-state -> expected final placement.

Trained on the placement-labeled trajectories from `econ_env` self-play. This is
the learned replacement for the hand-tuned trajectory term in the multi-turn
planner / whole-game value — it captures how (turn, tier, board-strength-vs-curve,
HP, players-left) map to a *finish*, learned from emergent lobby outcomes rather
than weights we picked.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .econ_env import features


class EconValueNet(nn.Module):
    def __init__(self, n_in: int = 6, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(X, y, epochs: int = 30, lr: float = 0.01, batch: int = 512,
          val_frac: float = 0.15, seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = Xt.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(n * val_frac))
    vi, ti = perm[:n_val], perm[n_val:]
    model = EconValueNet(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        p = ti[torch.randperm(ti.shape[0])]
        for i in range(0, p.shape[0], batch):
            idx = p[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            mae, r = _evaluate(model, Xt[vi], yt[vi])
            print(f"epoch {ep:2d}  val MAE {mae:.3f}  val r {r:.3f}")
    mae, r = _evaluate(model, Xt[vi], yt[vi])
    return model, {"val_mae": mae, "val_r": r}


def _evaluate(model, Xv, yv) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        pred = model(Xv).numpy()
    yv = yv.numpy()
    mae = float(np.mean(np.abs(pred - yv)))
    r = 0.0 if pred.std() < 1e-9 else float(np.corrcoef(pred, yv)[0, 1])
    return mae, r


class EconValue:
    """Inference wrapper: economy state -> expected placement (1..8, lower better)."""

    def __init__(self, model: EconValueNet):
        self.model = model

    def predict(self, turn, tier, strength, ratio, hp, players_left=4) -> float:
        x = torch.tensor([features(turn, tier, strength, ratio, hp, players_left)],
                         dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return max(1.0, min(8.0, float(self.model(x).item())))

    def save(self, path: str):
        torch.save({"state": self.model.state_dict(),
                    "n_in": self.model.net[0].in_features}, path)

    @classmethod
    def load(cls, path: str) -> "EconValue":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = EconValueNet(blob["n_in"])
        m.load_state_dict(blob["state"])
        return cls(m)
