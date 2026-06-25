"""Learned trajectory value: economy-state -> placement *distribution*.

Trained on placement-labeled self-play lobbies from `econ_env`. Predicts the full
P(finish = 1..8) via an 8-way head with cross-entropy — a proper scoring rule, so
the probabilities are calibrated, not just a point estimate. Exposes:

  * `predict`     -> expected placement  Σ k·P(k)   (the scalar the planner uses)
  * `predict_dist`-> the 8 probabilities
  * `top4`        -> P(finish ≤ 4)       (a useful overlay number)

This replaces the hand-tuned trajectory term in the whole-game value with a value
learned from emergent lobby outcomes.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .econ_env import features

PLACES = 8


class EconValueNet(nn.Module):
    def __init__(self, n_in: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, PLACES),          # logits over placements 1..8
        )

    def forward(self, x):
        return self.net(x)


def train(X, y, epochs: int = 30, lr: float = 0.01, batch: int = 512,
          val_frac: float = 0.15, seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor([p - 1 for p in y], dtype=torch.long)   # 0..7
    n = Xt.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    nv = max(1, int(n * val_frac))
    vi, ti = perm[:nv], perm[nv:]
    model = EconValueNet(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        p = ti[torch.randperm(ti.shape[0])]
        for i in range(0, p.shape[0], batch):
            idx = p[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            mae, r = _evaluate(model, Xt[vi], yt[vi])
            print(f"epoch {ep:2d}  val MAE {mae:.3f}  val r {r:.3f}")
    mae, r = _evaluate(model, Xt[vi], yt[vi])
    return model, {"val_mae": mae, "val_r": r}


def _expected(logits) -> np.ndarray:
    probs = F.softmax(logits, dim=-1).numpy()
    return probs @ np.arange(1, PLACES + 1)


def _evaluate(model, Xv, yv) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        pred = _expected(model(Xv))
    actual = yv.numpy() + 1
    mae = float(np.mean(np.abs(pred - actual)))
    r = 0.0 if pred.std() < 1e-9 else float(np.corrcoef(pred, actual)[0, 1])
    return mae, r


def reliability(model, X, y, bins: int = 7):
    """Per prediction-bin actual mean placement — the calibration check."""
    model.eval()
    with torch.no_grad():
        pred = _expected(model(torch.tensor(X, dtype=torch.float32)))
    y = np.asarray(y)
    out = []
    for lo in range(1, bins + 1):
        m = (pred >= lo) & (pred < lo + 1)
        if m.sum() > 10:
            out.append((lo, int(m.sum()), float(y[m].mean())))
    return out


class EconValue:
    def __init__(self, model: EconValueNet):
        self.model = model

    def _logits(self, turn, tier, strength, ratio, hp, players_left):
        x = torch.tensor([features(turn, tier, strength, ratio, hp, players_left)],
                         dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def predict(self, turn, tier, strength, ratio, hp, players_left=5) -> float:
        v = float(_expected(self._logits(turn, tier, strength, ratio, hp, players_left))[0])
        return max(1.0, min(8.0, v))

    def predict_dist(self, turn, tier, strength, ratio, hp, players_left=5) -> List[float]:
        return F.softmax(self._logits(turn, tier, strength, ratio, hp, players_left),
                         dim=-1).numpy()[0].tolist()

    def top4(self, turn, tier, strength, ratio, hp, players_left=5) -> float:
        return float(sum(self.predict_dist(turn, tier, strength, ratio, hp, players_left)[:4]))

    def save(self, path: str):
        torch.save({"state": self.model.state_dict(),
                    "n_in": self.model.net[0].in_features}, path)

    @classmethod
    def load(cls, path: str) -> "EconValue":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = EconValueNet(blob["n_in"])
        m.load_state_dict(blob["state"])
        return cls(m)
