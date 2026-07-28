"""The deep board-evaluation net: board + hero -> expected finish (1st–8th).

A small MLP over the board features (`board_features`) with a learned hero
embedding. It's the "brain" the move-recommender will query: to compare actions
(buy / sell / roll / reposition), score the resulting boards and prefer the one
with the lower expected placement.

Trained by `train_eval_net.py`; persisted as a .pt (state + meta) so inference
needs no dataset. Lower predicted value = stronger board.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .board_features import board_vector, feature_dim


class EvalNet(nn.Module):
    def __init__(self, n_dense: int, n_heroes: int, hero_dim: int = 8,
                 hidden: int = 64):
        super().__init__()
        self.hero_emb = nn.Embedding(n_heroes, hero_dim)
        self.net = nn.Sequential(
            nn.Linear(n_dense + hero_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, hero):
        z = torch.cat([x, self.hero_emb(hero)], dim=1)
        return self.net(z).squeeze(-1)


def train(X: np.ndarray, hero: np.ndarray, y: np.ndarray, n_heroes: int,
          epochs: int = 40, lr: float = 0.005, batch: int = 256,
          weight_decay: float = 1e-4, val: Optional[Tuple] = None,
          seed: int = 0, verbose: bool = True) -> Tuple[EvalNet, dict]:
    import copy
    torch.manual_seed(seed)
    model = EvalNet(X.shape[1], n_heroes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.MSELoss()
    Xt, ht, yt = (torch.from_numpy(X), torch.from_numpy(hero), torch.from_numpy(y))
    n = Xt.shape[0]
    history = {"val_mae": None, "val_r": None}
    best_mae, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx], ht[idx]), yt[idx])
            loss.backward()
            opt.step()
        if val is not None:
            mae, r = evaluate(model, *val)
            if mae < best_mae:                      # keep the best-val snapshot
                best_mae, best_state = mae, copy.deepcopy(model.state_dict())
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"epoch {ep:2d}  val MAE {mae:.3f}  val r {r:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    if val is not None:
        history["val_mae"], history["val_r"] = evaluate(model, *val)
    return model, history


def evaluate(model: EvalNet, X, hero, y) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X), torch.from_numpy(hero)).numpy()
    mae = float(np.mean(np.abs(pred - y)))
    if pred.std() < 1e-9 or y.std() < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(pred, y)[0, 1])
    return mae, r


class EvalModel:
    """Inference wrapper: load a trained net and score boards."""

    def __init__(self, model: EvalNet, hero_stoi: Dict[str, int],
                 stats: Tuple[np.ndarray, np.ndarray], emb: Dict[str, List[float]],
                 with_context: bool = False):
        self.model = model
        self.hero_stoi = hero_stoi
        self.mean, self.std = stats
        self.emb = emb
        self.with_context = with_context

    def predict(self, minions: List[Dict], hero_id: str = "UNKNOWN",
                state=None) -> Dict:
        if self.with_context:
            from .board_features import full_vector
            raw = full_vector(minions, self.emb, state)
        else:
            raw = board_vector(minions, self.emb)
        x = (raw - self.mean) / self.std
        hidx = self.hero_stoi.get(hero_id, self.hero_stoi.get("UNKNOWN", 0))
        self.model.eval()
        with torch.no_grad():
            placement = float(self.model(
                torch.from_numpy(x.astype(np.float32)).unsqueeze(0),
                torch.tensor([hidx])).item())
        placement = max(1.0, min(8.0, placement))
        return {"placement": placement, "equity": (8.0 - placement) / 7.0}

    def save(self, path: str):
        torch.save({
            "state": self.model.state_dict(),
            "hero_stoi": self.hero_stoi,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "n_dense": self.model.net[0].in_features - self.model.hero_emb.embedding_dim,
            "with_context": self.with_context,
        }, path)
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump({"heroes": len(self.hero_stoi), "n_dense": self.mean.shape[0],
                       "with_context": self.with_context}, fh)

    @classmethod
    def load(cls, path: str, emb: Dict[str, List[float]]) -> "EvalModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = EvalNet(blob["n_dense"], len(blob["hero_stoi"]))
        model.load_state_dict(blob["state"])
        stats = (np.asarray(blob["mean"]), np.asarray(blob["std"]))
        return cls(model, blob["hero_stoi"], stats, emb,
                   with_context=bool(blob.get("with_context", False)))
