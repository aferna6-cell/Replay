"""The set-transformer board brain: minion tokens -> placement *distribution*.

Two structural upgrades over ml/eval_net.py:

  * **Attention over minions, not a mean-pooled bag.** Each minion is a token
    (see `tokens.py`); 2 self-attention layers model minion↔minion interaction
    directly — synergy emerges from the network instead of being hand-written
    (the architecture the RL spec §6b recommends). A context token (tavern
    tier / gold / HP / turn / lobby + learned hero embedding) conditions the
    whole read, so the same board is valued differently on turn 6 vs turn 12.
  * **Distributional head.** Placement is ordinal and high-variance; scalar MSE
    collapses everything toward 4.5. The net predicts P(finish 1st..8th) with a
    soft (two-hot interpolated) cross-entropy, and inference reports the
    expected placement plus the full distribution — top-1 equity and bust risk
    become readable, not just the mean.

`SetEvalModel` is a drop-in for `EvalModel` (same predict/save/load surface),
so `board_value.get_scorer()` picks it up transparently via `eval_scorer.py`.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .board_features import CONTEXT_DIM
from .tokens import board_tokens, token_dim, examples_to_arrays, default_byname

N_PLACES = 8
PLACES = torch.arange(1, N_PLACES + 1, dtype=torch.float32)


def soft_targets(y: np.ndarray) -> np.ndarray:
    """Float placements -> two-hot distributions (3.25 => 75% on 3, 25% on 4)."""
    y = np.clip(np.asarray(y, dtype=np.float32), 1.0, float(N_PLACES))
    lo = np.floor(y).astype(int)
    frac = y - lo
    t = np.zeros((len(y), N_PLACES), dtype=np.float32)
    for i, (l, f) in enumerate(zip(lo, frac)):
        l = min(l, N_PLACES)
        t[i, l - 1] = 1.0 - f
        if l < N_PLACES and f > 0:
            t[i, l] = f
    return t


class SetEvalNet(nn.Module):
    def __init__(self, tok_dim: int, n_heroes: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Linear(tok_dim, d_model)
        self.ctx = nn.Linear(CONTEXT_DIM, d_model)
        self.hero_emb = nn.Embedding(n_heroes, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, N_PLACES))

    def forward(self, tokens, mask, ctx, hero):
        b = tokens.shape[0]
        t = self.tok(tokens)                                    # [B, 7, d]
        c = (self.ctx(ctx) + self.hero_emb(hero)).unsqueeze(1)  # [B, 1, d]
        x = torch.cat([c, t], dim=1)
        ones = torch.ones(b, 1, device=tokens.device)
        full = torch.cat([ones, mask], dim=1)                   # ctx token always on
        h = self.encoder(x, src_key_padding_mask=(full < 0.5))
        pooled = (h * full.unsqueeze(-1)).sum(1) / full.sum(1, keepdim=True)
        return self.head(pooled)                                # logits [B, 8]

    def expected_placement(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) * PLACES.to(logits.device)).sum(-1)


def _soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def train(data: Tuple, n_heroes: int, epochs: int = 30, lr: float = 1e-3,
          batch: int = 256, weight_decay: float = 1e-4,
          val: Optional[Tuple] = None, seed: int = 0, verbose: bool = True
          ) -> Tuple[SetEvalNet, dict]:
    """data/val = (tokens, mask, ctx, hero, y). Returns (best model, history)."""
    import copy
    torch.manual_seed(seed)
    toks, mask, ctx, hero, y = [torch.from_numpy(a) for a in data]
    tgt = torch.from_numpy(soft_targets(y.numpy()))
    model = SetEvalNet(toks.shape[-1], n_heroes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = toks.shape[0]
    best_mae, best_state = float("inf"), None
    history = {"val_mae": None, "val_r": None}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            opt.zero_grad()
            logits = model(toks[ix], mask[ix], ctx[ix], hero[ix])
            loss = _soft_ce(logits, tgt[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val is not None:
            mae, r = evaluate(model, val)
            if mae < best_mae:
                best_mae, best_state = mae, copy.deepcopy(model.state_dict())
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"epoch {ep:2d}  val MAE {mae:.3f}  val r {r:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    if val is not None:
        history["val_mae"], history["val_r"] = evaluate(model, val)
    return model, history


def evaluate(model: SetEvalNet, data: Tuple) -> Tuple[float, float]:
    toks, mask, ctx, hero, y = [torch.from_numpy(a) for a in data]
    model.eval()
    with torch.no_grad():
        pred = model.expected_placement(model(toks, mask, ctx, hero)).numpy()
    mae = float(np.mean(np.abs(pred - y.numpy())))
    ya = y.numpy()
    if pred.std() < 1e-9 or ya.std() < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(pred, ya)[0, 1])
    return mae, r


class SetEvalModel:
    """Inference wrapper — same surface as EvalModel, plus the distribution."""

    def __init__(self, model: SetEvalNet, hero_stoi: Dict[str, int],
                 ctx_stats: Tuple[np.ndarray, np.ndarray],
                 emb: Dict[str, List[float]]):
        self.model = model
        self.hero_stoi = hero_stoi
        self.ctx_mean, self.ctx_std = ctx_stats
        self.emb = emb
        self._byname = default_byname()

    def predict(self, minions: List[Dict], hero_id: str = "UNKNOWN",
                state=None) -> Dict:
        from .board_features import context_vector
        toks, mask = board_tokens(minions, self.emb, self._byname)
        ctx = (context_vector(state) - self.ctx_mean) / self.ctx_std
        hidx = self.hero_stoi.get(hero_id, self.hero_stoi.get("UNKNOWN", 0))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(toks).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
                torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0),
                torch.tensor([hidx]))
            dist = F.softmax(logits, dim=-1)[0].numpy()
        placement = float(np.dot(dist, np.arange(1, N_PLACES + 1)))
        placement = max(1.0, min(8.0, placement))
        return {"placement": placement, "equity": (8.0 - placement) / 7.0,
                "dist": dist.tolist()}

    def save(self, path: str):
        torch.save({
            "state": self.model.state_dict(),
            "hero_stoi": self.hero_stoi,
            "ctx_mean": self.ctx_mean.tolist(),
            "ctx_std": self.ctx_std.tolist(),
            "tok_dim": self.model.tok.in_features,
        }, path)
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump({"heroes": len(self.hero_stoi),
                       "tok_dim": self.model.tok.in_features,
                       "arch": "set_transformer_v1"}, fh)

    @classmethod
    def load(cls, path: str, emb: Dict[str, List[float]]) -> "SetEvalModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = SetEvalNet(blob["tok_dim"], len(blob["hero_stoi"]))
        model.load_state_dict(blob["state"])
        stats = (np.asarray(blob["ctx_mean"]), np.asarray(blob["ctx_std"]))
        return cls(model, blob["hero_stoi"], stats, emb)


__all__ = ["SetEvalNet", "SetEvalModel", "train", "evaluate", "soft_targets",
           "examples_to_arrays", "N_PLACES"]
