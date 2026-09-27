"""Behaviour-cloned NEXT policy: score every legal option at a decision point.

score(state, option) = MLP([encode_state | encode_option]) with two hidden
layers of 128 and ReLU, one score per option. A decision's legal options are
softmaxed and training is per-row weighted cross-entropy on the chosen option.
Checkpoints carry the encoder version and dims; a mismatched checkpoint raises
CheckpointMismatch instead of loading.

Features come from hsbg_coach/encode.py, the single encoder shared with the
live overlay. Unrelated to ml/policy_net.py (old PPO experiments).
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from hsbg_coach import encode as enc

KIND = "bc_option_scorer"
HIDDEN = 128


class CheckpointMismatch(ValueError):
    """Checkpoint was written for a different encoder or model shape."""


class OptionScorer(nn.Module):
    def __init__(self, state_dim: int = enc.STATE_DIM,
                 option_dim: int = enc.OPTION_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.state_dim, self.option_dim, self.hidden = state_dim, option_dim, hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim + option_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """state [..., S], options [..., N, O] -> scores [..., N]."""
        s = state.unsqueeze(-2).expand(*options.shape[:-1], state.shape[-1])
        return self.net(torch.cat([s, options], dim=-1)).squeeze(-1)


def pad_batch(items):
    """[(state[S], options[N,O], chosen, weight)] -> padded tensors + mask."""
    n_max = max(o.shape[0] for _, o, _, _ in items)
    b = len(items)
    states = torch.zeros(b, items[0][0].shape[0])
    options = torch.zeros(b, n_max, items[0][1].shape[1])
    mask = torch.zeros(b, n_max, dtype=torch.bool)
    chosen = torch.zeros(b, dtype=torch.long)
    weight = torch.zeros(b)
    for i, (s, o, c, w) in enumerate(items):
        states[i] = torch.from_numpy(np.asarray(s, dtype=np.float32))
        options[i, :o.shape[0]] = torch.from_numpy(np.asarray(o, dtype=np.float32))
        mask[i, :o.shape[0]] = True
        chosen[i] = int(c)
        weight[i] = float(w)
    return states, options, mask, chosen, weight


def decision_loss(model, states, options, mask, chosen, weight) -> torch.Tensor:
    """Weighted cross-entropy of the chosen option under a softmax over legal ones."""
    scores = model(states, options).masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(scores, dim=-1)
    nll = -logp.gather(1, chosen.unsqueeze(1)).squeeze(1)
    return (weight * nll).sum() / weight.sum().clamp_min(1e-8)


def checkpoint_meta(model: OptionScorer, extra: Optional[Dict] = None) -> Dict:
    meta = {"kind": KIND, "encoder_version": enc.ENCODER_VERSION,
            "state_dim": model.state_dim, "option_dim": model.option_dim,
            "hidden": model.hidden}
    meta.update(extra or {})
    return meta


def save_policy(model: OptionScorer, path: str, extra: Optional[Dict] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"meta": checkpoint_meta(model, extra),
                "state_dict": model.state_dict()}, path)
    return path


def load_bc_policy(path: str) -> "BCPolicy":
    """Load a checkpoint; raise CheckpointMismatch if it was built for another
    encoder version or other state/option dims."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    if not isinstance(meta, dict) or meta.get("kind") != KIND:
        raise CheckpointMismatch(f"{path}: not a {KIND} checkpoint")
    want = {"encoder_version": enc.ENCODER_VERSION,
            "state_dim": enc.STATE_DIM, "option_dim": enc.OPTION_DIM}
    for key, value in want.items():
        if meta.get(key) != value:
            raise CheckpointMismatch(
                f"{path}: {key}={meta.get(key)!r}, this encoder needs {value!r}")
    model = OptionScorer(meta["state_dim"], meta["option_dim"],
                         int(meta.get("hidden", HIDDEN)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return BCPolicy(model, meta)


class BCPolicy:
    """Scores options for a snapshot; best() is the live NEXT pick."""

    def __init__(self, model: OptionScorer, meta: Optional[Dict] = None):
        self.model = model.eval()
        self.meta = dict(meta or checkpoint_meta(model))

    @torch.no_grad()
    def score_encoded(self, state, options) -> np.ndarray:
        s = torch.from_numpy(np.asarray(state, dtype=np.float32))
        o = torch.from_numpy(np.asarray(options, dtype=np.float32))
        return self.model(s, o).numpy()

    def score(self, snapshot, options: List[Dict]) -> np.ndarray:
        if not options:
            return np.zeros(0, dtype=np.float32)
        return self.score_encoded(
            enc.encode_state(snapshot),
            np.stack([enc.encode_option(snapshot, o) for o in options]))

    def best(self, snapshot, options: Optional[List[Dict]] = None) -> Optional[Dict]:
        """Top-scored option (legal_options(snapshot) unless given); None if none."""
        options = enc.legal_options(snapshot) if options is None else list(options)
        if not options:
            return None
        return options[int(np.argmax(self.score(snapshot, options)))]