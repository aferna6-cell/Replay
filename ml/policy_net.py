"""The policy/value net for the self-play RL agent (spec §6b).

Shared set encoder (same token space as the eval net, plus a zone embedding
for board / shop / hand) → two heads:

  * **policy** — logits over the env's fixed 28-action space, with illegal
    actions masked to -inf so the distribution only covers legal moves;
  * **value** — expected zero-mean placement return, used by GAE/PPO.

Small on purpose (d=64, 2 layers): the env is CPU-bound, and Phase 1's goal
is "beats random + greedy", not superhuman.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env_obs import N_ZONES, POLICY_CTX_DIM, N_ACTIONS


class PolicyNet(nn.Module):
    def __init__(self, tok_dim: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ff: int = 128, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Linear(tok_dim, d_model)
        self.zone = nn.Embedding(N_ZONES, d_model)
        self.ctx = nn.Linear(POLICY_CTX_DIM, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)
        self.pi = nn.Sequential(nn.LayerNorm(d_model),
                                nn.Linear(d_model, N_ACTIONS))
        self.v = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, tokens, mask, zones, ctx
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = tokens.shape[0]
        t = self.tok(tokens) + self.zone(zones)
        c = self.ctx(ctx).unsqueeze(1)
        x = torch.cat([c, t], dim=1)
        ones = torch.ones(b, 1, device=tokens.device)
        full = torch.cat([ones, mask], dim=1)
        h = self.encoder(x, src_key_padding_mask=(full < 0.5))
        pooled = (h * full.unsqueeze(-1)).sum(1) / full.sum(1, keepdim=True)
        return self.pi(pooled), self.v(pooled).squeeze(-1)

    @staticmethod
    def masked_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(legal < 0.5, float("-inf"))

    def act(self, arrays, legal_mask: List[bool], greedy: bool = False
            ) -> Tuple[int, float, float]:
        """One decision: (action, logprob, value). arrays = encode_obs output."""
        toks, mask, zones, ctx = arrays
        legal = torch.tensor([legal_mask], dtype=torch.float32)
        with torch.no_grad():
            logits, value = self(
                torch.from_numpy(toks).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
                torch.from_numpy(zones).unsqueeze(0),
                torch.from_numpy(ctx).unsqueeze(0))
            logits = self.masked_logits(logits, legal)
            dist = torch.distributions.Categorical(logits=logits)
            a = logits.argmax(-1) if greedy else dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(value.item())


def save_policy(net: PolicyNet, path: str, meta: Optional[Dict] = None) -> None:
    torch.save({"state": net.state_dict(),
                "tok_dim": net.tok.in_features,
                "meta": meta or {}}, path)


def load_policy(path: str) -> PolicyNet:
    # weights_only=True: checkpoints are tensors + primitives only, and this
    # refuses pickled code — loading a third-party .pt must not execute it.
    blob = torch.load(path, map_location="cpu", weights_only=True)
    net = PolicyNet(blob["tok_dim"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net


def as_env_policy(net: PolicyNet, emb: Dict, byname: Optional[Dict] = None,
                  greedy: bool = True):
    """Wrap a net as a bg_env scripted-seat policy(obs, mask, rng) -> action —
    this is how league checkpoints occupy opponent seats."""
    from .env_obs import encode_obs

    def policy(obs, mask, rng):
        arrays = encode_obs(obs, emb, byname)
        a, _, _ = net.act(arrays, mask, greedy=greedy)
        return a
    return policy
