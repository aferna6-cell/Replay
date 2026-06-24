"""Combat-evaluation network (DeepSets board encoder + win/tie/loss head).

DeepSets: a per-minion MLP, then a masked sum-pool over the board — permutation
invariant, handles variable board size. Both boards share the encoder (Siamese);
we feed [enc_a, enc_b, enc_a - enc_b] to a head that predicts the win/tie/loss
distribution. A Transformer encoder (attention over minions) is the natural
upgrade for synergy modeling — see specs/self-play-rl-agent.md §6b — but DeepSets
is the right, fast first model.
"""

import torch
import torch.nn as nn

from .encode import NUM_FEATURES


class BoardEncoder(nn.Module):
    def __init__(self, feats: int = NUM_FEATURES, hidden: int = 64, emb: int = 64):
        super().__init__()
        self.minion = nn.Sequential(
            nn.Linear(feats, hidden), nn.ReLU(),
            nn.Linear(hidden, emb), nn.ReLU(),
        )

    def forward(self, x, mask):                 # x:[B,7,F]  mask:[B,7]
        h = self.minion(x)                      # [B,7,emb]
        return (h * mask.unsqueeze(-1)).sum(dim=1)   # masked sum-pool -> [B,emb]


class CombatValueNet(nn.Module):
    def __init__(self, emb: int = 64, hidden: int = 128):
        super().__init__()
        self.enc = BoardEncoder(emb=emb)
        self.head = nn.Sequential(
            nn.Linear(emb * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),                # logits: win, tie, loss
        )

    def forward(self, xa, ma, xb, mb):
        ea, eb = self.enc(xa, ma), self.enc(xb, mb)
        z = torch.cat([ea, eb, ea - eb], dim=-1)
        return self.head(z)

    @torch.no_grad()
    def predict_probs(self, xa, ma, xb, mb):
        return torch.softmax(self.forward(xa, ma, xb, mb), dim=-1)
