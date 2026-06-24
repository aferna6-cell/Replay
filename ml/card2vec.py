"""Skip-gram with negative sampling (SGNS) for card embeddings.

A real (shallow) neural embedding trained by gradient descent — the canonical
representation-learning method, here over winning boards instead of sentences.
Outputs a vector per card such that cosine similarity ≈ "win together-ness."
"""

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .board_corpus import build_vocab


class SGNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 48):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, center, pos, neg):
        c = self.center(center)                       # [B, dim]
        p = self.context(pos)                         # [B, dim]
        n = self.context(neg)                         # [B, K, dim]
        pos_score = F.logsigmoid((c * p).sum(-1))     # [B]
        neg_score = F.logsigmoid(-(n @ c.unsqueeze(-1)).squeeze(-1)).sum(-1)  # [B]
        return -(pos_score + neg_score).mean()


def _pairs(boards: List[List[str]], stoi: Dict[str, int]) -> np.ndarray:
    """All (center, context) index pairs co-occurring within a board."""
    out = []
    for board in boards:
        idx = [stoi[c] for c in board if c in stoi]
        for i, ci in enumerate(idx):
            for j, cj in enumerate(idx):
                if i != j:
                    out.append((ci, cj))
    return np.asarray(out, dtype=np.int64)


def train_embeddings(boards: List[List[str]], dim: int = 48, epochs: int = 5,
                     neg_k: int = 5, min_count: int = 20, batch: int = 4096,
                     lr: float = 0.01, seed: int = 0, verbose: bool = True
                     ) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    itos, stoi, counts = build_vocab(boards, min_count)
    if not itos:
        return {}
    pairs = _pairs(boards, stoi)
    # Negative-sampling distribution: unigram^0.75.
    freq = np.asarray(counts, dtype=np.float64) ** 0.75
    neg_p = torch.tensor(freq / freq.sum(), dtype=torch.float)

    model = SGNS(len(itos), dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    centers = torch.from_numpy(pairs[:, 0])
    contexts = torch.from_numpy(pairs[:, 1])
    n = centers.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            c, p = centers[idx], contexts[idx]
            neg = torch.multinomial(neg_p, len(idx) * neg_k, replacement=True
                                    ).view(len(idx), neg_k)
            opt.zero_grad()
            loss = model(c, p, neg)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"epoch {ep:2d}  loss {total / n:.4f}  pairs {n}  vocab {len(itos)}")

    emb = model.center.weight.detach().numpy()
    return {itos[i]: emb[i] for i in range(len(itos))}


# --- similarity helpers ------------------------------------------------------
def _normalize(vectors: Dict[str, np.ndarray]):
    names = list(vectors)
    mat = np.stack([vectors[n] for n in names])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return names, mat


def most_similar(vectors: Dict[str, np.ndarray], card: str, k: int = 8):
    if card not in vectors:
        return []
    names, mat = _normalize(vectors)
    q = vectors[card] / (np.linalg.norm(vectors[card]) + 1e-9)
    sims = mat @ q
    order = np.argsort(-sims)
    out = [(names[i], float(sims[i])) for i in order if names[i] != card]
    return out[:k]


def similarity(vectors: Dict[str, np.ndarray], a: str, b: str):
    if a not in vectors or b not in vectors:
        return None
    va, vb = vectors[a], vectors[b]
    return float(va @ vb / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9))
