"""Score a board → equity (0–1, higher = stronger / better expected finish).

This is the value function the advisor uses to compare actions: to judge "buy X"
vs "sell Y" vs "do nothing", score the resulting board and prefer the higher
equity. Two implementations share one interface:

  * `EvalNetScorer` (in ml/) — the deep board-evaluation net. Best, needs torch.
  * `HeuristicScorer` (here)  — stdlib: total stats + card2vec board cohesion.
    Always available, so the advisor works even without the ML stack installed.

`get_scorer()` returns the eval net if a trained model is present, else the
heuristic — same `.equity(board, hero_id)` call either way.
"""

import math
from typing import Dict, List, Optional

from .synergy import load_embeddings, _cosine


def _val(m) -> int:
    if isinstance(m, dict):
        a, h = m.get("attack") or 0, m.get("health") or 0
    else:
        a, h = getattr(m, "attack", 0) or 0, getattr(m, "health", 0) or 0
    return int(a) + int(h)


def _name(m) -> str:
    if isinstance(m, dict):
        return m.get("name") or m.get("card_id") or "?"
    return getattr(m, "name", None) or getattr(m, "card_id", None) or "?"


class HeuristicScorer:
    """Stdlib board value: raw stats + learned card2vec cohesion, soft-normalized."""

    name = "heuristic"

    def __init__(self, embeddings: Optional[Dict[str, List[float]]] = None):
        self.emb = embeddings if embeddings is not None else load_embeddings()

    def _cohesion(self, board) -> float:
        vecs = [self.emb[_name(m)] for m in board if _name(m) in self.emb]
        if len(vecs) < 2:
            return 0.0
        sims, n = 0.0, 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims += _cosine(vecs[i], vecs[j])
                n += 1
        return sims / n if n else 0.0

    def equity(self, board, hero_id: str = "UNKNOWN") -> float:
        stat = sum(_val(m) for m in board)
        raw = stat + max(self._cohesion(board), 0.0) * 25.0
        return raw / (raw + 50.0)            # monotonic squash into (0, 1)


def get_scorer(prefer_eval: bool = True):
    """Return the best available scorer (deep eval net if trained, else heuristic)."""
    if prefer_eval:
        try:
            from ml.eval_scorer import load_default_scorer
            scorer = load_default_scorer()
            if scorer is not None:
                return scorer
        except Exception:
            pass                              # torch/model absent → heuristic
    return HeuristicScorer()
