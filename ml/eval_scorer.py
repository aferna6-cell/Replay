"""Adapt the deep board-evaluation net to the advisor's scorer interface.

`load_default_scorer()` loads ml/eval_net.pt (if present) and returns an object
with `.equity(board, hero_id)` — the same call the heuristic scorer answers — so
the advisor can prefer the learned brain transparently. Returns None when no
trained model exists, letting the advisor fall back to the heuristic.
"""

import os
from typing import List, Optional

from hsbg_coach.synergy import load_embeddings
from hsbg_coach.cards import by_name, load_kb
from .board_features import minion_from_snapshot

_MODEL = os.path.join(os.path.dirname(__file__), "eval_net.pt")


class EvalNetScorer:
    name = "eval_net"

    def __init__(self, model, byname):
        self.model = model
        self.byname = byname

    def equity(self, board, hero_id: str = "UNKNOWN", state=None) -> float:
        minions = [m for m in (minion_from_snapshot(_as_dict(x), self.byname)
                               for x in board) if m]
        if not minions:
            return 0.0
        return self.model.predict(minions, hero_id or "UNKNOWN", state=state)["equity"]


def _as_dict(m):
    if isinstance(m, dict):
        return m
    return {"name": getattr(m, "name", None), "card_id": getattr(m, "card_id", None),
            "attack": getattr(m, "attack", None), "health": getattr(m, "health", None),
            "tags": getattr(m, "tags", {}) or {}}


def load_default_scorer(model_path: str = _MODEL) -> Optional[EvalNetScorer]:
    if not os.path.isfile(model_path):
        return None
    from .eval_net import EvalModel          # imports torch — kept lazy
    emb = load_embeddings()
    model = EvalModel.load(model_path, emb)
    return EvalNetScorer(model, by_name(load_kb()))
