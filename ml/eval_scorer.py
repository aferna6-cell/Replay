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
_SET_MODEL = os.path.join(os.path.dirname(__file__), "set_net.pt")


class EvalNetScorer:
    name = "eval_net"

    def __init__(self, model, byname, name: str = "eval_net"):
        self.model = model
        self.byname = byname
        self.name = name

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


def load_default_scorer(model_path: str = _MODEL,
                        set_model_path: str = _SET_MODEL
                        ) -> Optional[EvalNetScorer]:
    """Best available learned scorer: the set-transformer (attention over
    minions, mid-game trained) when present, else the original MLP."""
    emb = load_embeddings()
    if os.path.isfile(set_model_path):
        from .set_net import SetEvalModel    # imports torch — kept lazy
        model = SetEvalModel.load(set_model_path, emb)
        return EvalNetScorer(model, by_name(load_kb()), name="set_net")
    if not os.path.isfile(model_path):
        return None
    from .eval_net import EvalModel          # imports torch — kept lazy
    model = EvalModel.load(model_path, emb)
    return EvalNetScorer(model, by_name(load_kb()))
