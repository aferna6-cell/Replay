"""Assemble the (board -> expected finish) training set for the eval net.

Two sources, same feature pipeline:

  * **Population** (`build_examples`) — every Firestone final board, labeled with
    its hero-within-archetype `averagePlacement`. A graded prior over the whole
    meta: strong comps carry low placement, weak comps high.
  * **Your games** (`trajectory_examples`) — recorded snapshots labeled with the
    *actual* placement they led to. This is the sharp, personal signal that the
    continual-training loop folds in as you play.

`group` is the unit of the train/val split (hero id, or game id for trajectories)
so we measure generalization to *unseen* comps/games, not memorized labels.
"""

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from hsbg_coach import cards
from hsbg_coach.firestone_stats import (
    _fetch_json, _card_meta, _names_from_meta, COMP_URL, CARDS_URL,
)
from .board_features import (
    minion_from_population, minion_from_snapshot, board_vector,
)

UNKNOWN_HERO = "UNKNOWN"


def build_examples(comp_source: Optional[str] = None, period: str = "past-seven",
                   cards_source: str = CARDS_URL) -> List[Dict]:
    raw = _fetch_json(comp_source or COMP_URL.format(period=period))
    names = _names_from_meta(_card_meta(cards_source))
    byname = cards.by_name(cards.load_kb())
    out: List[Dict] = []
    for comp in raw.get("compStats", []):
        for hero in comp.get("heroStats", []):
            label = hero.get("averagePlacement") or comp.get("averagePlacement")
            hid = hero.get("heroCardId") or UNKNOWN_HERO
            if not label:
                continue
            for fb in hero.get("finalBoards", []):
                board = (fb.get("finalComp") or {}).get("board") or []
                minions = [m for m in (minion_from_population(x, names, byname)
                                       for x in board) if m]
                if len(minions) >= 2:
                    out.append({"minions": minions, "hero": hid,
                                "label": float(label), "state": _ENDGAME_CONTEXT,
                                "group": hid})
    return out


# Population final boards are late-game winning boards, so give them a neutral
# endgame context (so the context channel isn't all-zeros for the population bulk
# while trajectories carry real state). Tuned to a typical strong endgame.
_ENDGAME_CONTEXT = {"tavern_tier": 6, "gold": 0, "hero_health": 25, "turn": 13,
                    "opponent_profiles": [], "trinkets": [], "anomaly": None}


def trajectory_examples(data_dir: str, weight: float = 1.0) -> List[Dict]:
    """Labeled examples from recorded games (placement-tagged states).
    `weight` is the training sample weight attached to every example — used to
    up-weight scarcer/higher-quality sources (e.g. VOD-reconstructed games of
    top players) relative to the population boards."""
    byname = cards.by_name(cards.load_kb())
    out: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            pl = d.get("placement")
            if pl is None:
                continue
            state = d.get("state") or {}
            board = state.get("board") or []
            minions = [m for m in (minion_from_snapshot(x, byname) for x in board) if m]
            if len(minions) >= 2:
                out.append({"minions": minions, "hero": UNKNOWN_HERO,
                            "label": float(pl),
                            "state": state,          # whole-state context for training
                            "group": d.get("game_id") or path,
                            "weight": weight})
    return out


def example_weights(examples: List[Dict]):
    """Per-sample training weights (1.0 where unset), aligned with to_arrays."""
    import numpy as _np
    return _np.array([float(e.get("weight", 1.0)) for e in examples],
                     dtype=_np.float32)


def build_hero_vocab(examples: List[Dict]) -> Dict[str, int]:
    heroes = sorted({e["hero"] for e in examples} | {UNKNOWN_HERO})
    return {h: i for i, h in enumerate(heroes)}


def to_arrays(examples: List[Dict], emb: Dict[str, List[float]],
              hero_stoi: Dict[str, int],
              stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              with_context: bool = False
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple]:
    """Return (X_dense, hero_idx, y, (mean, std)). Fits standardization if stats=None.

    with_context=True appends the whole-state context block (tier/gold/hp/turn/
    opponents/trinkets/anomaly) to each board vector, so the net learns the SAME
    board is worth more or less given the state around it. Population examples carry
    a neutral endgame context (set by build_examples), trajectories carry the real
    recorded state — so the context channel is meaningful across both."""
    if with_context:
        from .board_features import full_vector
        X = np.stack([full_vector(e["minions"], emb, e.get("state")) for e in examples])
    else:
        X = np.stack([board_vector(e["minions"], emb) for e in examples])
    hero = np.array([hero_stoi.get(e["hero"], hero_stoi[UNKNOWN_HERO])
                     for e in examples], dtype=np.int64)
    y = np.array([e["label"] for e in examples], dtype=np.float32)
    if stats is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        stats = (mean, std)
    X = (X - stats[0]) / stats[1]
    return X.astype(np.float32), hero, y, stats


def group_split(examples: List[Dict], val_frac: float = 0.15, seed: int = 0
                ) -> Tuple[List[Dict], List[Dict]]:
    """Split by `group` so val comps/games are unseen in training."""
    groups = sorted({e["group"] for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    val_groups = set(groups[:n_val])
    train = [e for e in examples if e["group"] not in val_groups]
    val = [e for e in examples if e["group"] in val_groups]
    return train, val
