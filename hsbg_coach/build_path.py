"""Build-path navigation: value a move by how much it advances you toward a
reachable *winning* end-comp — not just by how the board looks right now.

The eval net knows what a strong final board looks like. It does NOT know that a
scrappy turn-4 board with Ingenious Inventor + a Mech or two is *on the path* to
Mech Magnet, one of the best end-comps. This module supplies that missing signal.

We have, from Firestone, 34 archetypes of real winning boards — each with its core
cards (by share of winning boards), its tribe, and how many winning boards it
represents (a strength/consistency prior). card2vec places cards near the cards
they win with. Together:

  * ``infer_target(board)`` — which winning archetype your board is closest to /
    building toward (core coverage + centroid similarity + a popularity prior).
  * ``path_value(board, card, tier)`` — how much buying ``card`` advances that
    build. A core piece you're missing is a big step; an off-tribe scatter in the
    mid-game is a step back. Commitment ramps with tier: stay flexible early, lock
    in the comp by tier 4-5 (the mid-game navigation the player asked for).

Output is a placement adjustment (negative = better finish) so it drops straight
into the whole-game ranker alongside the eval net, tech and economy terms.
"""

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .synergy import load_embeddings, _cosine

_BOARDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stats",
                            "firestone_final_boards.json")

# How strongly the build-path term can move placement, and how commitment ramps
# with tavern tier (flexible early, committed mid-game).
_MAX_ADJUST = 0.8
_TIER_COMMIT = {1: 0.25, 2: 0.4, 3: 0.7, 4: 1.0, 5: 1.15, 6: 1.15}


@dataclass
class Archetype:
    key: str
    name: str
    tribe: Optional[str]
    core: Dict[str, float]            # card name -> share of winning boards
    board_count: int
    centroid: List[float] = field(default_factory=list)


@dataclass
class TargetFit:
    arch: Archetype
    coverage: float                   # how much of this comp you've assembled (0..1)
    have: int                         # core cards you already hold
    core_total: int


def _name(m) -> Optional[str]:
    if isinstance(m, dict):
        return m.get("name") or m.get("card_id")
    return getattr(m, "name", None) or getattr(m, "card_id", None)


def board_names(board) -> List[str]:
    return [n for n in (_name(m) for m in (board or [])) if n]


@lru_cache(maxsize=1)
def load_archetypes() -> List[Archetype]:
    """Archetypes with card2vec centroids, sorted by winning-board support."""
    if not os.path.isfile(_BOARDS_PATH):
        return []
    raw = json.load(open(_BOARDS_PATH, encoding="utf-8"))
    emb = load_embeddings()
    out: List[Archetype] = []
    for a in raw.get("boards", []):
        core = {c["name"]: float(c.get("frequency") or 0.0)
                for c in a.get("coreCards", []) if c.get("name")}
        if not core:
            continue
        centroid = _weighted_centroid(core, emb)
        out.append(Archetype(
            key=a.get("archetype", ""), name=a.get("name") or a.get("archetype", ""),
            tribe=a.get("tribe"), core=core,
            board_count=int(a.get("boardCount") or 0), centroid=centroid,
        ))
    out.sort(key=lambda x: x.board_count, reverse=True)
    return out


def _weighted_centroid(core: Dict[str, float], emb: Dict[str, List[float]]) -> List[float]:
    vecs = [(emb[n], w) for n, w in core.items() if n in emb]
    if not vecs:
        return []
    dim = len(vecs[0][0])
    acc = [0.0] * dim
    tw = 0.0
    for v, w in vecs:
        for i in range(dim):
            acc[i] += v[i] * w
        tw += w
    return [x / tw for x in acc] if tw else []


def _board_centroid(names: List[str], emb: Dict[str, List[float]]) -> List[float]:
    vecs = [emb[n] for n in names if n in emb]
    if not vecs:
        return []
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


# board_count prior: a comp backed by thousands of winning boards is a more
# reliable target than a fringe one. Compressed via log so it nudges, not dominates.
def _support_prior(board_count: int) -> float:
    return math.log10(board_count + 10) / 3.0      # ~0.4 (count 0) .. ~1.0 (count 1000)


def infer_target(board, archetypes: Optional[List[Archetype]] = None,
                 emb: Optional[Dict[str, List[float]]] = None) -> Optional[TargetFit]:
    """The winning archetype the board is best positioned to complete."""
    archetypes = archetypes if archetypes is not None else load_archetypes()
    if not archetypes:
        return None
    emb = emb if emb is not None else load_embeddings()
    names = board_names(board)
    bc = _board_centroid(names, emb)

    best, best_score = None, -1.0
    for a in archetypes:
        have_w = sum(a.core.get(n, 0.0) for n in names)      # weighted overlap
        core_sum = sum(a.core.values()) or 1.0
        coverage = have_w / core_sum
        sim = _cosine(bc, a.centroid) if bc and a.centroid else 0.0
        score = (0.6 * coverage + 0.4 * max(0.0, sim)) * _support_prior(a.board_count)
        if score > best_score:
            best, best_score = a, score
    if best is None:
        return None
    have = sum(1 for n in names if n in best.core)
    return TargetFit(best, coverage=min(1.0, sum(best.core.get(n, 0.0) for n in names)
                                        / (sum(best.core.values()) or 1.0)),
                     have=have, core_total=len(best.core))


def _tier_commit(tier: Optional[int]) -> float:
    return _TIER_COMMIT.get(int(tier or 1), 1.0)


def path_value(board, candidate_name: Optional[str], tier: Optional[int],
               candidate_tribe: Optional[str] = None,
               archetypes: Optional[List[Archetype]] = None,
               emb: Optional[Dict[str, List[float]]] = None
               ) -> Tuple[float, Optional[str]]:
    """(placement_adjustment, reason) for adding ``candidate_name`` to the board.

    Negative adjustment = it advances a reachable winning comp (recommend more).
    Returns (0.0, None) when we can't place it (no data / unknown card)."""
    if not candidate_name:
        return 0.0, None
    archetypes = archetypes if archetypes is not None else load_archetypes()
    emb = emb if emb is not None else load_embeddings()
    fit = infer_target(board, archetypes, emb)
    if fit is None:
        return 0.0, None
    a = fit.arch
    commit = _tier_commit(tier)

    # 1) A core piece of your target you don't already hold — the biggest step.
    if candidate_name in a.core and candidate_name not in board_names(board):
        freq = a.core[candidate_name]
        adj = -min(_MAX_ADJUST, freq * commit)
        return adj, (f"core {a.name} piece — in {freq:.0%} of winning {a.name} "
                     f"boards (you have {fit.have}/{fit.core_total} core)")

    # 2) An enabler: not named core, but it pulls the board toward the comp in
    #    card2vec space (wins alongside the core).
    if candidate_name in emb and a.centroid:
        sim = _cosine(emb[candidate_name], a.centroid)
        if sim > 0.35:
            adj = -min(_MAX_ADJUST * 0.5, sim * 0.5 * commit)
            return adj, f"fits the {a.name} build (synergizes with your direction)"

    # 3) Mid-game scatter: off-tribe filler once you should be committing.
    if (candidate_tribe and a.tribe and candidate_tribe != a.tribe
            and int(tier or 1) >= 4 and fit.have >= 2):
        return min(_MAX_ADJUST * 0.4, 0.3 * commit), (
            f"off-path for your {a.name} build — dilutes the comp")

    return 0.0, None


def build_note(board, tier: Optional[int]) -> Optional[str]:
    """One-line 'what you're building' summary for the overlay."""
    fit = infer_target(board)
    if fit is None or fit.have == 0:
        return None
    missing = [n for n in sorted(fit.arch.core, key=lambda k: -fit.arch.core[k])
               if n not in board_names(board)][:3]
    tail = f" — next: {', '.join(missing)}" if missing else ""
    return f"Building toward {fit.arch.name} ({fit.have}/{fit.core_total} core){tail}"
