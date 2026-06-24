"""Draft/choice recommender: rank a set of *offered* options and pick the best.

Distinct from the action recommender (`advisor.py`, which ranks buy/sell/roll on
your board). A draft is "choose 1 of N offered": hero select, trinket pick,
Discover. Each has its own best signal:

  * **hero**    — population average placement (lower = better), at your MMR.
  * **trinket** — population average placement.
  * **discover**— board fit: how much adding this minion raises your board's
    expected finish (the eval net + card2vec synergy on your *live* board).

So hero/trinket lean on the meta stats; Discover leans on the deep brain + your
current board. Everything is ranked best-first.
"""

from dataclasses import dataclass
from typing import List, Optional

from .board_value import get_scorer, _name as _minion_name
from .cards import by_name
from .economy import HeroContext
from .stats import StatsDB, HeroStats, TrinketStats
from .synergy import score_card, load_embeddings


@dataclass
class Choice:
    name: str
    rank_value: float        # sort key (best = smallest); internal
    reason: str
    metric: str              # what rank_value means, for display

    def line(self) -> str:
        return f"  {self.name} — {self.reason}"


def _match(name: str, pool):
    """Find a stats row for an offered option by name or card id (fuzzy)."""
    key = (name or "").strip().lower()
    for s in pool:
        if s.name.lower() == key or getattr(s, "card_id", "").lower() == key:
            return s
    for s in pool:                                  # partial / substring
        if key and (key in s.name.lower() or s.name.lower() in key):
            return s
    return None


def rank_heroes(offered: List[str], db: StatsDB) -> List[Choice]:
    out = []
    for nm in offered:
        h: Optional[HeroStats] = _match(nm, db.heroes)
        if h:
            tribes = ("favors " + "/".join(h.best_tribes)) if h.best_tribes else "flexible tribes"
            out.append(Choice(h.name, h.average_position,
                              f"avg {h.average_position:.2f} · {tribes} · {h.playstyle}",
                              "avg placement"))
        else:
            out.append(Choice(nm, 4.5, "no stats for this hero (defaulting to average)",
                              "avg placement"))
    out.sort(key=lambda c: c.rank_value)            # lower placement = better
    return out


def rank_trinkets(offered: List[str], db: StatsDB) -> List[Choice]:
    out = []
    for nm in offered:
        t: Optional[TrinketStats] = _match(nm, db.trinkets)
        if t:
            out.append(Choice(t.name, t.average_position,
                              f"avg {t.average_position:.2f} · tier {t.tier}",
                              "avg placement"))
        else:
            out.append(Choice(nm, 4.5, "no stats for this trinket", "avg placement"))
    out.sort(key=lambda c: c.rank_value)
    return out


def _minion_from_name(ck, name):
    if ck is None:
        return {"name": name, "attack": 3, "health": 3}
    return {"name": ck.name, "attack": ck.attack, "health": ck.health}


def rank_discover(offered: List[str], board, kb, scorer=None,
                  hero_ctx: Optional[HeroContext] = None) -> List[Choice]:
    """Rank Discover options by how much each improves your live board."""
    scorer = scorer or get_scorer()
    emb = load_embeddings()
    idx = by_name(kb) if kb is not None else {}
    hero_id = hero_ctx.hero if hero_ctx and hero_ctx.hero else "UNKNOWN"
    target = hero_ctx.target_tribe if hero_ctx else None
    board = list(board or [])
    base = scorer.equity(board, hero_id)
    board_cks = [idx.get(_minion_name(m)) for m in board if idx.get(_minion_name(m))]

    out = []
    for nm in offered:
        ck = idx.get(nm)
        cand = _minion_from_name(ck, nm)
        delta = scorer.equity(board + [cand], hero_id) - base
        bits = []
        if ck is not None:
            bits = score_card(ck, board_cks, target_tribe=target, embeddings=emb).reasons[:2]
        reason = f"{delta:+.0%} equity"
        if bits:
            reason += " — " + "; ".join(bits)
        out.append(Choice(nm, -delta, reason, "board equity"))   # higher equity first
    out.sort(key=lambda c: c.rank_value)
    return out


def recommend_choice(kind: str, offered: List[str], *, db: Optional[StatsDB] = None,
                     board=None, kb=None, scorer=None,
                     hero_ctx: Optional[HeroContext] = None) -> List[Choice]:
    """Dispatch to the right ranker. kind: 'hero' | 'trinket' | 'discover'."""
    if kind == "hero":
        return rank_heroes(offered, db or StatsDB.load())
    if kind == "trinket":
        return rank_trinkets(offered, db or StatsDB.load())
    if kind == "discover":
        return rank_discover(offered, board or [], kb, scorer=scorer, hero_ctx=hero_ctx)
    raise ValueError(f"unknown choice kind: {kind}")
