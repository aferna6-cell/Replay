"""Detect in-game choices (Discover / hero / trinket) from the live log + rank them.

Hearthstone logs every offered choice via ``GameState.DebugPrintEntityChoices()``:

    ...DebugPrintEntityChoices() - id=1 Player=Bob ChoiceType=GENERAL CountMin=1 CountMax=1
    ...DebugPrintEntityChoices() -   Source=...
    ...DebugPrintEntityChoices() -   Entities[0]=[entityName=Foo id=70 zone=... cardId=BGS_039 player=1]
    ...DebugPrintEntityChoices() -   Entities[1]=[... cardId=BG30_MagicItem_403 ...]

We accumulate that block, pull the offered ``cardId``s, classify them
(hero / trinket / minion-discover by id pattern), resolve names from our committed
data, and hand them to the draft recommender — so the overlay can say which to
pick. The pick is cleared when the player resolves it (``SendChoices``).

CALIBRATION: the two regexes below (`_CHOICE_LINE`, `_CARDID`) follow the
documented Power.log shape. If a real client logs choices differently, those two
lines are the only thing to adjust — everything downstream (classify, name
resolution, ranking, overlay display) is client-agnostic and unit-tested.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from . import cards, stats

_CHOICE_LINE = re.compile(r"DebugPrintEntityChoices\(\)\s*-\s*(.*)$")
_SEND_CHOICES = re.compile(r"SendChoices\(\)")
_CARDID = re.compile(r"cardId=([A-Za-z0-9_]+)")
_FIELD = lambda s, k: (re.search(rf"{k}=(\w+)", s) or [None, None])[1]


def classify(card_ids: List[str]) -> str:
    """hero / trinket / discover, from BG cardId conventions."""
    joined = " ".join(card_ids)
    if "MagicItem" in joined:          # BG trinkets, e.g. BG30_MagicItem_403
        return "trinket"
    if "HERO" in joined:               # BG heroes, e.g. BG..._HERO_...
        return "hero"
    return "discover"


@dataclass
class ChoiceOffer:
    kind: str
    card_ids: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    @property
    def key(self):
        return (self.kind, tuple(self.card_ids))


@lru_cache(maxsize=1)
def _name_index():
    """cardId -> display name across minions, heroes and trinkets (committed data)."""
    idx = {c.card_id: c.name for c in cards.load_kb().values()}
    try:
        for h in stats.load_hero_stats(stats.default_hero_source()):
            idx[h.card_id] = h.name
        for t in stats.load_trinket_stats(stats.default_trinket_source()):
            idx[t.card_id] = t.name
    except Exception:
        pass
    return idx


def name_for(card_id: str) -> str:
    return _name_index().get(card_id, card_id)


class ChoiceParser:
    """Feed raw log lines; emits a ChoiceOffer when an offer block completes."""

    def __init__(self):
        self._ids: List[str] = []
        self._open = False

    def feed(self, line: str) -> Optional[ChoiceOffer]:
        if _SEND_CHOICES.search(line):     # player resolved a choice
            self._reset()
            return None
        m = _CHOICE_LINE.search(line)
        if not m:
            # First non-choice line after a block ends it.
            if self._open and self._ids:
                return self._finalize()
            return None
        payload = m.group(1)
        if payload.startswith("Entities["):
            cid = _CARDID.search(payload)
            if cid:
                self._ids.append(cid.group(1))
        elif "ChoiceType=" in payload:     # header → start a fresh block
            self._ids = []
            self._open = True
        return None

    def _finalize(self) -> ChoiceOffer:
        ids = list(self._ids)
        self._reset()
        offer = ChoiceOffer(classify(ids), ids, [name_for(c) for c in ids])
        return offer

    def _reset(self):
        self._ids = []
        self._open = False


def rank_offer(offer: ChoiceOffer, board=None, kb=None, scorer=None,
               hero_ctx=None, db=None):
    """Rank an offer's options via the draft recommender (best first)."""
    from .draft import recommend_choice
    return recommend_choice(offer.kind, offer.names, db=db, board=board, kb=kb,
                            scorer=scorer, hero_ctx=hero_ctx)
