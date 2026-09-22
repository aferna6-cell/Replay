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
# entityName runs up to " id=" — names contain spaces/punctuation (e.g. "A. F. Kay").
_ENTNAME = re.compile(r"entityName=(.+?)\s+id=")
_FIELD = lambda s, k: (re.search(rf"{k}=(\w+)", s) or [None, None])[1]


def classify(card_ids: List[str]) -> str:
    """hero / trinket / hero_power / quest / discover, from BG cardId conventions."""
    joined = " ".join(card_ids).lower()
    if "magicitem" in joined:          # BG trinkets, e.g. BG30_MagicItem_403
        return "trinket"
    # Hero-power picks (e.g. Nguyen): TB_BaconShop_HP_### or a hero id with a
    # trailing 'p' (BG28_HERO_800p). Check before "hero" so they don't read as a
    # hero select.
    if "_hp_" in joined or re.search(r"_hero_\d+p\b", joined):
        return "hero_power"
    # Quest / reward picks (e.g. Sire Denathrius).
    if "quest" in joined or "reward" in joined:
        return "quest"
    if "hero" in joined:               # BG heroes, e.g. BG..._HERO_...
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
        self._names: List[str] = []
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
                nm = _ENTNAME.search(payload)   # prefer the name straight from the log
                self._names.append(nm.group(1).strip() if nm else "")
        elif "ChoiceType=" in payload:     # header → start a fresh block
            self._ids, self._names = [], []
            self._open = True
        return None

    def _finalize(self) -> ChoiceOffer:
        ids, names = list(self._ids), list(self._names)
        self._reset()
        # Use the log's entityName; fall back to our committed name lookup.
        disp = [names[i] or name_for(ids[i]) for i in range(len(ids))]
        return ChoiceOffer(classify(ids), ids, disp)

    def _reset(self):
        self._ids = []
        self._names = []
        self._open = False


def rank_offer(offer: ChoiceOffer, board=None, kb=None, scorer=None,
               hero_ctx=None, db=None, tier=None):
    """Rank an offer's options via the draft recommender (best first).

    For heroes this returns the full ranking (does not drop the reroll target).
    Prefer ``offer_advice_lines`` for overlay text that includes the reroll step.
    """
    from .draft import recommend_choice
    return recommend_choice(offer.kind, offer.names, db=db, board=board, kb=kb,
                            scorer=scorer, hero_ctx=hero_ctx, tier=tier)


def offer_advice_lines(offer: ChoiceOffer, board=None, kb=None, scorer=None,
                       hero_ctx=None, db=None, tier=None,
                       rerolls_available: int = 1):
    """Overlay lines for an offer. Heroes get reroll-then-pick; others get PICK."""
    if offer.kind == "hero":
        from .draft import hero_draft_plan
        from .stats import StatsDB
        plan = hero_draft_plan(
            offer.names, db or StatsDB.load(),
            rerolls_available=rerolls_available,
        )
        return plan["lines"]
    picks = rank_offer(offer, board=board, kb=kb, scorer=scorer,
                       hero_ctx=hero_ctx, db=db, tier=tier)
    return [f"PICK {c.name} — {c.reason}" for c in picks[:6]]
