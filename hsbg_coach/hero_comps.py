"""Which HSReplay comps a hero's HSReplay guide points toward (or away from).

Reads the ingested hero guide + buddy guide (``data/hsreplay_guides/heroes.json``)
and maps what the text names onto the live comps:

  * a card the guide names that is an enabler/core card of a comp — counted
    only when the card is that comp's tribe, or appears in at most two comps
    (so a universal neutral like Brann doesn't point everywhere)
  * a tribe the guide names ("go Beasts"), plus HSReplay's favorable_tribes
  * negative sentences ("try not to go Beasts or Undead") mark avoided tribes
  * cards named in the guide's buy-preference bullets become hero buys

Nothing here writes strategy: every signal is a card or tribe HSReplay's hero
guide text names. The lobby playbook uses it to break ties between comps of
the same HSReplay tier and to steer tribe choice (see lobby_playbook).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

_MARK = re.compile(r"\[\[([^\]|]+)(?:\|\|\d+)?\]\]")
_SENT = re.compile(r"(?<=[.!?])\s+")
_NEG = re.compile(r"\b(not|avoid|don'?t|never|stay away|bad|weak|worse)\b", re.I)
_TRIBE_WORDS = {
    "Beast": r"beasts?", "Demon": r"demons?", "Dragon": r"dragons?",
    "Elemental": r"elementals?", "Mech": r"mechs?", "Murloc": r"murlocs?",
    "Pirate": r"pirates?", "Quilboar": r"quilboars?", "Undead": r"undeads?",
    "Aberration": r"aberrations?",
}
# Hearthstone CARDRACE ids used by HSReplay's favorable_tribes.
_RACE_IDS = {11: "UNDEAD", 14: "MURLOC", 15: "DEMON", 17: "MECHANICAL",
             18: "ELEMENTAL", 20: "BEAST", 23: "PIRATE", 24: "DRAGON",
             43: "QUILBOAR", 92: "NAGA"}
_MAX_SHARED = 2     # a card naming more comps than this is too generic to count


@dataclass
class HeroFit:
    hero: str
    comps: Dict[str, List[str]] = field(default_factory=dict)   # comp -> cards named
    tribes: List[str] = field(default_factory=list)             # favored tribes
    avoid: List[str] = field(default_factory=list)              # avoided tribes
    buys: List[str] = field(default_factory=list)               # buy-pref cards

    def favors(self, comp_name: str, tribe: Optional[str] = None) -> bool:
        return comp_name in self.comps or (tribe is not None and tribe in self.tribes)

    def rank(self, comp_name: str, tribe: Optional[str] = None) -> int:
        """0 = guide names this comp's cards, 1 = guide favors its tribe, 2 = neither."""
        if comp_name in self.comps:
            return 0
        return 1 if tribe is not None and tribe in self.tribes else 2


def _plain(text: str) -> str:
    return _MARK.sub(lambda m: m.group(1), text or "")


@lru_cache(maxsize=256)
def _fit_for(hero_name: str) -> HeroFit:
    from .hsreplay_guides import load_heroes
    from .lobby_playbook import card_tribes, in_live_pool, live_comp_infos
    from .tribe_policy import canonicalize

    hero = next((h for h in load_heroes().get("heroes") or []
                 if h.get("name") == hero_name), None)
    fit = HeroFit(hero_name)
    if hero is None:
        return fit
    comps = live_comp_infos()
    shared: Dict[str, int] = {}
    for ci in comps:
        for n in ci.core:
            shared[n] = shared.get(n, 0) + 1

    text = " ".join(x for x in (hero.get("guide_text"), hero.get("buddy_guide_text")) if x)
    for sent in _SENT.split(text):
        # Clause split at the first negation: "…Elementals - NOT Undeads,
        # Beast, or Mechs!" favors what comes before and avoids what follows.
        m = _NEG.search(sent)
        pos, neg = (sent[:m.start()], sent[m.start():]) if m else (sent, "")
        for clause, bucket in ((pos, fit.tribes), (neg, fit.avoid)):
            low = _plain(clause).lower()
            for tribe, pat in _TRIBE_WORDS.items():
                if re.search(rf"\b{pat}\b", low) and tribe not in bucket:
                    bucket.append(tribe)
        for card in _MARK.findall(pos):
            card = card.strip()
            if not in_live_pool(card):
                continue
            own = set(card_tribes(card))
            for ci in comps:
                if card not in ci.core:
                    continue
                if ci.tribe in own or shared.get(card, 0) <= _MAX_SHARED:
                    fit.comps.setdefault(ci.name, [])
                    if card not in fit.comps[ci.name]:
                        fit.comps[ci.name].append(card)
    # HSReplay's own favorable_tribes (card race ids) are authoritative.
    hs_fav = []
    for t in hero.get("favorable_tribes") or []:
        c = canonicalize(_RACE_IDS.get(t) if isinstance(t, int) else t)
        if c and c not in hs_fav:
            hs_fav.append(c)
    fit.avoid = [t for t in fit.avoid if t not in hs_fav]
    fit.tribes = hs_fav + [t for t in fit.tribes if t not in hs_fav and t not in fit.avoid]
    for bullet in (hero.get("structured") or {}).get("buy_prefs") or []:
        for card in _MARK.findall(bullet):
            card = card.strip()
            if in_live_pool(card) and card not in fit.buys:
                fit.buys.append(card)
    return fit


def hero_fit(snapshot) -> Optional[HeroFit]:
    """HeroFit for the snapshot's hero, or None when the hero is unknown."""
    try:
        from .hsreplay_guides import lookup_hero
        hero = lookup_hero(snapshot)
    except Exception:
        hero = None
    if not hero or not hero.get("name"):
        return None
    return _fit_for(hero["name"])


def fit_for_name(hero_name: str) -> HeroFit:
    return _fit_for(hero_name)


def all_fits() -> List[Tuple[dict, HeroFit]]:
    from .hsreplay_guides import load_heroes
    return [(h, _fit_for(h.get("name"))) for h in load_heroes().get("heroes") or []
            if h.get("name")]
