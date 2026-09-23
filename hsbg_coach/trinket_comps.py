"""What each HSReplay trinket guide says to play — the trinket twin of hero_comps.

Reads the ingested trinket guides (``data/hsreplay_guides/trinkets.json``) and
maps what the text names onto the live comps:

  * a card the guide names (``[[Card]]`` marks, or a live-pool card's full
    name in plain text) that is an enabler/core card of a comp — counted when
    the card is that comp's tribe or appears in at most two comps
  * a comp the guide names in words: "Commit Attack Scaling Undead",
    "Commit shop buff Demons", "Leviathan Beasts" (descriptor + tribe, or a
    distinctive descriptor like Beetles / Leviathan on its own)
  * tribes it favors ("commit to a beast comp", "Shop Buff Eles"), and tribes
    a negative clause names ("… NOT Undeads") as avoided
  * tribes outside this patch (Nagas) — a guide that only works there is dead
  * every live-pool card it names becomes a trinket buy

HSReplay's trinket ``favorable_tribes`` field is empty for every trinket, so
the text is the only signal. Nothing here writes strategy: every signal is a
card, comp or tribe HSReplay's trinket guide names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from .hero_comps import _MARK, _MAX_SHARED, _NEG, _SENT, _TRIBE_WORDS, _plain

# Trinket guides also use nicknames ("Shop Buff Eles", "Commit Quils").
_TRIBE_PATTERNS = dict(_TRIBE_WORDS, Elemental=r"elementals?|eles?",
                       Quilboar=r"quilboars?|quils?")
_OFF_POOL = {"Naga": r"nagas?"}
# Comp descriptors that identify the comp without the tribe word next to them.
_DISTINCT = {"beetle", "leviathan", "lobstah", "tidecaller", "venom", "evoker",
             "bristlemane", "tempest", "unbound"}
_MIN_PLAIN = 6          # plain-text card names shorter than this are too noisy


@dataclass
class TrinketFit:
    trinket: str
    comps: Dict[str, List[str]] = field(default_factory=dict)  # comp -> why (cards / "named")
    tribes: List[str] = field(default_factory=list)            # favored tribes
    avoid: List[str] = field(default_factory=list)             # avoided tribes
    buys: List[str] = field(default_factory=list)              # live-pool cards named
    off_pool: List[str] = field(default_factory=list)          # tribes not in this patch

    @property
    def leans(self) -> bool:
        """The guide points at a comp or tribe (live or not)."""
        return bool(self.comps or self.tribes or self.off_pool)

    @property
    def dead(self) -> bool:
        """The guide only works with tribes that aren't in this patch."""
        return bool(self.off_pool) and not (self.comps or self.tribes)

    def rank(self, comp_name: str, tribe: Optional[str] = None) -> int:
        """0 = guide names this comp, 1 = guide favors its tribe, 2 = neither."""
        if comp_name in self.comps:
            return 0
        return 1 if tribe is not None and tribe in self.tribes else 2

    def fits(self, comp_name: str, tribe: Optional[str] = None) -> bool:
        return self.rank(comp_name, tribe) < 2


def _stem(word: str) -> str:
    w = word.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _words(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z]+", text.lower())}


def _tribe_in(text_low: str, tribe: str) -> bool:
    pat = _TRIBE_PATTERNS.get(tribe)
    return bool(pat and re.search(rf"\b(?:{pat})\b", text_low))


def _comps_named(clause: str, comps) -> List[str]:
    """Live comps a clause names in words (descriptor + tribe)."""
    low = clause.lower()
    words = _words(low)
    out = []
    for ci in comps:
        desc = ci.name.split(" - ", 1)[1] if " - " in ci.name else ci.name
        for alt in desc.split("/"):
            need = _words(alt)
            if not need or not need <= words:
                continue
            if _tribe_in(low, ci.tribe) or need & _DISTINCT:
                out.append(ci.name)
                break
    return out


@lru_cache(maxsize=1)
def _plain_names() -> Dict[str, str]:
    from .lobby_playbook import _pool_names_cased
    return {n.lower(): n for n in _pool_names_cased() if len(n) >= _MIN_PLAIN}


@lru_cache(maxsize=512)
def _fit_for(key: str) -> TrinketFit:
    """Fit for one trinket row, keyed by card id (Lesser/Greater share names)."""
    from .hsreplay_guides import lookup_trinket
    from .lobby_playbook import card_tribes, in_live_pool, live_comp_infos

    row = lookup_trinket(key)
    fit = TrinketFit((row or {}).get("name") or key)
    text = (row or {}).get("guide_text") or ""
    if not text:
        return fit
    comps = live_comp_infos()
    shared: Dict[str, int] = {}
    for ci in comps:
        for n in ci.core:
            shared[n] = shared.get(n, 0) + 1
    plain_names = _plain_names()

    for sent in _SENT.split(text):
        m = _NEG.search(sent)
        pos, neg = (sent[:m.start()], sent[m.start():]) if m else (sent, "")
        for clause, bucket in ((pos, fit.tribes), (neg, fit.avoid)):
            low = _plain(clause).lower()
            for tribe in _TRIBE_PATTERNS:
                if _tribe_in(low, tribe) and tribe not in bucket:
                    bucket.append(tribe)
            if bucket is fit.tribes:
                for tribe, pat in _OFF_POOL.items():
                    if re.search(rf"\b(?:{pat})\b", low) and tribe not in fit.off_pool:
                        fit.off_pool.append(tribe)
        pos_plain = _plain(pos)
        for name in _comps_named(pos_plain, comps):
            fit.comps.setdefault(name, [])
        # Cards: [[marks]] first, then full live-pool names written plainly.
        cards = [c.strip() for c in _MARK.findall(pos)]
        unmarked = _MARK.sub(" ", pos).lower()
        cards += [real for low_name, real in plain_names.items()
                  if re.search(rf"\b{re.escape(low_name)}\b", unmarked)]
        for card in cards:
            if not in_live_pool(card):
                continue
            if card not in fit.buys:
                fit.buys.append(card)
            own = set(card_tribes(card))
            for ci in comps:
                if card in ci.core and (ci.tribe in own or shared.get(card, 0) <= _MAX_SHARED):
                    got = fit.comps.setdefault(ci.name, [])
                    if card not in got:
                        got.append(card)
    fit.avoid = [t for t in fit.avoid if t not in fit.tribes]
    # A comp the guide names implies its tribe.
    for name in fit.comps:
        tribe = next((ci.tribe for ci in comps if ci.name == name), None)
        if tribe and tribe not in fit.tribes and tribe not in fit.avoid:
            fit.tribes.append(tribe)
    return fit


def fit_for_name(trinket_name: Optional[str]) -> TrinketFit:
    """TrinketFit for a trinket name or card id (empty fit when unknown)."""
    if not trinket_name:
        return TrinketFit("")
    from .hsreplay_guides import lookup_trinket
    row = lookup_trinket(trinket_name)
    if not row:
        return TrinketFit(str(trinket_name))
    return _fit_for(row.get("card_id") or row.get("name"))


def owned_fits(snapshot) -> List[TrinketFit]:
    """Fits for the trinkets the player has equipped (snapshot ``trinkets``)."""
    items = (snapshot.get("trinkets") if isinstance(snapshot, dict)
             else getattr(snapshot, "trinkets", None)) or []
    out = []
    for t in items:
        keys = [t.get("card_id"), t.get("name")] if isinstance(t, dict) else [t]
        fit = TrinketFit("")
        for key in keys:
            fit = fit_for_name(key)
            if fit.leans or fit.buys:
                break
        if fit.leans or fit.buys:
            out.append(fit)
    return out


# Pick-time weights, placement units (negative = better); they convert to
# 1st-place points in draft.rank_trinkets.
_PLAN_COMP = -0.90      # guide names the locked comp
_PLAN_TRIBE = -0.50     # guide favors the locked comp's tribe
_OFF_PLAN = 0.60        # guide wants another comp / tribe than the lock
_LOBBY_FIT = -0.35      # (no lock yet) guide favors a strong lobby tribe
_NOT_IN_LOBBY = 0.60    # guide's tribes are all missing from this lobby
_DEAD = 1.00            # guide only works with an out-of-patch tribe


def pick_adjust(key: Optional[str], snapshot) -> tuple:
    """(placement delta, reason bits) for offering this trinket in this game.

    Adheres to the trinket's HSReplay guide the way the hero guide steers comp
    choice: after PLAN locks, a trinket whose guide is for that comp (or its
    tribe) is promoted and one whose guide wants something else is demoted;
    before the lock, the lobby's strong tribes decide.
    """
    fit = fit_for_name(key)
    if not fit.leans:
        return 0.0, []
    if fit.dead:
        return _DEAD, [f"HSReplay guide needs {'/'.join(fit.off_pool)}s — not in this patch"]
    from .lobby_playbook import comp_by_name, state_of
    st = state_of(snapshot) if snapshot is not None else None
    ci = comp_by_name(st.plan) if st is not None and st.committed else None
    if ci is not None:
        r = fit.rank(ci.name, ci.tribe)
        if r == 0:
            return _PLAN_COMP, [f"HSReplay guide: made for {ci.name} (your PLAN)"]
        if r == 1:
            return _PLAN_TRIBE, [f"HSReplay guide fits {ci.tribe}s (your PLAN)"]
        want = next(iter(fit.comps), None) or "/".join(fit.tribes)
        return _OFF_PLAN, [f"HSReplay guide wants {want} — not your PLAN"]
    if st is None or not st.lobby_known or not fit.tribes:
        return 0.0, []
    hits = [t for t in fit.tribes if t in st.strong]
    if hits:
        return _LOBBY_FIT, [f"HSReplay guide fits {'/'.join(hits)} (strong in lobby)"]
    if not [t for t in fit.tribes if t in st.lobby]:
        return _NOT_IN_LOBBY, [f"HSReplay guide wants {'/'.join(fit.tribes)} — not in this lobby"]
    return 0.0, []


def best_rank(fits: Sequence[TrinketFit], comp_name: str, tribe: Optional[str]) -> int:
    return min((f.rank(comp_name, tribe) for f in fits), default=2)


def all_fits() -> List[TrinketFit]:
    from .hsreplay_guides import load_trinkets
    return [_fit_for(t.get("card_id") or t["name"])
            for t in load_trinkets().get("trinkets") or []
            if t.get("name") and t.get("guide_text")]
