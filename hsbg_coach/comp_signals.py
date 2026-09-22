"""Comp signals — what your trinkets / hero / board are telling you to build, so
the coach can make you AWARE of what to leverage and look for in the mid-late game.

Two outputs:
  * ``buy_bias(snapshot)`` — adjustments the recommender folds in (e.g. a
    spell-reward trinket makes tavern-spell buys worth more than usual).
  * ``guidance(snapshot)`` — a one-line strategic readout for the overlay
    ("Your <trinket> rewards Tavern spells — buy/play more spells than usual").

Trinket effects are read from their real text (firestone_trinket_stats.json), so
this generalizes across the whole trinket pool, not a hand-coded few.

Soft only: battlecry / deathrattle / divine shield / tribe leans nudge matching
shop buys and demote selling enablers — never forced locks.
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stats",
                     "firestone_trinket_stats.json")

# Text markers that mean "this rewards casting/buying Tavern spells".
_SPELL_MARKERS = ("tavern spell", "cast a spell", "whenever you cast", "spellcraft",
                  "after you cast")
_TRIBES = ("murloc", "beast", "dragon", "mech", "elemental", "undead", "demon",
           "pirate", "naga", "quilboar", "aberration")

# Keyword synergies distilled from trinket text → HSJSON mechanic token.
# Soft placement nudge when a shop minion's keywords/text match.
_KEYWORD_MARKERS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("battlecry", ("battlecry",), "BATTLECRY"),
    ("deathrattle", ("deathrattle",), "DEATHRATTLE"),
    ("divine_shield", ("divine shield",), "DIVINE_SHIELD"),
    ("reborn", ("reborn",), "REBORN"),
    ("taunt", ("taunt",), "TAUNT"),
    ("avenge", ("avenge",), "AVENGE"),
    ("magnetic", ("magnetic", "magnetize"), "MAGNETIC"),
    ("windfury", ("windfury",), "WINDFURY"),
)

# Soft placement units (negative = promote buy / prefer keep).
_KEYWORD_BUY_BOOST = -0.38
_TRIBE_BUY_BOOST = -0.34
_KEYWORD_SELL_PEN = 0.55
_TRIBE_SELL_PEN = 0.42


@lru_cache(maxsize=1)
def _trinket_text():
    by_id, by_name = {}, {}
    try:
        data = json.load(open(_PATH, encoding="utf-8"))
        rows = data.get("trinkets", data) if isinstance(data, dict) else data
    except Exception:
        return by_id, by_name
    for r in rows:
        if not isinstance(r, dict):
            continue
        txt = (r.get("text") or "").lower().replace("[x]", "")
        if r.get("cardId"):
            by_id[r["cardId"]] = txt
        if r.get("name"):
            by_name[r["name"].lower()] = txt
    return by_id, by_name


def _text_for(card_id: Optional[str], name: Optional[str]) -> str:
    by_id, by_name = _trinket_text()
    if card_id and card_id in by_id:
        return by_id[card_id]
    if name and name.lower() in by_name:
        return by_name[name.lower()]
    return ""


def _trinket_texts(snapshot) -> List[Tuple[str, str]]:
    out = []
    for t in (snapshot.get("trinkets") or []):
        name = t.get("name") if isinstance(t, dict) else None
        cid = t.get("card_id") if isinstance(t, dict) else None
        # Prefer live text on the snapshot (discover/equip parse); else DB.
        live = ""
        if isinstance(t, dict):
            live = (t.get("text") or "").lower().replace("[x]", "")
        txt = live or _text_for(cid, name)
        if txt:
            out.append((name or cid or "trinket", txt))
    return out


def spell_lean(snapshot) -> Tuple[float, Optional[str]]:
    """(strength 0..1, trinket_name) if a trinket rewards Tavern spells."""
    for name, txt in _trinket_texts(snapshot):
        if any(m in txt for m in _SPELL_MARKERS):
            return 1.0, name
    return 0.0, None


def tribe_lean(snapshot) -> Tuple[Optional[str], Optional[str]]:
    """(tribe, trinket_name) if a trinket clearly pushes one tribe."""
    for name, txt in _trinket_texts(snapshot):
        for tr in _TRIBES:
            if txt.count(tr) >= 1 and (f"your {tr}" in txt or f"{tr}s" in txt
                                       or f"a {tr}" in txt or f"friendly {tr}" in txt):
                return tr, name
    return None, None


def keyword_leans(snapshot) -> List[Tuple[str, str, str]]:
    """[(tag, MECHANIC, trinket_name), ...] from equipped trinket text."""
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for name, txt in _trinket_texts(snapshot):
        for tag, needles, mechanic in _KEYWORD_MARKERS:
            if any(n in txt for n in needles) and tag not in seen:
                seen.add(tag)
                out.append((tag, mechanic, name))
    return out


def buy_bias(snapshot) -> Dict[str, float]:
    """Buy-priority biases the recommender folds in. Negative = promote (placement
    units). Spell-reward trinkets + keyword/tribe leans."""
    bias: Dict[str, float] = {}
    s, _ = spell_lean(snapshot)
    if s:
        bias["spell"] = -0.6 * s            # buy more tavern spells than usual
    for tag, _mech, _name in keyword_leans(snapshot):
        bias[tag] = _KEYWORD_BUY_BOOST
    tribe, _ = tribe_lean(snapshot)
    if tribe:
        bias[f"tribe:{tribe}"] = _TRIBE_BUY_BOOST
    return bias


def _minion_keywords(minion, kb=None) -> List[str]:
    """Uppercase mechanic tokens for a shop/board minion."""
    kws: List[str] = []
    if isinstance(minion, dict):
        raw = minion.get("keywords") or []
        kws.extend(str(k).upper() for k in raw)
        tags = minion.get("tags") or {}
        for tok in ("BATTLECRY", "DEATHRATTLE", "DIVINE_SHIELD", "REBORN",
                    "TAUNT", "AVENGE", "MAGNETIC", "WINDFURY"):
            if tags.get(tok) in (1, "1", True):
                kws.append(tok)
        text = (minion.get("text") or "").lower()
    else:
        raw = getattr(minion, "keywords", None) or []
        kws.extend(str(k).upper() for k in raw)
        text = (getattr(minion, "text", None) or "").lower()
    # Fall back to card knowledge base.
    if kb is not None:
        try:
            from .cards import by_name
            cid = (minion.get("card_id") if isinstance(minion, dict)
                   else getattr(minion, "card_id", None))
            name = (minion.get("name") if isinstance(minion, dict)
                    else getattr(minion, "name", None))
            ck = (kb.get(cid) if cid else None) or by_name(kb).get(name or "")
            if ck is not None:
                kws.extend(str(k).upper() for k in (ck.keywords or []))
                text = text or (ck.text or "").lower()
        except Exception:
            pass
    # Text fallback for missing mechanic tags.
    for tag, needles, mechanic in _KEYWORD_MARKERS:
        if mechanic not in kws and any(n in text for n in needles):
            kws.append(mechanic)
    return list(dict.fromkeys(kws))


def _minion_tribes(minion, kb=None) -> List[str]:
    tribes = []
    if isinstance(minion, dict):
        raw = minion.get("tribes") or []
        if isinstance(raw, str):
            raw = [raw]
        tribes.extend(str(t).lower() for t in raw)
        if minion.get("tribe"):
            tribes.append(str(minion["tribe"]).lower())
    else:
        raw = getattr(minion, "tribes", None) or []
        tribes.extend(str(t).lower() for t in raw)
    if kb is not None and not tribes:
        try:
            from .cards import by_name
            cid = (minion.get("card_id") if isinstance(minion, dict)
                   else getattr(minion, "card_id", None))
            name = (minion.get("name") if isinstance(minion, dict)
                    else getattr(minion, "name", None))
            ck = (kb.get(cid) if cid else None) or by_name(kb).get(name or "")
            if ck is not None:
                tribes.extend(str(t).lower() for t in (ck.tribes or []))
        except Exception:
            pass
    return list(dict.fromkeys(tribes))


def minion_trinket_buy_adjust(minion, snapshot, kb=None
                              ) -> Tuple[float, Optional[str]]:
    """Soft placement nudge for a BUY that plays into equipped trinkets.

    Negative = promote. Example: Battlecry trinket → boost Battlecry minions.
    """
    if minion is None or not (snapshot.get("trinkets") if isinstance(snapshot, dict)
                              else getattr(snapshot, "trinkets", None)):
        return 0.0, None
    snap = snapshot if isinstance(snapshot, dict) else (
        snapshot.to_dict() if hasattr(snapshot, "to_dict") else {"trinkets": []})
    adj = 0.0
    reason = None
    kws = set(_minion_keywords(minion, kb))
    for tag, mechanic, tname in keyword_leans(snap):
        if mechanic in kws:
            adj += _KEYWORD_BUY_BOOST
            pretty = tag.replace("_", " ")
            reason = f"{tname} rewards {pretty} — play into your trinket"
            break  # one keyword hit is enough for the reason line
    tribe, tname = tribe_lean(snap)
    if tribe and tribe in _minion_tribes(minion, kb):
        adj += _TRIBE_BUY_BOOST
        reason = reason or f"{tname} pushes {tribe.capitalize()}s — play into your trinket"
    if not adj:
        return 0.0, None
    # Soft cap so trinket never overwhelms triples / direction gates.
    return max(-0.55, adj), reason


def sell_trinket_penalty(minion, snapshot, kb=None) -> float:
    """Extra placement penalty for selling a piece that enables equipped trinkets."""
    if minion is None:
        return 0.0
    snap = snapshot if isinstance(snapshot, dict) else (
        snapshot.to_dict() if hasattr(snapshot, "to_dict") else {"trinkets": []})
    if not snap.get("trinkets"):
        return 0.0
    pen = 0.0
    kws = set(_minion_keywords(minion, kb))
    for _tag, mechanic, _tname in keyword_leans(snap):
        if mechanic in kws:
            pen += _KEYWORD_SELL_PEN
            break
    tribe, _ = tribe_lean(snap)
    if tribe and tribe in _minion_tribes(minion, kb):
        pen += _TRIBE_SELL_PEN
    return min(0.9, pen)


def guidance(snapshot) -> Optional[str]:
    """One-line mid/late-game strategic readout: what to leverage and look for."""
    bits = []
    s, sname = spell_lean(snapshot)
    if s and sname:
        bits.append(f"{sname} rewards Tavern spells — buy/play more spells than usual")
    tribe, tname = tribe_lean(snapshot)
    if tribe and tname:
        bits.append(f"{tname} pushes {tribe.capitalize()}s — prioritize that tribe")
    for tag, _mech, kname in keyword_leans(snapshot):
        pretty = tag.replace("_", " ")
        bits.append(f"{kname} rewards {pretty} — prioritize matching minions")
    return " · ".join(bits) if bits else None



def off_trinket_buy_penalty(minion, snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Soft placement penalty for buys that fight an equipped keyword/tribe trinket.

    Positive = demote. Only fires when we have a clear trinket plan AND the minion
    contributes neither the keyword nor the tribe lean — keeps the play-into loop
    honest without blocking every flex body.
    """
    if minion is None:
        return 0.0, None
    snap = snapshot if isinstance(snapshot, dict) else (
        snapshot.to_dict() if hasattr(snapshot, "to_dict") else {"trinkets": []})
    if not snap.get("trinkets"):
        return 0.0, None
    leans = keyword_leans(snap)
    tribe, tname = tribe_lean(snap)
    if not leans and not tribe:
        return 0.0, None
    kws = set(_minion_keywords(minion, kb))
    tribes = set(_minion_tribes(minion, kb))
    # Contributes to at least one lean → no penalty.
    for _tag, mechanic, _tn in leans:
        if mechanic in kws:
            return 0.0, None
    if tribe and tribe in tribes:
        return 0.0, None
    # Board still thin → allow flexible bodies.
    board = snap.get("board") or []
    if len(board) < 4:
        return 0.0, None
    if leans:
        tag, _mech, tn = leans[0]
        pretty = tag.replace("_", " ")
        return 0.28, f"off-plan for {tn} ({pretty}) — prefer enabling pieces"
    if tribe and tname:
        return 0.22, f"off-{tribe} for {tname} — prefer tribe pieces"
    return 0.0, None
