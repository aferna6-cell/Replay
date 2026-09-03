"""Comp signals — what your trinkets / hero / board are telling you to build, so
the coach can make you AWARE of what to leverage and look for in the mid-late game.

Two outputs:
  * ``buy_bias(snapshot)`` — adjustments the recommender folds in (e.g. a
    spell-reward trinket makes tavern-spell buys worth more than usual).
  * ``guidance(snapshot)`` — a one-line strategic readout for the overlay
    ("Your <trinket> rewards Tavern spells — buy/play more spells than usual").

Trinket effects are read from their real text (firestone_trinket_stats.json), so
this generalizes across the whole trinket pool, not a hand-coded few.
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
           "pirate", "naga", "quilboar")


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
        txt = _text_for(cid, name)
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
            if txt.count(tr) >= 1 and (f"your {tr}" in txt or f"{tr}s" in txt):
                return tr, name
    return None, None


def buy_bias(snapshot) -> Dict[str, float]:
    """Buy-priority biases the recommender folds in. Negative = promote (placement
    units). Currently: a spell-reward trinket makes tavern-spell buys worth more."""
    bias = {}
    s, _ = spell_lean(snapshot)
    if s:
        bias["spell"] = -0.6 * s            # buy more tavern spells than usual
    return bias


def guidance(snapshot) -> Optional[str]:
    """One-line mid/late-game strategic readout: what to leverage and look for."""
    bits = []
    s, sname = spell_lean(snapshot)
    if s and sname:
        bits.append(f"{sname} rewards Tavern spells — buy/play more spells than usual")
    tribe, tname = tribe_lean(snapshot)
    if tribe and tname:
        bits.append(f"{tname} pushes {tribe.capitalize()}s — prioritize that tribe")
    return " · ".join(bits) if bits else None
