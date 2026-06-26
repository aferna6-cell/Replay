"""Where to magnetize a Magnetic mech.

A Magnetic mech can attach to another Mech on your board, fusing its stats and
keywords onto the host. The question the coach must answer is *which host*: a
magnetized buff is worth the most where it's protected and multiplied —

  * Divine Shield — the added stats sit behind the shield, so they stick.
  * Reborn        — the body (now bigger) comes back once.
  * Windfury / Cleave — the added attack hits more than once / more targets.
  * Taunt         — fusing onto your wall keeps the whole board alive.

Among hosts with those payoffs, prefer the strongest; with none, fuse onto your
biggest Mech (your scaling target). Heuristic + board-aware — no per-card data
needed, it works for any Magnetic mech.

Magnetizing also keeps your board count down (no extra slot used), which is why
it's almost always better than playing the mech as a standalone body when a Mech
host exists.
"""

from typing import List, Optional, Tuple

# Same keyword amplifiers as spell_target — a buff/keyword carries best onto a
# host with these.
_KW = {
    "DIVINE_SHIELD": 3.0,
    "WINDFURY": 3.0,
    "MEGA_WINDFURY": 4.0,
    "REBORN": 2.0,
    "CLEAVE": 2.0,
    "TAUNT": 1.5,
    "POISONOUS": 1.0,
}
_KW_PRETTY = {
    "DIVINE_SHIELD": "Divine Shield", "WINDFURY": "Windfury",
    "MEGA_WINDFURY": "Mega-Windfury", "REBORN": "Reborn", "CLEAVE": "Cleave",
    "TAUNT": "Taunt", "POISONOUS": "Poisonous",
}

_MECH_TRIBES = {"mech", "mechs"}


def _name(m) -> Optional[str]:
    if isinstance(m, dict):
        return m.get("name") or m.get("card_id")
    return getattr(m, "name", None) or getattr(m, "card_id", None)


def _card_id(m) -> Optional[str]:
    return m.get("card_id") if isinstance(m, dict) else getattr(m, "card_id", None)


def _tags(m) -> dict:
    return (m.get("tags") if isinstance(m, dict) else getattr(m, "tags", None)) or {}


def _stat(m, key) -> int:
    v = m.get(key) if isinstance(m, dict) else getattr(m, key, None)
    if v is None:
        v = _tags(m).get("ATK" if key == "attack" else "HEALTH")
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _knowledge(m, kb):
    """KB entry for a minion view, by card_id then by name."""
    if not kb:
        return None
    cid = _card_id(m)
    if cid and cid in kb:
        return kb[cid]
    name = _name(m)
    for c in kb.values():
        if c.name == name:
            return c
    return None


def _is_mech(m, kb) -> bool:
    ck = _knowledge(m, kb)
    tribes = [t.lower() for t in (getattr(ck, "tribes", None) or [])] if ck else []
    if any(t in _MECH_TRIBES for t in tribes) or "all" in tribes:
        return True
    # Fall back to the live tag if the KB lookup missed (custom/unknown cards).
    return "MECH" in str(_tags(m).get("CARDRACE", "")).upper()


def is_magnetic(m, kb) -> bool:
    """True if this minion has the Magnetic keyword (can be fused onto a Mech)."""
    ck = _knowledge(m, kb)
    if ck and ck.has("MAGNETIC"):
        return True
    return str(_tags(m).get("MODULAR", "")) == "1"      # MODULAR == Magnetic tag


def _keywords(m) -> List[str]:
    tags = _tags(m)
    return [k for k in _KW if str(tags.get(k, "")).strip() not in ("", "0", "False")]


def _keywords_kb(m, kb) -> List[str]:
    ck = _knowledge(m, kb)
    kws = _keywords(m)
    if ck:
        for k in _KW:
            if k not in kws and ck.has(k):
                kws.append(k)
    return kws


def best_magnetize_target(board, kb) -> Optional[Tuple[object, str]]:
    """(host_mech, reason) for the best Mech to magnetize onto, or None if you have
    no Mech host (then the mech is played as a standalone body instead)."""
    mechs = [m for m in (board or []) if _is_mech(m, kb)]
    if not mechs:
        return None
    best, best_score, best_kws = None, -1.0, []
    for m in mechs:
        kws = _keywords_kb(m, kb)
        score = sum(_KW[k] for k in kws) + (_stat(m, "attack") + _stat(m, "health")) / 20.0
        if score > best_score:
            best, best_score, best_kws = m, score, kws
    if best is None:
        return None
    if best_kws:
        pretty = " + ".join(_KW_PRETTY[k] for k in best_kws[:2])
        reason = f"fuse onto your {pretty} mech — the buff is protected and scales"
    else:
        reason = "fuse onto your biggest mech (your scaling target)"
    return best, reason
