"""Opponent profiles — who you're up against, so the coach can steer you toward a
board that beats THIS lobby, not a generic strong board.

During each combat the log reveals the opponent's hero, board, tribes, keywords,
strength and health. We profile every opponent we fight and keep the latest read,
then summarize the lobby's threats (how strong, which tribes, how many Divine
Shields / Taunts / Poisonous) so the recommender can:
  * value tech cards against what the lobby actually runs, and
  * tell you what to scale past to keep winning.

Pure functions over the GameState entity map — bg.py persists the results.
"""

from typing import Dict, List, Optional

_TRIBES = ("murloc", "beast", "dragon", "mech", "elemental", "undead", "demon",
           "pirate", "naga", "quilboar")
_KEYWORDS = ("DIVINE_SHIELD", "TAUNT", "POISONOUS", "VENOMOUS", "REBORN", "WINDFURY")


def _int(ent, tag):
    try:
        return int(ent.tags.get(tag))
    except (TypeError, ValueError):
        return None


def _stat(ent, atk_key="ATK"):
    a = _int(ent, atk_key) or 0
    h = _int(ent, "HEALTH") or 0
    return a + h


def build_profiles(entities, local_player, kb=None) -> Dict[str, dict]:
    """{controller: profile} for every foreign player whose board is on the field.
    Profile: hero, tier, dominant tribe, strength, keyword counts, health."""
    foreign_minions: Dict[str, list] = {}
    for ent in entities.values():
        if ent.tags.get("ZONE") != "PLAY":
            continue
        ctrl = ent.tags.get("CONTROLLER")
        if ctrl in (None, str(local_player)):
            continue
        if ent.tags.get("CARDTYPE") != "MINION":
            continue
        if not ent.card_id or (ent.name and "UNKNOWN ENTITY" in ent.name):
            continue
        foreign_minions.setdefault(ctrl, []).append(ent)

    idx = None
    if kb is not None:
        from .cards import by_name
        idx = by_name(kb)

    profiles: Dict[str, dict] = {}
    for ctrl, minions in foreign_minions.items():
        if not minions:
            continue
        tribes: Dict[str, int] = {}
        keywords: Dict[str, int] = {}
        strength = 0
        for m in minions:
            strength += _stat(m)
            for kw in _KEYWORDS:
                if str(m.tags.get(kw, "")) not in ("", "0", "False"):
                    keywords[kw] = keywords.get(kw, 0) + 1
            ck = idx.get(m.name) if idx and m.name else None
            for t in (getattr(ck, "tribes", None) or []):
                tribes[t.lower()] = tribes.get(t.lower(), 0) + 1
        dom = max(tribes, key=tribes.get) if tribes else None
        prof = {
            "controller": ctrl,
            "tribe": dom,
            "strength": strength,
            "keywords": keywords,
            "board_size": len(minions),
        }
        hero = _foreign_hero(entities, ctrl)
        if hero:
            prof["hero"], prof["tier"], prof["health"] = hero
        prof["trinkets"] = _foreign_trinkets(entities, ctrl)
        profiles[ctrl] = prof
    return profiles


def _foreign_hero(entities, ctrl):
    """(hero_name, tier, health) for a controller's hero, or None. Skips hero-power
    cards (the '…p' variants) — only the real hero card."""
    for ent in entities.values():
        if ent.tags.get("CARDTYPE") != "HERO" or ent.tags.get("CONTROLLER") != ctrl:
            continue
        cid = ent.card_id or ""
        if "HERO" not in cid or cid.rstrip("_se").endswith("p"):
            continue
        from .bg import _card_name
        name = ent.name if ent.name and "UNKNOWN" not in ent.name else _card_name(cid)
        tier = _int(ent, "PLAYER_TECH_LEVEL") or _int(ent, "TECH_LEVEL")
        h = _int(ent, "HEALTH")
        armor = _int(ent, "ARMOR") or 0
        dmg = _int(ent, "DAMAGE") or 0
        health = (h + armor - dmg) if h is not None else None
        return name, tier, health
    return None


def _foreign_trinkets(entities, ctrl) -> List[str]:
    out = []
    for ent in entities.values():
        if ent.tags.get("CARDTYPE") != "BATTLEGROUND_TRINKET":
            continue
        if ent.tags.get("CONTROLLER") != ctrl or "MagicItem" not in (ent.card_id or ""):
            continue
        if ent.name and ent.card_id not in out:
            out.append(ent.name)
    return out


def threats(profiles: List[dict]) -> dict:
    """Lobby-wide threat summary from all known opponent profiles."""
    profs = [p for p in (profiles or []) if p]
    if not profs:
        return {}
    kw_total: Dict[str, int] = {}
    tribes: Dict[str, int] = {}
    for p in profs:
        for k, n in (p.get("keywords") or {}).items():
            kw_total[k] = kw_total.get(k, 0) + n
        if p.get("tribe"):
            tribes[p["tribe"]] = tribes.get(p["tribe"], 0) + 1
    return {
        "max_strength": max((p.get("strength") or 0) for p in profs),
        "keywords": kw_total,
        "tribes": tribes,
        "count": len(profs),
    }


def threat_note(profiles: List[dict]) -> Optional[str]:
    """One-line readout of the lobby's threats and what to tech against."""
    t = threats(profiles)
    if not t:
        return None
    bits = []
    ds = t["keywords"].get("DIVINE_SHIELD", 0)
    if ds >= 3:
        bits.append(f"{ds} Divine Shields in the lobby — value AOE/Divine-Shield-pop")
    pois = t["keywords"].get("POISONOUS", 0) + t["keywords"].get("VENOMOUS", 0)
    if pois >= 2:
        bits.append(f"{pois} Poisonous — big single minions get sniped; go wide/Divine Shield")
    if t["tribes"]:
        top = max(t["tribes"], key=t["tribes"].get)
        bits.append(f"top enemy tribe: {top.capitalize()}")
    bits.append(f"strongest board ~{t['max_strength']} stats")
    return "Threats — " + "; ".join(bits)
