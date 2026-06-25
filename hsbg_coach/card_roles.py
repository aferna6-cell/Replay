"""Situational tech cards — valued by the matchup, not in a vacuum.

The eval net scores a minion by stats + keywords, so *tech* cards — whose value
is "they answer a specific enemy board" — get rated as if their effect always
applies. The fix is to read the opponent board we last fought and adjust:

  * Tunnel Blaster (Deathrattle: 3 damage to all): strong into wide boards, stacks
    of Divine Shields (3 dmg pops a shield), and low-health swarms. Mediocre into a
    tall board of a few big minions.
  * Deadly Spore (Venomous): a one-shot answer to a single big threat. Strong when
    the enemy has a standout giant; weak into a go-wide board.

So instead of a flat discount, ``tech_assessment`` returns a *placement
adjustment* (negative = better finish) conditioned on the live opponent, plus a
reason that names the read ("pops 3 Divine Shields"). With no opponent known we
fall back to the default discount — situational, so don't make it the auto-pick.

Keyed by cardId (stable) with name fallbacks. Extend as more get flagged.
"""

from typing import List, Optional, Tuple

TUNNEL_BLASTER = "BG_DAL_775"
DEADLY_SPORE = "BGS_131"

_NAMES = {"Tunnel Blaster": TUNNEL_BLASTER, "Deadly Spore": DEADLY_SPORE}

# Placement units. Positive = worse finish (discourage); negative = better.
_BLIND_DISCOUNT = 0.5      # bought with no opponent read -> demote from auto-pick
_FAVORED_BONUS = 0.8       # the matchup clearly wants this tech -> promote


def _card_id(card_id: Optional[str], name: Optional[str]) -> Optional[str]:
    if card_id in (TUNNEL_BLASTER, DEADLY_SPORE):
        return card_id
    return _NAMES.get(name or "")


def is_tech(card_id: Optional[str], name: Optional[str]) -> bool:
    return _card_id(card_id, name) is not None


def tech_note(card_id: Optional[str], name: Optional[str]) -> Optional[str]:
    cid = _card_id(card_id, name)
    if cid == TUNNEL_BLASTER:
        return "board-clear tech — best into wide / Divine-Shield / low-health boards"
    if cid == DEADLY_SPORE:
        return "Venomous tech — best as a one-shot answer to a single big threat"
    return None


def _stat(m, key):
    if isinstance(m, dict):
        v = m.get(key)
        if v is None and key in ("attack", "health"):
            v = (m.get("tags") or {}).get("ATK" if key == "attack" else "HEALTH")
        return v
    return getattr(m, key, None)


def _has_divine_shield(m) -> bool:
    tags = m.get("tags") if isinstance(m, dict) else getattr(m, "tags", None)
    return bool(tags) and str(tags.get("DIVINE_SHIELD", "")) == "1"


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def tech_assessment(card_id: Optional[str], name: Optional[str],
                    opponent_board: Optional[List]) -> Optional[Tuple[float, str]]:
    """(placement_adjustment, reason) for a tech card given the last enemy board,
    or None if this isn't a tech card. Negative adjustment = recommend it more."""
    cid = _card_id(card_id, name)
    if cid is None:
        return None
    opp = opponent_board or []

    if cid == TUNNEL_BLASTER:
        shields = sum(1 for m in opp if _has_divine_shield(m))
        width = len(opp)
        low_hp = sum(1 for m in opp if 0 < _as_int(_stat(m, "health")) <= 3)
        if shields >= 2:
            return -_FAVORED_BONUS, f"pops {shields} Divine Shields + clears chip damage"
        if width >= 5 or low_hp >= 3:
            return -_FAVORED_BONUS, f"clears their wide board ({width} minions, 3 to all)"
        if not opp:
            return _BLIND_DISCOUNT, tech_note(cid, name)
        return _BLIND_DISCOUNT, "board-clear tech — their board isn't wide/shielded enough"

    if cid == DEADLY_SPORE:
        if opp:
            big = max(opp, key=lambda m: _as_int(_stat(m, "health")) + _as_int(_stat(m, "attack")))
            atk, hp = _as_int(_stat(big, "attack")), _as_int(_stat(big, "health"))
            nm = (big.get("name") if isinstance(big, dict) else getattr(big, "name", None)) or "their biggest"
            if atk >= 12 or hp >= 12:
                return -_FAVORED_BONUS, f"Venomous trades up into {nm} ({atk}/{hp})"
            return _BLIND_DISCOUNT, "Venomous tech — no single big threat to answer yet"
        return _BLIND_DISCOUNT, tech_note(cid, name)

    return None
