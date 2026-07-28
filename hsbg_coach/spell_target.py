"""Where to play a targeted tavern spell (a hand spell that buffs a minion).

When a spell like "Tavern Dish Banana" (+stats to a minion) lands in your hand,
the question is *which minion to put it on*. A permanent stat buff is worth the
most on a minion that multiplies or protects it:

  * Divine Shield / Reborn — the buff survives a hit, so it sticks.
  * Windfury / Cleave      — extra attack value is multiplied.
  * Poisonous              — already trades up; extra health keeps it alive to do
                             it again.
  * Taunt                  — protects the rest of the board.

Among minions with those payoffs, prefer the strongest; with none, buff your
biggest minion (your win condition). Heuristic, board-aware, no per-spell data
needed — it works for any "+X/+Y to a minion" tavern spell.
"""

from typing import List, Optional, Tuple

# Keyword -> how much a stat buff is amplified by having it. Tuned so a
# Divine-Shield/Windfury minion clearly beats a vanilla bigger one.
_KW = {
    "DIVINE_SHIELD": 3.0,
    "WINDFURY": 3.0,
    "MEGA_WINDFURY": 4.0,
    "REBORN": 2.0,
    "POISONOUS": 2.0,
    "VENOMOUS": 2.0,
    "CLEAVE": 2.0,
    "TAUNT": 1.0,
}

_KW_PRETTY = {
    "DIVINE_SHIELD": "Divine Shield", "WINDFURY": "Windfury",
    "MEGA_WINDFURY": "Mega-Windfury", "REBORN": "Reborn", "POISONOUS": "Poisonous",
    "VENOMOUS": "Venomous", "CLEAVE": "Cleave", "TAUNT": "Taunt",
}


def _name(m) -> Optional[str]:
    if isinstance(m, dict):
        return m.get("name") or m.get("card_id")
    return getattr(m, "name", None) or getattr(m, "card_id", None)


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


def _keywords(m) -> List[str]:
    tags = _tags(m)
    return [k for k in _KW if str(tags.get(k, "")).strip() not in ("", "0", "False")]


# Hand spells that don't target a minion (gold/economy/utility). Coins are caught
# reliably by the COIN_CARD tag; the rest by name. Everything else is treated as a
# minion buff to be placed on the best target.
_NON_TARGETED = ("coin", "recruit", "investment", "telescope", "search through time",
                 "hasty excavation", "strike oil", "staff of enrichment", "fortify")


def is_targeted(spell: dict) -> bool:
    """True if this hand spell should be played on a minion (a +stats buff)."""
    if spell.get("coin"):
        return False
    name = (spell.get("name") or "").lower()
    return not any(k in name for k in _NON_TARGETED)


def best_buff_target(board) -> Optional[Tuple[object, str]]:
    """(minion, reason) for the best place to play a +stats tavern spell, or None
    if the board is empty."""
    board = list(board or [])
    if not board:
        return None
    best, best_score, best_kws = None, -1.0, []
    for m in board:
        kws = _keywords(m)
        score = sum(_KW[k] for k in kws) + (_stat(m, "attack") + _stat(m, "health")) / 20.0
        if score > best_score:
            best, best_score, best_kws = m, score, kws
    if best is None:
        return None
    if best_kws:
        pretty = " + ".join(_KW_PRETTY[k] for k in best_kws[:2])
        reason = f"it has {pretty} — the buff scales best there"
    else:
        reason = "your strongest minion (win condition)"
    return best, reason
