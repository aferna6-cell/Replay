"""Hero-aware contextual scripts that can become NEXT (Aidan playtest).

Generic shop EV is not enough — infer from hero + hand + board. Small table of
named scripts that emit concrete advice lines (and soft placement nudges) so the
coach actually plays into the hero / hand items.

Start set:
  * Lens Case (and similar hand MagicItems) — never silent; Play when in hand
  * Trade Prince Gallywix — cycle minions (buy/sell churn for economy HP)
  * Stronger hero-power use lives in jeef_priors.hero_power_adjust
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Playable hand trinkets / items (not board-equipped). Lens Case generates
# Duplicating Lens every 2 turns — must be played from hand, never ignored.
_HAND_ITEM_IDS = {
    "BG35_MagicItem_817": "Lens Case",
}
_HAND_ITEM_NAMES = {v.lower(): k for k, v in _HAND_ITEM_IDS.items()}
_HAND_ITEM_NAME_NEEDLES = ("lens case",)

# Heroes that want buy/sell cycling (gold → HP value).
_GALLYWIX_IDS = {"TB_BaconShop_HERO_10"}
_GALLYWIX_NAMES = {"trade prince gallywix", "gallywix"}


def _get(snap, key, default=None):
    return snap.get(key, default) if isinstance(snap, dict) else getattr(snap, key, default)


def _cid(m) -> str:
    if isinstance(m, dict):
        return str(m.get("card_id") or m.get("id") or "")
    return str(getattr(m, "card_id", None) or getattr(m, "id", None) or "")


def _name(m) -> str:
    if isinstance(m, dict):
        return str(m.get("name") or m.get("card_id") or "?")
    return str(getattr(m, "name", None) or getattr(m, "card_id", None) or "?")


def is_hand_playable_item(m) -> bool:
    """True for Lens Case / similar MagicItems that sit in HAND and must be played."""
    cid = _cid(m)
    if cid in _HAND_ITEM_IDS:
        return True
    if "MagicItem" in cid:
        nm = _name(m).lower()
        if any(n in nm for n in _HAND_ITEM_NAME_NEEDLES):
            return True
        # Unknown MagicItem in hand — still playable (better than silent).
        tags = (m.get("tags") if isinstance(m, dict) else getattr(m, "tags", None)) or {}
        ctype = str(tags.get("CARDTYPE") or "")
        if ctype in ("BATTLEGROUND_TRINKET", "BATTLEGROUND_ITEM", ""):
            # Equipped trinkets are zone PLAY; hand MagicItems are playable.
            return True
    nm = _name(m).lower()
    return nm in _HAND_ITEM_NAMES or any(n in nm for n in _HAND_ITEM_NAME_NEEDLES)


def _is_gallywix(snapshot) -> bool:
    hero = (_get(snapshot, "hero") or "")
    hname = (_get(snapshot, "hero_name") or "")
    if str(hero) in _GALLYWIX_IDS:
        return True
    blob = f"{hero} {hname}".lower()
    return any(n in blob for n in _GALLYWIX_NAMES)


def hand_item_play_lines(snapshot) -> List[str]:
    """Advice lines for playable hand items (Lens Case etc.). Lead-quality NEXT."""
    hand = list(_get(snapshot, "hand", []) or [])
    out = []
    for m in hand:
        if not is_hand_playable_item(m):
            continue
        nm = _name(m)
        if "lens" in nm.lower():
            out.append(
                f"Play {nm} from hand — generates Duplicating Lens (hero script)"
            )
        else:
            out.append(f"Play {nm} from hand — hand item, play it (hero script)")
    return out


def gallywix_cycle_lines(snapshot, kb=None) -> List[str]:
    """When hero is Gallywix: bias sell+buy cycle when gold allows and shop has bodies."""
    if not _is_gallywix(snapshot):
        return []
    gold = int(_get(snapshot, "gold") or 0)
    if gold < 3:
        return []
    board = list(_get(snapshot, "board", []) or [])
    shop = list(_get(snapshot, "shop", []) or [])
    if not shop:
        return []
    # Prefer cycling a weak/off-plan board piece for a shop body.
    from .game_value import _keep_value
    from .board_value import _name as _mname
    if not board:
        # Empty-ish: just buy something to start the cycle.
        tgt = shop[0]
        return [f"Buy {_name(tgt)} — Gallywix: start cycling (hero script)"]
    weakest = min(board, key=lambda m: _keep_value(m, board, kb))
    wname = _mname(weakest)
    # Pick a shop unit that isn't trash vs board (any body to churn).
    def stats(m):
        if isinstance(m, dict):
            return int(m.get("attack") or 0) + int(m.get("health") or 0)
        return 0
    best = max(shop, key=stats)
    bname = _name(best)
    if gold >= 4 and len(board) >= 7:
        return [
            f"Sell {wname}, then buy {bname} — Gallywix cycle (hero script)"
        ]
    if gold >= 3:
        return [
            f"Buy {bname} (cycle) — Gallywix wants spend/churn (hero script)"
        ]
    return []


def hero_script_lines(snapshot, kb=None) -> List[str]:
    """All hero/hand contextual lines that can lead NEXT (best-effort order)."""
    lines: List[str] = []
    lines.extend(hand_item_play_lines(snapshot))
    lines.extend(gallywix_cycle_lines(snapshot, kb=kb))
    return lines


def gallywix_buy_adjust(action, snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Soft placement nudge: Gallywix prefers BUY (cycle) over idle rolling."""
    from .actions import BUY
    if not _is_gallywix(snapshot):
        return 0.0, None
    if getattr(action, "kind", None) != BUY:
        return 0.0, None
    gold = int(_get(snapshot, "gold") or 0)
    if gold < 3:
        return 0.0, None
    return -0.22, "Gallywix — cycle: buy to churn gold into HP value"


def gallywix_roll_adjust(snapshot, kb=None) -> Tuple[float, Optional[str]]:
    """Demote ROLL when Gallywix has gold and a non-empty shop to cycle."""
    if not _is_gallywix(snapshot):
        return 0.0, None
    gold = int(_get(snapshot, "gold") or 0)
    shop = list(_get(snapshot, "shop", []) or [])
    if gold >= 3 and shop:
        return 0.35, "Gallywix — don't roll past cycle fuel"
    return 0.0, None
