"""Season 14 Activate-keyword minion cardIds.

HAS_ACTIVATE_POWER is set on nearly every recruit minion (buy/sell chrome), so
it alone cannot mean the Activate keyword. Gate Activate advice on this allowlist
(from HearthstoneJSON card text ``<b>Activate (N):</b>``) plus HAS_ACTIVATE_POWER
on a friendly board minion — never shop UI / DragBuy / DragSell.
"""

from __future__ import annotations

from typing import Optional

# Normal + golden copies with ``<b>Activate (N):</b>`` in card text (2026-09).
ACTIVATE_CARD_IDS = frozenset({
    "BG28_582",
    "BG28_582_G",
    "BG36_099",
    "BG36_099_G",
    "BG36_180",
    "BG36_180_G",
    "BG36_201",
    "BG36_201_G",
    "BG36_240",
    "BG36_240_G",
    "BG36_243",
    "BG36_243_G",
    "BG36_300",
    "BG36_300_G",
    "BG36_311",
    "BG36_311_G",
    "BG36_312",
    "BG36_312_G",
    "BG36_342",
    "BG36_342_G",
    "BG36_345",
    "BG36_345_G",
    "BG36_346",
    "BG36_346_G",
    "BG36_354",
    "BG36_354_G",
    "BG36_356",
    "BG36_356_G",
    "BG36_362",
    "BG36_362_G",
    "BG36_370",
    "BG36_370_G",
    "BG36_503",
    "BG36_503_G",
    "BG36_506",
    "BG36_506_G",
    "BG36_507",
    "BG36_507_G",
    "BG36_508",
    "BG36_508_G",
    "BG36_509",
    "BG36_509_G",
    "BG36_511",
    "BG36_511_G",
    "BG36_621",
    "BG36_621_G",
    "BG36_700",
    "BG36_700_G",
    "BG36_701",
    "BG36_701_G",
})


def is_activate_minion(card_id: Optional[str], name: Optional[str] = None) -> bool:
    """True if this cardId is a real Activate-keyword BG minion (not sell chrome)."""
    if not card_id:
        return False
    if card_id in ACTIVATE_CARD_IDS:
        return True
    # Triple/enchantment suffixes occasionally appear; strip common golden marker.
    if card_id.endswith("_G") and card_id[:-2] in ACTIVATE_CARD_IDS:
        return True
    return False
