"""Season 14 Activate-keyword minion cardIds (pool-accurate).

HAS_ACTIVATE_POWER is set on nearly every recruit minion (buy/sell chrome), so it
alone cannot mean the Activate keyword. Gate Activate advice on this allowlist
(from HearthstoneJSON card text ``<b>Activate (N):</b>`` on current pool minions)
plus HAS_ACTIVATE_POWER on a friendly board minion — never shop UI / DragBuy /
DragSell.

Strict matching: cardId must be in ACTIVATE_CARD_IDS (or its ``_G`` twin already
listed). Name fallback is **exact** allowlist names only — no substring/fuzzy.
Naga Activate cards are excluded (tribe rotated out in 36.6.1).
"""

from __future__ import annotations

from typing import Optional

# Normal + golden copies currently in the Battlegrounds pool with Activate text.
ACTIVATE_CARD_IDS = frozenset({
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
    "BG36_503",
    "BG36_503_G",
    "BG36_506",
    "BG36_506_G",
    "BG36_507",
    "BG36_507_G",
    "BG36_509",
    "BG36_509_G",
    "BG36_511",
    "BG36_511_G",
    "BG36_621",
    "BG36_621_G",
    "BG36_701",
    "BG36_701_G",
})

# Exact display names -> cardId for the rare case the log omits cardId.
ACTIVATE_NAMES = {
    "Abyssal Envoy": "BG36_311",
    "Brain Rotter": "BG36_099",
    "Breakout Mastermind": "BG36_507",
    "Clever Castaway": "BG36_342",
    "Dead Bellringer": "BG36_511",
    "Decoy Conjurer": "BG36_354",
    "Deft Deserter": "BG36_621",
    "Drone Duplicator": "BG36_506",
    "Fruit Vendor": "BG36_346",
    "Hired Mount": "BG36_240",
    "Kelp Keeper": "BG36_701",
    "Living Prison": "BG36_180",
    "Lurking Lionfish": "BG36_201",
    "Mindbending Recruiter": "BG36_312",
    "N'raqi Frostcaller": "BG36_300",
    "Private Investigator": "BG36_509",
    "Sky-hatch Runaway": "BG36_243",
    "Soulkeeping Jailer": "BG36_503",
    "Suspicious Prisonguard": "BG36_345",
    "Tyrael": "BG36_356",
}


def is_activate_minion(card_id: Optional[str], name: Optional[str] = None) -> bool:
    """True iff this is a real Activate-keyword BG minion (not sell chrome)."""
    if card_id:
        if card_id in ACTIVATE_CARD_IDS:
            return True
        # Golden twin already listed as *_G; also accept base+_G if base listed.
        if card_id.endswith("_G") and card_id[:-2] in ACTIVATE_CARD_IDS:
            return True
        return False
    if name:
        return name.strip() in ACTIVATE_NAMES
    return False
