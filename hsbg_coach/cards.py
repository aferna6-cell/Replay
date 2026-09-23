"""Card knowledge base — what the model needs to *understand* each minion.

Beyond win-rate numbers, the model has to know what a minion IS: its tavern
**tier**, its **tribe(s)**, its **keywords/effects** (Battlecry, Deathrattle,
Divine Shield, Magnetic, …), and its rules **text**. That's the difference
between "this card places 3.4" and knowing *why* — and it's the substrate the
synergy layer (`synergy.py`) reads.

Source: HearthstoneJSON (free). We keep only Battlegrounds minions (those with a
`techLevel`) and store a slim knowledge file at ``data/cards/bg_cards.json``,
refreshable via the `refresh-cards` CLI.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .firestone_stats import _RACE_TO_TRIBE, _fetch_json, CARDS_URL

_CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cards")
BG_CARDS = os.path.join(_CARDS_DIR, "bg_cards.json")

# Mechanics (HearthstoneJSON `mechanics` strings) that matter in BG combat/scaling.
KEYWORD_MECHANICS = {
    "BATTLECRY", "DEATHRATTLE", "DIVINE_SHIELD", "TAUNT", "POISONOUS", "VENOMOUS",
    "REBORN", "WINDFURY", "MEGA_WINDFURY", "MAGNETIC", "FRENZY", "STEALTH",
    "OVERKILL", "SPELLPOWER", "CLEAVE", "AVENGE", "CHOOSE_ONE",
}


@dataclass
class CardKnowledge:
    card_id: str
    name: str
    tier: Optional[int]                 # techLevel = tavern tier
    attack: Optional[int]
    health: Optional[int]
    tribes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    text: str = ""

    def has(self, keyword: str) -> bool:
        return keyword.upper() in self.keywords


def _tribes(card: dict) -> List[str]:
    races = card.get("races") or ([card["race"]] if card.get("race") else [])
    if "ALL" in races:
        return ["All"]
    return [_RACE_TO_TRIBE[r] for r in races if r in _RACE_TO_TRIBE]


def build_card_kb(cards_source: str = CARDS_URL) -> Dict[str, CardKnowledge]:
    """Build the BG minion knowledge base from a HearthstoneJSON cards source."""
    cards = _fetch_json(cards_source) if isinstance(cards_source, str) else cards_source
    kb: Dict[str, CardKnowledge] = {}
    for c in cards:
        tier = c.get("techLevel")
        if tier is None or c.get("type") != "MINION":   # BG minions carry techLevel
            continue
        if c.get("battlegroundsNormalDbfId"):           # skip golden/triple copies
            continue
        # Prefer the live shop-pool flag when present (tokens/buddies often have
        # techLevel but are not discoverable in the tavern).
        if "isBattlegroundsPoolMinion" in c and not c.get("isBattlegroundsPoolMinion"):
            continue
        if c.get("isBattlegroundsDuosExclusive"):
            continue
        # Patch 36.6.1: Naga rotated out. HSJSON may still flag some Naga as pool
        # — hard-exclude by race so the coach never recommends them.
        races = c.get("races") or ([c["race"]] if c.get("race") else [])
        if any(str(r).upper() == "NAGA" for r in races):
            continue
        mechanics = c.get("mechanics") or []
        kb[c["id"]] = CardKnowledge(
            card_id=c["id"],
            name=c.get("name", c["id"]),
            tier=tier,
            attack=c.get("attack"),
            health=c.get("health"),
            tribes=_tribes(c),
            keywords=sorted(m for m in mechanics if m in KEYWORD_MECHANICS),
            text=(c.get("text") or "").replace("\n", " ").replace("[x]", "").strip(),
        )
    return kb


def save_kb(kb: Dict[str, CardKnowledge], path: str = BG_CARDS) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [c.__dict__ for c in sorted(kb.values(), key=lambda c: (c.tier or 0, c.name))]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_source": "HearthstoneJSON", "cards": rows}, fh, indent=1)
    return path


def load_kb(path: str = BG_CARDS) -> Dict[str, CardKnowledge]:
    if not os.path.isfile(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in data.get("cards", []):
        out[r["card_id"]] = CardKnowledge(
            card_id=r["card_id"], name=r.get("name", ""), tier=r.get("tier"),
            attack=r.get("attack"), health=r.get("health"),
            tribes=list(r.get("tribes", [])), keywords=list(r.get("keywords", [])),
            text=r.get("text", ""))
    return out


def by_name(kb: Dict[str, CardKnowledge]) -> Dict[str, CardKnowledge]:
    return {c.name: c for c in kb.values()}


# ---------------------------------------------------------------------------
# Every-Battlegrounds-card name/tribe fallback (data/cards/bg_card_names.json,
# built by scripts/build_bg_card_names.py from HearthstoneJSON). The KB above
# is only the live minion pool; shop cards outside it (tavern spells, dual-
# tribe Demon/Naga cards, cards HSJSON doesn't flag) resolve here.
# ---------------------------------------------------------------------------

BG_CARD_NAMES = os.path.join(_CARDS_DIR, "bg_card_names.json")
_NAMES_CACHE: Optional[Dict[str, dict]] = None
_NAME_INDEX: Optional[Dict[str, dict]] = None


def card_names() -> Dict[str, dict]:
    """card id -> {name, type, tier, tribes} for every Battlegrounds card."""
    global _NAMES_CACHE
    if _NAMES_CACHE is None:
        try:
            with open(BG_CARD_NAMES, encoding="utf-8") as fh:
                _NAMES_CACHE = json.load(fh).get("cards") or {}
        except (OSError, ValueError):
            _NAMES_CACHE = {}
    return _NAMES_CACHE


def fallback_card(card_id: Optional[str] = None,
                  name: Optional[str] = None) -> Optional[dict]:
    """The fallback entry for a card id (preferred) or exact name."""
    global _NAME_INDEX
    names = card_names()
    if card_id and card_id in names:
        return names[card_id]
    if not name:
        return None
    if _NAME_INDEX is None:
        # Prefer non-golden base ids so a name maps to its normal version.
        _NAME_INDEX = {}
        for cid in sorted(names, key=lambda c: (c.endswith("_G"), c)):
            _NAME_INDEX.setdefault(names[cid]["name"].lower(), names[cid])
    return _NAME_INDEX.get(name.lower())
