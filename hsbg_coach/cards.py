"""Card knowledge base — what the model needs to *understand* each minion.

Beyond win-rate numbers, the model has to know what a minion IS: its tavern
**tier**, its **tribe(s)**, its **keywords/effects** (Battlecry, Deathrattle,
Divine Shield, Magnetic, Activate, …), and its rules **text**. That's the
difference between "this card places 3.4" and knowing *why* — and it's the
substrate the synergy layer (`synergy.py`) reads.

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
# Season 14's public data represents the clickable Activate keyword as
# INTERACTABLE_OBJECT; keep it so the model can identify these cards explicitly.
KEYWORD_MECHANICS = {
    "BATTLECRY", "DEATHRATTLE", "DIVINE_SHIELD", "TAUNT", "POISONOUS", "VENOMOUS",
    "REBORN", "WINDFURY", "MEGA_WINDFURY", "MAGNETIC", "FRENZY", "STEALTH",
    "OVERKILL", "SPELLPOWER", "CLEAVE", "AVENGE", "CHOOSE_ONE",
    "INTERACTABLE_OBJECT",
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
