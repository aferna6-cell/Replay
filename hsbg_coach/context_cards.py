"""Context cards — heroes' powers, trinkets and anomalies.

`cards.py` builds the *minion* knowledge base. This module builds the other half
of the state: the cards that don't sit on the board but decide how the whole game
plays — **hero powers**, **trinkets** and **anomalies**.

They are not a long tail. HearthstoneJSON tags them as first-class types, they are
few (hundreds, not thousands), every game has them, and they rewrite what a good
line is. A model that ignores them is playing a different game.

Source: HearthstoneJSON (same feed as `cards.py`), joined to the Firestone value
priors we already commit — `firestone_trinket_stats.json` (average position, pick
rate, tier) and `firestone_hero_stats.json` (average position, best tribes,
playstyle). Stored at ``data/cards/bg_context.json``, refreshed via the
`refresh-context` CLI.

Anomalies carry no public per-anomaly stats, so they arrive with text only.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .firestone_stats import _fetch_json, CARDS_URL

_CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cards")
_STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stats")
BG_CONTEXT = os.path.join(_CARDS_DIR, "bg_context.json")

TRINKET_STATS = os.path.join(_STATS_DIR, "firestone_trinket_stats.json")
HERO_STATS = os.path.join(_STATS_DIR, "firestone_hero_stats.json")

# HearthstoneJSON `type` values, and the kind we file them under.
HERO_POWER = "hero_power"
TRINKET = "trinket"
ANOMALY = "anomaly"

_TYPE_TO_KIND = {
    "BATTLEGROUND_TRINKET": TRINKET,
    "BATTLEGROUND_ANOMALY": ANOMALY,
}
# Hero powers are plain HERO_POWER cards; the BG ones are the BATTLEGROUNDS set.
_HERO_POWER_SETS = {"BATTLEGROUNDS"}

KINDS = (HERO_POWER, TRINKET, ANOMALY)


@dataclass
class ContextCard:
    card_id: str
    name: str
    kind: str                              # hero_power | trinket | anomaly
    text: str = ""
    cost: Optional[int] = None             # hero-power / trinket gold cost
    # Firestone priors — present for trinkets and hero powers, absent for anomalies.
    avg_position: Optional[float] = None
    pick_rate: Optional[float] = None
    tier: Optional[str] = None             # Firestone letter tier (S/A/B/…)
    best_tribes: List[str] = field(default_factory=list)
    playstyle: Optional[str] = None

    @property
    def known_strength(self) -> Optional[float]:
        """Lower is better (average finishing position). None when unmeasured."""
        return self.avg_position


def _clean(text: Optional[str]) -> str:
    """HearthstoneJSON text carries layout markup we never want downstream."""
    out = (text or "").replace("[x]", " ")
    for tag in ("<b>", "</b>", "<i>", "</i>"):
        out = out.replace(tag, "")
    return " ".join(out.split())


def _load_priors() -> Dict[str, dict]:
    """card_id -> Firestone prior row, across trinket + hero stat files."""
    priors: Dict[str, dict] = {}
    for path, key in ((TRINKET_STATS, "trinkets"), (HERO_STATS, "heroes")):
        if not os.path.isfile(path):
            continue
        rows = json.load(open(path, encoding="utf-8")).get(key) or []
        for r in rows:
            cid = r.get("cardId")
            if cid:
                priors[cid] = r
    return priors


def build_context_kb(cards_source: str = CARDS_URL) -> Dict[str, ContextCard]:
    """Build the hero-power / trinket / anomaly KB from a HearthstoneJSON source."""
    cards = _fetch_json(cards_source) if isinstance(cards_source, str) else cards_source
    priors = _load_priors()
    kb: Dict[str, ContextCard] = {}
    for c in cards:
        ctype = c.get("type")
        kind = _TYPE_TO_KIND.get(ctype)
        if kind is None:
            if ctype != "HERO_POWER" or c.get("set") not in _HERO_POWER_SETS:
                continue
            kind = HERO_POWER
        text = _clean(c.get("text"))
        if not text:                        # textless entries carry no mechanics
            continue
        prior = priors.get(c["id"], {})
        kb[c["id"]] = ContextCard(
            card_id=c["id"],
            name=c.get("name", c["id"]),
            kind=kind,
            text=text,
            cost=c.get("cost"),
            avg_position=prior.get("averagePosition"),
            pick_rate=prior.get("pickRate"),
            tier=prior.get("tier"),
            best_tribes=list(prior.get("bestTribes") or []),
            playstyle=prior.get("playstyle"),
        )
    return kb


def save_context_kb(kb: Dict[str, ContextCard], path: str = BG_CONTEXT) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [c.__dict__ for c in sorted(kb.values(), key=lambda c: (c.kind, c.name))]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_source": "HearthstoneJSON + Firestone", "context": rows},
                  fh, indent=1)
    return path


def load_context_kb(path: str = BG_CONTEXT) -> Dict[str, ContextCard]:
    if not os.path.isfile(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    out: Dict[str, ContextCard] = {}
    for r in data.get("context", []):
        out[r["card_id"]] = ContextCard(
            card_id=r["card_id"], name=r.get("name", ""), kind=r.get("kind", ""),
            text=r.get("text", ""), cost=r.get("cost"),
            avg_position=r.get("avg_position"), pick_rate=r.get("pick_rate"),
            tier=r.get("tier"), best_tribes=list(r.get("best_tribes") or []),
            playstyle=r.get("playstyle"))
    return out


def of_kind(kb: Dict[str, ContextCard], kind: str) -> List[ContextCard]:
    return sorted((c for c in kb.values() if c.kind == kind), key=lambda c: c.name)
