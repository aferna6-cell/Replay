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
ACTIVE_POOL = os.path.join(_CARDS_DIR, "bg_active_pool.json")

TRINKET_STATS = os.path.join(_STATS_DIR, "firestone_trinket_stats.json")
HERO_STATS = os.path.join(_STATS_DIR, "firestone_hero_stats.json")

# HearthstoneJSON `type` values, and the kind we file them under.
HERO = "hero"
HERO_POWER = "hero_power"
TRINKET = "trinket"
ANOMALY = "anomaly"

_TYPE_TO_KIND = {
    "BATTLEGROUND_TRINKET": TRINKET,
    "BATTLEGROUND_ANOMALY": ANOMALY,
}
# Hero powers are plain HERO_POWER cards; the BG ones are the BATTLEGROUNDS set.
# Heroes reach theirs by `heroPowerDbfId`, which is also how a hero power inherits
# its hero's play data — the stats are measured per hero, not per power.
_HERO_POWER_SETS = {"BATTLEGROUNDS"}

KINDS = (HERO, HERO_POWER, TRINKET, ANOMALY)


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
    # In rotation right now. Anomalies come from the hand-maintained patch list;
    # everything else is active iff live play data measured it.
    active: bool = False
    hero_power_id: Optional[str] = None    # heroes only — the power they grant

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


def load_active_anomalies(path: str = ACTIVE_POOL) -> set:
    """Names of the anomalies currently in rotation (see `bg_active_pool.json`)."""
    if not os.path.isfile(path):
        return set()
    return set(json.load(open(path, encoding="utf-8")).get("anomalies") or [])


def build_context_kb(cards_source: str = CARDS_URL) -> Dict[str, ContextCard]:
    """Build the hero-power / trinket / anomaly KB from a HearthstoneJSON source."""
    cards = _fetch_json(cards_source) if isinstance(cards_source, str) else cards_source
    priors = _load_priors()
    active_anomalies = load_active_anomalies()
    by_dbf = {c.get("dbfId"): c for c in cards if c.get("dbfId")}

    # A hero power is in rotation when an active hero grants it, so resolve the
    # hero -> power links first and let the power inherit that hero's play data.
    power_prior: Dict[str, dict] = {}
    hero_power_of: Dict[str, str] = {}
    for c in cards:
        if c.get("type") != "HERO" or not c.get("battlegroundsHero"):
            continue
        power = by_dbf.get(c.get("heroPowerDbfId"))
        if not power:
            continue
        hero_power_of[c["id"]] = power["id"]
        prior = priors.get(c["id"])
        if prior and power["id"] not in power_prior:
            power_prior[power["id"]] = prior

    kb: Dict[str, ContextCard] = {}
    for c in cards:
        ctype = c.get("type")
        kind = _TYPE_TO_KIND.get(ctype)
        if kind is None:
            if ctype == "HERO" and c.get("battlegroundsHero"):
                kind = HERO
            elif ctype == "HERO_POWER" and c.get("set") in _HERO_POWER_SETS:
                kind = HERO_POWER
            else:
                continue
        text = _clean(c.get("text"))
        if not text and kind != HERO:       # a hero's mechanics live in its power
            continue
        prior = (power_prior if kind == HERO_POWER else priors).get(c["id"], {})
        name = c.get("name", c["id"])
        # Anomalies are gated by the patch list; everything else by measured play.
        active = (name in active_anomalies) if kind == ANOMALY else bool(prior)
        kb[c["id"]] = ContextCard(
            card_id=c["id"],
            name=name,
            kind=kind,
            text=text,
            cost=c.get("cost"),
            active=active,
            avg_position=prior.get("averagePosition"),
            pick_rate=prior.get("pickRate"),
            tier=prior.get("tier"),
            best_tribes=list(prior.get("bestTribes") or []),
            playstyle=prior.get("playstyle"),
            hero_power_id=hero_power_of.get(c["id"]),
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
            playstyle=r.get("playstyle"), active=bool(r.get("active")),
            hero_power_id=r.get("hero_power_id"))
    return out


def of_kind(kb: Dict[str, ContextCard], kind: str,
            active_only: bool = False) -> List[ContextCard]:
    return sorted((c for c in kb.values()
                   if c.kind == kind and (c.active or not active_only)),
                  key=lambda c: c.name)
