"""Fetch REAL Battlegrounds stats from Firestone's public data + normalize.

Firestone (Zero-to-Heroes) publishes its aggregated BG stats to a public CDN —
no account, no auth. Confirmed live 2026-06-24 (see ADR
`decisions/2026-06-24-firestone-bridge.md`). Endpoints (gzipped JSON):

  hero:  https://static.zerotoheroes.com/api/bgs/hero-stats/mmr-{mmr}/{period}/overview-from-hourly.gz.json
  comp:  https://static.zerotoheroes.com/api/bgs/comp-stats/{period}/overview-from-hourly.gz.json

  mmr:    100 (all), 50, 25, 10, 1 (top 1%)   — percentile cutoffs
  period: last-patch | past-three | past-seven

We fetch these, map Firestone's schema (BgsHeroStatsV2 / comp archetypes) onto the
normalized schema in ``stats.py`` (so the existing StatsDB / HeroContext logic
works unchanged), resolve hero card-ids to names via HearthstoneJSON, and write
``data/stats/firestone_hero_stats.json`` + ``firestone_comp_stats.json``.

`refresh-stats` (CLI) does this; the result is the default StatsDB source.
"""

import datetime
import gzip
import json
import os
import re
import urllib.request
from typing import Dict, List, Optional

BASE = "https://static.zerotoheroes.com/api/bgs"
HERO_URL = BASE + "/hero-stats/mmr-{mmr}/{period}/overview-from-hourly.gz.json"
COMP_URL = BASE + "/comp-stats/{period}/overview-from-hourly.gz.json"
CARD_URL = BASE + "/card-stats/mmr-{mmr}/{period}/overview-from-hourly.gz.json"
TRINKET_URL = BASE + "/trinket-stats/{period}/overview-from-hourly.gz.json"
CARDS_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"

# HearthstoneJSON `races` strings -> our tribe names (note MECHANICAL, not MECH).
_RACE_TO_TRIBE = {
    "MURLOC": "Murloc", "BEAST": "Beast", "DRAGON": "Dragon", "MECHANICAL": "Mech",
    "ELEMENTAL": "Elemental", "UNDEAD": "Undead", "DEMON": "Demon",
    "PIRATE": "Pirate", "QUILBOAR": "Quilboar", "NAGA": "Naga",
}

VALID_PERIODS = ("last-patch", "past-three", "past-seven")
VALID_MMR = (100, 50, 25, 10, 1)

# HearthStone CARDRACE ids -> tribe name (BG-relevant subset).
RACE_NAMES = {
    11: "Undead", 14: "Murloc", 15: "Demon", 17: "Mech", 18: "Elemental",
    20: "Beast", 23: "Pirate", 24: "Dragon", 43: "Quilboar", 92: "Naga",
}
_TRIBE_KEYWORDS = {
    "murloc": "Murloc", "beast": "Beast", "dragon": "Dragon", "mech": "Mech",
    "elemental": "Elemental", "undead": "Undead", "demon": "Demon",
    "pirate": "Pirate", "quilboar": "Quilboar", "naga": "Naga",
}


def _fetch_json(source: str) -> dict:
    """Fetch JSON from a URL (handles gzip) or read a local file."""
    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            raw = resp.read()
        if raw[:2] == b"\x1f\x8b":               # gzip magic
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))
    with open(source, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _card_meta(cards_source: str = CARDS_URL) -> Dict[str, dict]:
    """id -> {name, tribes:[...], techLevel} from HearthstoneJSON."""
    cards = _fetch_json(cards_source)
    meta = {}
    for c in cards:
        cid = c.get("id")
        if not cid:
            continue
        races = c.get("races") or ([c["race"]] if c.get("race") else [])
        tribes = [_RACE_TO_TRIBE[r] for r in races if r in _RACE_TO_TRIBE]
        meta[cid] = {"name": c.get("name", cid), "tribes": tribes,
                     "techLevel": c.get("techLevel")}
    return meta


def _names_from_meta(meta: Dict[str, dict]) -> Dict[str, str]:
    return {cid: m["name"] for cid, m in meta.items()}


def _playstyle(combat_winrate: List[dict]) -> str:
    """Derive a rough playstyle from the per-turn combat win-rate curve:
    stronger late => scaling/economy hero; stronger early => tempo hero."""
    def avg(lo, hi):
        vals = [w["winrate"] for w in combat_winrate
                if lo <= w.get("turn", 0) <= hi and w.get("winrate")]
        return sum(vals) / len(vals) if vals else 0.0
    early, late = avg(2, 5), avg(9, 13)
    if late - early > 2:
        return "economy"
    if early - late > 2:
        return "tempo"
    return "flexible"


def normalize_heroes(raw: dict, names: Dict[str, str],
                     min_data: int = 500) -> List[dict]:
    total = raw.get("dataPoints") or 1
    out = []
    for h in raw.get("heroStats", []):
        if (h.get("dataPoints") or 0) < min_data:
            continue
        cid = h.get("heroCardId", "")
        tribes = [
            t for t in (h.get("tribeStats") or [])
            if (t.get("dataPoints") or 0) >= min_data and t.get("tribe") in RACE_NAMES
        ]
        tribes.sort(key=lambda t: t.get("averagePosition", 9))
        best = [RACE_NAMES[t["tribe"]] for t in tribes[:3]]
        out.append({
            "cardId": cid,
            "name": names.get(cid, cid),
            "averagePosition": round(h.get("averagePosition", 4.5), 3),
            "pickRate": round((h.get("totalPicked") or 0) / total, 4),
            "bestTribes": best,
            "playstyle": _playstyle(h.get("combatWinrate") or []),
        })
    out.sort(key=lambda x: x["averagePosition"])
    return out


def _archetype_tribe(name: str) -> Optional[str]:
    low = name.lower()
    for kw, tribe in _TRIBE_KEYWORDS.items():
        if kw in low:
            return tribe
    return None


def _tier(avg: float) -> str:
    return ("S" if avg < 3.6 else "A" if avg < 4.0 else "B" if avg < 4.3
            else "C" if avg < 4.6 else "D")


def normalize_comps(raw: dict, min_data: int = 1000) -> List[dict]:
    total = sum(c.get("dataPoints", 0) for c in raw.get("compStats", [])) or 1
    out = []
    for c in raw.get("compStats", []):
        if (c.get("dataPoints") or 0) < min_data:
            continue
        arch = c.get("archetype", "")
        pretty = arch.replace("_", " ").title()
        avg = round(c.get("averagePlacement", 4.5), 3)
        out.append({
            "name": pretty,
            "tribe": _archetype_tribe(arch),
            "averagePosition": avg,
            "popularity": round((c.get("dataPoints") or 0) / total, 4),
            "coreCards": [],            # not in this endpoint; tribe drives advice
            "powerTurns": [],
            "tier": _tier(avg),
        })
    out.sort(key=lambda x: x["averagePosition"])
    return out


def normalize_cards(raw: dict, meta: Dict[str, dict],
                    min_play: int = 2000) -> List[dict]:
    """Per-minion stats. `impact` = placement-without minus placement-with
    (positive = you place better when you have it)."""
    out = []
    for c in raw.get("cardStats", []):
        if (c.get("totalPlayed") or 0) < min_play:
            continue
        cid = c.get("cardId", "")
        m = meta.get(cid, {})
        avg = c.get("averagePlacement")
        other = c.get("averagePlacementOther")
        impact = round(other - avg, 3) if (avg is not None and other is not None) else None
        out.append({
            "cardId": cid,
            "name": m.get("name", cid),
            "tribes": m.get("tribes", []),
            "techLevel": m.get("techLevel"),
            "averagePlacement": round(avg, 3) if avg is not None else None,
            "impact": impact,
            "totalPlayed": c.get("totalPlayed"),
        })
    out.sort(key=lambda x: x["averagePlacement"] if x["averagePlacement"] is not None else 9)
    return out


def normalize_trinkets(raw: dict, meta: Dict[str, dict],
                       min_data: int = 200) -> List[dict]:
    out = []
    for t in raw.get("trinketStats", []):
        if (t.get("dataPoints") or 0) < min_data:
            continue
        cid = t.get("trinketCardId", "")
        avg = t.get("averagePlacement")
        out.append({
            "cardId": cid,
            "name": meta.get(cid, {}).get("name", cid),
            "averagePosition": round(avg, 3) if avg is not None else None,
            "pickRate": round(t.get("pickRate") or 0, 4),
            "tier": _tier(avg) if avg is not None else "?",
        })
    out.sort(key=lambda x: x["averagePosition"] if x["averagePosition"] is not None else 9)
    return out


def inject_core_cards(comps: List[dict], cards: List[dict], per_comp: int = 4) -> None:
    """Fill each comp's coreCards with the strongest minions of its tribe."""
    by_tribe: Dict[str, List[dict]] = {}
    for c in cards:                              # cards already sorted best-first
        for tr in c.get("tribes", []):
            by_tribe.setdefault(tr, []).append(c)
    for comp in comps:
        tribe = comp.get("tribe")
        if tribe and by_tribe.get(tribe):
            comp["coreCards"] = [c["name"] for c in by_tribe[tribe][:per_comp]]


def refresh(out_dir: str, mmr: int = 100, period: str = "last-patch",
            hero_source: Optional[str] = None, comp_source: Optional[str] = None,
            card_source: Optional[str] = None, trinket_source: Optional[str] = None,
            cards_source: str = CARDS_URL) -> Dict[str, str]:
    """Fetch + normalize + write firestone_{hero,comp}_stats.json. Returns paths.

    hero_source/comp_source override the URLs (e.g. with local files) for tests
    and snapshot generation.
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {VALID_PERIODS}")
    if mmr not in VALID_MMR:
        raise ValueError(f"mmr must be one of {VALID_MMR}")
    os.makedirs(out_dir, exist_ok=True)

    hero_raw = _fetch_json(hero_source or HERO_URL.format(mmr=mmr, period=period))
    comp_raw = _fetch_json(comp_source or COMP_URL.format(period=period))
    card_raw = _fetch_json(card_source or CARD_URL.format(mmr=mmr, period=period))
    trinket_raw = _fetch_json(trinket_source or TRINKET_URL.format(period=period))

    card_meta = _card_meta(cards_source)
    names = _names_from_meta(card_meta)

    heroes = normalize_heroes(hero_raw, names)
    comps = normalize_comps(comp_raw)
    cards = normalize_cards(card_raw, card_meta)
    trinkets = normalize_trinkets(trinket_raw, card_meta)
    inject_core_cards(comps, cards)             # fill comp coreCards from card stats

    meta = {
        "_source": "Firestone (static.zerotoheroes.com/api/bgs)",
        "_fetched": datetime.date.today().isoformat() if not hero_source else "from-local",
        "_mmr": mmr, "_period": period,
        "_heroDataPoints": hero_raw.get("dataPoints"),
    }

    paths = {}
    for name, key, rows in (
        ("firestone_hero_stats.json", "heroes", heroes),
        ("firestone_comp_stats.json", "comps", comps),
        ("firestone_card_stats.json", "cards", cards),
        ("firestone_trinket_stats.json", "trinkets", trinkets),
    ):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({**meta, key: rows}, fh, indent=1)
        paths[key] = path
    return {**paths, "num_heroes": len(heroes), "num_comps": len(comps),
            "num_cards": len(cards), "num_trinkets": len(trinkets)}
