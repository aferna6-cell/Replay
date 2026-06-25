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


_UA = "Mozilla/5.0 (hsbg-coach stats fetcher)"


def _decode(raw: bytes) -> dict:
    if raw[:2] == b"\x1f\x8b":                   # gzip magic
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _fetch_url(url: str) -> bytes:
    """Fetch a URL, falling back to curl for large responses that trip urllib's
    IncompleteRead through some proxies (the comp file is ~80MB)."""
    try:
        req = urllib.request.Request(
            url, headers={"Accept-Encoding": "gzip", "User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            return resp.read()
    except Exception:
        import shutil
        import subprocess
        if not shutil.which("curl"):
            raise
        proc = subprocess.run(
            ["curl", "-fsSL", "--max-time", "300", "-A", _UA, url],
            capture_output=True, timeout=320)
        if proc.returncode != 0:
            raise RuntimeError(f"curl failed for {url}: {proc.stderr[:200]!r}")
        return proc.stdout


def _fetch_json(source: str) -> dict:
    """Fetch JSON from a URL (handles gzip) or read a local file."""
    if source.startswith("http"):
        return _decode(_fetch_url(source))
    with open(source, "rb") as fh:
        return _decode(fh.read())


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
        text = re.sub(r"<[^>]+>", "", c.get("text") or "").replace("\n", " ").strip()
        meta[cid] = {"name": c.get("name", cid), "tribes": tribes,
                     "techLevel": c.get("techLevel"), "text": text}
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
                     min_data: int = 200) -> List[dict]:
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


def _at_mmr(rows: list, mmr: int, field: str, default=None):
    """Pull a per-MMR value (placement/pickRate) from an *AtMmr array."""
    for r in rows or []:
        if r.get("mmr") == mmr:
            return r.get(field, default)
    return default


def _archetype_tribe(name: str) -> Optional[str]:
    low = name.lower()
    for kw, tribe in _TRIBE_KEYWORDS.items():
        if kw in low:
            return tribe
    return None


def _tier(avg: float) -> str:
    return ("S" if avg < 3.6 else "A" if avg < 4.0 else "B" if avg < 4.3
            else "C" if avg < 4.6 else "D")


def normalize_comps(raw: dict, min_data: int = 300, mmr: int = 100) -> List[dict]:
    out = []
    rows = raw.get("compStats", [])
    total = sum(_at_mmr(c.get("placementDistributionAtMmr"), mmr, "dataPoints", 0)
                or (c.get("dataPoints", 0) if mmr == 100 else 0) for c in rows) or 1
    for c in rows:
        # Filter + score on the requested MMR bracket, not the all-MMR overall.
        dp = (c.get("dataPoints", 0) if mmr == 100
              else _at_mmr(c.get("averagePlacementAtMmr"), mmr, "dataPoints", 0))
        if (dp or 0) < min_data:
            continue
        avg_raw = (c.get("averagePlacement") if mmr == 100
                   else _at_mmr(c.get("averagePlacementAtMmr"), mmr, "placement"))
        if avg_raw is None:
            continue
        arch = c.get("archetype", "")
        avg = round(avg_raw, 3)
        out.append({
            "name": arch.replace("_", " ").title(),
            "archetype": arch,
            "tribe": _archetype_tribe(arch),
            "averagePosition": avg,
            "popularity": round((dp or 0) / total, 4),
            "coreCards": [],            # filled by inject_core_cards
            "powerTurns": [],
            "tier": _tier(avg),
        })
    out.sort(key=lambda x: x["averagePosition"])
    return out


def normalize_cards(raw: dict, meta: Dict[str, dict],
                    min_play: int = 500) -> List[dict]:
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
                       min_data: int = 50, mmr: int = 100) -> List[dict]:
    out = []
    for t in raw.get("trinketStats", []):
        dp = (t.get("dataPoints", 0) if mmr == 100
              else _at_mmr(t.get("averagePlacementAtMmr"), mmr, "dataPoints", 0))
        if (dp or 0) < min_data:
            continue
        avg = (t.get("averagePlacement") if mmr == 100
               else _at_mmr(t.get("averagePlacementAtMmr"), mmr, "placement"))
        pick = (t.get("pickRate") if mmr == 100
                else _at_mmr(t.get("pickRateAtMmr"), mmr, "pickRate"))
        cid = t.get("trinketCardId", "")
        out.append({
            "cardId": cid,
            "name": meta.get(cid, {}).get("name", cid),
            "averagePosition": round(avg, 3) if avg is not None else None,
            "pickRate": round(pick or 0, 4),
            "tier": _tier(avg) if avg is not None else "?",
            "text": meta.get(cid, {}).get("text", ""),
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
        if comp.get("coreCards"):                # keep real winning-board cards
            continue
        tribe = comp.get("tribe")
        if tribe and by_tribe.get(tribe):
            comp["coreCards"] = [c["name"] for c in by_tribe[tribe][:per_comp]]


def leveling_pace(card_raw: dict, card_meta: Dict[str, dict],
                  max_turn: int = 14) -> Dict[int, float]:
    """Avg tavern tier of minions *played* each turn (proxy for tier reached).
    Weighted by how often each card is played that turn (card turnStats)."""
    num: Dict[int, float] = {}
    den: Dict[int, float] = {}
    for c in card_raw.get("cardStats", []):
        lvl = card_meta.get(c.get("cardId"), {}).get("techLevel")
        if lvl is None:
            continue
        for ts in c.get("turnStats", []):
            t, p = ts.get("turn"), ts.get("totalPlayed") or 0
            if t is None or t < 1 or t > max_turn:
                continue
            num[t] = num.get(t, 0) + lvl * p
            den[t] = den.get(t, 0) + p
    return {t: round(num[t] / den[t], 2) for t in sorted(num) if den[t]}


def scaling_pace(hero_raw: dict, max_turn: int = 14) -> Dict[int, float]:
    """Avg total board stats each turn, across heroes weighted by data points."""
    num: Dict[int, float] = {}
    den: Dict[int, float] = {}
    for h in hero_raw.get("heroStats", []):
        w = h.get("dataPoints") or 0
        for ws in h.get("warbandStats") or []:
            t, s = ws.get("turn"), ws.get("averageStats")
            if t is None or s is None or t < 1 or t > max_turn:
                continue
            num[t] = num.get(t, 0) + s * w
            den[t] = den.get(t, 0) + w
    return {t: round(num[t] / den[t], 1) for t in sorted(num) if den[t]}


def refresh(out_dir: str, mmr: int = 10, period: str = "past-seven",
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
    comps = normalize_comps(comp_raw, mmr=mmr)
    cards = normalize_cards(card_raw, card_meta)
    trinkets = normalize_trinkets(trinket_raw, card_meta, mmr=mmr)

    # Real winning boards: per-comp core cards by frequency + example boards.
    from . import final_boards
    min_mmr = next((p["mmr"] for p in hero_raw.get("mmrPercentiles", [])
                    if p.get("percentile") == mmr), 0)
    boards = final_boards.extract(comp_raw, names, min_mmr=min_mmr)
    real_core = final_boards.core_cards_by_archetype(boards)
    for comp in comps:                          # prefer real board frequency...
        rc = real_core.get(comp.get("archetype"))
        if rc:
            comp["coreCards"] = rc
    inject_core_cards(comps, cards)             # ...fallback to tribe-strength

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
        ("firestone_final_boards.json", "boards", boards),
    ):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({**meta, key: rows}, fh, indent=1)
        paths[key] = path

    # Pace benchmarks: how fast top players level + scale, by turn.
    pace_path = os.path.join(out_dir, "firestone_pace.json")
    with open(pace_path, "w", encoding="utf-8") as fh:
        from .pace import STANDARD_TAVERN_TIER
        json.dump({**meta,
                   # Real "when to level" benchmark (tavern tier by turn). The
                   # card data only gives avg tier *played*, which isn't tavern
                   # tier, so we carry the high-level standard curve here.
                   "tavern_tier": {str(k): v for k, v in STANDARD_TAVERN_TIER.items()},
                   "leveling": leveling_pace(card_raw, card_meta),
                   "scaling": scaling_pace(hero_raw)}, fh, indent=1)
    paths["pace"] = pace_path

    return {**paths, "num_heroes": len(heroes), "num_comps": len(comps),
            "num_cards": len(cards), "num_trinkets": len(trinkets),
            "num_board_sets": len(boards),
            "total_boards": sum(b["boardCount"] for b in boards)}
