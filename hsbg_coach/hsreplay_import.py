"""Import an HSReplay.net Battlegrounds minions export as a card-stats source.

HSReplay publishes BG minion performance (avg placement etc.) on
https://hsreplay.net/battlegrounds/minions/ but has NO public API and its
servers refuse non-browser traffic, so this cannot be auto-fetched the way the
Firestone stats are. Instead, export the data yourself from a logged-in
browser session and feed the file to::

    python -m hsbg_coach import-hsreplay path/to/export.json

Getting an export (either works):
  * DevTools → Network tab while loading the minions page → copy the JSON
    response that contains the per-minion rows → save to a file.
  * Select the stats table on the page → copy → paste into a ``.csv``/``.tsv``
    file (a header row with a name column and an avg-placement column is
    enough).

The importer is deliberately loose about key names (HSReplay's frontend schema
is undocumented and shifts), normalizes into the same shape as
``firestone_card_stats.json``, and writes ``data/stats/hsreplay_card_stats.json``.
From then on ``card_meta_stats`` blends it with the Firestone numbers
automatically — both for the live advisor and the eval net's anti-survivorship
feature block.
"""

import csv
import datetime
import glob
import json
import os
import re
from typing import Dict, List, Optional
from urllib.parse import parse_qsl, urlsplit

_STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stats")
OUT_PATH = os.path.join(_STATS_DIR, "hsreplay_card_stats.json")
BY_TURN_PATH = os.path.join(_STATS_DIR, "hsreplay_card_stats_by_turn.json")

# Loose key aliases (lowercased, spaces/underscores stripped) -> canonical.
_KEYS = {
    "name": ("name", "cardname", "card", "minion", "minionname"),
    "cardId": ("cardid", "dbfid", "id"),
    "averagePlacement": ("averageplacement", "avgplacement", "avgplace",
                         "averageplace", "avgfinalplacement", "placement"),
    "impact": ("impact", "placementdelta", "netimpact"),
    "pickRate": ("pickrate", "playrate", "popularity"),
    "totalPlayed": ("totalplayed", "timesplayed", "games", "count",
                    "samplesize"),
}


def _canon(key: str) -> Optional[str]:
    flat = key.strip().lower().replace(" ", "").replace("_", "").replace("%", "")
    for canonical, aliases in _KEYS.items():
        if flat in aliases:
            return canonical
    return None


def _rows_from_json(blob) -> List[dict]:
    """Find the list of per-minion rows in whatever envelope the export uses."""
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if isinstance(blob, dict):
        for key in ("cards", "data", "minions", "series", "rows", "results"):
            inner = blob.get(key)
            if isinstance(inner, (list, dict)):
                rows = _rows_from_json(inner)
                if rows:
                    return rows
        # depth-first: some exports nest one level deeper (e.g. series.data.ALL)
        for inner in blob.values():
            if isinstance(inner, (list, dict)):
                rows = _rows_from_json(inner)
                if rows:
                    return rows
    return []


def _rows_from_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return list(csv.DictReader(fh, dialect=dialect))


def _num(v) -> Optional[float]:
    try:
        return float(str(v).replace("%", "").replace(",", "."))
    except (TypeError, ValueError):
        return None


def normalize(rows: List[dict]) -> List[dict]:
    """Map loose export rows onto the firestone_card_stats card schema."""
    out = []
    for row in rows:
        mapped: Dict[str, object] = {}
        for key, value in row.items():
            canonical = _canon(str(key))
            if canonical and canonical not in mapped:
                mapped[canonical] = value
        name = mapped.get("name")
        ap = _num(mapped.get("averagePlacement"))
        if not name or ap is None or not (1.0 <= ap <= 8.0):
            continue
        card = {"name": str(name).strip(), "averagePlacement": round(ap, 3)}
        impact = _num(mapped.get("impact"))
        if impact is not None:
            card["impact"] = round(impact, 3)
        if mapped.get("cardId") is not None:
            card["cardId"] = str(mapped["cardId"])
        played = _num(mapped.get("totalPlayed"))
        if played is not None:
            card["totalPlayed"] = int(played)
        out.append(card)
    out.sort(key=lambda c: c["averagePlacement"])
    return out


def _turn_of(wrapper: dict) -> Optional[int]:
    """Turn filter a captured request was made with, if any — from the URL
    query (any param whose name contains 'turn') or the POST body."""
    for key, val in parse_qsl(urlsplit(wrapper.get("url") or "").query):
        if "turn" in key.lower():
            m = re.search(r"\d+", val)
            if m:
                return int(m.group())
    post = wrapper.get("post_data") or ""
    m = re.search(r'"turns?"\s*:\s*"?(\d+)', post, re.I)
    return int(m.group(1)) if m else None


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")


def import_captures(dir_path: str, out_path: str = OUT_PATH,
                    by_turn_path: str = BY_TURN_PATH) -> Dict:
    """Ingest a scripts/hsreplay_capture.py capture directory (or any dir of
    exported JSON files). Un-turn-filtered payloads merge into the overall
    stats file; turn-filtered ones into the by-turn file. Later captures of
    the same card/turn override earlier ones (a re-capture is a refresh).
    Returns {"overall": n_cards, "turns": [turns seen]}."""
    overall: Dict[str, dict] = {}
    by_turn: Dict[int, Dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        try:
            with open(path, encoding="utf-8-sig") as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            continue
        wrapper = blob if isinstance(blob, dict) and "body" in blob else {}
        cards = normalize(_rows_from_json(wrapper.get("body", blob)))
        if not cards:
            continue
        turn = _turn_of(wrapper)
        bucket = by_turn.setdefault(turn, {}) if turn is not None else overall
        for c in cards:
            bucket[c["name"]] = c
    stamp = datetime.date.today().isoformat()
    if overall:
        _write(out_path, {
            "_source": f"hsreplay.net capture ({os.path.basename(dir_path)})",
            "_imported": stamp,
            "cards": sorted(overall.values(),
                            key=lambda c: c["averagePlacement"]),
        })
    if by_turn:
        _write(by_turn_path, {
            "_source": f"hsreplay.net capture ({os.path.basename(dir_path)})",
            "_imported": stamp,
            "turns": {str(t): sorted(cards.values(),
                                     key=lambda c: c["averagePlacement"])
                      for t, cards in sorted(by_turn.items())},
        })
    return {"overall": len(overall), "turns": sorted(by_turn)}


def import_file(path: str, out_path: str = OUT_PATH) -> int:
    """Parse an HSReplay export (JSON/CSV/TSV) and write the stats file.
    Returns the number of cards imported."""
    if path.lower().endswith((".csv", ".tsv", ".txt")):
        rows = _rows_from_csv(path)
    else:
        with open(path, encoding="utf-8-sig") as fh:
            rows = _rows_from_json(json.load(fh))
    cards = normalize(rows)
    if not cards:
        return 0
    payload = {
        "_source": f"hsreplay.net export ({os.path.basename(path)})",
        "_imported": datetime.date.today().isoformat(),
        "cards": cards,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    return len(cards)
