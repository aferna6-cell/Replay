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

# Everything HSReplay publishes for BG, not just minions. Categorized by
# request-URL keywords (first match wins); each category gets its own
# data/stats/hsreplay_<category>_stats.json. Minions additionally keep the
# legacy OUT_PATH/BY_TURN_PATH files that card_meta_stats blends from.
CATEGORIES = (
    ("comps", ("comp", "composition", "archetype")),
    ("trinkets", ("trinket",)),
    ("dark_gifts", ("gift", "darkgift", "dark-gift", "anomal")),
    ("quests", ("quest", "reward")),
    ("heroes", ("hero",)),
    ("minions", ("minion", "card")),
)


def category_of(wrapper: dict) -> str:
    haystack = ((wrapper.get("url") or "") + " "
                + (wrapper.get("post_data") or "")).lower()
    for cat, keys in CATEGORIES:
        if any(k in haystack for k in keys):
            return cat
    return "minions"       # rows with a name + 1-8 avg placement default here


def category_path(cat: str) -> str:
    return os.path.join(_STATS_DIR, f"hsreplay_{cat}_stats.json")

# Loose key aliases (lowercased, spaces/underscores stripped) -> canonical.
_KEYS = {
    "name": ("name", "cardname", "card", "minion", "minionname"),
    "cardId": ("cardid", "dbfid", "id"),
    "averagePlacement": ("averageplacement", "avgplacement", "avgplace",
                         "averageplace", "avgfinalplacement", "placement",
                         "averagefinalplacement", "finalplacement",
                         "avgfinalplace", "meanplacement", "avgplacementall"),
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


# --- card-id -> name resolution ------------------------------------------
# HSReplay analytics rows usually carry only a numeric dbf_id (or a card id
# like "BG28_573") — the site joins names client-side. We resolve via the
# local BG card KB plus HearthstoneJSON (downloaded once, cached locally).
_HSJSON_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"
_HSJSON_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cards",
                             "hsjson_index.json")
_BG_KB = os.path.join(os.path.dirname(__file__), "..", "data", "cards",
                      "bg_cards.json")
_NAME_INDEX: Optional[Dict[str, str]] = None   # id (str) -> name


def _load_name_index() -> Dict[str, str]:
    global _NAME_INDEX
    if _NAME_INDEX is not None:
        return _NAME_INDEX
    idx: Dict[str, str] = {}
    try:                                   # local BG KB: card_id -> name
        kb = json.load(open(_BG_KB, encoding="utf-8"))
        rows = kb.values() if isinstance(kb, dict) else kb
        for c in rows:
            if isinstance(c, dict) and c.get("card_id") and c.get("name"):
                idx[str(c["card_id"])] = c["name"]
    except (OSError, ValueError):
        pass
    try:                                   # cached HearthstoneJSON index
        idx.update(json.load(open(_HSJSON_CACHE, encoding="utf-8")))
    except (OSError, ValueError):
        try:                               # one-time download + cache
            import urllib.request
            req = urllib.request.Request(_HSJSON_URL,
                                         headers={"User-Agent": "hsbg-coach"})
            cards = json.load(urllib.request.urlopen(req, timeout=60))
            fresh = {}
            for c in cards:
                name = c.get("name")
                if not name:
                    continue
                if c.get("dbfId") is not None:
                    fresh[str(c["dbfId"])] = name
                if c.get("id"):
                    fresh[str(c["id"])] = name
            os.makedirs(os.path.dirname(_HSJSON_CACHE), exist_ok=True)
            with open(_HSJSON_CACHE, "w", encoding="utf-8") as fh:
                json.dump(fresh, fh)
            idx.update(fresh)
        except Exception:
            pass                           # offline -> id-only rows are skipped
    _NAME_INDEX = idx
    return idx


def resolve_name(raw_id) -> Optional[str]:
    if raw_id is None:
        return None
    return _load_name_index().get(str(raw_id).strip())


def _rows_from_json(blob) -> List[dict]:
    """Find the list of per-minion rows in whatever envelope the export uses."""
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if isinstance(blob, dict):
        # dict keyed by numeric dbf id -> one row per entry
        if (len(blob) >= 4
                and all(re.fullmatch(r"\d+", str(k)) for k in blob)
                and all(isinstance(v, dict) for v in blob.values())):
            return [{"dbf_id": k, **v} for k, v in blob.items()]
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
        name = mapped.get("name") or resolve_name(mapped.get("cardId"))
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
                    by_turn_path: str = BY_TURN_PATH,
                    stats_dir: Optional[str] = None) -> Dict:
    """Ingest a scripts/hsreplay_capture.py capture directory (or any dir of
    exported JSON files). Every payload is categorized (minions / comps /
    heroes / trinkets / dark_gifts / quests) from its request URL and split
    by the turn filter it was captured under. Later captures of the same
    (category, turn, name) override earlier ones (a re-capture is a refresh).

    Writes one hsreplay_<category>_stats.json per non-empty category
    ({"items": [...], "by_turn": {turn: [...]}}) — plus, for minions, the
    legacy hsreplay_card_stats.json / _by_turn.json that card_meta_stats and
    the eval-net priors blend from.

    Returns {"overall": n_minions, "turns": [minion turns],
             "categories": {cat: n_items}}."""
    wrappers = []
    for path in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        try:
            with open(path, encoding="utf-8-sig") as fh:
                wrappers.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return ingest_wrappers(wrappers,
                           f"hsreplay.net capture ({os.path.basename(dir_path)})",
                           out_path, by_turn_path, stats_dir)


def ingest_wrappers(wrappers: List[dict], src: str,
                    out_path: Optional[str] = None,
                    by_turn_path: Optional[str] = None,
                    stats_dir: Optional[str] = None) -> Dict:
    """Shared merge for captured payload wrappers ({url, post_data, body}) —
    used by both the Playwright capture dir and the userscript bundle.
    Path params default to the module-level locations, resolved at call time
    so tests can redirect them; stats_dir hosts the per-category files."""
    out_path = out_path or OUT_PATH
    by_turn_path = by_turn_path or (
        os.path.join(stats_dir, os.path.basename(BY_TURN_PATH))
        if stats_dir else BY_TURN_PATH)
    # buckets[cat][turn or None][name] = row
    buckets: Dict[str, Dict[Optional[int], Dict[str, dict]]] = {}
    for blob in wrappers:
        wrapper = blob if isinstance(blob, dict) and "body" in blob else {}
        rows = normalize(_rows_from_json(wrapper.get("body", blob)))
        if not rows:
            continue
        cat = category_of(wrapper)
        turn = _turn_of(wrapper)
        bucket = buckets.setdefault(cat, {}).setdefault(turn, {})
        for r in rows:
            bucket[r["name"]] = r

    stamp = datetime.date.today().isoformat()

    def ranked(d: Dict[str, dict]) -> list:
        return sorted(d.values(), key=lambda c: c["averagePlacement"])

    categories: Dict[str, int] = {}
    for cat, per_turn in buckets.items():
        overall = per_turn.get(None, {})
        turns = {str(t): ranked(per_turn[t])
                 for t in sorted(t for t in per_turn if t is not None)}
        categories[cat] = len(overall) or sum(len(v) for v in turns.values())
        payload = {"_source": src, "_imported": stamp,
                   "items": ranked(overall)}
        if turns:
            payload["by_turn"] = turns
        base = stats_dir or _STATS_DIR
        _write(os.path.join(base, f"hsreplay_{cat}_stats.json"), payload)

    minions = buckets.get("minions", {})
    if minions.get(None):
        _write(out_path, {"_source": src, "_imported": stamp,
                          "cards": ranked(minions[None])})
    minion_turns = sorted(t for t in minions if t is not None)
    if minion_turns:
        _write(by_turn_path, {
            "_source": src, "_imported": stamp,
            "turns": {str(t): ranked(minions[t]) for t in minion_turns}})
    return {"overall": len(minions.get(None, {})), "turns": minion_turns,
            "categories": categories}


def _sketch(value, depth: int = 0) -> str:
    """Compact structure sketch of a payload (keys + shapes, values elided) —
    safe to paste when reporting an unrecognized capture."""
    pad = "  " * depth
    if isinstance(value, dict):
        if depth >= 3:
            return f"{pad}{{…{len(value)} keys}}"
        keys = list(value)[:12]
        lines = [f"{pad}{{"]
        for k in keys:
            inner = _sketch(value[k], depth + 1).lstrip()
            lines.append(f"{pad}  {k}: {inner}")
        if len(value) > 12:
            lines.append(f"{pad}  …{len(value) - 12} more keys")
        return "\n".join(lines + [f"{pad}}}"])
    if isinstance(value, list):
        if not value:
            return f"{pad}[]"
        head = _sketch(value[0], depth + 1).lstrip()
        return f"{pad}[{len(value)} × {head}]"
    return f"{pad}<{type(value).__name__}>"


def diagnose(dir_path: str) -> str:
    """Human-readable report of what a capture dir holds and why rows did or
    didn't normalize — paste this when the import finds nothing."""
    lines: List[str] = []
    seen_urls = set()
    for path in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        try:
            with open(path, encoding="utf-8-sig") as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            lines.append(f"{os.path.basename(path)}: unreadable JSON")
            continue
        wrapper = blob if isinstance(blob, dict) and "body" in blob else {}
        url = wrapper.get("url", "(no url — raw export)")
        short = re.sub(r"^https?://", "", url)[:110]
        dedup_key = short.split("?")[0]
        body = wrapper.get("body", blob)
        rows = _rows_from_json(body)
        ok = normalize(rows)
        with_name = sum(1 for r in rows if any(
            _canon(str(k)) == "name" for k in r))
        with_id = sum(1 for r in rows if any(
            _canon(str(k)) == "cardId" for k in r))
        with_ap = sum(1 for r in rows if any(
            _canon(str(k)) == "averagePlacement" for k in r))
        lines.append(f"{os.path.basename(path)}  {short}")
        lines.append(f"  rows found {len(rows)} | name-key {with_name} | "
                     f"id-key {with_id} | avg-place-key {with_ap} | "
                     f"normalized {len(ok)} | category {category_of(wrapper)}"
                     + (f" | turn {_turn_of(wrapper)}"
                        if _turn_of(wrapper) is not None else ""))
        if rows and not ok and dedup_key not in seen_urls:
            lines.append("  first row keys: "
                         + ", ".join(list(rows[0])[:20]))
        elif not rows and dedup_key not in seen_urls:
            lines.append("  body shape:\n"
                         + "\n".join("    " + l for l in
                                     _sketch(body).splitlines()[:18]))
        seen_urls.add(dedup_key)
    return "\n".join(lines) or f"No *.json files in {dir_path}"


def is_wrapper_bundle(blob) -> bool:
    """A userscript-exported bundle: a list of {url, body} capture wrappers
    (optionally under a top-level {"captures": [...]})."""
    if isinstance(blob, dict):
        blob = blob.get("captures")
    return (isinstance(blob, list) and len(blob) > 0
            and all(isinstance(w, dict) and "url" in w and "body" in w
                    for w in blob))


def import_file(path: str, out_path: str = OUT_PATH) -> int:
    """Parse an HSReplay export (JSON/CSV/TSV) and write the stats file.
    Returns the number of cards imported. A userscript capture bundle is
    detected and routed through the categorized ingest instead."""
    if path.lower().endswith((".csv", ".tsv", ".txt")):
        rows = _rows_from_csv(path)
    else:
        with open(path, encoding="utf-8-sig") as fh:
            blob = json.load(fh)
        if is_wrapper_bundle(blob):
            wrappers = blob.get("captures") if isinstance(blob, dict) else blob
            result = ingest_wrappers(
                wrappers, f"hsreplay.net userscript ({os.path.basename(path)})",
                out_path=out_path, stats_dir=os.path.dirname(out_path))
            return sum(result["categories"].values())
        rows = _rows_from_json(blob)
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
