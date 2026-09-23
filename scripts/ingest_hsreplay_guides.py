#!/usr/bin/env python3
"""Ingest ALL HSReplay Battlegrounds guides into data/hsreplay_guides/.

Sources (cite HSReplay only — never invent strategy text):
  * Heroes API + per-hero react_context: hero_guide / hero_buddy_guide
  * Comps page react_context + per-comp how_to_play / enablers / core cards
  * Trinkets page guide array + API placement stats

Auth: HSREPLAY_COOKIE_FILE (Netscape jar or Cookie header). Tier7 optional;
comps / trinket / hero guide pages are public with a session cookie.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hsreplay_guides"
POOL_PATH = ROOT / "data" / "cards" / "bg_live_pool_36_6_1.json"
HSJSON_CACHE = Path("/tmp/bg_pool_live/hsjson_latest.json")
BASE = "https://hsreplay.net"
UA = "hsbg-coach/0.1 (hsreplay guide ingest)"

REACT_CTX = re.compile(
    r'<script[^>]*id=["\']react_context["\'][^>]*>(.*?)</script>', re.S
)
TRINKET_ARR = re.compile(
    r'(\[\s*\{\s*"trinket_id"\s*:.*?\}\s*\])\s*</script>', re.S
)


def now_et_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %I:%M %p ET")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "unknown"


def read_cookie(path: Optional[str]) -> Optional[str]:
    p = path or os.environ.get("HSREPLAY_COOKIE_FILE") or ""
    if not p:
        for cand in ("/tmp/hsreplay.cookie.header", "/tmp/hsreplay.cookies.txt"):
            if os.path.isfile(cand):
                p = cand
                break
    if not p or not os.path.isfile(p):
        return None
    text = Path(p).read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0] if text.strip() else ""
    if text.startswith("# Netscape") or "\t" in first:
        cookies = []
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies.append(f"{parts[5]}={parts[6]}")
        return "; ".join(cookies) if cookies else None
    return text.strip() or None


def headers(cookie: Optional[str]) -> Dict[str, str]:
    h = {"User-Agent": UA, "Accept": "application/json,text/html,*/*"}
    if cookie:
        h["Cookie"] = cookie
    tok = (os.environ.get("HSREPLAY_API_TOKEN") or "").strip()
    if tok:
        h["Authorization"] = f"Token {tok}"
    return h


def http_get(url: str, cookie: Optional[str], timeout: float = 60.0) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers(cookie))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read() if e.fp else b""
    except Exception as e:
        return 0, str(e).encode()


def parse_react_context(html: str) -> Any:
    m = REACT_CTX.search(html or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def load_hsjson() -> Dict[int, dict]:
    for p in (HSJSON_CACHE, ROOT / "data" / "cards" / "hsjson_latest.json"):
        if p.is_file():
            rows = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rows, list) and rows and "dbfId" in rows[0]:
                return {int(c["dbfId"]): c for c in rows if "dbfId" in c}
    url = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"
    code, body = http_get(url, None, timeout=120)
    if code != 200:
        raise SystemExit(f"HSJSON download failed ({code})")
    HSJSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HSJSON_CACHE.write_bytes(body)
    rows = json.loads(body)
    return {int(c["dbfId"]): c for c in rows if "dbfId" in c}


def load_live_pool_names() -> set:
    if not POOL_PATH.is_file():
        return set()
    doc = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    names = set(doc.get("names") or [])
    for m in doc.get("minions") or []:
        if isinstance(m, dict) and m.get("name"):
            names.add(m["name"])
    return names


def bullets_from_hero_guide(text: str) -> Dict[str, List[str]]:
    """Partition HSReplay hero_guide sentences into hp / cycle / buy_prefs."""
    t = (text or "").strip()
    if not t:
        return {"hp": [], "cycle": [], "buy_prefs": []}
    parts = re.split(r"(?<=[.!?])\s+", t)
    hp, cycle, buy = [], [], []
    for p in parts:
        if not p.strip():
            continue
        low = p.lower()
        if any(k in low for k in (
            "hero power", "hero-power", "heropower", "use your hero",
            "don't hero", "do not hero", "hp every", "hp the",
            "hero power the", "don't have to hero",
        )):
            hp.append(p.strip())
        elif any(k in low for k in ("cycle", "sell", "refresh", "roll", "buddy")):
            cycle.append(p.strip())
        else:
            buy.append(p.strip())
    return {"hp": hp, "cycle": cycle, "buy_prefs": buy}


def ingest_heroes(cookie: Optional[str], by_dbf: Dict[int, dict], workers: int) -> dict:
    url = (
        f"{BASE}/api/v1/battlegrounds/heroes/"
        "?BattlegroundsMMRPercentile=ALL&BattlegroundsTimeRange=LAST_7_DAYS"
    )
    code, body = http_get(url, cookie)
    if code != 200:
        raise SystemExit(f"heroes API {code}: {body[:200]!r}")
    rows = json.loads(body)
    code2, body2 = http_get(
        f"{BASE}/api/v1/battlegrounds/heroes/free/"
        "?BattlegroundsMMRPercentile=ALL&BattlegroundsTimeRange=LAST_7_DAYS",
        cookie,
    )
    if code2 == 200:
        seen = {r["hero_dbf_id"] for r in rows}
        for r in json.loads(body2):
            if r.get("hero_dbf_id") not in seen:
                rows.append(r)
    stats_by = {int(r["hero_dbf_id"]): r for r in rows if r.get("hero_dbf_id")}

    def fetch_one(dbf_id: int) -> dict:
        page = f"{BASE}/battlegrounds/heroes/{dbf_id}/"
        c, b = http_get(page, cookie)
        card = by_dbf.get(int(dbf_id)) or {}
        name = card.get("name") or f"Hero_{dbf_id}"
        guide = buddy = ""
        fav: List[Any] = []
        updated = False
        if c == 200:
            rc = parse_react_context(b.decode("utf-8", "replace"))
            if isinstance(rc, dict):
                guide = (rc.get("hero_guide") or "").strip()
                buddy = (rc.get("hero_buddy_guide") or "").strip()
                fav = rc.get("hero_guide_favorable_tribes") or []
                updated = bool(rc.get("hero_guide_recently_updated"))
        structured = bullets_from_hero_guide(guide)
        if buddy:
            structured.setdefault("cycle", []).append(buddy)
        return {
            "id": int(dbf_id),
            "dbf_id": int(dbf_id),
            "name": name,
            "card_id": card.get("id"),
            "slug": slugify(name),
            "guide_text": guide,
            "buddy_guide_text": buddy,
            "favorable_tribes": fav,
            "guide_recently_updated": updated,
            "structured": structured,
            "source_url": f"{BASE}/battlegrounds/heroes/{dbf_id}/{slugify(name)}",
            "stats": stats_by.get(int(dbf_id), {}),
        }

    ids = sorted(stats_by.keys())
    print(f"Fetching {len(ids)} hero guides…", flush=True)
    heroes: List[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, i): i for i in ids}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            heroes.append(fut.result())
            if n % 20 == 0 or n == len(ids):
                print(f"  heroes {n}/{len(ids)}", flush=True)
    heroes.sort(key=lambda h: h["name"].lower())
    return {
        "as_of": now_et_label(),
        "patch": "36.6.1",
        "source": "HSReplay hero pages react_context (How to Play / Hero Guide)",
        "source_urls": [f"{BASE}/battlegrounds/heroes/"],
        "count": len(heroes),
        "with_guide_text": sum(1 for h in heroes if h["guide_text"]),
        "heroes": heroes,
    }


def ingest_comps(
    cookie: Optional[str], by_dbf: Dict[int, dict], pool: set, workers: int
) -> dict:
    code, body = http_get(f"{BASE}/battlegrounds/comps/", cookie)
    if code != 200:
        raise SystemExit(f"comps page {code}")
    rc = parse_react_context(body.decode("utf-8", "replace"))
    if not isinstance(rc, dict) or "comps" not in rc:
        raise SystemExit("comps react_context missing")
    listing = rc["comps"]

    def card_name(dbf: int) -> Optional[str]:
        c = by_dbf.get(int(dbf))
        return c.get("name") if c else None

    def card_id(dbf: int) -> Optional[str]:
        c = by_dbf.get(int(dbf))
        return c.get("id") if c else None

    def resolve_cards(ids: List[int]) -> List[dict]:
        out = []
        for d in ids or []:
            nm = card_name(d)
            out.append({
                "dbf_id": int(d),
                "name": nm,
                "card_id": card_id(d),
                "in_live_pool": (nm in pool) if pool and nm else None,
            })
        return out

    def fetch_one(comp: dict) -> dict:
        cid = int(comp["comp_id"])
        c, b = http_get(f"{BASE}/battlegrounds/comps/{cid}/", cookie)
        detail: dict = {}
        if c == 200:
            d = parse_react_context(b.decode("utf-8", "replace"))
            if isinstance(d, dict):
                detail = d
        name = detail.get("comp_name") or comp.get("comp_name") or f"comp-{cid}"
        tribe = name.split(" - ", 1)[0].strip() if " - " in name else None
        core_ids = detail.get("comp_core_cards") or comp.get("comp_core_cards") or []
        addon_ids = detail.get("comp_addon_cards") or []
        core = resolve_cards(core_ids)
        addon = resolve_cards(addon_ids)
        enabler_text = detail.get("comp_common_enablers") or ""
        enabler_names = re.findall(r"\[\[([^\]|]+)\|\|(\d+)\]\]", enabler_text)
        enablers = [
            {
                "name": n,
                "dbf_id": int(d),
                "card_id": card_id(int(d)),
                "in_live_pool": (n in pool) if pool else None,
            }
            for n, d in enabler_names
        ]
        seen = {x["dbf_id"] for x in core}
        key_cards = core + [e for e in enablers if e["dbf_id"] not in seen]
        naga = (tribe or "").lower() == "naga" or "naga" in name.lower()
        return {
            "id": cid,
            "name": name,
            "slug": detail.get("comp_slug") or comp.get("comp_slug") or slugify(name),
            "tribe": tribe,
            "tier": detail.get("comp_tier", comp.get("comp_tier")),
            "tier_rank": comp.get("comp_tier_rank"),
            "difficulty": detail.get("comp_difficulty", comp.get("comp_difficulty")),
            "summary": detail.get("comp_summary") or comp.get("comp_summary") or "",
            "how_to_play": detail.get("comp_how_to_play") or "",
            "when_to_commit": detail.get("comp_when_to_commit") or "",
            "common_enablers_text": enabler_text,
            "core_cards": core,
            "addon_cards": addon,
            "enabler_cards": enablers,
            "key_cards": key_cards,
            "key_names": [c["name"] for c in key_cards if c.get("name")],
            "naga": naga,
            "out_of_pool_core": [x for x in core if x.get("in_live_pool") is False],
            "hidden": bool(detail.get("comp_hidden", comp.get("comp_hidden"))),
            "source_url": f"{BASE}/battlegrounds/comps/{cid}/",
            "last_updated": detail.get("comp_last_updated") or comp.get("comp_last_updated"),
        }

    print(f"Fetching {len(listing)} comp guides…", flush=True)
    comps: List[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, c) for c in listing]
        for n, fut in enumerate(cf.as_completed(futs), 1):
            comps.append(fut.result())
            if n % 15 == 0 or n == len(listing):
                print(f"  comps {n}/{len(listing)}", flush=True)
    comps.sort(key=lambda c: (c.get("tier") or 99, c.get("tier_rank") or 99, c["name"]))
    live, dropped = [], []
    for c in comps:
        if c["naga"] or c["hidden"]:
            dropped.append({"name": c["name"], "reason": "naga_or_hidden"})
            continue
        if pool and c["core_cards"] and all(
            x.get("in_live_pool") is False for x in c["core_cards"]
        ):
            dropped.append({"name": c["name"], "reason": "all_core_out_of_pool"})
            continue
        live.append(c)
    return {
        "as_of": now_et_label(),
        "patch": "36.6.1",
        "source": "HSReplay comps react_context + per-comp How to Play",
        "source_urls": [f"{BASE}/battlegrounds/comps/"],
        "count_listed": len(listing),
        "count_live": len(live),
        "dropped": dropped,
        "comps": live,
        "all_comps_including_dropped": comps,
    }


def ingest_trinkets(
    cookie: Optional[str], by_dbf: Dict[int, dict], by_card_id: Dict[str, dict]
) -> dict:
    code, body = http_get(f"{BASE}/battlegrounds/trinkets/", cookie)
    if code != 200:
        raise SystemExit(f"trinkets page {code}")
    html = body.decode("utf-8", "replace")
    m = TRINKET_ARR.search(html)
    if not m:
        raise SystemExit("trinket_guide array not found on trinkets page")
    guides = json.loads(m.group(1))

    stats_by_dbf: Dict[int, dict] = {}
    sc, sb = http_get(
        f"{BASE}/api/v1/battlegrounds/trinkets/"
        "?BattlegroundsMMRPercentile=ALL&BattlegroundsTimeRange=LAST_7_DAYS",
        cookie,
    )
    if sc == 200:
        for r in json.loads(sb):
            stats_by_dbf[int(r["trinket_dbf_id"])] = r

    top1: Dict[int, dict] = {}
    sc2, sb2 = http_get(
        f"{BASE}/api/v1/battlegrounds/trinkets/"
        "?BattlegroundsMMRPercentile=TOP_1_PERCENT&BattlegroundsTimeRange=LAST_7_DAYS",
        cookie,
    )
    if sc2 == 200:
        for r in json.loads(sb2):
            top1[int(r["trinket_dbf_id"])] = r

    out = []
    for g in guides:
        tid = g.get("trinket_id") or ""
        card = by_card_id.get(tid) or {}
        dbf = card.get("dbfId")
        name = card.get("name") or tid
        text = card.get("text") or ""
        guide = (g.get("trinket_guide") or "").strip()
        st = stats_by_dbf.get(int(dbf), {}) if dbf else {}
        t1 = top1.get(int(dbf), {}) if dbf else {}
        out.append({
            "id": tid,
            "card_id": tid,
            "dbf_id": dbf,
            "name": name,
            "type": g.get("trinket_type"),
            "guide_text": guide,
            "favorable_tribes": g.get("trinket_guide_favorable_tribes") or [],
            "guide_recently_updated": bool(g.get("trinket_guide_recently_updated")),
            "effect_summary": re.sub(r"<[^>]+>", " ", text or "").replace("[x]", "").strip(),
            "stats": {
                "pick_rate": st.get("pick_rate"),
                "avg_final_placement": st.get("avg_final_placement"),
                "tier": st.get("tier"),
                "group": st.get("group"),
                "top1_avg_final_placement": t1.get("avg_final_placement"),
                "top1_tier": t1.get("tier"),
            },
            "source_url": f"{BASE}/battlegrounds/trinkets/",
        })
    out.sort(key=lambda t: (t.get("type") or "", t.get("name") or ""))
    return {
        "as_of": now_et_label(),
        "patch": "36.6.1",
        "source": "HSReplay trinkets page guide array + API stats",
        "source_urls": [f"{BASE}/battlegrounds/trinkets/"],
        "count": len(out),
        "with_guide_text": sum(1 for t in out if t["guide_text"]),
        "trinkets": out,
    }


def write_per_hero_files(heroes_doc: dict, out_dir: Path) -> None:
    hdir = out_dir / "heroes"
    hdir.mkdir(parents=True, exist_ok=True)
    for h in heroes_doc["heroes"]:
        path = hdir / f"{h['dbf_id']}_{h.get('slug') or slugify(h['name'])}.json"
        path.write_text(json.dumps(h, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cookie-file", default=os.environ.get("HSREPLAY_COOKIE_FILE"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-heroes", action="store_true")
    ap.add_argument("--skip-comps", action="store_true")
    ap.add_argument("--skip-trinkets", action="store_true")
    args = ap.parse_args(argv)

    cookie = read_cookie(args.cookie_file)
    if not cookie:
        print("WARNING: no HSREPLAY cookie — public pages may still work", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    by_dbf = load_hsjson()
    by_card_id = {c["id"]: c for c in by_dbf.values() if c.get("id")}
    pool = load_live_pool_names()
    print(f"HSJSON cards={len(by_dbf)} live_pool_names={len(pool)}", flush=True)

    manifest: Dict[str, Any] = {"as_of": now_et_label(), "patch": "36.6.1", "files": {}}

    if not args.skip_heroes:
        heroes = ingest_heroes(cookie, by_dbf, workers=args.workers)
        (args.out / "heroes.json").write_text(
            json.dumps(heroes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_per_hero_files(heroes, args.out)
        manifest["files"]["heroes.json"] = {
            "count": heroes["count"], "with_guide_text": heroes["with_guide_text"]
        }
        print(f"Wrote heroes.json ({heroes['count']}, guides={heroes['with_guide_text']})")

    if not args.skip_comps:
        comps = ingest_comps(cookie, by_dbf, pool, workers=args.workers)
        (args.out / "comps.json").write_text(
            json.dumps(comps, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["files"]["comps.json"] = {
            "count_live": comps["count_live"],
            "count_listed": comps["count_listed"],
            "dropped": len(comps["dropped"]),
        }
        print(f"Wrote comps.json (live={comps['count_live']}, dropped={len(comps['dropped'])})")

    if not args.skip_trinkets:
        trinkets = ingest_trinkets(cookie, by_dbf, by_card_id)
        (args.out / "trinkets.json").write_text(
            json.dumps(trinkets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["files"]["trinkets.json"] = {
            "count": trinkets["count"], "with_guide_text": trinkets["with_guide_text"]
        }
        print(f"Wrote trinkets.json ({trinkets['count']}, guides={trinkets['with_guide_text']})")

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Done.", json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
