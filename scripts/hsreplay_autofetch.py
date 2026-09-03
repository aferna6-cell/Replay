#!/usr/bin/env python3
"""Autonomous HSReplay BG stats fetch — no clicking, built for a weekly cron.

Uses the SAME persistent browser profile as scripts/hsreplay_capture.py (log
in once via that script; the session + Cloudflare clearance persist), then:

  1. opens hsreplay.net so Cloudflare re-validates the stored cookies,
  2. fetches the known BG endpoints (discovered from a real capture:
     minions, comps, heroes, trinkets, purchase rates, minion curves,
     tavern-up) directly from inside the page — browser TLS + cookies, so
     nothing to detect,
  3. ALSO walks the BG pages with the response recorder on, catching
     anything the endpoint list misses,
  4. imports everything through the categorized per-turn ingest.

    python scripts/hsreplay_autofetch.py                # fetch + import
    python scripts/hsreplay_autofetch.py --percentile TOP_10_PERCENT

If it reports a Cloudflare wall, run the interactive capture once
(scripts/hsreplay_capture.py) to refresh the session, then retry. Weekly
automation: scripts/schedule_hsreplay.sh installs the cron job that runs
this + retrain + push (scripts/weekly_hsreplay_update.sh).
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STATS_DIR = REPO_ROOT / "data" / "stats"
PROFILE_DIR = STATS_DIR / "hsreplay_profile"
CAPTURES_ROOT = STATS_DIR / "hsreplay_captures"

BASE = "https://hsreplay.net"
# Real endpoints, confirmed from the 2026-08-27 capture diagnose.
ENDPOINTS = (
    "/api/v1/battlegrounds/compositions/?hl=en",
    "/analytics/query/battlegrounds_minion_list/?{f}",
    "/analytics/query/battlegrounds_comp_stats/?{f}",
    "/analytics/query/battlegrounds_minion_stats/?{f}",
    "/analytics/query/battlegrounds_purchase_rates_by_turn/?{f}",
    "/analytics/query/battlegrounds_tavern_up_stats_all/?{f}",
    "/api/v1/battlegrounds/heroes/?{f}",
    "/api/v1/battlegrounds/trinkets/?{f}",
)
# Pages to walk as a safety net (their onload queries get captured too).
PAGES = ("/battlegrounds/minions/", "/battlegrounds/comps/",
         "/battlegrounds/heroes/", "/battlegrounds/trinkets/")

_BLOCKED = re.compile(
    r"google|gstatic|doubleclick|facebook|sentry|amplitude|braze|"
    r"cloudflareinsights|adsystem|quantserve|scorecard|adnxs|criteo|"
    r"prebid|rubicon|pubmatic|onetrust|cookielaw", re.I)

_FETCH_JS = """async (url) => {
  const r = await fetch(url, {credentials: 'include',
                              headers: {'Accept': 'application/json'}});
  if (!r.ok) return {__status: r.status};
  const ct = r.headers.get('content-type') || '';
  if (!ct.includes('json')) return {__status: 'non-json'};
  return {__status: 200, body: await r.json()};
}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--percentile", default="TOP_20_PERCENT",
                    help="MMR filter (default TOP_20_PERCENT)")
    ap.add_argument("--time-range", default="LAST_7_DAYS")
    ap.add_argument("--headless", action="store_true",
                    help="try headless (works once the profile holds a valid "
                         "Cloudflare clearance; falls back is NOT automatic)")
    ap.add_argument("--no-import", action="store_true")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright missing: pip install playwright && "
              "python -m playwright install chromium")
        return 1

    capture_dir = (CAPTURES_ROOT
                   / f"auto-{datetime.date.today().isoformat()}")
    capture_dir.mkdir(parents=True, exist_ok=True)
    count = [0]

    def save(url: str, body, post_data=None):
        count[0] += 1
        (capture_dir / f"cap_{count[0]:04d}.json").write_text(json.dumps(
            {"url": url, "method": "GET", "post_data": post_data,
             "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
             "body": body}), encoding="utf-8")

    def on_response(response):
        url = response.url
        if _BLOCKED.search(url):
            return
        try:
            if "json" not in response.headers.get("content-type", ""):
                return
            save(url, response.json(), response.request.post_data)
        except Exception:
            pass

    filters = (f"BattlegroundsMMRPercentile={args.percentile}"
               f"&BattlegroundsTimeRange={args.time_range}")
    with sync_playwright() as p:
        launch = dict(
            headless=args.headless, viewport=None,
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"])
        try:
            ctx = p.chromium.launch_persistent_context(
                str(PROFILE_DIR), channel="chrome", **launch)
        except Exception:
            ctx = p.chromium.launch_persistent_context(
                str(PROFILE_DIR), **launch)
        ctx.add_init_script("Object.defineProperty(navigator, 'webdriver',"
                            " {get: () => undefined})")
        ctx.on("response", on_response)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        print("Validating session…")
        page.goto(BASE + PAGES[0], wait_until="domcontentloaded")
        page.wait_for_timeout(6000)          # give Cloudflare a beat
        if "just a moment" in (page.title() or "").lower():
            print("Cloudflare wall — run scripts/hsreplay_capture.py once "
                  "interactively to refresh the session, then retry.")
            ctx.close()
            return 2

        print("Fetching known endpoints…")
        ok = 0
        for ep in ENDPOINTS:
            url = BASE + ep.format(f=filters)
            try:
                res = page.evaluate(_FETCH_JS, url)
            except Exception as exc:
                print(f"  ERR  {ep.split('?')[0]}: {exc}")
                continue
            if res.get("__status") == 200:
                save(url, res["body"])
                ok += 1
                print(f"  ok   {ep.split('?')[0]}")
            else:
                print(f"  {res.get('__status')}   {ep.split('?')[0]}")

        print("Walking BG pages (safety net)…")
        for path in PAGES:
            try:
                page.goto(BASE + path, wait_until="networkidle", timeout=30000)
            except Exception:
                pass
        ctx.close()

    print(f"{count[0]} payload(s) -> {capture_dir}  "
          f"({ok}/{len(ENDPOINTS)} direct endpoint fetches)")
    if not count[0]:
        return 1
    if args.no_import:
        return 0
    from hsbg_coach import card_meta_stats, hsreplay_import
    result = hsreplay_import.import_captures(str(capture_dir))
    card_meta_stats.reload()
    for cat, n in sorted(result["categories"].items()):
        print(f"  {cat:16s} {n:5d} items")
    return 0 if result["categories"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
