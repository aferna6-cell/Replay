#!/usr/bin/env python3
"""Capture HSReplay Battlegrounds stats from your own browser session — no
DevTools needed, fully repeatable.

HSReplay has no public API and 403s non-browser traffic, so this opens a REAL
browser window via Playwright (your login persists in a dedicated profile
between runs) and silently records every stats/analytics JSON the site fetches
while you flip filters. The workflow:

    pip install playwright && python -m playwright install chromium   # once
    python scripts/hsreplay_capture.py

    1. A browser window opens on the BG minions page. First run: log in
       (your session is saved to data/stats/hsreplay_profile/ — next runs
       skip this).
    2. Flip filters in the page: set Rank = Top 10%%, then step through the
       TURN filter one value at a time (each change fires a request that gets
       captured — the terminal prints a line per capture so you can see it
       land). Any other filter combos you want, same deal.
    3. Press Enter in this terminal when done. Captures are saved under
       data/stats/hsreplay_captures/<date>/ and imported automatically:
       overall stats -> hsreplay_card_stats.json, turn-filtered stats ->
       hsreplay_card_stats_by_turn.json.

Then fold into the model:  ./scripts/retrain.sh

Repeat any time (patch day, weekly): the login, the capture dir, and the
import are all persistent/idempotent — it's the same 2-minute loop.
"""

import argparse
import datetime
import json
import re
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STATS_DIR = REPO_ROOT / "data" / "stats"
PROFILE_DIR = STATS_DIR / "hsreplay_profile"
CAPTURES_ROOT = STATS_DIR / "hsreplay_captures"
DEFAULT_URL = "https://hsreplay.net/battlegrounds/minions/"

# Capture EVERY JSON response except known trackers/ads — the stats payloads
# have shipped from several hosts/paths over time, so an allowlist misses data;
# the importer sorts out what's usable.
_BLOCKED = re.compile(
    r"google|gstatic|doubleclick|facebook|sentry|amplitude|braze|"
    r"cloudflareinsights|adsystem|quantserve|scorecard|adnxs|criteo|"
    r"prebid|rubicon|pubmatic|onetrust|cookielaw", re.I)


def _interesting(url: str) -> bool:
    return not _BLOCKED.search(url)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="page to open (default: BG minions)")
    ap.add_argument("--no-import", action="store_true",
                    help="capture only; skip the automatic import at the end")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright missing. Install once with:\n"
              "  pip install playwright && python -m playwright install chromium")
        return 1

    capture_dir = CAPTURES_ROOT / datetime.date.today().isoformat()
    capture_dir.mkdir(parents=True, exist_ok=True)
    count = [len(list(capture_dir.glob("cap_*.json")))]

    def on_response(response):
        url = response.url
        if not _interesting(url):
            return
        try:
            ctype = response.headers.get("content-type", "")
            if "json" not in ctype:
                return
            body = response.json()
        except Exception:
            return
        count[0] += 1
        wrapper = {
            "url": url,
            "method": response.request.method,
            "post_data": response.request.post_data,
            "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "body": body,
        }
        out = capture_dir / f"cap_{count[0]:04d}.json"
        out.write_text(json.dumps(wrapper), encoding="utf-8")
        short = url.split("hsreplay.net", 1)[-1][:90]
        print(f"  captured #{count[0]:03d}  {short}")

    print(f"Captures -> {capture_dir}")
    print("""Opening browser… log in if asked, then walk the WHOLE BG section —
every stats request any page makes is captured automatically:

  1. Minions page: set Rank = Top 10%, then step the TURN filter through
     each value (let the table reload between clicks)
  2. Comps page: same — overall, then per-turn if offered
  3. Heroes, Trinkets, Dark Gifts / Quests pages: open each, flip the
     filters you care about

Watch this terminal — a line prints per captured payload.""")
    with sync_playwright() as p:
        # Cloudflare Turnstile rejects obviously-automated browsers, so drop
        # the automation tells: no --enable-automation banner, no
        # navigator.webdriver, and prefer real installed Chrome over the
        # bundled "Chrome for Testing" build (whose branding alone can flag).
        launch = dict(
            headless=False, viewport=None,
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        try:      # your installed Chrome looks most like a normal user
            ctx = p.chromium.launch_persistent_context(
                str(PROFILE_DIR), channel="chrome", **launch)
        except Exception:
            print("(no system Chrome found — using bundled Chromium; if "
                  "Cloudflare still blocks, install Google Chrome or use the "
                  "userscript fallback: scripts/hsreplay_capture.user.js)")
            ctx = p.chromium.launch_persistent_context(
                str(PROFILE_DIR), **launch)
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        ctx.on("response", on_response)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(args.url)

        done = threading.Event()
        threading.Thread(target=lambda: (input("\nPress Enter here when "
                                               "you're done flipping filters… "),
                                         done.set()), daemon=True).start()
        while not done.is_set():
            page.wait_for_timeout(300)
        ctx.close()

    total = len(list(capture_dir.glob("cap_*.json")))
    print(f"\n{total} captured payload(s) in {capture_dir}")
    if not total:
        print("Nothing captured — did the page load and filters change? "
              "(An adblocker can also eat the requests.)")
        return 1

    if args.no_import:
        print(f"Import later with:  python -m hsbg_coach import-hsreplay {capture_dir}")
        return 0
    from hsbg_coach import card_meta_stats, hsreplay_import
    result = hsreplay_import.import_captures(str(capture_dir))
    card_meta_stats.reload()
    print(f"Imported: {result['overall']} minions overall"
          + (f", minion turn splits {result['turns']}" if result["turns"] else ""))
    for cat, n in sorted(result["categories"].items()):
        print(f"  {cat:11s} {n:4d} items -> "
              f"{hsreplay_import.category_path(cat)}")
    if not result["categories"]:
        print("No minion rows recognized in the captures. Run again and make "
              "sure the stats table actually reloads when you flip filters; "
              "if it still finds nothing, share one cap_*.json so the "
              "importer's key mapping can be extended.")
        return 1
    print("Fold into the model:  ./scripts/retrain.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
