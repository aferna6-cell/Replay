"""Pull new Firestone Battlegrounds replays + a manifest the replay pipeline reads.

Source: Firestone's public "perfect games" list (the one its app shows under
Battlegrounds > Replays), a rolling window of the latest ~1,000 first-place
games with their review ids, replay keys, player MMR and final board:

    https://static.zerotoheroes.com/api/bgs/bgs-perfect-games.json

Each replay is a zip holding ``replay.xml`` at ``https://xml.firestoneapp.com/<replayKey>``.
It is re-written as ``<reviewId>.xml.gz``, the name ``replay_states`` and
``replay_labels`` expect, and each game gets a manifest entry in the shape they
read (``reviewId``, ``buildNumber``, ``mmr``, ``placement``, ``finalComp`` with
``turn`` + ``board[{cardId, golden}]``, ``tribes.available[{name}]``,
``creationDate``/``creationTimestamp``).

    python -m hsbg_coach.firestone_replays --out data/firestone/incoming \\
        --manifest data/firestone/raw/manifest.json

Only games whose ``reviewId`` (or ``originalReviewId``) is not already in the
manifest or in the replay folder are downloaded. Stdlib only. Everything under
``data/firestone/`` is gitignored and must never be committed.
"""

import argparse
import base64
import gzip
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .hs_enums import ENUM_VALUES

PERFECT_GAMES_URL = "https://static.zerotoheroes.com/api/bgs/bgs-perfect-games.json"
REPLAY_URL = "https://xml.firestoneapp.com/{key}"
USER_AGENT = "hsbg-coach replay pipeline (github.com/aferna6-cell/Replay)"
SOURCE = "firestone:bgs-perfect-games"
RACE_NAMES = ENUM_VALUES["CARDRACE"]

HttpGet = Callable[[str], bytes]


def http_get(url: str, timeout: float = 60.0, retries: int = 3, backoff: float = 2.0) -> bytes:
    """GET with a few retries on network errors and 5xx/429. Handles gzip bodies."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def fetch_listing(get: HttpGet = http_get, url: str = PERFECT_GAMES_URL) -> List[Dict]:
    data = json.loads(get(url))
    if not isinstance(data, list):
        raise ValueError(f"unexpected listing shape from {url}: {type(data).__name__}")
    return data


def decode_final_comp(value) -> Dict:
    """finalComp arrives as base64(zlib(json)); older records may be plain dicts."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    return json.loads(zlib.decompress(base64.b64decode(value)))


def _int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tribes(ids: Iterable) -> List[Dict]:
    out = []
    for raw in ids or []:
        tid = _int(raw)
        if tid is not None:
            out.append({"id": tid, "name": RACE_NAMES.get(tid, str(tid))})
    return out


def to_manifest_entry(game: Dict) -> Dict:
    """One listing record -> the manifest entry replay_states / replay_labels read."""
    fc = decode_final_comp(game.get("finalComp"))
    board = []
    for m in sorted(fc.get("board") or [],
                    key=lambda m: _int((m.get("tags") or {}).get("ZONE_POSITION")) or 0):
        tags = m.get("tags") or {}
        board.append({"cardId": m.get("cardID") or m.get("cardId"),
                      "golden": _int(tags.get("PREMIUM")) == 1,
                      "attack": _int(tags.get("ATK")), "health": _int(tags.get("HEALTH"))})
    return {
        "reviewId": game["reviewId"],
        "originalReviewId": game.get("originalReviewId"),
        "replayKey": game.get("replayKey"),
        "buildNumber": _int(game.get("buildNumber")),
        "mmr": _int(game.get("playerRank")),
        # Firestone lists only won games here (its app shows them as result "1").
        # replay_states still checks this against the replay's final leaderboard place.
        "placement": 1,
        "playerCardId": game.get("playerCardId"),
        "archetype": game.get("bgsCompArchetype") or game.get("archetype") or None,
        "finalComp": {"turn": _int(fc.get("turn")), "board": board},
        "tribes": {"available": _tribes(game.get("bgsAvailableTribes")),
                   "banned": _tribes(game.get("bgsBannedTribes"))},
        "anomalies": game.get("bgsAnomalies") or [],
        "creationDate": game.get("creationDate"),
        "creationTimestamp": _int(game.get("creationTimestamp")),
        "gameDurationTurns": _int(game.get("gameDurationTurns")),
        "gameDurationSeconds": _int(game.get("gameDurationSeconds")),
        "source": SOURCE,
    }


def replay_bytes_to_xml(data: bytes, key: str = "") -> bytes:
    """Replay download -> raw XML. Firestone serves ``.xml.zip`` (one replay.xml)."""
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xml")] or zf.namelist()
            if len(names) != 1:
                raise ValueError(f"{key}: expected one XML in the zip, got {names}")
            return zf.read(names[0])
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    if data.lstrip()[:5] == b"<?xml" or data.lstrip()[:9] == b"<HSReplay":
        return data
    raise ValueError(f"{key}: not a zip, gzip or XML replay ({data[:16]!r})")


def write_xml_gz(xml: bytes, path: str) -> None:
    if b"<HSReplay" not in xml[:4096]:
        raise ValueError(f"{os.path.basename(path)}: no <HSReplay> root")
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as fh:
        fh.write(xml)
    os.replace(tmp, path)


# --- manifest ----------------------------------------------------------------

def load_manifest(path: str) -> Tuple[object, List[Dict]]:
    """(container, games). Container is the parsed JSON ({games:[...]} or a list)."""
    if not os.path.isfile(path):
        return {"games": []}, []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    games = data.get("games", []) if isinstance(data, dict) else data
    return data, games


def save_manifest(path: str, container, games: List[Dict]) -> None:
    """Atomic write that keeps the container shape ({games: [...]} or a bare list)."""
    if isinstance(container, dict):
        container = dict(container, games=games)
    else:
        container = games
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(container, fh, indent=1)
    os.replace(tmp, path)


def append_to_manifest(path: str, entries: List[Dict]) -> int:
    """Add entries whose reviewId is not in the manifest yet. Returns how many were added."""
    container, games = load_manifest(path)
    have = {g.get("reviewId") for g in games}
    new = [e for e in entries if e["reviewId"] not in have]
    if new:
        save_manifest(path, container, list(games) + new)
    return len(new)


def known_ids(games: Iterable[Dict], replay_dirs: Iterable[str] = ()) -> Set[str]:
    ids: Set[str] = set()
    for g in games:
        for key in ("reviewId", "originalReviewId"):
            if g.get(key):
                ids.add(g[key])
    for d in replay_dirs:
        if os.path.isdir(d):
            ids.update(f[:-len(".xml.gz")] for f in os.listdir(d) if f.endswith(".xml.gz"))
    return ids


def select_new(listing: List[Dict], known: Set[str], limit: Optional[int] = None) -> List[Dict]:
    """Listing records not seen before, oldest first (so a --limit keeps up in order)."""
    seen: Set[str] = set()
    out = []
    for g in sorted(listing, key=lambda g: g.get("creationTimestamp") or 0):
        rid, orig = g.get("reviewId"), g.get("originalReviewId")
        if not rid or not g.get("replayKey") or rid in known or rid in seen \
                or (orig and orig in known):
            continue
        seen.add(rid)
        out.append(g)
    return out if limit is None else out[:limit]


def download(games: List[Dict], out_dir: str, get: HttpGet = http_get,
             pause_s: float = 0.5, log=print) -> Tuple[List[Dict], List[Dict]]:
    """Download each replay to ``<out_dir>/<reviewId>.xml.gz``.

    Returns (manifest entries for the games written, failures). A failed game is
    not added to anything, so the next run tries it again while it is still listed.
    """
    os.makedirs(out_dir, exist_ok=True)
    ok, failed = [], []
    for i, g in enumerate(games):
        rid = g["reviewId"]
        dest = os.path.join(out_dir, f"{rid}.xml.gz")
        try:
            entry = to_manifest_entry(g)
            if not os.path.isfile(dest):
                xml = replay_bytes_to_xml(get(REPLAY_URL.format(key=g["replayKey"])), rid)
                write_xml_gz(xml, dest)
            ok.append(entry)
            log(f"[{i + 1}/{len(games)}] {rid} ok")
        except Exception as exc:                 # one bad game never stops the batch
            failed.append({"reviewId": rid, "error": f"{type(exc).__name__}: {exc}"})
            log(f"[{i + 1}/{len(games)}] {rid} FAILED {type(exc).__name__}: {exc}")
        if pause_s and i + 1 < len(games):
            time.sleep(pause_s)                  # be polite to Firestone's bucket
    return ok, failed


def fetch_new(out_dir: str, manifest: str, replay_dirs: Iterable[str] = (),
              limit: Optional[int] = None, get: HttpGet = http_get,
              pause_s: float = 0.5, log=print) -> Dict:
    """List, select unseen games, download them into out_dir. Does not touch the manifest."""
    _, games = load_manifest(manifest)
    listing = fetch_listing(get)
    known = known_ids(games, [out_dir, *replay_dirs])
    todo = select_new(listing, known, limit)
    log(f"listing: {len(listing)} games, already have {len(listing) - len(select_new(listing, known))}, "
        f"downloading {len(todo)}")
    ok, failed = download(todo, out_dir, get, pause_s, log)
    return {"listed": len(listing), "selected": len(todo), "entries": ok, "failed": failed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="folder for new <reviewId>.xml.gz files")
    ap.add_argument("--manifest", required=True,
                    help="corpus manifest; games already in it are skipped")
    ap.add_argument("--replays", action="append", default=[],
                    help="extra replay folder whose games count as already had (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="max games to download")
    ap.add_argument("--batch-manifest", default=None,
                    help="write the new games' entries here ({games:[...]})")
    ap.add_argument("--append", action="store_true",
                    help="also append the new entries to --manifest")
    args = ap.parse_args(argv)
    res = fetch_new(args.out, args.manifest, args.replays, args.limit)
    if args.batch_manifest:
        save_manifest(args.batch_manifest, {"games": []}, res["entries"])
    if args.append:
        append_to_manifest(args.manifest, res["entries"])
    print(f"downloaded {len(res['entries'])}/{res['selected']} new games "
          f"({len(res['failed'])} failed; {res['listed']} listed)")
    return 1 if res["failed"] and not res["entries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
