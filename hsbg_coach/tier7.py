"""Tier7 (HSReplay.net premium) Battlegrounds stats bridge.

HSReplay's Tier7 subscription serves **lobby-conditioned** stats the free
Firestone aggregates don't: hero pick tiers/placements *given the tribes in
your lobby and your MMR*, first-place comps *for this lobby's tribe set*, and
trinket/quest pick stats. There is no documented public API; these are the
private client endpoints HDT/HSTracker use, recovered from HSTracker's
open-source Swift client (see ADR ``decisions/2026-08-20-tier7-bridge.md``):

  POST https://api.hsreplay.net/battlegrounds/hero_pick/
  POST https://api.hsreplay.net/battlegrounds/duos/hero_pick/
  POST https://api.hsreplay.net/battlegrounds/quest_pick/
  POST https://api.hsreplay.net/battlegrounds/trinket_pick/
  POST https://api.hsreplay.net/battlegrounds/first_place_comps/
  GET  https://api.hsreplay.net/battlegrounds/alltime/

Auth: OAuth2 Bearer token from an HSReplay.net account with an **active Tier7
subscription** — i.e. *your* account. Token discovery (in order):

  1. ``HSREPLAY_TOKEN`` env var (the raw access token)
  2. ``HSREPLAY_OAUTH_FILE`` env var → path to HDT's ``hsreplay_oauth.json``
  3. HDT's default location (Windows: ``%APPDATA%/HearthstoneDeckTracker/``)

We deliberately do NOT use the refresh token: refresh-token grants rotate in
django-oauth-toolkit, so refreshing from here could invalidate HDT's own copy
and silently log the tracker out. If the access token has expired (HTTP 401),
launch HDT once (it refreshes + re-saves) and retry.

Requests/responses mirror HSTracker's ``BattlegroundsTrinketPickParams`` /
``BattlegroundsQuestPickParams`` / ``BattlegroundsHeroPickStats`` /
``BattlegroundsCompStats`` structs (verified from source). The hero-pick
*request* struct isn't fetchable in the open repo, so its field names follow
the sibling structs and HDT's documented signature — marked ``# CALIBRATE``;
on a 400/422 the server's error body is surfaced verbatim to fix field names
in one edit.

Every successful response is appended to ``data/tier7_log.jsonl`` (gitignored,
like game trajectories): each lobby-conditioned answer is a labeled example
for the ML pick-trainer, so querying during play grows the corpus for free.

Personal-use note: these endpoints exist for HSReplay's own clients. Pulling
them with your own subscribed account for your own overlay is the same data
your tracker fetches every game; do not commit or redistribute the responses.
"""

import datetime
import json
import os
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

from .firestone_stats import CARDS_URL, RACE_NAMES, _fetch_json

BASE_API = "https://api.hsreplay.net"
HERO_PICK_URL = BASE_API + "/battlegrounds/hero_pick/"
DUOS_HERO_PICK_URL = BASE_API + "/battlegrounds/duos/hero_pick/"
QUEST_PICK_URL = BASE_API + "/battlegrounds/quest_pick/"
TRINKET_PICK_URL = BASE_API + "/battlegrounds/trinket_pick/"
COMPS_URL = BASE_API + "/battlegrounds/first_place_comps/"
ALLTIME_URL = BASE_API + "/battlegrounds/alltime/"

# Our tribe names -> Hearthstone CARDRACE ids (the API's `minion_types`).
TRIBE_TO_MINION_TYPE = {name.lower(): rid for rid, name in RACE_NAMES.items()}

# Hearthstone GameType enum: 23 = GT_BATTLEGROUNDS. # CALIBRATE vs a live
# request if the server rejects it (duos may want its own value).
GAME_TYPE_BG = 23

_STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stats")
DBF_MAP = os.path.join(_STATS_DIR, "dbf_map.json")
QUERY_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "tier7_log.jsonl")


# --- Token discovery --------------------------------------------------------

def _token_in(obj) -> Optional[str]:
    """Recursively find an access token in HDT's ``hsreplay_oauth.json``.

    The file is C#-serialized (`OAuthData` with a nested `TokenData`), and key
    casing isn't guaranteed across HDT versions — match case-insensitively.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower().replace("_", "") == "accesstoken" and isinstance(v, str) and v:
                return v
        for v in obj.values():
            tok = _token_in(v)
            if tok:
                return tok
    elif isinstance(obj, list):
        for v in obj:
            tok = _token_in(v)
            if tok:
                return tok
    return None


def _oauth_file_candidates() -> List[str]:
    out = []
    env_file = os.environ.get("HSREPLAY_OAUTH_FILE")
    if env_file:
        out.append(env_file)
    appdata = os.environ.get("APPDATA")           # Windows (HDT's home)
    if appdata:
        out.append(os.path.join(appdata, "HearthstoneDeckTracker", "hsreplay_oauth.json"))
    out.append(os.path.expanduser(
        "~/AppData/Roaming/HearthstoneDeckTracker/hsreplay_oauth.json"))
    return out


def find_token() -> Optional[str]:
    """Locate a Tier7-capable access token. Returns None (with no side effects)
    if nothing is found — callers print the how-to."""
    env = os.environ.get("HSREPLAY_TOKEN")
    if env:
        return env.strip()
    for path in _oauth_file_candidates():
        try:
            with open(path, encoding="utf-8") as fh:
                tok = _token_in(json.load(fh))
            if tok:
                return tok
        except (OSError, ValueError):
            continue
    return None


TOKEN_HELP = (
    "No HSReplay token found. Either:\n"
    "  * set HSREPLAY_TOKEN to your access token, or\n"
    "  * set HSREPLAY_OAUTH_FILE to HDT's hsreplay_oauth.json, or\n"
    "  * run this on the machine where HDT is logged in (Windows:\n"
    "    %APPDATA%/HearthstoneDeckTracker/hsreplay_oauth.json).\n"
    "A 401 means the token expired — launch HDT once to refresh it."
)


# --- HTTP -------------------------------------------------------------------

def _request(url: str, token: str, payload: Optional[dict] = None,
             timeout: int = 30) -> dict:
    """POST (or GET when payload is None) with Bearer auth. On an HTTP error
    the server body is included verbatim — it names bad/missing fields, which
    is how the # CALIBRATE items get pinned down."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "hsbg-coach tier7 bridge",
    }, method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        if exc.code == 401:
            raise RuntimeError(
                "HSReplay rejected the token (401). Launch HDT once to refresh "
                "it, then retry.") from exc
        if exc.code in (402, 403):
            raise RuntimeError(
                f"HSReplay says this account can't access Tier7 data "
                f"({exc.code}): {body}") from exc
        raise RuntimeError(f"Tier7 request failed ({exc.code}) for {url}: {body}") from exc


# --- dbf-id map (the API speaks dbf ids, not card ids/names) ---------------

def refresh_dbf_map(cards_source=CARDS_URL, path: str = DBF_MAP) -> Dict[str, dict]:
    """Build name<->dbfId maps from HearthstoneJSON and cache them.
    Keeps every card (heroes, trinkets, quests and minions all have dbfIds)."""
    cards = _fetch_json(cards_source) if isinstance(cards_source, str) else cards_source
    by_name: Dict[str, int] = {}
    by_id: Dict[str, str] = {}
    for c in cards:
        dbf, name = c.get("dbfId"), c.get("name")
        if not dbf or not name:
            continue
        by_id[str(dbf)] = name
        # First writer wins on name collisions (skins/copies share names);
        # ids stay unambiguous via by_id.
        by_name.setdefault(name.lower(), dbf)
    out = {"_fetched": datetime.date.today().isoformat(),
           "by_name": by_name, "by_id": by_id}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def load_dbf_map(path: str = DBF_MAP, auto_refresh: bool = True) -> Dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        if not auto_refresh:
            raise
        return refresh_dbf_map(path=path)


def resolve_dbf_ids(names: List[str], dbf_map: Dict[str, dict]) -> List[int]:
    """Names (or literal dbf ids) -> dbf ids. Raises naming every miss so a
    typo'd hero doesn't silently vanish from the request."""
    by_name = dbf_map.get("by_name", {})
    out, missing = [], []
    for n in names:
        key = (n or "").strip()
        if key.isdigit():
            out.append(int(key))
            continue
        hit = by_name.get(key.lower())
        if hit is None:                          # substring fallback (skins etc.)
            partial = [v for k, v in by_name.items() if key.lower() in k]
            hit = partial[0] if len(partial) == 1 else None
        if hit is None:
            missing.append(key)
        else:
            out.append(hit)
    if missing:
        raise ValueError(f"unknown card name(s): {missing} — check spelling or "
                         f"run `refresh-cards` / delete {DBF_MAP} to rebuild the map")
    return out


def tribes_to_minion_types(tribes: List[str]) -> List[int]:
    out, missing = [], []
    for t in tribes:
        rid = TRIBE_TO_MINION_TYPE.get((t or "").strip().lower())
        (out if rid else missing).append(rid or (t or "").strip())
    if missing:
        raise ValueError(f"unknown tribe(s): {missing} — expected "
                         f"{sorted(TRIBE_TO_MINION_TYPE)}")
    return out


# --- Param builders (mirror HSTracker's request structs) --------------------

def hero_pick_params(hero_names: List[str], tribes: List[str], dbf_map: dict,
                     rating: Optional[int] = None, anomaly: Optional[str] = None,
                     language: str = "enUS") -> dict:
    # CALIBRATE: field names follow the verified sibling structs
    # (BattlegroundsQuestPickParams/TrinketPickParams) + HDT's signature
    # "hero dbf ids, minion types, language, rating". A 400 will name any
    # mismatch in its body.
    p = {
        "hero_dbf_ids": resolve_dbf_ids(hero_names, dbf_map),
        "minion_types": tribes_to_minion_types(tribes),
        "game_language": language,
        "game_type": GAME_TYPE_BG,
        "battlegrounds_rating": rating,
    }
    if anomaly:
        p["anomaly_dbf_id"] = resolve_dbf_ids([anomaly], dbf_map)[0]
    return p


def comps_params(tribes: List[str], language: str = "enUS") -> dict:
    # Verified: BattlegroundsCompStatsParams {minion_types, game_language,
    # include_toast}.
    return {"minion_types": tribes_to_minion_types(tribes),
            "game_language": language, "include_toast": True}


def trinket_pick_params(hero: str, offered: List[str], tribes: List[str],
                        turn: int, dbf_map: dict, rating: Optional[int] = None,
                        hero_powers: Optional[List[str]] = None,
                        anomaly: Optional[str] = None, source: Optional[str] = None,
                        language: str = "enUS") -> dict:
    # Verified: BattlegroundsTrinketPickParams. `source_dbf_id` is the entity
    # offering the trinkets; 0 until live calibration. # CALIBRATE
    return {
        "hero_dbf_id": resolve_dbf_ids([hero], dbf_map)[0],
        "hero_power_dbf_ids": resolve_dbf_ids(hero_powers or [], dbf_map),
        "minion_types": tribes_to_minion_types(tribes),
        "anomaly_dbf_id": resolve_dbf_ids([anomaly], dbf_map)[0] if anomaly else None,
        "turn": turn,
        "source_dbf_id": resolve_dbf_ids([source], dbf_map)[0] if source else 0,
        "offered_trinkets": [{"trinket_dbf_id": d}
                             for d in resolve_dbf_ids(offered, dbf_map)],
        "game_language": language,
        "game_type": GAME_TYPE_BG,
        "battlegrounds_rating": rating,
    }


def quest_pick_params(hero: str, offered: List[str], tribes: List[str],
                      turn: int, dbf_map: dict, rating: Optional[int] = None,
                      hero_powers: Optional[List[str]] = None,
                      anomaly: Optional[str] = None, language: str = "enUS") -> dict:
    # Verified: BattlegroundsQuestPickParams.
    return {
        "hero_dbf_id": resolve_dbf_ids([hero], dbf_map)[0],
        "hero_power_dbf_ids": resolve_dbf_ids(hero_powers or [], dbf_map),
        "turn": turn,
        "minion_types": tribes_to_minion_types(tribes),
        "anomaly_dbf_id": resolve_dbf_ids([anomaly], dbf_map)[0] if anomaly else None,
        "offered_rewards": [{"reward_dbf_id": d}
                            for d in resolve_dbf_ids(offered, dbf_map)],
        "game_language": language,
        "game_type": GAME_TYPE_BG,
        "battlegrounds_rating": rating,
    }


# --- Response normalization (mirror HSTracker's response structs) -----------

def normalize_hero_pick(resp: dict, by_id: Dict[str, str]) -> List[dict]:
    """BattlegroundsHeroPickStats -> rows sorted best-first (lowest avg place).
    Rows without an avg_placement sort last but are kept (tier may still show)."""
    rows = []
    for h in resp.get("data", []) or []:
        dbf = h.get("hero_dbf_id")
        comps = [c.get("name") for c in (h.get("first_place_comp_popularity") or [])
                 if c.get("name")][:3]
        rows.append({
            "dbf_id": dbf,
            "name": by_id.get(str(dbf), str(dbf)),
            "tier": h.get("tier_v2"),
            "pick_rate": h.get("pick_rate"),
            "avg_placement": h.get("avg_placement"),
            "placement_distribution": h.get("placement_distribution"),
            "top_comps": comps,
        })
    rows.sort(key=lambda r: r["avg_placement"] if r["avg_placement"] is not None else 9.0)
    toast = resp.get("toast") or {}
    if toast.get("min_mmr") is not None:
        for r in rows:
            r["min_mmr"] = toast["min_mmr"]
    return rows


def normalize_comps(resp: dict, by_id: Dict[str, str]) -> List[dict]:
    """BattlegroundsCompStats -> rows sorted by avg final placement."""
    data = resp.get("data") or {}
    rows = []
    for c in data.get("first_place_comps_lobby_races", []) or []:
        rows.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "popularity": c.get("popularity"),
            "avg_final_placement": c.get("avg_final_placement"),
            "key_minions": [by_id.get(str(d), str(d))
                            for d in (c.get("key_minions_top3") or [])],
        })
    rows.sort(key=lambda r: r["avg_final_placement"]
              if r["avg_final_placement"] is not None else 9.0)
    return rows


# --- ML corpus log ----------------------------------------------------------

def log_query(kind: str, params: dict, resp: dict, path: str = QUERY_LOG) -> None:
    """Append (context, answer) to the local JSONL corpus. Best-effort — a
    logging failure never breaks a live pick."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "kind": kind, "params": params, "response": resp,
            }) + "\n")
    except OSError:
        pass


# --- High-level queries (fetch injectable for tests) ------------------------

Fetch = Callable[[str, str, Optional[dict]], dict]


def query_hero_pick(hero_names: List[str], tribes: List[str], *,
                    token: str, rating: Optional[int] = None,
                    anomaly: Optional[str] = None, duos: bool = False,
                    dbf_map: Optional[dict] = None,
                    fetch: Fetch = _request, log_path: str = QUERY_LOG) -> List[dict]:
    dbf_map = dbf_map or load_dbf_map()
    params = hero_pick_params(hero_names, tribes, dbf_map, rating=rating,
                              anomaly=anomaly)
    url = DUOS_HERO_PICK_URL if duos else HERO_PICK_URL
    resp = fetch(url, token, params)
    log_query("duos_hero_pick" if duos else "hero_pick", params, resp, log_path)
    return normalize_hero_pick(resp, dbf_map.get("by_id", {}))


def query_comps(tribes: List[str], *, token: str,
                dbf_map: Optional[dict] = None,
                fetch: Fetch = _request, log_path: str = QUERY_LOG) -> List[dict]:
    dbf_map = dbf_map or load_dbf_map()
    params = comps_params(tribes)
    resp = fetch(COMPS_URL, token, params)
    log_query("first_place_comps", params, resp, log_path)
    return normalize_comps(resp, dbf_map.get("by_id", {}))


def query_trinket_pick(hero: str, offered: List[str], tribes: List[str], turn: int, *,
                       token: str, rating: Optional[int] = None,
                       dbf_map: Optional[dict] = None,
                       fetch: Fetch = _request, log_path: str = QUERY_LOG) -> dict:
    """Raw trinket-pick response (response schema not yet normalized — logged
    for calibration + the ML corpus, and returned for display)."""
    dbf_map = dbf_map or load_dbf_map()
    params = trinket_pick_params(hero, offered, tribes, turn, dbf_map, rating=rating)
    resp = fetch(TRINKET_PICK_URL, token, params)
    log_query("trinket_pick", params, resp, log_path)
    return resp
