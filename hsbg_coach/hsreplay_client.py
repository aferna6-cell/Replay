"""Authenticated HSReplay.net / Tier7 client (stdlib only).

HSReplay does **not** expose a public bulk Battlegrounds trajectory dump.
`GET /api/v1/games/` returns 401 without credentials. Battlegrounds also has no
replay viewer pages on hsreplay.net (HearthSim help: BG combats are too long).

What *is* usable with a Tier7 subscription (and/or session cookie):

  Tier7-gated (403/401 without auth or entitlement):
    GET /api/v1/battlegrounds/heroes/              full hero pick/placement stats
    GET /api/v1/battlegrounds/perfect_games/       composition "perfect game" boards
    GET /api/v1/battlegrounds/inspiration/         board inspiration tool
    GET /api/v1/battlegrounds/duos/heroes/         duos hero stats
    GET /api/v1/games/{shortid}/                  processed replay metadata (+ replay_xml
                                                   when present — usually constructed, not BG)

  Public / lightly gated (no Tier7 required; spiked live 2026-09-22):
    GET /api/v1/battlegrounds/compositions/       id+name list
    GET /api/v1/battlegrounds/trinkets/?BattlegroundsMMRPercentile=TOP_1_PERCENT
    GET /api/v1/battlegrounds/heroes/free/?BattlegroundsMMRPercentile=...
                                                   pick_rate only (teaser)
    GET /api/v1/battlegrounds/meta_periods/        patch/season metadata

Auth (never commit secrets; never paste tokens into chat/PRs):

  HSREPLAY_API_TOKEN   — sent as ``Authorization: Token <value>``
                         (HSReplay DRF AuthToken style used by HDT uploads)
  HSREPLAY_BEARER      — alternative ``Authorization: Bearer <value>`` (OAuth access token)
  HSREPLAY_COOKIE_FILE — path to a cookie header file (one line ``name=value; ...``)
                         or a Netscape cookie jar exported from the browser while
                         logged into hsreplay.net with Tier7 active
  HSREPLAY_API_KEY     — optional ``X-Api-Key`` (agent/app key; rarely needed for
                         personal Tier7 GETs)

How to obtain credentials (Tier7 account, local only):
  1. Log into https://hsreplay.net in a browser (Tier7 active).
  2. Prefer a short-lived OAuth / session cookie export into a file chmod 600,
     then ``export HSREPLAY_COOKIE_FILE=/path/to/hsreplay.cookies``.
  3. Or, if you already use HDT linked to the same account, copy the HSReplay
     auth token from HDT's local store into ``HSREPLAY_API_TOKEN`` (env only).
  4. Never commit the file or token. CI runs without these vars and skips live calls.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BASE = "https://hsreplay.net"
API = BASE + "/api/v1"

# Known Battlegrounds MMR percentile filter members (confirmed live via
# /battlegrounds/trinkets/ error messages + successful GETs).
MMR_PERCENTILES = (
    "TOP_1_PERCENT",
    "TOP_5_PERCENT",
    "TOP_20_PERCENT",
    "TOP_50_PERCENT",
    "ALL",
)

# Surfaces we document + spike. ``auth`` = needs credentials; ``tier7`` = needs
# entitlement (often returns 403 "You do not have access to this query.").
TIER7_SURFACES: Tuple[Dict[str, Any], ...] = (
    {"path": "/api/v1/battlegrounds/heroes/",
     "auth": True, "tier7": True,
     "notes": "Full hero stats for the lobby/meta; free teaser is /heroes/free/."},
    {"path": "/api/v1/battlegrounds/perfect_games/",
     "auth": True, "tier7": True, "params": {"composition_id": "<id>"},
     "notes": "Example top boards per composition — best structured expert boards."},
    {"path": "/api/v1/battlegrounds/inspiration/",
     "auth": True, "tier7": True,
     "notes": "Board inspiration tool (Tier7)."},
    {"path": "/api/v1/battlegrounds/duos/heroes/",
     "auth": True, "tier7": True, "notes": "Duos hero stats."},
    {"path": "/api/v1/games/",
     "auth": True, "tier7": False,
     "notes": "Account/game list. BG games generally have no public replay pages."},
    {"path": "/api/v1/games/{shortid}/",
     "auth": True, "tier7": False,
     "notes": "Replay metadata; may include replay_xml URL for constructed games."},
    {"path": "/api/v1/battlegrounds/trinkets/",
     "auth": False, "tier7": False,
     "params": {"BattlegroundsMMRPercentile": "TOP_1_PERCENT"},
     "notes": "Public top-MMR trinket placements — useful expert prior."},
    {"path": "/api/v1/battlegrounds/heroes/free/",
     "auth": False, "tier7": False,
     "params": {"BattlegroundsMMRPercentile": "TOP_1_PERCENT"},
     "notes": "Public pick_rate teaser only (no placement)."},
    {"path": "/api/v1/battlegrounds/compositions/",
     "auth": False, "tier7": False, "notes": "Public composition id/name list."},
    {"path": "/api/v1/battlegrounds/meta_periods/",
     "auth": False, "tier7": False, "notes": "Patch/season mechanics + tribes."},
)

_UA = "hsbg-coach/0.1 (+https://github.com/aferna6-cell/Replay; Tier7 ingest)"


@dataclass
class AuthConfig:
    """Resolved auth material. Empty => live Tier7 calls are skipped."""
    token: Optional[str] = None          # Authorization: Token …
    bearer: Optional[str] = None         # Authorization: Bearer …
    cookie: Optional[str] = None         # Cookie: …
    api_key: Optional[str] = None        # X-Api-Key: …
    source: str = "none"                 # which env provided credentials

    @property
    def configured(self) -> bool:
        return bool(self.token or self.bearer or self.cookie)

    def headers(self) -> Dict[str, str]:
        h = {"User-Agent": _UA, "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Token {self.token}"
        elif self.bearer:
            h["Authorization"] = f"Bearer {self.bearer}"
        if self.cookie:
            h["Cookie"] = self.cookie
        if self.api_key:
            h["X-Api-Key"] = self.api_key
        return h


def load_auth_from_env(
    environ: Optional[Dict[str, str]] = None,
) -> AuthConfig:
    env = environ if environ is not None else os.environ
    cfg = AuthConfig()
    tok = (env.get("HSREPLAY_API_TOKEN") or "").strip()
    bearer = (env.get("HSREPLAY_BEARER") or "").strip()
    api_key = (env.get("HSREPLAY_API_KEY") or "").strip()
    cookie_file = (env.get("HSREPLAY_COOKIE_FILE") or "").strip()
    if tok:
        cfg.token, cfg.source = tok, "HSREPLAY_API_TOKEN"
    if bearer and not cfg.token:
        cfg.bearer, cfg.source = bearer, "HSREPLAY_BEARER"
    if api_key:
        cfg.api_key = api_key
    if cookie_file:
        cookie = _read_cookie_file(cookie_file)
        if cookie:
            cfg.cookie = cookie
            if cfg.source == "none":
                cfg.source = "HSREPLAY_COOKIE_FILE"
    return cfg


def _read_cookie_file(path: str) -> str:
    """Accept a raw Cookie header line, or a Netscape cookie jar for hsreplay.net."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return ""
    # Netscape jar: lines with tabs; build Cookie header for hsreplay.net hosts.
    if "\t" in text and ("hsreplay.net" in text or text.lstrip().startswith("#")):
        parts = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) >= 7 and "hsreplay.net" in cols[0]:
                parts.append(f"{cols[5]}={cols[6]}")
        return "; ".join(parts)
    # Otherwise treat the whole file as a Cookie header value (possibly multi-line).
    return " ".join(text.splitlines()).strip()


@dataclass
class HttpResult:
    ok: bool
    status: int
    url: str
    data: Any = None
    error: Optional[str] = None
    skipped: bool = False


class HSReplayClient:
    """Thin GET client. Live network only when auth is configured (or force_public)."""

    def __init__(self, auth: Optional[AuthConfig] = None, timeout: float = 30.0):
        self.auth = auth if auth is not None else load_auth_from_env()
        self.timeout = timeout

    def get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        require_auth: bool = False,
        allow_public: bool = True,
    ) -> HttpResult:
        if require_auth and not self.auth.configured:
            return HttpResult(
                ok=False, status=0, url=API + path, skipped=True,
                error="No HSREPLAY_API_TOKEN / HSREPLAY_BEARER / HSREPLAY_COOKIE_FILE; "
                      "skipping authenticated call (CI-safe).",
            )
        if not allow_public and not self.auth.configured:
            return HttpResult(
                ok=False, status=0, url=API + path, skipped=True,
                error="Auth required and not configured.",
            )
        url = path if path.startswith("http") else (API + path if path.startswith("/")
                                                     else f"{API}/{path}")
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self.auth.headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                raw = resp.read()
                status = getattr(resp, "status", 200) or 200
                try:
                    data = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    data = raw.decode("utf-8", errors="replace")
                return HttpResult(ok=True, status=status, url=url, data=data)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            try:
                data = json.loads(body) if body else None
            except json.JSONDecodeError:
                data = body
            detail = None
            if isinstance(data, dict):
                detail = data.get("detail") or data.get("error")
            elif isinstance(data, list) and data:
                detail = str(data[0])
            return HttpResult(
                ok=False, status=int(exc.code), url=url, data=data,
                error=detail or body[:200] or exc.reason,
            )
        except Exception as exc:  # noqa: BLE001 — surface network errors cleanly
            return HttpResult(ok=False, status=0, url=url, error=str(exc))

    # --- convenience wrappers -------------------------------------------------

    def list_compositions(self) -> HttpResult:
        return self.get_json("/battlegrounds/compositions/", require_auth=False)

    def list_trinkets(self, mmr: str = "TOP_1_PERCENT") -> HttpResult:
        return self.get_json(
            "/battlegrounds/trinkets/",
            {"BattlegroundsMMRPercentile": mmr},
            require_auth=False,
        )

    def list_heroes_free(self, mmr: str = "TOP_1_PERCENT") -> HttpResult:
        return self.get_json(
            "/battlegrounds/heroes/free/",
            {"BattlegroundsMMRPercentile": mmr},
            require_auth=False,
        )

    def list_heroes(self, mmr: str = "TOP_1_PERCENT") -> HttpResult:
        return self.get_json(
            "/battlegrounds/heroes/",
            {"BattlegroundsMMRPercentile": mmr},
            require_auth=True,
        )

    def perfect_games(self, composition_id: int,
                      mmr: str = "TOP_1_PERCENT") -> HttpResult:
        return self.get_json(
            "/battlegrounds/perfect_games/",
            {"composition_id": composition_id,
             "BattlegroundsMMRPercentile": mmr},
            require_auth=True,
        )

    def inspiration(self, **params: Any) -> HttpResult:
        return self.get_json("/battlegrounds/inspiration/", params or None,
                             require_auth=True)

    def meta_periods(self) -> HttpResult:
        return self.get_json("/battlegrounds/meta_periods/", require_auth=False)

    def game(self, shortid: str) -> HttpResult:
        return self.get_json(f"/games/{shortid}/", require_auth=True)

    def fetch_replay_xml(self, shortid: str) -> HttpResult:
        """Fetch game metadata, then download ``replay_xml`` if the payload has one.

        Note: Battlegrounds games typically do **not** expose replay pages / XML
        on HSReplay. This path is mainly for constructed shortids or rare cases
        where a ``replay_xml`` URL is present. Prefer ``.hsreplay`` file ingest
        for BG trajectories.
        """
        meta = self.game(shortid)
        if not meta.ok:
            return meta
        if not isinstance(meta.data, dict):
            return HttpResult(ok=False, status=meta.status, url=meta.url,
                              error="Unexpected game payload", data=meta.data)
        xml_url = meta.data.get("replay_xml")
        if not xml_url:
            return HttpResult(
                ok=False, status=meta.status, url=meta.url, data=meta.data,
                error="No replay_xml on this game (common for Battlegrounds).",
            )
        if xml_url.startswith("/"):
            xml_url = BASE + xml_url
        req = urllib.request.Request(xml_url, headers=self.auth.headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                text = resp.read().decode("utf-8", errors="replace")
                return HttpResult(ok=True, status=200, url=xml_url, data=text)
        except Exception as exc:  # noqa: BLE001
            return HttpResult(ok=False, status=0, url=xml_url, error=str(exc),
                              data=meta.data)


def _api_rel(path: str) -> str:
    """Normalize a documented /api/v1/... path to the client-relative form."""
    if path.startswith("/api/v1/"):
        return path[len("/api/v1"):]
    if path.startswith("http"):
        # strip scheme+host+/api/v1
        idx = path.find("/api/v1/")
        return path[idx + len("/api/v1"):] if idx >= 0 else path
    return path


def spike_surfaces(client: Optional[HSReplayClient] = None,
                   *, live: bool = True) -> List[Dict[str, Any]]:
    """Document + optionally probe each known surface.

    When ``live`` is False, or auth is missing for auth-required paths, entries
    are marked skipped without a network call (safe for CI).
    """
    client = client or HSReplayClient()
    out: List[Dict[str, Any]] = []
    for surf in TIER7_SURFACES:
        row = dict(surf)
        path = surf["path"]
        if "{shortid}" in path:
            row["probe"] = {"skipped": True, "reason": "needs a real shortid"}
            out.append(row)
            continue
        needs_auth = bool(surf.get("auth") or surf.get("tier7"))
        if not live or (needs_auth and not client.auth.configured):
            row["probe"] = {
                "skipped": True,
                "reason": ("auth not configured" if needs_auth
                           else "live probe disabled"),
            }
            out.append(row)
            continue
        params = {k: v for k, v in (surf.get("params") or {}).items()
                  if not str(v).startswith("<")}
        if path.endswith("perfect_games/") and "composition_id" not in params:
            params["composition_id"] = 1
        result = client.get_json(_api_rel(path), params or None,
                                 require_auth=needs_auth)
        n = None
        if isinstance(result.data, list):
            n = len(result.data)
        elif isinstance(result.data, dict):
            n = len(result.data)
        row["probe"] = {
            "ok": result.ok,
            "status": result.status,
            "skipped": result.skipped,
            "error": result.error,
            "url": result.url,
            "n": n,
        }
        out.append(row)
    return out


def describe_auth(auth: Optional[AuthConfig] = None) -> str:
    auth = auth if auth is not None else load_auth_from_env()
    if not auth.configured:
        return ("Auth: not configured (set HSREPLAY_API_TOKEN, HSREPLAY_BEARER, "
                "or HSREPLAY_COOKIE_FILE). Live Tier7 GETs will be skipped.")
    return f"Auth: configured via {auth.source} (value never printed)."
