# ADR: Tier7 (HSReplay premium) bridge for lobby-conditioned stats

Date: 2026-08-20
Status: Accepted (supplements, does not replace, the Firestone bridge)

## Context

The 2026-06-24 Firestone ADR said "HSReplay's detailed BG data is effectively
subscription-gated with no clean public API" and picked Firestone. That was
half-right: there is no *documented* public API (HSReplay's api-docs repo says
so explicitly), but the actual Tier7 endpoints are plain REST, visible in
HearthSim's own open-source clients. Recovered 2026-08-20 from HSTracker's
Swift source (`HSTracker/HSReplay/HSReplay.swift`, `HSReplayAPI.swift`,
`Data/*.swift`) and HDT (`HsReplay/HSReplayNetOAuth.cs`):

- `POST https://api.hsreplay.net/battlegrounds/hero_pick/` (+ `duos/hero_pick/`)
- `POST https://api.hsreplay.net/battlegrounds/quest_pick/`
- `POST https://api.hsreplay.net/battlegrounds/trinket_pick/`
- `POST https://api.hsreplay.net/battlegrounds/first_place_comps/`
- `GET  https://api.hsreplay.net/battlegrounds/alltime/`

Auth: OAuth2 Bearer token for an account with an active Tier7 subscription
(the owner is subscribed). HDT persists its token to
`%APPDATA%/HearthstoneDeckTracker/hsreplay_oauth.json`; its OAuth client_id is
public in its source.

Why this matters when Firestone is free: Tier7 is **lobby-conditioned** —
hero pick tiers/placements *given the tribes in this lobby* and MMR, and
first-place comps *for this lobby's tribe set*. Firestone's public aggregates
don't slice by lobby. That's exactly the context the draft ranker
(`draft.py`) and the future ML pick-trainer want. Tier7 is still aggregates,
not trajectories — it does not change the ML data plan.

## Decision

1. **Bridge, opt-in, personal token** — `hsbg_coach/tier7.py` + the `tier7`
   CLI. Token discovery: `HSREPLAY_TOKEN` env → `HSREPLAY_OAUTH_FILE` env →
   HDT's default file. Firestone stays the default StatsDB source; nothing
   breaks without a token.
2. **Never use the refresh token.** django-oauth-toolkit rotates refresh
   tokens; refreshing here could invalidate HDT's stored copy and log the
   tracker out. On 401 the fix is "launch HDT once" — the tool says so.
3. **Log every response** to `data/tier7_log.jsonl` (gitignored, like game
   trajectories): each lobby-conditioned answer is a labeled example for the
   pick-trainer. Do not commit or redistribute responses — these endpoints
   exist for HSReplay's own clients; pulling with your own subscription for
   your own overlay is personal use, republishing the data is not.
4. **dbf-id map** — the API speaks dbf ids; `data/stats/dbf_map.json` is
   built from HearthstoneJSON (which carries `dbfId`) and cached.

## Verified vs CALIBRATE

Verified against HSTracker source: trinket/quest *request* structs, hero-pick
and comps *response* structs, comps request struct, all six URLs, Bearer auth.

`# CALIBRATE` (confirm on first live call; error bodies name bad fields):
- hero-pick *request* field names (struct not in the open repo; follows the
  sibling structs + HDT's documented signature)
- `game_type` enum value (23 = GT_BATTLEGROUNDS assumed; duos may differ)
- `source_dbf_id` on trinket picks (0 until a live game shows the real value)
- exact `hsreplay_oauth.json` key casing (parser matches case-insensitively)

## Consequences

- New module + CLI, stdlib-only, fully offline-testable (`tests/test_tier7.py`,
  fetch injectable). No new dependencies.
- The live path needs a real token + a live call to close the CALIBRATE list —
  same "calibrated on capture, shaken down in game" posture as the parser.
- If HSReplay changes or gates these endpoints, the bridge degrades to a clear
  error and Firestone remains the working default.
