# ADR: HSReplay Tier7 authenticated expert ingest

Date: 2026-09-22
Status: Accepted

## Context

Aidan wants expert training signal from HSReplay / top players before investing
in Jeef VOD pipelines. He has an active Tier7 subscription.

Reality checks (live, 2026-09-22):

- `GET https://hsreplay.net/api/v1/games/` → **401** without credentials.
- HearthSim documents that **Battlegrounds has no replay pages / My Replays**.
- Public BG API surfaces that work without Tier7: compositions list, trinkets
  (with `BattlegroundsMMRPercentile`), heroes/free (pick_rate only), meta_periods.
- Tier7-gated: heroes (full), perfect_games (403 "You do not have access…"),
  inspiration, duos heroes.

Firestone’s public CDN already supplies top-MMR aggregate priors (`mmr=1|10`).

## Decision

1. Add an authenticated client (`hsreplay_client.py`) reading
   `HSREPLAY_API_TOKEN` / `HSREPLAY_BEARER` / `HSREPLAY_COOKIE_FILE` — never
   commit secrets; CI skips live auth calls when unset.
2. Prefer Tier7 **perfect_games** (+ inspiration when available) as structured
   expert boards → `data/*.jsonl` with `source=hsreplay_expert`.
3. Keep **`.hsreplay` / `.xml` file ingest** as the fallback for per-game XML
   when the user has exports (shortid `replay_xml` is unreliable for BG).
4. Upweight expert rows in `train_eval_net --expert-weight` (default 3).
5. Keep Firestone `refresh-stats --mmr 10|1` as the expert **population** prior
   wired into advise/stats via `expert_prior_note()`.
6. Jeef VOD remains a manual fallback for narrated mid-game lines; out of scope
   for this automation.

## Consequences

- Expert signal is strongest for **endgame boards** and **meta priors**, weaker
  for click-by-click mid-game actions unless local HSReplay XML is available.
- Token mishandling risk is mitigated by env-only auth and README warnings.
