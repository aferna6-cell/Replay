# hsbg-coach

A Hearthstone **Battlegrounds** advisory tool (HDT-style), with a future ML move
recommender. This repo is **milestone 1+2**: read the game from Hearthstone's
`Power.log`, reconstruct Battlegrounds game state, and log every
`(state, action, outcome)` tuple to disk as you play.

That trajectory log is the foundation for *everything* downstream — the live
overlay, the combat-odds panel, and the eventual learned move policy all consume
it. Build this first; lose nothing even if the ML never ships.

## Why the parser is step one

```
Power.log ──► tail ──► parse ──► BG state ──► record (state, action, outcome) ──► [overlay / sim / ML]
```

The parser is ~70% of the total effort and the prerequisite for every other
feature. Get reliable state reconstruction and the rest is incremental.

## Data sourcing (design intent)

Two data sources, two different roles — they are **not** interchangeable:

| Source | Role | Notes |
|---|---|---|
| Population stats (HDT/Firestone-style) | Priors / features: "what's objectively strong" | Aggregates only — no public raw-trajectory dump exists. Used as model *features*, not training examples. |
| Your own games (this logger) | Personalization: "what works for *you*" | The only raw `(state, action, outcome)` trajectories you'll ever have. |

Weighting between them is **adaptive**: population-heavy early (larger, less
noisy sample), shifting toward personal as your dataset grows and personalized
recommendations measurably improve placement. See `WEIGHTING` in
`hsbg_coach/config.py`.

## Status

| Layer | State |
|---|---|
| Log path detection (Mac + Windows) | implemented, **paths need real-machine verification** |
| `log.config` writer (enables Power/Zone/LoadingScreen loggers) | implemented |
| Log tailer (handles truncation/rotation) | implemented |
| Raw line parser (CREATE_GAME / FULL_ENTITY / SHOW_ENTITY / TAG_CHANGE / scene loads) | implemented + unit-tested |
| Entity/game state model | implemented |
| **BG semantic layer** (phase detection, board/shop snapshots) | scaffolded — **needs calibration against a real Power.log** |
| `(state, action, outcome)` recorder (JSONL) | implemented |

The raw parser is tested against synthetic log lines. The **BG semantic layer**
(`hsbg_coach/bg.py`) is the part that must be calibrated against a captured
real-game log — the exact tag names for tavern tier, health, and combat
transitions need confirmation from your machine before recommendations are
trustworthy.

## Quick start

```bash
# 1. Find your Hearthstone log locations
python -m hsbg_coach detect

# 2. Enable the loggers Hearthstone needs (then RESTART Hearthstone)
python -m hsbg_coach setup

# 3. Watch a live game: prints board on combat + records trajectories
python -m hsbg_coach watch

# Offline: parse a previously captured log (no Hearthstone needed — good for dev)
python -m hsbg_coach parse-file path/to/Power.log
```

## Next milestones (not in this repo yet)

3. Combat simulator integration → combat odds + positioning/buy advice (no ML)
4. Encoded heuristic economy advisor (tempo/greed, level breakpoints, freeze)
5. Population-stats features (hero/comp/trinket/minion tiers)
6. Learned move policy (the real ML — only after 1–2 have produced enough data)
