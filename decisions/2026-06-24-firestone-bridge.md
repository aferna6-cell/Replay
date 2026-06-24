# ADR: Bridge to Firestone for combat sim accuracy and hero/comp stats

Date: 2026-06-24
Status: Accepted

## Context

Two needs surfaced:

1. **Combat-sim accuracy.** Our pure-Python sim (`sim.py` + `effects.py`) models
   the core loop + keywords + a representative set of deathrattles/start-of-combat
   effects. Full coverage of ~500 BG cards (and chasing every patch) by hand is
   not viable.
2. **Hero/comp-specific advice.** `HeroContext` is wired but needs real win-rate
   data to populate.

Research (2026-06-24) found the Zero-to-Heroes / Firestone ecosystem covers both,
open-source:

- **`@firestone-hs/simulate-bgs-battle`** — the actual Firestone BG combat
  simulator, published to npm, actively maintained (1.1.6xx, 470+ releases).
  Exports `simulateBattle(battleInput: BgsBattleInfo, cards, cardsData)` →
  `Generator<SimulationResult>` with `wonPercent` / `tiedPercent` / `lostPercent`
  / `averageDamageWon` / `averageDamageLost`.
- **`api-simulate-battlegrounds-battle`** — a runnable API wrapper around it.
- **`@firestone-hs/bgs-global-stats`** + **`firestone-data`** (S3) — hero/comp/
  minion meta stats.

HSReplay (the HDT data) is higher-volume but its detailed BG data is effectively
subscription-gated with no clean public API. Firestone is open and free.

## Decision

1. **Sim accuracy: bridge to `@firestone-hs/simulate-bgs-battle` via a Node
   sidecar.** Python builds a `BgsBattleInfo` JSON, shells out to a small Node
   script that runs `simulateBattle`, and parses the `SimulationResult` back.
2. **Keep the pure-Python sim as the default fallback.** The bridge is opt-in:
   if Node or the sidecar's `node_modules` is absent, `firestone_bridge.simulate`
   transparently falls back to `sim.simulate`. Nothing breaks without Node.
3. **Stats: use Firestone**, not HSReplay. Load hero/comp stats from Firestone's
   open data into `HeroContext`. (Loader is a follow-up; this ADR locks the
   source choice.)

## Why a sidecar (not a port, not a hosted call)

- **Port to Python**: rejected — the sim is large TS that changes every patch;
  a hand-port would rot immediately. The whole point of bridging is to inherit
  Firestone's maintenance.
- **Call their hosted API**: rejected as the primary path — adds a network
  dependency and rate limits to a per-turn hot path, and we'd rather not lean on
  someone else's infra for every combat. (Still fine as an optional backend.)
- **Local Node sidecar**: chosen — runs the maintained package locally, no
  network per sim, and `npm update` pulls patch fixes.

## Consequences

- New optional dependency: Node + `npm install` in `bridge/`. Documented; not
  required for the Python package to work.
- The sidecar (`bridge/firestone_sim.js`) is a **template** — exact `BoardEntity`
  field names and the `AllCardsService`/`CardsData` init must be validated
  against the installed `dist/*.d.ts` on first local run, since they can shift
  between package versions. The Python-side conversion is isolated in
  `firestone_bridge.py` so corrections are one-file.
- Accuracy upgrade is incremental: pure sim today, Firestone-grade once the user
  runs `npm install`.

## Verification plan

- Python conversion (board → `BgsBattleInfo` dict) is unit-tested headless.
- Fallback path is unit-tested (no Node → pure sim result returned).
- End-to-end Node run validated locally by the user after `npm install` (can't
  run Node in the current headless env).

## Cross-refs
- `hsbg_coach/firestone_bridge.py`
- `bridge/firestone_sim.js`, `bridge/README.md`
- `specs/hsbg-coach_spec.md` §5–6
- `hsbg_coach/sim.py` (fallback engine)
