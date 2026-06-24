# HSBG Coach — Spec & Roadmap

A Hearthstone **Battlegrounds** advisory tool (HDT-style stats panels) with an ML
move recommender that advises on **every** decision, always conditioned on the
**full board + game state**.

Status: **milestone 1–2 + early 3 built** (log parser, state reconstruction,
trajectory recorder, Monte Carlo combat sim, overlay shell). Calibrated against
public reference parsers; final calibration awaits a real captured `Power.log`.

---

## 1. Goals

- **Advise on every BG decision** — hero pick, trinket pick, buy, sell, play,
  positioning, roll, freeze, tier-up, hero power, tavern spells, targeting,
  end-turn. Not just shop buys.
- **Always condition on full state** — every recommendation sees the whole
  observable game: board, shop, hand, tavern tier, gold, health, turn, and
  last-known opponent boards.
- **HDT-style stat panels** — recommended heroes / comps / trinkets / minion
  tiers, sourced from population data.
- **Learn from the player's own games** — personalize on top of population
  priors.
- **No ban risk** — advisory only, log-parsing only. No automation, no memory
  reads. (HDT operates this way and is tolerated; full automation is bannable.)

## 2. Non-goals

- No auto-play / botting (ToS violation).
- No screen-capture/CV in v1 — log parsing only (CV is a possible later input).
- Not reproducing HSReplay's full population dataset — we *consume* aggregate
  stats, we don't re-aggregate millions of games.
- Combat sim does not (yet) model every deathrattle/battlecry/aura — see §6.

## 3. Architecture

```
Power.log ─► tail ─► parse ─► GameState ─► BG layer ─► Snapshot ─┬─► Recorder (JSONL trajectories)
                                                                 ├─► Combat sim (win/tie/loss)
                                                                 ├─► Heuristics + population stats
                                                                 └─► Overlay UI
```

Modules (all shipped except where noted):

| Module | Role |
|---|---|
| `config.py` | log path detection (Mac/Win); adaptive weighting knobs |
| `logfix.py` | writes `log.config` to enable Power/Zone/LoadingScreen loggers |
| `tail.py` | `tail -f` surviving truncation/rotation |
| `parser.py` | raw log-line → typed events (unit-tested, provider-agnostic) |
| `state.py` | entity/tag game-state accumulator |
| `bg.py` | BG semantics: phases (turn parity), local player, full Snapshot, action space |
| `recorder.py` | append-only `(state, action, outcome)` JSONL with outcome backfill |
| `sim.py` | Monte Carlo combat sim → win/tie/loss + expected damage |
| `overlay.py` | transparent always-on-top panel (pure formatter + Tk shell) |
| `cli.py` | `detect` / `setup` / `watch` / `parse-file` / `overlay` |

## 4. Data model

**Snapshot** (the model's input) — full observable state at a decision point:
`game_counter, turn, phase, tavern_tier, gold, hero_health, board[], shop[],
hand[], opponents_seen[], notes[]`. Each minion: `entity_id, card_id, name,
attack, health, position, tags{}`.

**Decision** (one training row) — `state` (Snapshot), `action_type`,
`action_detail`, `placement` (1–8, backfilled at game end), `wall_clock`.
Stored as JSONL, one game per file, append-only.

**Action space** (`bg.ActionType`) — `hero_pick, trinket_pick, buy, sell, play,
position, roll, freeze, unfreeze, tier_up, hero_power, tavern_spell, target,
end_turn`. The recorder labels each transition so the policy can recommend
across the whole game.

## 5. ML design

### Two data sources, two roles (not interchangeable)

| Source | Role | Mechanics |
|---|---|---|
| Population stats (HDT/Firestone-style) | priors / features: "what's objectively strong" | Aggregates only — **no public raw-trajectory dump exists**. Enter the model as input *features*, not training examples. |
| Player's own games (the recorder) | personalization: "what works for *you*" | The only raw `(state, action, outcome)` trajectories obtainable. |

### Adaptive weighting (`config.WEIGHTING`)

Population priors dominate early (larger, less noisy sample). Personal weight
grows as (a) the personal dataset grows and (b) personalized recommendations
*measurably* improve placement. Starting default 0.80 population, floor 0.40,
personal maxing around 1500 games. **A dial, not a constant.**

### Why "train on my games" alone fails (and the fix)

- **Behavioral cloning** (predict the move I made) copies the player, mistakes
  included — ceiling = player's own skill.
- Learning what's *good* needs an outcome signal → **offline RL** weighting
  decisions by final placement. Hard walls: credit assignment (~50–100 decisions
  per game), data volume (need 10⁴–10⁶ games), meta decay (patches shift values).
- **Mitigation**: population priors warm-start the policy so it learns *deltas*
  from "the community already knows comp X is strong," not from zero. Plus
  simulator-grounded labels (§6) give dense, causal signal for combat-adjacent
  decisions without waiting for sparse placement outcomes.

### Recommended layering (value-to-effort order)

1. **Simulator lookahead** — buy/position decisions scored by next-fight win%.
   Deterministic, no training.
2. **Encoded heuristics** — tempo/greed curves, level breakpoints, freeze logic.
3. **Population-stats features** — hero/comp/trinket/minion tiers.
4. **Learned policy** — offline RL/IL on accumulated trajectories. Last, hardest.

### Data sources (decision pending)

| Need | Source | Access |
|---|---|---|
| Card mechanics, tribes, tavern tiers | [HearthstoneJSON](https://hearthstonejson.com/), [BG JSON](https://bgknowhow.com/bgjson/) | **Free, open** — powers `effects.py` coverage + tribe-aware advice |
| Full combat-effect coverage | [Firestone sim](https://github.com/Zero-to-Heroes/firestone) (open source) | Port/bridge — biggest accuracy win |
| Hero / comp / minion win rates | HSReplay.net (the HDT data) · Firestone · community tier lists | HSReplay detailed data ~needs subscription/no clean public API; Firestone open-source; tier-list scraping has ToS caveats |

`HeroContext` (in `economy.py`) is the hook these feed. Card data is the easy
free win; hero/comp win rates need a source decision.

## 6. Combat simulator scope

Models the core auto-battle loop + the four dominant keywords: **Divine Shield,
Taunt, Poisonous, Reborn**. Returns win/tie/loss probabilities and expected
damage over N Monte Carlo runs (deterministic given a seed).

**Not yet modeled**: deathrattles, battlecries, auras, minion-specific text,
true per-minion tier for damage calc (approximated as 1). These are the long
tail that a full **Bob's Buddy** port (HDT's simulator) handles — milestone 3b.
Odds are a strong approximation, most accurate on stat-stick boards, least
accurate on deathrattle/scam comps.

## 7. Calibration plan

Calibrated against `twanvl/hearthstone-battlegrounds-simulator` + HearthSim tag
conventions (2026-06-24):

- ✅ Phase via **turn parity** (odd=recruit, even=combat) + `STEP=MAIN_READY`.
- ✅ Tavern tier from `TECH_LEVEL` / `PLAYER_TECH_LEVEL` (both checked).
- ✅ Hero health = `HEALTH − DAMAGE` on `CARDTYPE=HERO`.
- ✅ Exclude `BACON_DUMMY_PLAYER`.
- ⚠️ `# CALIBRATE` (needs a real captured log): shop-minion zone/controller,
  `RESOURCES` gold tag, `PLAYER_LEADERBOARD_PLACE` placement, local-player
  detection edge cases, exact log file paths on Mac/Windows.

**Action required from the player**: run `setup`, play one BG game, share the
resulting `Power.log`. That confirms the ⚠️ items against ground truth.

## 8. Milestones

| # | Milestone | State |
|---|---|---|
| 1 | Log parser + state reconstruction | ✅ built, tested |
| 2 | `(state, action, outcome)` recorder | ✅ built, tested |
| 3a | Monte Carlo combat sim (core + 4 keywords) | ✅ built, tested |
| 3b | Effect engine: Windfury, Cleave, deathrattle summons, start-of-combat | ✅ engine built + tested; **card list partial** (`effects.py`) |
| 3b+ | Bridge to Firestone's open-source sim (full card coverage) | ✅ bridge built + tested; **needs local `npm install` to activate** (`bridge/`, `firestone_bridge.py`) |
| 3c | Per-action labeling from log blocks | ⬜ needs real log |
| 3d | Sim-based positioning optimizer | ✅ built, tested (`position.py`) |
| 4 | Heuristic economy advisor (buy/level/roll/freeze/sell) | ✅ built, tested (`economy.py`) |
| 4b | Recommendation facade (economy + positioning + odds) | ✅ built, tested (`recommend.py`) |
| 4c | Hero/comp-aware advice plumbing (`HeroContext`) | ✅ built, tested; **needs stats data to populate** |
| 5 | Population-stats features (hero/comp/trinket/minion) | ⬜ pick a source (see §5) |
| 6 | Learned move policy (offline RL/IL) | ⬜ after dataset accrues |
| 7 | Live overlay wired to watch loop (threaded) | ⬜ shell built |

The advice engine (3d, 4, 4b) is **fully log-independent** — it runs off the
Snapshot/board contracts and needs no real `Power.log`. Only *reading your live
game* is gated on calibration; the recommendations themselves are buildable and
testable now. Once calibration lands, the live `watch` loop feeds real Snapshots
into `recommend()` and the overlay.

## Cross-refs
- `README.md` — quick start
- `hsbg_coach/bg.py` — `# CALIBRATE` markers
- `hsbg_coach/sim.py` — combat sim scope notes
