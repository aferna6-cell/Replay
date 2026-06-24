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
| **BG semantic layer** (phase detection, board/shop snapshots) | calibrated vs reference parsers; **shop zone + placement still need a real log** |
| `(state, action, outcome)` recorder (JSONL) | implemented |
| Monte Carlo combat sim (core + Divine Shield/Taunt/Poisonous/Reborn) | implemented + unit-tested (`hsbg_coach/sim.py`) |
| Overlay UI shell (transparent always-on-top panel) | implemented; pure formatter unit-tested (`hsbg_coach/overlay.py`) |

The raw parser is tested against synthetic log lines. The **BG semantic layer**
(`hsbg_coach/bg.py`) is calibrated against public reference parsers
(`twanvl/hearthstone-battlegrounds-simulator` + HearthSim tag conventions):
phase detection (turn parity), tavern tier, and hero health are pinned down.
The remaining `# CALIBRATE` items — shop-minion zone, gold tag, final placement,
and exact log paths — need confirmation from a real captured `Power.log` before
recommendations are fully trustworthy. See `specs/hsbg-coach_spec.md` §7.

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

# Show the overlay with sample data (needs a graphical display)
python -m hsbg_coach overlay
```

## Combat odds (no ML)

```python
from hsbg_coach.sim import Combatant, simulate

mine  = [Combatant(3, 3), Combatant(2, 4, taunt=True)]
enemy = [Combatant(4, 4, divine_shield=True)]
print(simulate(mine, enemy, runs=1000, seed=0).summary())
# -> win 71% / tie 6% / loss 23% (avg dmg dealt 2.1, taken 1.4)
```

`simulate()` also accepts the `MinionView`s produced by `bg.py` directly.

## Recommendations (no ML, no log needed)

The advice engine runs off the Snapshot/board contracts — fully testable without
a real game:

```python
from hsbg_coach.recommend import recommend, combat_odds
from hsbg_coach.sim import Combatant as C

snap = {"turn": 7, "tavern_tier": 2, "gold": 8, "hero_health": 29,
        "board": [{"name": "Felfin", "attack": 4, "health": 4}],
        "shop": [{"name": "Primalfin", "attack": 2, "health": 3}], "notes": []}
enemy = [[C(3, 3), C(4, 4), C(2, 5, taunt=True)]]   # a likely opponent (field)

print(combat_odds(snap, enemy))                     # win/tie/loss
for r in recommend(snap, enemy_boards=enemy):       # ranked across layers
    print(f"[{r.priority:.2f}] {r.source}: {r.rationale}")
```

- `economy.py` — buy / level / roll / freeze / sell heuristics over a Snapshot.
- `position.py` — sim-based best board ordering (averaged over a field of
  likely enemies during recruit; exact vs a known board at combat).
- `recommend.py` — merges both into one ranked move list + combat odds.

### Hero/comp-specific advice (population stats)

```bash
python -m hsbg_coach stats --hero "Old Murk-Eye"
# Target comp: Murlocs (tribe Murloc, avg place 3.7, tier S) ...
python -m hsbg_coach stats --hero "Old Murk-Eye" --tribes "Beast,Mech,Dragon"
# pivots to the best comp actually available this lobby
```

`stats.py` loads hero/comp stats and builds a `HeroContext` (target tribe, core
minions, leveling bias) that makes `recommend()` hero/comp-specific. It ships
with sample data — point `--hero-source` / `--comp-source` at a real Firestone or
HSReplay export to use live numbers.

### Full-accuracy combat sim (optional)

`bridge/` wraps Firestone's open-source simulator. `cd bridge && npm install`
activates it; `recommend()` then uses it automatically, else the built-in sim.

## Roadmap

See `specs/hsbg-coach_spec.md` for the full spec, data model, and ML design.
Built: log parser, state reconstruction, recorder, combat sim, positioning
optimizer, economy advisor, recommendation facade, overlay shell. Next:
per-action labeling (needs a real log), fuller sim (deathrattles),
population-stats panels, then the learned move policy.
