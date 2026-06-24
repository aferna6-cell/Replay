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
minions, leveling bias) that makes `recommend()` hero/comp-specific.

**It runs on real data out of the box** — a Firestone snapshot (114 heroes / 29
comps, ~983k games) is committed. Pull the latest anytime (free, no account):

```bash
python -m hsbg_coach refresh-stats                 # latest, all-MMR, last patch
python -m hsbg_coach refresh-stats --mmr 1 --period past-seven   # top 1%, 7 days
```

It pulls **hero, comp, card, and trinket** stats, defaulting to **top-10% MMR
over the past week** (`refresh-stats --mmr 10 --period past-seven`). Comp **core
cards** come from joining card-stats to tribes; **trinket** rankings come from
the trinket endpoint.

Source: Firestone's public CDN (`static.zerotoheroes.com/api/bgs`), hero/card
names via HearthstoneJSON. See `decisions/2026-06-24-firestone-bridge.md`.

### Card knowledge + synergy

The model also *understands* each minion, not just its win rate:

- `cards.py` (+ committed `data/cards/bg_cards.json`, refresh with
  `python -m hsbg_coach refresh-cards`) — every BG minion's **tier**, **tribes**,
  **keywords/effects**, and rules **text**.
- `synergy.py` — derives synergy tags (tribe buffs, keyword payoffs, Battlecry/
  Deathrattle doublers, hero-power/trinket tribe care) and ranks a shop against
  *your* board. Pass `recommend(snap, kb=load_kb())` to get synergy-aware buys
  on top of the population stats. (Heuristic from card text; learned synergy is
  a later ML step.)
- `final_boards.py` — Firestone embeds **real example winning boards**; we
  extract per-comp **core cards by board frequency** (so comp core cards are
  data-driven, not tribe-approximated) plus example boards (`stats.example_boards`)
  with positions/keywords — a target board and ML reference data.
- `pace.py` (+ `firestone_pace.json`, `python -m hsbg_coach pace`) — the
  **early-game process**, derived from real top-10% data: average tavern **tier
  by turn** and board **stats by turn**. `recommend(snap, pace=load_pace())`
  nudges leveling when you're behind the real curve. (Data shows top players
  level ~0.2-0.25 tiers ahead on turns 5-7 — more aggressive than guides say.)

### Full-accuracy combat sim (optional)

`bridge/` wraps Firestone's open-source simulator. `cd bridge && npm install`
activates it; `recommend()` then uses it automatically, else the built-in sim.

## Deep learning (optional `ml/` track)

The core package is stdlib-only. The `ml/` track adds the first real neural
model — a **combat-evaluation net** that learns to approximate the combat sim
from *unlimited sim-generated data* (no human logs needed). It's a fast,
differentiable win/tie/loss estimator for the future RL agent's lookahead.

```bash
pip install -r requirements-ml.txt
python -m ml.train --train 8000 --epochs 40 --save combat_net.pt
# val win%-MAE ~0.05, outcome-acc ~0.94 vs the simulator
```

**card2vec** learns a vector per card from the 20k winning boards (cards that win
together land near each other — it rediscovers tribes unsupervised). It's wired
into the synergy layer, and you can explore it (no torch needed to *use* it):

```bash
python -m ml.train_card2vec --epochs 5     # train (needs torch); commits card2vec.json
python -m hsbg_coach similar --card "Brann Bronzebeard"   # query (stdlib only)
```

**The board-evaluation net** is the brain that scores a whole board → expected
finish (1st–8th). It learns card+board+hero relationships from the meta, and
keeps learning from *your* games:

```bash
# train on the meta (20k labeled final boards)
python -m ml.train_eval_net --epochs 40        # val MAE ~0.26 placements, r ~0.66 on unseen comps
# fold in your own recorded games as you play (continual learning)
python -m ml.train_eval_net --trajectories data/
```

Every game you `watch` is recorded with its final placement, so retraining with
`--trajectories data/` sharpens the model on the live meta *and your playstyle*.
The move-recommender will query this net: to compare buy/sell/roll/reposition, it
scores the resulting boards and prefers the lower expected finish.

**The move recommender** (`hsbg advise`) is the payoff: it enumerates *every*
legal action — buy each shop minion, sell each board minion, roll, tier up,
reposition, freeze, end — and ranks them by how much each improves your expected
finish. Buy/sell are scored by one-ply lookahead through the eval net (heuristic
fallback if it isn't trained); roll/level/freeze by pace/gold heuristics;
reposition by the combat sim.

```bash
python -m hsbg_coach advise                 # demo board built from real card data
python -m hsbg_coach advise --snapshot game.json --tribe Murloc
```

```
Best move: Buy Monstrous Macaw — learned synergy with board (0.32)
  0.85  Buy Monstrous Macaw   (+5% equity) — learned synergy with board
  0.72  Sell Deflect-o-Bot    (+6% equity) — removing it improves the board
  0.55  Buy Holo Rover        (+1% equity) — on-tribe with board (Mech)
  ...
```

It's one-ply (ranks each immediate action, not full multi-buy turn plans), and
roll/level/freeze are heuristic — see the caveats it prints.

Design for the full self-play RL agent: `specs/self-play-rl-agent.md`.

## Roadmap

See `specs/hsbg-coach_spec.md` for the full spec, data model, and ML design.
Built: log parser, state reconstruction, recorder, combat sim, positioning
optimizer, economy advisor, recommendation facade, overlay shell. Next:
per-action labeling (needs a real log), fuller sim (deathrattles),
population-stats panels, then the learned move policy.
