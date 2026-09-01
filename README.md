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

# 3. Launch the HDT-style overlay. It can start before Hearthstone and detects
#    each new game/session automatically; use borderless/windowed game mode.
python -m hsbg_coach watch

# Optional: keep the live recommendation panel inside your terminal instead
python -m hsbg_coach watch --terminal

# Offline: parse a previously captured log (no Hearthstone needed — good for dev)
python -m hsbg_coach parse-file path/to/Power.log

# Show the overlay with sample data only (needs a graphical display)
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

Every move is scored by **expected final placement** (whole-game value), not just
immediate board strength — so buy / sell / level / roll are compared on one axis,
with the rest of the game baked in (`game_value.py`: eval-net board value +
multi-turn trajectory + HP risk):

```
Whole-game ranking — expected final placement (now: 3.8):
  finish 3.3 (+0.50)  Buy Monstrous Macaw — learned synergy with board
  finish 3.5 (+0.34)  Buy Holo Rover — on-tribe with board (Mech)
  finish 4.8 (-1.00)  Sell Ingenious Inventor — weakens the board
```

`advise` also prints a **full-turn plan** — it greedily applies the best move,
re-scores the resulting board, and repeats, so you get the whole turn in order:

```
Full-turn plan (follow in order):
  1. Buy Monstrous Macaw (+5%)
  2. Sell Deflect-o-Bot (+5%)
  3. Roll — surplus gold and no improving buys left
  4. End turn
```

This is how you'd "play the turn by the recommender." It's greedy (locally best
each step), not globally optimal turn search, and roll/level are heuristic.

## Multi-turn plan (tempo vs greed)

`hsbg plan` looks several turns ahead — the strategic layer above single-turn
moves. It projects the next K turns (tier, board strength vs the top-10% pace
curve, HP) under candidate strategies and ranks them:

```bash
python -m hsbg_coach plan --horizon 4
```

```
Strategy lookahead (4 turns) — best first:
  1. Level next turn: value 10.0
  2. Tempo:           value  9.0
  4. Double level:    value  4.2  ⚠ DIES
Best strategy: Level next turn — THIS TURN: TEMPO
  T6: tempo  tier 3 · 91% of pace · hp 24
  T7: level  tier 4 · 63% of pace · hp 21
  ...
```

This is the "skip the buy now, tier up for a spike" reasoning: low HP ⇒ tempo to
survive, under-tiered ⇒ level to catch up, and the further ahead you plan the
more leveling pays. It's an economy model grounded in the real pace curves (not
the eval net) — an honest approximation; eval-net/RL terminal valuation is the
next refinement.

## Draft picks (hero / trinket / Discover)

Beyond board actions, it ranks "choose 1 of N" decisions. Hero and trinket use
the meta's average placement; Discover uses board-fit (the eval net + card2vec
synergy against your current board):

```bash
python -m hsbg_coach pick hero "Rafaam" "Pyramad" "Galakrond"
python -m hsbg_coach pick trinket "Ironforge Anvil" "Accord-o-Tron Portrait" --board "Holo Rover,Scrap Scraper" --tribe Mech
python -m hsbg_coach pick discover "Monstrous Macaw" "Holo Rover" --board "Holo Rover,Scrap Scraper" --tribe Mech
```

Trinkets blend meta placement *with board fit* — a trinket that buffs your tribe
gets a bonus, one for a tribe you don't run gets a penalty (from the trinket's
effect text vs your board).

```
Pick (discover) — best first:
  1. Monstrous Macaw — +5% equity — learned synergy with board (0.31)  ◀ PICK
  2. Holo Rover      — -0% equity — on-tribe with board (Mech)
```

Auto-detecting these offers from the live log (so they pop in the overlay) is the
next live step — pending real-game log calibration. For now, type the offered
names into `pick`.

## Take it into a game (live overlay)

On your gaming PC (Windows/Mac), with Hearthstone installed:

```bash
python -m hsbg_coach setup            # one-time: make Hearthstone emit the logs
# launch Hearthstone, start a Battlegrounds game
python -m hsbg_coach watch --overlay  # on-screen, always-on-top recommendations
```

A small draggable panel pins to a screen corner and updates as you play — each
recruit turn it shows the ranked moves (best first), recomputed whenever your
board, shop, or gold changes. It runs the full stack: the eval net scores buys,
card2vec drives synergy, the combat sim handles positioning.

> First real-game use is a shakedown: the live path is calibrated on captured
> logs, not yet against a running client. Expect to file rough edges.

## It learns from your games

Every game you `watch` is recorded to `data/<game>.jsonl` — each decision with the
final placement it led to. Fold those into the brain after a session:

```bash
./scripts/retrain.sh                  # retrain the eval net on the meta + your games
```

The more you play, the sharper it gets on the live meta *and your playstyle* —
your per-game placements are a sharper signal than the population averages.

## Keeping it fresh (weekly meta pull)

```bash
./scripts/weekly_update.sh            # refresh cards + stats from Firestone, retrain everything
```

Schedule it weekly so the models track the current patch:

```
cron (mac/linux):   0 9 * * 1  cd /path/to/Replay && ./scripts/weekly_update.sh
Windows Task Scheduler: weekly action -> bash scripts/weekly_update.sh
```

Design for the full self-play RL agent: `specs/self-play-rl-agent.md`.

## The Phase 0 environment + self-play RL track (built)

The RL spec's make-or-break gate — a faithful-enough recruit-phase simulator —
is built and validated (`hsbg_coach/bg_env.py`, stdlib only): an 8-player
lobby with real cards from the committed KB, a finite shared pool (real copy
counts), shop generation by tier, buy/sell/roll/freeze, real tavern-up
discounts, triples → golden + discover, combat resolved by `sim.py`, and an
abstract end-of-turn scaling layer standing in for the buff long tail so
boards track the measured Firestone curve. Validation (`python -m
hsbg_coach.bg_env --lobbies 100`): tavern tier within ~0.5 of the real curve,
board stats on-curve, ~14-turn games, eliminations matching the real
alive-by-turn table.

That unlocks the whole learning stack:

```bash
# 1. The set-transformer board brain (replaces the mean-pooled MLP):
#    one token per minion -> self-attention -> P(finish 1st..8th).
#    Trained on env self-play MID-GAME states (the states the advisor
#    actually queries — the old net only ever saw final boards).
python -m ml.train_set_net --midgame-lobbies 300 --epochs 30
python -m ml.calibrate                  # per-stage calibration check

# 2. Behavior-clone the greedy baseline (RL warm start + first league member)
python -m ml.bc --lobbies 150

# 3. PPO against a league (scripted baselines + past selves)
python -m ml.train_ppo --iters 40 --episodes 16
```

Phase 1 status: the learned policy **beats the random field decisively**
(avg placement 1.00) and is still short of the all-greedy field (6.5 after
~35 CPU-minutes; the bar is <4.5) — stat-max greedy is near-optimal in the
simplified env, and RL compute is the known cost (spec §9). The training
loop, league, and eval gates are all in place; it's a `--iters` dial now.

`get_scorer()` prefers `ml/set_net.pt` automatically, so the advisor, the
whole-game ranking, and the overlay all read the new brain once trained.

## Replay Benchmark

`python -m ml.benchmark` is the one canonical way to measure agent strength
(`ml/benchmark.py`). It exists so every future ML change — PPO tuning, network
redesigns, new features — is judged against the **same fixed standard** instead
of ad-hoc eval loops with drifting seeds and fields.

**How it works.** The tested agent sits in seat 0 of a `BGEnv` lobby against
seven copies of one field policy. Game `i` runs on seed `base_seed + i`, so the
whole run is deterministic and every agent is evaluated from the **same
deterministic initial conditions** — same seed, environment configuration,
field policy, and seat. (Trajectories legitimately diverge once agents make
different decisions, since actions feed back into the shared pool and combat;
what's fixed is the starting point and every rule around it.) Learned
checkpoints act by argmax (no sampling), so their evaluation is deterministic
too. Integrity is enforced, not assumed: a game that fails to terminate within
the decision cap raises instead of being scored 8th, and every agent action is
validated as an in-range, unmasked action index before use.

**Train/eval seed separation.** `ml/seeds.py` is the single authority.
Evaluation owns the finite reserved interval **[10,250,000 – 10,299,999]**
(50,000 seeds), and the benchmark refuses any run whose seed range leaves it.
The interval was placed after auditing every `BGEnv`-seeding training scheme —
BC/DAgger (`base + i`, `base + 10000·round + i`), PPO (`base·1000003 + k`),
the midgame dataset (`base·100003 + i`), the legacy 9000+ eval loop: no
multiple of either multiplicative stride falls inside the interval, so under
current defaults and any reasonable run size (a PPO run needs ≥249,970
episodes, a midgame run ≥49,695 lobbies to reach it) training cannot touch
benchmark seeds. That is separation under stated bounds, **not a mathematical
guarantee** — the exact collision conditions are documented in `ml/seeds.py`,
training entry points warn loudly if a planned run would overlap, and future
training code must derive its seeds through `ml/seeds.py` helpers and keep
calling `check_training_range`. Never intentionally train on reserved seeds.

**Supported agents** — `--agent random`, `--agent greedy`, and
`--agent policy --checkpoint <file>.pt` for any `PolicyNet` checkpoint (BC,
BC + DAgger, and PPO all share the same checkpoint format, so they're all
evaluated through `--checkpoint`; use `--name` to label them).
**Supported fields** — `--field greedy` (default, the field that matters) and
`--field random`.

```bash
# Full default suite (random, greedy, + BC/PPO checkpoints when present):
python -m ml.benchmark --games 1000

# Individual agents vs the all-greedy field:
python -m ml.benchmark --agent random --games 1000 --field greedy
python -m ml.benchmark --agent greedy --games 1000 --field greedy
python -m ml.benchmark --agent policy --checkpoint ml/policy_bc.pt \
    --name "BC + DAgger" --games 1000 --field greedy
python -m ml.benchmark --agent policy --checkpoint ml/policy_ppo.pt \
    --name PPO --games 1000 --field greedy --json-out results/ppo_vs_greedy.json

# Compare saved result JSONs (sorted by avg placement; warns on
# mismatched version/field/games/seed):
python -m ml.benchmark compare results/*.json
```

**Metrics**: average / median placement, placement std dev, Top-4 rate, win
rate, the full 1st–8th placement distribution, a seeded percentile-bootstrap
95% CI on average placement, and agent decision latency (mean/p50/p95 of the
decision function only, not env stepping). `--json-out` writes a
machine-readable record including the benchmark version, seeds, environment
config, and **model identity**: the checkpoint's basename plus the SHA-256 of
its exact bytes (a filename like `policy_ppo.pt` can't identify a model), and
the repo Git commit (suffixed `-dirty` for uncommitted changes; `null` outside
a checkout). Single-agent runs write one result object; the default suite
writes a versioned wrapper `{"benchmark_version", "timestamp", "results":
[...]}` — `compare` consumes both shapes, warns when run conditions
(version/field/games/seed/environment) differ, and prints each agent's
checkpoint fingerprint (differing model hashes are the point of a comparison,
never an error).

**The 4.5 threshold**: an 8-player lobby averages placement 4.5 by
construction, so average placement **below 4.5 means the tested agent
outperformed the field's average** — nothing more. It is not a claim of
optimal play.

### Measured baseline (Replay Baseline Experiment v1, 2026-08-28)

**Replay Benchmark v1 · 1000 games per agent · same reserved evaluation seeds
(10,250,000–10,250,999) · vs 7 greedy opponents.** Every number comes from the
committed result JSON in `results/benchmark_v1/`; full setup, paired
statistics, and failure analysis in
[`experiments/replay_baseline_v1.md`](experiments/replay_baseline_v1.md).

| Agent | Avg Place | 95% CI | Top-4 | Win | p95 latency |
|---|---|---|---|---|---|
| Greedy | **4.445** | [4.308, 4.600] | 52.6% | 11.8% | 0.008 ms |
| BC | 6.497 | [6.375, 6.618] | 17.2% | 3.1% | 1.15 ms |
| BC + DAgger | 6.527 | [6.409, 6.652] | 15.8% | 2.8% | 1.10 ms |
| PPO | 6.798 | [6.686, 6.910] | 12.5% | 1.9% | 1.10 ms |
| Random | 7.989 | [7.982, 7.995] | 0.0% | 0.0% | 0.004 ms |

Honest reading: no learned agent beats the field yet. Paired comparisons on
the same seeds show BC vs BC + DAgger has no clear difference, and the
shipped PPO recipe (640 episodes) currently makes its warm start *worse* by
0.27 places (CI [0.15, 0.39]) on this benchmark — while still crushing a
random field (training diagnostic 1.00). These are the "before" numbers that
every future change must beat.

**Comparability**: results are only comparable when the benchmark version,
environment, field, game count, and base seed all match. If `bg_env` rules or
pace assumptions change materially, that's a new benchmark version
(`BENCHMARK_VERSION` in `ml/benchmark.py`) and old numbers belong to the old
one. v1 is single-process by design (reproducibility over speed); parallel
execution is a documented future optimization.

## Research status (2026-09-01)

**PPO tuning on Simulator v1 is closed.** Experiments 4b–6 established that KL
anchoring stabilizes PPO, β=0.03 is the best fixed recipe, and scheduled
relaxation after iter 160 does not beat the BC warm start reliably. See
[`experiments/INDEX.md`](experiments/INDEX.md).

**Active work: Simulator Fidelity Phase 2C** — composition assembly diagnostic
(sim winner coverage ~0.009 vs real ~0.766). Phase 2B merged **Simulator v1.1**.

```bash
python -m ml.fidelity_phase_2c --lobbies 200      # composition funnel diagnostic
python -m ml.fidelity_phase_2b --lobbies 200      # paired scaling study (v1 vs v1.1)
python -m ml.fidelity_benchmark --lobbies 200     # aggregate fidelity curves
```

Artifacts: [`experiments/sim_fidelity_phase_2c.md`](experiments/sim_fidelity_phase_2c.md),
[`results/sim_fidelity_phase_2c/`](results/sim_fidelity_phase_2c/),
[`results/sim_fidelity_v1_1/`](results/sim_fidelity_v1_1/).

Replay Benchmark v1 remains the historical agent-strength standard. **Replay
Benchmark v2** will be defined only after the simulator is materially improved;
do not reuse TEST casually across simulator versions.

## Turn planning is beam search now

`hsbg advise` plans the full turn by beam search over action *sequences*
(`hsbg_coach/turn_search.py`) instead of greedy best-move-repeat: it finds
sell→buy room-making lines, triple completions, and level-then-buy plans the
greedy planner structurally cannot, scores every line by whole-game expected
placement, and reports the expected finish of the chosen line. Roll/freeze
stay honest heuristics (future shops aren't simulated).

## Roadmap

See `specs/hsbg-coach_spec.md` for the full spec, data model, and ML design.
Built: log parser, state reconstruction, recorder, combat sim, positioning
optimizer, economy advisor, recommendation facade, overlay shell. Next:
per-action labeling (needs a real log), fuller sim (deathrattles),
population-stats panels, then the learned move policy.
