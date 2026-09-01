# Simulator Fidelity Benchmark v1 — Phase 2A baseline

Date: 2026-09-01 · Status: **measurement only** ·
Artifacts: [`results/sim_fidelity_v1/`](../results/sim_fidelity_v1/)

## Question

> Where, quantitatively, does Replay's Simulator v1 diverge from the frozen
> **Firestone 2026-09-01 reference distribution** (top-10% MMR, past-seven)?

This is the simulator analogue of Replay Benchmark v1 for agents: an immutable
**before** measurement. No simulation mechanics are changed in Phase 2A.

Reference data was refreshed via `python -m hsbg_coach refresh-stats --mmr 10
--period past-seven` on 2026-09-01 before freezing the baseline.

## What this is / is not

| | Fidelity Benchmark v1 | Replay Benchmark v1 |
|---|---|---|
| Measures | Sim vs frozen Firestone curves | Agent placement strength |
| Changes sim? | No | No |
| Uses TEST seeds? | No (rollout seeds 0…) | Yes (10.25M–10.299M) |
| When to re-run | Before/after sim fixes | After Benchmark v2 defined |

## Frozen Simulator v1 contract

Recorded in `results/sim_fidelity_v1/contract.json`:

- Git commit at baseline run
- `ml/experiment_contract.env_config()` hash
- SHA256 fingerprints: `firestone_pace.json`, `firestone_final_boards.json`,
  `bg_cards.json`, `card2vec.json`, other Firestone stats
- Reference label: Firestone fetch date + MMR/period metadata
- Evaluation: greedy policy, N lobbies, base seed 0

## Metrics (v1)

| Dimension | Sim source | Real / reference source |
|---|---|---|
| Board stats by turn | `BGEnv.play_scripted` | `firestone_pace.scaling` (turns 1–14 only) |
| Tavern tier by turn | same | `firestone_pace.tavern_tier` (turns 1–14 only) |
| Board size by turn | same | (sim only — no public curve) |
| Gold at end-recruit | same | `econ_env.gold_at` prior |
| Alive players by turn | same | `econ_env.alive_at` prior |
| Game length | max turn per lobby, averaged | (sim only) |
| Midgame composition | `infer_target` turns 8–14 | Real final winning examples (**diagnostic only**) |
| Final winner composition | Sim 1st-place final board | Real final winning examples (**calibrated baseline**) |
| Combat accuracy | **Not measured** | Prior ~97% spot-check note |
| Shop/recruit mix | **Deferred** | Requires trajectory logs |

### Reference lookup rules

Firestone pace curves are compared with **strict exact-turn lookup**. Turns without
measured reference data report `reference_status: unmeasured` and `N/A` ratios —
no silent forward-fill from turn 14.

## Commands

```bash
pytest tests/test_fidelity_benchmark.py
python -m ml.fidelity_benchmark --lobbies 200
python -m ml.fidelity_benchmark --lobbies 500   # fuller baseline
python -m hsbg_coach refresh-stats --mmr 10 --period past-seven
```

## Phase 2B preview (not started)

One intervention only: **fix late-game scaling runaway** on measured reference
turns. Re-run this benchmark as `Simulator v1` vs `Simulator v1 + scaling
correction`. Do not rewrite combat.

Pre-specified success (tolerances derived from bootstrap variance across lobby
samples):

- Primary: turn-14 stats ratio ≤ derived threshold
- Secondary: turn-12 ratio ≤ derived threshold
- Regression guards: turns 8–10, tavern tier, alive curve, game length, throughput

### Baseline snapshot (200 greedy lobbies, seed 0)

See [`results/sim_fidelity_v1/baseline.json`](../results/sim_fidelity_v1/baseline.json)
for current numbers after reference refresh and validity fixes.

## Phase 2C preview (not started)

Diagnose **stage-matched** final-winner composition realism before adding heroes,
trinkets, anomalies, or broad card-effect ports from stale branches.
