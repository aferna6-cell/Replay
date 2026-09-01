# Simulator Fidelity Benchmark v1 — Phase 2A baseline

Date: 2026-09-01 · Status: **measurement only** ·
Artifacts: [`results/sim_fidelity_v1/`](../results/sim_fidelity_v1/)

## Question

> Where, quantitatively, does Replay's Simulator v1 diverge from real
> Battlegrounds?

This is the simulator analogue of Replay Benchmark v1 for agents: an immutable
**before** measurement. No simulation mechanics are changed in Phase 2A.

## What this is / is not

| | Fidelity Benchmark v1 | Replay Benchmark v1 |
|---|---|---|
| Measures | Sim vs real curves | Agent placement strength |
| Changes sim? | No | No |
| Uses TEST seeds? | No (rollout seeds 0…) | Yes (10.25M–10.299M) |
| When to re-run | Before/after sim fixes | After Benchmark v2 defined |

## Frozen Simulator v1 contract

Recorded in `results/sim_fidelity_v1/contract.json`:

- Git commit at baseline run
- `ml/experiment_contract.env_config()` hash
- SHA256 fingerprints: `firestone_pace.json`, `firestone_final_boards.json`,
  `bg_cards.json`, `card2vec.json`, other Firestone stats
- Evaluation: greedy policy, N lobbies, base seed 0

## Metrics (v1)

| Dimension | Sim source | Real / reference source |
|---|---|---|
| Board stats by turn | `BGEnv.play_scripted` | `firestone_pace.scaling` |
| Tavern tier by turn | same | `firestone_pace.tavern_tier` |
| Board size by turn | same | (sim only — no public curve) |
| Gold at end-recruit | same | `econ_env.gold_at` prior |
| Alive players by turn | same | `econ_env.alive_at` prior |
| Game length | same | (sim only) |
| Composition coverage | `build_path.infer_target` turns 8–14 | Real winning example boards |
| Combat accuracy | **Not measured** | Prior ~97% spot-check note |
| Shop/recruit mix | **Deferred** | Requires trajectory logs |

## Commands

```bash
pytest tests/test_fidelity_benchmark.py
python -m ml.fidelity_benchmark --lobbies 200
python -m ml.fidelity_benchmark --lobbies 500   # fuller baseline
```

## Phase 2B preview (not started)

One intervention only: **fix late-game scaling runaway**. Re-run this benchmark
as `Simulator v1` vs `Simulator v1 + scaling correction`. Do not rewrite combat.

### Baseline snapshot (200 greedy lobbies, seed 0)

| Turn | Real stats | Sim stats | Ratio |
|---|---|---|---|
| 10 | 1,171 | 1,435 | 1.23× |
| 12 | 3,383 | 5,053 | 1.49× |
| 14 | 6,047 | 17,314 | **2.86×** |
| 16 | 6,047 | 25,763 | **4.26×** |

Composition coverage (turns 8–14): sim **0.014** vs real **0.777**.

See [`results/sim_fidelity_v1/baseline.json`](../results/sim_fidelity_v1/baseline.json).

## Phase 2C preview (not started)

Diagnose composition realism (sim ~0.02 vs real ~0.84 coverage) before adding
heroes, trinkets, anomalies, or broad card-effect ports from stale branches.
