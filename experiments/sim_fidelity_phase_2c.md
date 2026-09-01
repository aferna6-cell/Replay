# Simulator Fidelity Phase 2C — composition assembly diagnostic

Date: 2026-09-01 · Status: **measurement only** ·
Artifacts: [`results/sim_fidelity_phase_2c/`](../results/sim_fidelity_phase_2c/)

## Question

> Why do Simulator v1.1 winners achieve ~0.008 final composition coverage when
> real Firestone winners achieve ~0.766?

No sim mechanics, policies, card effects, shop, scaling, or combat changes.

## Trace schema (per recruit action)

Each event records: lobby, seat, turn, placement (post-game), tavern tier,
shop offered, buy/play/sell/roll/level/freeze/end, triple/discover, board
before/after recruit, and `infer_target()` summary with core cards.

## Funnel (per Firestone archetype)

```text
core card in shop (all seats — availability)
        ↓
core affordable when offered
        ↓
core purchased (winner seat)
        ↓
core played
        ↓
core retained ≥ 2 turns
        ↓
2+ core assembled
        ↓
4+ core assembled
        ↓
final winner coverage
```

## Failure classification (per lobby × archetype)

| Code | Meaning |
|---|---|
| A_IMPOSSIBLE | Required core rarely offered |
| B_AVAILABLE_NOT_BOUGHT | Offered + affordable but greedy rejected |
| C_BOUGHT_NOT_RETAINED | Bought then sold/replaced |
| D_ASSEMBLED_NO_PAYOFF | Core on board but coverage still ~0 |
| E_SUCCESSFULLY_ASSEMBLED | Meaningful comp formed |

## Commands

```bash
pytest tests/test_composition_diagnostic.py
python -m ml.fidelity_phase_2c --lobbies 200 --seed 0
```

## Results (200 greedy lobbies, seed 0)

| Metric | Value |
|---|---|
| Sim final winner coverage | **0.009** |
| Real final winner coverage | **0.766** |
| Recruit events traced | 80,563 |

### Failure classification (lobby × archetype, n=3,800)

| Class | Count | Meaning |
|---|---|---|
| B_AVAILABLE_NOT_BOUGHT | 2,516 | Core offered + affordable; greedy bought stats instead |
| A_IMPOSSIBLE | 1,185 | Core rarely offered enough to assemble |
| C_BOUGHT_NOT_RETAINED | 99 | Bought core then sold/replaced |

### Recommended Phase 2D (single intervention)

**Build-aware recruit policy / evaluator** — available-but-not-bought dominates.
Greedy frequently prefers slightly larger off-comp bodies when core pieces are in shop.

See [`results/sim_fidelity_phase_2c/phase_2c_report.json`](../results/sim_fidelity_phase_2c/phase_2c_report.json)
for per-archetype funnels, opportunity-loss patterns, and offer rates by tier.

## Frozen for Phase 2C

Residual scaling (v1.1), combat, greedy policy, card effects, shop distribution,
triples/discovers, heroes, trinkets, anomalies.
