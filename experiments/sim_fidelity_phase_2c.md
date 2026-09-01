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

## Output

[`results/sim_fidelity_phase_2c/phase_2c_report.json`](../results/sim_fidelity_phase_2c/phase_2c_report.json)
ends with exactly **one** recommended Phase 2D intervention.

## Frozen for Phase 2C

Residual scaling (v1.1), combat, greedy policy, card effects, shop distribution,
triples/discovers, heroes, trinkets, anomalies.
