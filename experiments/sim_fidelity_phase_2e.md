# Simulator Fidelity Phase 2E — seeded-core conversion stress test

Date: 2026-09-01 · Status: **oracle diagnostic** ·
Artifacts: [`results/sim_fidelity_phase_2e/`](../results/sim_fidelity_phase_2e/)

## Research question

> If we **force** conversion of seeded core opportunities (oracle stress), do
> coherent compositions emerge?

This is **not** a proposed production policy. It is a causal stress test after
Phase 2D showed `path_adj/5` is insufficient (0 seeded fulfillment).

## Arms

| Arm | Policy | Buy behavior |
|---|---|---|
| Control | `greedy_policy` | raw `attack + health` |
| Treatment | `seeded_core_stress_greedy_policy` | when seeded + missing core buyable → buy best matching core; else greedy |

## Design

- Paired **200 lobbies**, **fresh seeds `1000–1199`** (not used in 2C/2D)
- Phase 2C `2c_v3` tracing + macro fidelity on both arms
- No coefficient sweep, no post-hoc tuning

## Decision tree

| Outcome | Next step |
|---|---|
| fulfillment ↑, 2+ core ↑, coverage ↑ | Recruit is causal → design realistic tempo-aware policy (DEV calibration) |
| fulfillment ↑, 2+ core ↑, coverage flat | **Card-effect fidelity** |
| fulfillment ↑, 2+ core ~0 | Retention / triples / target transitions |
| fulfillment still ~0 | Tracing or policy-path issue |

## Commands

```bash
pytest tests/test_seeded_core_stress.py
python -m ml.fidelity_phase_2e --lobbies 200 --seed 1000
```

## Frozen

Simulator v1.1, scaling, shop, combat, card effects, BC/PPO/TEST unchanged.
