# Simulator Fidelity Phase 2G — seeded-core deployment / board-slot stress test

Date: 2026-09-01 · Status: **oracle diagnostic** ·
Artifacts: [`results/sim_fidelity_phase_2g/`](../results/sim_fidelity_phase_2g/)

## Research question

> If we **guarantee a board slot** for oracle-acquired seeded cores, does
> acquisition finally become persistent 2+ core assembly?

Phase 2F showed 33/34 fulfilled cores stuck in hand on full boards. Phase 2G
adds one deployment rule to the Phase 2E buy oracle.

## Arms

| Arm | Policy | Behavior |
|---|---|---|
| Control | `seeded_core_stress_greedy_policy` | Phase 2E buy oracle only |
| Treatment | `seeded_core_deploy_stress_greedy_policy` | + sell weakest **non-core** when hand core blocked on full board |

Treatment never sells seed/core board pieces to make room.

## Design

- Paired **200 lobbies**, **fresh seeds `2000–2199`** (not inspected in 2E/2F)
- Primary mechanisms: **played rate** and **persistent 2+ core assembly** (not fulfillment)
- Phase 2F lifecycle tracing on both arms + macro fidelity

## Decision tree

| Outcome | Next step |
|---|---|
| macro regression | Reject intervention |
| played ↑, 2+ core ↑, coverage ↑ | Board-slot handling causal → realistic board-management policy |
| played ↑, 2+ core ↑, coverage flat | Retention or card effects |
| played ↑, 2+ core ~0 | Seed loss / target / identity |
| played ~0 | Deployment oracle insufficient |

## Commands

```bash
pytest tests/test_seeded_core_deploy.py
python -m ml.fidelity_phase_2g --lobbies 200 --seed 2000
```

## Frozen

Simulator v1.1, scaling, shop, combat, card effects, BC/PPO/TEST unchanged.
