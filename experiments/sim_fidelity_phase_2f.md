# Simulator Fidelity Phase 2F — post-purchase core lifecycle diagnosis

Date: 2026-09-01 · Status: **measurement-only diagnostic** ·
Artifacts: [`results/sim_fidelity_phase_2f/`](../results/sim_fidelity_phase_2f/)

## Research question

> What happened to the fulfilled seeded core purchases after they were acquired
> under Phase 2E oracle stress?

Phase 2E showed **34/49 seeded fulfillment** but **0 end-of-recruit 2+ core
assembly**. Phase 2F traces each fulfilled purchase from buy through
disappearance or game end.

## Policy

Same oracle as Phase 2E treatment: `seeded_core_stress_greedy_policy` on fresh
seeds **1000–1199**. No retention, triple, target, or card-effect changes.

## Lifecycle fates (mutually exclusive)

| Fate | Meaning |
|---|---|
| A | Purchased but never played |
| B | Played then sold same recruit turn |
| C | Played but no persistent 2+ core assembly (see `sold_after_play` flag) |
| D | Core survives but infer_target changes before second-piece assembly |
| E | New core survives but original seeded core disappears |
| F | Transformed / tripled |
| G | Two cores coexist at action level but not end-of-recruit |
| H | Two+ cores coexist through end-of-recruit |

## Funnel

```text
fulfilled purchases → played → coexist with seed → survive end-recruit
  → survive 1 turn → survive 2 turns
```

## Decision output

Exactly **one** recommended next intervention (play policy, retention/sell,
seed retention, target hysteresis, triple/discover fidelity, or card effects if
H dominates with flat coverage).

## Commands

```bash
pytest tests/test_core_lifecycle_diagnostic.py
python -m ml.fidelity_phase_2f --lobbies 200 --seed 1000
```

## Frozen

Simulator v1.1, scaling, shop, combat, card effects, BC/PPO/TEST unchanged.
