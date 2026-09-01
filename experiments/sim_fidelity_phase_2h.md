# Simulator Fidelity Phase 2H — tempo-aware board management

Date: 2026-09-01 · Status: **candidate policy + DEV calibration** ·
Artifacts: [`results/sim_fidelity_phase_2h/`](../results/sim_fidelity_phase_2h/)

## Research question

> Can one realistic policy recover meaningful portions of the Phase 2E/2G oracle
> gains by explicitly trading immediate board strength against composition
> progress and replacement cost?

First **candidate real simulator policy** — not another hard oracle.

## Policy

When `infer_target(board).have >= 1`:

```text
candidate_value = raw_stats + λ_build × candidate_build_gain
replacement_cost = repl_raw + λ_build × replacement_build_value
net_transition = candidate_value - replacement_cost
```

Commit only when `net > 0`. Jointly scores **buy** (with best replacement if
board full) and **deploy** (explicit hand-card × replacement pairing). Does not
use play-first-greedy ordering.

When not seeded: identical to raw greedy.

## DEV calibration

| Stage | Seeds | Lobbies |
|---|---:|---:|
| Screen | 3000–3099 | 100 |
| Replication (top-2 λ) | 3100–3499 | 400 |
| Confirmation (frozen) | 4000–4199 | 200 |

```text
λ_build ∈ {4, 8, 12}
```

Selection order: macro OK → persistent 2+ core → committed states → coverage →
lower action-deviation vs greedy.

## Confirmation arms

| Arm | Policy |
|---|---|
| Raw greedy | baseline |
| Frozen Phase 2H | `tempo_board_greedy_policy` with frozen λ |
| Phase 2E+2G oracle | upper-bound diagnostic only |

## Commands

```bash
pytest tests/test_tempo_board_policy.py
python -m ml.fidelity_phase_2h full
python -m ml.fidelity_phase_2h calibrate
python -m ml.fidelity_phase_2h confirm --lambda-build 8
```

## Policy lineage (explicit IDs)

```text
greedy_policy
build_aware_greedy_policy              # Phase 2D negative
seeded_core_stress_greedy_policy       # Phase 2E oracle
seeded_core_deploy_stress_greedy_policy  # Phase 2G oracle
tempo_board_greedy_policy              # Phase 2H candidate
```

## Frozen

Simulator v1.1, scaling, shop, combat, card effects, BC/PPO/TEST unchanged.
