# Simulator Fidelity Phase 2H — tempo-aware board management

Date: 2026-09-02 · Status: **2h_v3 confirmation complete — `transition_utility_inadequate` (draft PR #23)** ·
Artifacts: [`results/sim_fidelity_phase_2h/`](../results/sim_fidelity_phase_2h/)

**Invalidated runs (do not use for decisions):**

- v1 seeds 4000–4199: [`invalidated_v1/`](../results/sim_fidelity_phase_2h/invalidated_v1/)
- v2 seeds 5000–5199: [`invalidated_v2/`](../results/sim_fidelity_phase_2h/invalidated_v2/)

## Research question

> Can one realistic policy recover meaningful portions of the Phase 2E/2G oracle
> gains by explicitly trading immediate board strength against composition
> progress and replacement cost?

Methodology **2h_v3** fixes v2 issues: shop `sell→buy→play` compounds, fresh
policy instances per lobby, and completed-action telemetry with target-core
counters separate from tempo-selected actions.

## Policy

When `infer_target(board).have >= 1`:

```text
shop:     build_gain vs board+hand (duplicate check)
deploy:   build_gain vs board only (hand→board progress)
net = (raw + λ×gain) − (repl_raw + λ×repl_build); commit if net > 0
hand full-board:  SELL replacement → PLAY candidate
shop full-board:  SELL replacement → BUY candidate → PLAY candidate
```

When not seeded: identical to raw greedy.

## DEV calibration

| Stage | Seeds | Lobbies |
|---|---:|---:|
| Screen | 3000–3099 | 100 |
| Replication (top-2 λ) | 3100–3499 | 400 |
| Confirmation (frozen) | **6000–6199** | 200 |

```text
λ_build ∈ {4, 8, 12}
```

## Commands

```bash
pytest tests/test_tempo_board_policy.py
python -m ml.fidelity_phase_2h calibrate   # commit calibration
python -m ml.fidelity_phase_2h confirm     # after clean-tree commit
```

## Policy lineage (explicit IDs)

```text
greedy_policy
build_aware_greedy_policy
seeded_core_stress_greedy_policy
seeded_core_deploy_stress_greedy_policy
tempo_board_greedy_policy   # Phase 2H candidate (2h_v3)
```

## 2h_v3 confirmation (seeds 6000–6199, frozen λ=12, clean tree)

| Arm | Seeded fulfillment | 2+ core | Committed | Coverage |
|---|---:|---:|---:|---:|
| Greedy | 0/35 | 0 | 0 | 0.0062 |
| Tempo (λ=12) | 0/35 | 0 | 0 | 0.0061 |
| Oracle | 30/33 | 14 | 11 | 0.0111 |

Macro regression: **pass**. Mechanism gates: **fail** (no lift vs greedy).
Decision: `transition_utility_inadequate`.

Compound transition integrity: **176/176 completed** (100% completion rate).
Target-core actions: 1 buy, 1 deploy (vs 575 tempo-selected buys, 577 deploys).

## Frozen

Simulator v1.1 unchanged. No BC/PPO/card-effect/shop/combat/scaling changes.
Do not start Phase 2I until v3 confirmation completes on clean tree.
