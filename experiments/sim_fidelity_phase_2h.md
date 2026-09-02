# Simulator Fidelity Phase 2H — tempo-aware board management

Date: 2026-09-02 · Status: **merged — `transition_utility_inadequate` (2h_v3 negative)** ·
Artifacts: [`results/sim_fidelity_phase_2h/`](../results/sim_fidelity_phase_2h/)

**Invalidated runs (do not use for decisions):**

- v1 seeds 4000–4199: [`invalidated_v1/`](../results/sim_fidelity_phase_2h/invalidated_v1/)
- v2 seeds 5000–5199: [`invalidated_v2/`](../results/sim_fidelity_phase_2h/invalidated_v2/)

## Research question

> Can one realistic policy recover meaningful portions of the Phase 2E/2G oracle
> gains by explicitly trading immediate board strength against composition
> progress and replacement cost?

**Answer (2h_v3):** No. The tested raw-stat + λ×build-progress transition utility
does not recover acquisition/deployment gains demonstrated by the Phase 2E/2G oracle.

## Policy

Use `TempoBoardGreedyPolicy` via `policies_for_lobby(lambda_build, n)` — fresh
instances per lobby; no stateful default export through `bg_env`.

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

Frozen **λ_build = 12** (calibration commit `da18ca9`).

## 2h_v3 confirmation (seeds 6000–6199, clean tree)

| Arm | Seeded fulfillment | 2+ core | Committed | Coverage |
|---|---:|---:|---:|---:|
| Greedy | 0/35 | 0 | 0 | 0.00616 |
| Tempo (λ=12) | 0/35 | 0 | 0 | 0.00607 |
| Oracle | 30/38 | 14 | 16 | 0.01035 |

Macro regression: **pass**. Mechanism gates: **fail** (no lift vs greedy).
Decision: `transition_utility_inadequate`.

Compound transition integrity: **176/176 completed** (100% completion rate).
Target-core actions: 1 buy, 1 deploy (vs 575 tempo-selected buys, 577 deploys).

## Policy lineage (explicit IDs)

```text
greedy_policy
build_aware_greedy_policy
seeded_core_stress_greedy_policy
seeded_core_deploy_stress_greedy_policy
TempoBoardGreedyPolicy + policies_for_lobby   # Phase 2H candidate (2h_v3, rejected)
```

## Frozen

Simulator v1.1 unchanged. Next: Phase 2I decision-margin diagnostic (measurement only).
