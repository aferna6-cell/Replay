# Simulator Fidelity Phase 2H — tempo-aware board management

Date: 2026-09-01 · Status: **candidate policy + DEV calibration (2h_v2)** ·
Artifacts: [`results/sim_fidelity_phase_2h/`](../results/sim_fidelity_phase_2h/)

**Invalidated v1** (seeds 4000–4199): [`invalidated_v1/`](../results/sim_fidelity_phase_2h/invalidated_v1/) —
lobby-ID collapse + non-atomic transitions. Do not use for decisions.

## Research question

> Can one realistic policy recover meaningful portions of the Phase 2E/2G oracle
> gains by explicitly trading immediate board strength against composition
> progress and replacement cost?

Methodology **2h_v2** fixes: distinct lobby IDs in traces, separate
acquisition vs deployment build gain, latched sell→play/buy transitions, and
completed-action telemetry (not double-counted).

## Policy

When `infer_target(board).have >= 1`:

```text
shop:     build_gain vs board+hand (duplicate check)
deploy:   build_gain vs board only (hand→board progress)
net = (raw + λ×gain) − (repl_raw + λ×repl_build); commit if net > 0
compound: sell replacement → latch pending → complete play/buy next action
```

When not seeded: identical to raw greedy.

## DEV calibration

| Stage | Seeds | Lobbies |
|---|---:|---:|
| Screen | 3000–3099 | 100 |
| Replication (top-2 λ) | 3100–3499 | 400 |
| Confirmation (frozen) | **5000–5199** | 200 |

```text
λ_build ∈ {4, 8, 12}
```

## Commands

```bash
pytest tests/test_tempo_board_policy.py
python -m ml.fidelity_phase_2h full
```

## Policy lineage (explicit IDs)

```text
greedy_policy
build_aware_greedy_policy
seeded_core_stress_greedy_policy
seeded_core_deploy_stress_greedy_policy
tempo_board_greedy_policy   # Phase 2H candidate (2h_v2)
```

## Frozen

Simulator v1.1 unchanged. No BC/PPO/card-effect/shop/combat/scaling changes.
