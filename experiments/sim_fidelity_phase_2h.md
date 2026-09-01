# Simulator Fidelity Phase 2H — tempo-aware board management

Date: 2026-09-01 · Status: **2h_v2 confirmation complete — `transition_utility_inadequate` (draft PR #23)** ·
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

## 2h_v2 confirmation (seeds 5000–5199, frozen λ=12)

| Arm | Seeded fulfillment | 2+ core | Committed | Coverage |
|---|---:|---:|---:|---:|
| Greedy | 1/30 | 1 | 1 | 0.0079 |
| Tempo (λ=12) | 0/30 | 0 | 0 | 0.0041 |
| Oracle upper bound | 24/33 | 8 | 11 | 0.0111 |

Macro regression: **pass** (treatment within macro gates vs greedy).

Mechanism gates: **fail** — treatment does not beat greedy on seeded fulfillment,
2+ assembly, committed states, or coverage. Decision:
`transition_utility_inadequate`.

Completed-action telemetry (treatment, fidelity run only): 427 slot sells, 486 core
buys, 701 hand deploys — policy is active but does not translate into seeded
composition progress under the 2c diagnostic.

## Frozen

Simulator v1.1 unchanged. No BC/PPO/card-effect/shop/combat/scaling changes.
Do not start Phase 2I until policy reformulation is scoped.
