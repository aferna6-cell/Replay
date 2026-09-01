# Replay experiment index

Canonical agent-strength evaluation: **Replay Benchmark v1** (`ml/benchmark.py`,
seeds 10,250,000–10,299,999). Do not reuse TEST for a changed simulator.

## PPO research arc — **CLOSED** (2026-09-01)

| # | Report | Outcome |
|---|---|---|
| Baseline | [`replay_baseline_v1.md`](replay_baseline_v1.md) | Greedy / BC / PPO on Benchmark v1 |
| 1 | [`ppo_degradation_diagnosis_v1.md`](ppo_degradation_diagnosis_v1.md) | PPO instability diagnosed |
| 2 | [`ppo_budget_study_v1.md`](ppo_budget_study_v1.md) | Budget curve |
| 3 | [`ppo_multiseed_replication_v1.md`](ppo_multiseed_replication_v1.md) | Multi-seed replication |
| 4 | [`ppo_policy_anchoring_v1.md`](ppo_policy_anchoring_v1.md) | KL anchoring machinery |
| 4b | [`ppo_matched_ab_v1.md`](ppo_matched_ab_v1.md) | **Anchoring causally stabilizes PPO** |
| 5 | [`ppo_dose_v1.md`](ppo_dose_v1.md) | **β=0.03 best fixed tradeoff; no reliable BC beat** |
| 6 | [`ppo_schedule_v1.md`](ppo_schedule_v1.md) | **STOP — schedule fails all criteria** |

**Conclusion:** Fixed β=0.03 is the best known PPO recipe in Simulator v1, but
PPO does not reliably improve on the BC warm start. No further PPO coefficient
tuning on the current simulator.

## Simulator fidelity — **IN PROGRESS**

| Phase | Report | Status |
|---|---|---|
| 2A | [`sim_fidelity_benchmark_v1.md`](sim_fidelity_benchmark_v1.md) | Measurement baseline (merged) |
| 2B | [`sim_fidelity_phase_2b.md`](sim_fidelity_phase_2b.md) | **Residual scaling — Simulator v1.1 merged** |
| 2C | [`sim_fidelity_phase_2c.md`](sim_fidelity_phase_2c.md) | Composition assembly diagnostic (in progress) |

**Replay Benchmark v2** — define only after Simulator v1.x is frozen; retrain all
agents from scratch. Keep Benchmark v1 TEST untouched.
