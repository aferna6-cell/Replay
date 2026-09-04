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
| 2C | [`sim_fidelity_phase_2c.md`](sim_fidelity_phase_2c.md) | Composition diagnostic **merged** |
| 2D | [`sim_fidelity_phase_2d.md`](sim_fidelity_phase_2d.md) | Build-aware path/5 A/B **merged (negative)** |
| 2E | [`sim_fidelity_phase_2e.md`](sim_fidelity_phase_2e.md) | Seeded-core conversion stress test |
| 2F | [`sim_fidelity_phase_2f.md`](sim_fidelity_phase_2f.md) | Post-purchase core lifecycle diagnosis **merged** |
| 2G | [`sim_fidelity_phase_2g.md`](sim_fidelity_phase_2g.md) | Seeded-core deployment board-slot stress test **merged** |
| 2H | [`sim_fidelity_phase_2h.md`](sim_fidelity_phase_2h.md) | Tempo board policy **merged (negative, 2h_v3)** |
| 2I | [`sim_fidelity_phase_2i.md`](sim_fidelity_phase_2i.md) | Seeded opportunity decision-margin diagnostic (**merged 2i_v2**) |
| 2J | [`sim_fidelity_phase_2j.md`](sim_fidelity_phase_2j.md) | Board-relative opportunity-cost policy (**merged ACCEPT α=0.5**) |
| 2K | [`sim_fidelity_phase_2k.md`](sim_fidelity_phase_2k.md) | Post-assembly gap (**merged**; 92% never-available) |
| 2L | [`sim_fidelity_phase_2l.md`](sim_fidelity_phase_2l.md) | Availability decomp (`2l_v2`: **63% A3** → approve; Phase 2M pool audit) |
| 2M | [`sim_fidelity_phase_2m.md`](sim_fidelity_phase_2m.md) | Shop/pool audit (`2m_v2`: multi-mismatch; mild draw undershoot) |
| 2N | [`sim_fidelity_phase_2n.md`](sim_fidelity_phase_2n.md) | Active Tavern-pool OK; **HOLD v1.x** (`2n_v3` macro gates fail; confirm 11500 reserved) |
| 2O | [`sim_fidelity_phase_2o.md`](sim_fidelity_phase_2o.md) | Midgame scaling budget (`2o_v1`): **pre-scale hole, post-scale ~OK at T10; recruit Δ≈0**; #29 still HOLD |
| 2P | [`sim_fidelity_phase_2p.md`](sim_fidelity_phase_2p.md) | Replacement-value contamination (`2p_v2`): **confirmed** (golden fix; weakest-golden share 0); #29 HOLD |
| 2Q | [`sim_fidelity_phase_2q.md`](sim_fidelity_phase_2q.md) | Recruit vs combat split (`2q_v1`): **HOLD** — replaces unblocked; post-scale macro collapses; #29 HOLD |
| 2R | [`sim_fidelity_phase_2r.md`](sim_fidelity_phase_2r.md) | Replacement churn (`2r_v1`): **HOLD** — churn/loss explains collapse (frac 0.994); preserve combat on replace; #29/#33 HOLD |
| 2R QA | [`sim_fidelity_phase_2r_qa.md`](sim_fidelity_phase_2r_qa.md) | Independent recompute **0.9938 survives**; sell→buy→play / residual / carry clean |
| 2S | [`sim_fidelity_phase_2s.md`](sim_fidelity_phase_2s.md) | Board-level abstract scaling (`2s_v1`): **HOLD `inconclusive`** — replace 0.298 + T10 1.007 recover; net-board loss **−3.46**; game length Δ **−2.18**; #29/#33/#34/#35/#36 HOLD |

**Replay Benchmark v2** — define only after Simulator v1.x is frozen; retrain all
agents from scratch. Keep Benchmark v1 TEST untouched.
