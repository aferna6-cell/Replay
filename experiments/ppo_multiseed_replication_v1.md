# Replay Experiment 3 — Multi-Seed PPO Budget Replication

Date: 2026-08-31 · Split: **DEV only** (Benchmark v1 TEST never run) ·
Artifacts: [`results/ppo_multiseed_v1/`](../results/ppo_multiseed_v1/) ·
Manifest: [`manifest.json`](../results/ppo_multiseed_v1/manifest.json)

> Draft report scaffold — tables and outcome filled after seeds 1–3 complete.

## Question

Experiment 2 found a U-shaped PPO budget curve on training seed 0: significant
degradation at 640 episodes, significant improvement at 1,280, regression by
5,120, with unbounded policy drift. Is that shape a property of the frozen
algorithm, or of one lucky/unlucky trajectory?

**One variable changed: PPO training seed.** Architecture, optimizer, LR,
weight decay, γ, λ, clip, entropy/value coefficients, PPO epochs, batch size,
league behavior, reward, shaping schedule, episodes/iteration, observation
encoder, action space, and the BC+DAgger warm start are all frozen.

## Historical observation (Experiment 2, seed 0)

| iter | episodes | Greedy avg | vs iter 0 | Expert agree | KL |
|---|---|---|---|---|---|
| 0 | 0 | 6.554 | — | 84.5% | 0.000 |
| 40 | 640 | 6.761 | +0.207 worse | 77.2% | 0.371 |
| 80 | 1,280 | 6.325 | −0.229 better | 80.9% | 0.249 |
| 160 | 2,560 | 6.435 | −0.119 | 74.3% | 0.484 |
| 320 | 5,120 | 6.606 | +0.052 | 42.6% | 1.171 |

## Setup

- **Frozen warm start:** `policy_bc.pt` with
  `parameter_sha256 = 094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b`
  (matches Experiment 2). Reproduced via
  `PYTHONHASHSEED=0 python scripts/reproduce_warm_start_bc.py` under
  deterministic torch; verified before every new-seed run. Iter 0 of each new
  seed must reproduce this hash.
- **Seeds:** 0 = published Exp2 trajectory (not rerun). New runs: seeds **1, 2, 3**.
- **Protocol:** 16 episodes/iter × 320 iters = 5,120 episodes;
  `--shaping-horizon 40`; primary checkpoints at iters 0, 40, 80, 160, 320.
- **DEV evaluation only:** 1000 games vs 7× greedy on seeds
  10,550,000–10,550,999; secondary `greedy4_random3` diagnostic (500 games),
  no 4.5 threshold. **TEST locked.**
- **Drift corpus:** frozen Exp1/2 corpus (4,440 states, fingerprint
  `2ec217b353bd…`).

## Training-seed control

| Seed | Episode seed span | Overlaps DEV/TEST? |
|---|---|---|
| 0 | 1–5120 | no |
| 1 | 1000004–1005123 | no |
| 2 | 2000007–2005126 | no |
| 3 | 3000010–3005129 | no |

## Per-seed results

*(filled after training)*

## 1280-episode replication (Question A)

*(iter80 − iter0 across seeds)*

## Long-training regression (Question B)

*(iter320 − iter80 across seeds)*

## Drift

*(iter320 expert/warm-start/KL; best-checkpoint vs later drift)*

## Action-category

*(tempo roll/end/play; freeze appearance at iter320)*

## RL diagnostics

*(blocks 1–40 / 41–160 / 161–320 across seeds)*

## Limitations

- **n=4 training seeds is exploratory only.** Counts are descriptive, not
  confirmatory frequency estimates.
- Checkpoint binaries are gitignored; fingerprints are recorded.
- No hyperparameter tuning; no best-seed selection; no iter80 deployment;
  TEST untouched.

## Conclusion

*(Outcome A/B/C/D — filled after analysis)*

## Recommended Experiment 4

*(ONE recommendation per decision rule — filled after analysis)*
