# Replay Experiment 4 — PPO Policy Anchoring

Date: 2026-08-31 · Split: **DEV only** (Benchmark v1 TEST never run) ·
Artifacts: [`results/ppo_anchor_v1/`](../results/ppo_anchor_v1/) ·
Manifest: [`manifest.json`](../results/ppo_anchor_v1/manifest.json)

## Question

Experiment 3 found that unconstrained PPO drifts massively from its BC warm
start (KL reaching ~1.0–1.4, expert agreement falling to ~43%) across all
training seeds, while any placement improvements are seed-specific excursions.
Experiment 3 recommended anchoring the policy to the imitation prior.

> **Does a pre-specified KL penalty toward the frozen BC prior (β = 0.1) reduce
> unbounded policy drift while preserving or improving DEV placement vs
> unconstrained PPO (Experiment 2, β = 0) at the same 5,120-episode budget?**

One variable changed: **KL anchoring coefficient** (β = 0.1 vs baseline β = 0).
Architecture, learning rate, γ, λ, clip, entropy, value coefficient, optimizer,
league logic, reward function, training seed, budget, and shaping schedule are
all unchanged from Experiment 2.

## Setup

- **Intervention**: add `β · KL(π_BC ‖ π_θ)` to the PPO loss on every minibatch,
  using the same masked-KL definition as `ml.policy_drift`. The anchor network
  (`ml/policy_bc.pt`) is frozen; only the trainable policy is updated.
- **Pre-specified β = 0.1** — chosen before results, not tuned.
- **Training**: seed 0, 320 iterations × 16 episodes, `--shaping-horizon 40`,
  warm start from BC + DAgger (`policy_bc.pt`).
- **Baseline comparison**: committed Experiment 2 results (`results/ppo_budget_v1/`,
  unconstrained PPO, identical recipe except β = 0).
- **Primary evaluation**: 1000 DEV games vs **7× greedy**, seeds
  10,550,000–10,550,999, identical for every checkpoint.
- **Secondary diagnostic**: 500 DEV games vs `greedy4_random3`.
- **Drift corpus**: same frozen 4,440-state corpus as Experiments 1–3
  (fingerprint `2ec217b353bd`).
- **TEST set remained locked.**

## DEV performance curve

| iter | episodes | Anchored avg | 95% CI | Top-4 | Baseline avg | Expert agree (anch) | Expert agree (base) | KL (anch) | KL (base) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 6.550 | [6.427, 6.665] | 16.0% | 6.554 | 84.4% | 84.5% | 0.000 | 0.000 |
| 40 | 640 | 6.608 | [6.489, 6.731] | 13.8% | 6.761 | 83.2% | 77.2% | 0.048 | 0.371 |
| 80 | 1,280 | 6.580 | [6.460, 6.697] | 15.5% | **6.325** | 84.0% | 80.9% | 0.038 | 0.249 |
| 160 | 2,560 | 6.584 | [6.461, 6.699] | 15.0% | 6.435 | 85.3% | 74.3% | 0.052 | 0.484 |
| 320 | 5,120 | 6.673 | [6.556, 6.785] | 13.0% | 6.606 | **83.2%** | **42.6%** | **0.043** | **1.171** |

## Paired comparisons

Deterministic paired bootstrap over identical 1000 DEV games (B = 10,000, seed 0).
Positive Δ = anchored places **worse** than reference.

**Anchored vs warm start (iteration 0), greedy field:**

| Comparison | Mean Δ | 95% CI | Reading |
|---|---|---|---|
| iter 40 − iter 0 | +0.058 | [−0.063, +0.178] | no clear difference |
| iter 80 − iter 0 | +0.030 | [−0.130, +0.192] | no clear difference |
| iter 160 − iter 0 | +0.034 | [−0.043, +0.112] | no clear difference |
| iter 320 − iter 0 | +0.123 | [−0.035, +0.281] | no clear difference |

**Anchored vs unconstrained baseline (Experiment 2) at same iteration:**

| Comparison | Mean Δ | 95% CI | Reading |
|---|---|---|---|
| iter 0 − iter 0 | −0.004 | [−0.163, +0.158] | no clear difference |
| iter 40 − iter 40 | −0.153 | [−0.312, +0.008] | no clear difference (CI touches 0) |
| iter 80 − iter 80 | **+0.255** | **[+0.094, +0.410]** | **baseline places better** |
| iter 160 − iter 160 | +0.149 | [−0.002, +0.306] | no clear difference (CI touches 0) |
| iter 320 − iter 320 | +0.067 | [−0.090, +0.223] | no clear difference |

## Observations

1. **Drift control succeeded.** KL from warm start stayed below 0.05 across all
   anchored checkpoints (max 0.052 at iter 160). Expert agreement remained
   83–85% throughout — vs 43% and KL 1.17 for the unconstrained baseline at
   iter 320. Warm-start agreement stayed ~83–88% vs ~41% baseline. The
   intervention directly addresses the pathology Experiment 3 identified.

2. **Placement did not improve.** Anchored DEV placement stayed flat around
   6.55–6.67 across the full budget — no transient improvement at iter 80, no
   regression at iter 320. The anchored policy neither gained nor lost vs its
   warm start on the primary field.

3. **Anchoring forfeited the baseline's best checkpoint.** At iter 80 the
   unconstrained baseline scored 6.325 (−0.229 vs warm start in Experiment 2).
   Anchored iter 80 scored 6.580 — significantly worse than the baseline at
   the same budget (+0.255, CI [+0.094, +0.410]). The KL penalty appears to
   prevent the policy from exploring the region where that transient gain lived.

4. **At full budget, placement is indistinguishable.** Anchored iter 320
   (6.673) vs baseline iter 320 (6.606): +0.067, CI includes zero. But the
   baseline's poor placement at 320 came with catastrophic drift; anchored
   320 kept 83% expert agreement.

5. **Training-time anchor KL was stable.** Mean `anchor_kl` in the loss stayed
   ~0.05–0.07 across all 320 iterations — the penalty was active but not
   driving KL to zero (the policy still moved within a bounded neighborhood).

## Conclusion

**Partial success on the stated hypothesis:**

- **Drift half: confirmed.** β = 0.1 KL anchoring prevents the unbounded policy
  collapse observed in Experiments 2–3.
- **Placement half: not confirmed.** Anchoring does not improve DEV placement
  vs the warm start or vs unconstrained PPO at iter 320, and it blocks the
  transient improvement unconstrained PPO found at 2× budget.

The evidence supports treating drift control and placement optimization as
**separate problems**: anchoring fixes the former at the cost of preventing
exploration that occasionally helps placement.

## Limitations

- **Single β value (0.1), single training seed.** Stronger or weaker anchoring
  may trade off differently; this experiment tests one pre-specified point.
- **BC checkpoint regenerated** for this run (parameter hash differs from
  Experiment 2's historical warm start); iter 0 reproduces the current BC
  exactly, and baseline comparisons use committed Experiment 2 DEV JSON.
- Checkpoint binaries are gitignored; fingerprints are in the manifest.
- Benchmark v1 TEST was not run.

## Recommended Experiment 5 (do not run yet)

**Adaptive or scheduled anchoring** — start with stronger KL anchoring early
(when drift is most damaging) and anneal β toward zero after iter 40–80 to
allow the policy to capture transient placement gains without the iter-320
collapse. Alternatively, test whether a **trust-region constraint** (PPO clip
tightening toward BC) achieves the same drift control with less exploration
penalty.
