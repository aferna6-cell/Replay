# Replay Experiment 3 — Multi-Seed PPO Budget Replication

Date: 2026-08-31 · Split: **DEV only** (Benchmark v1 TEST never run) ·
Artifacts: [`results/ppo_multiseed_v1/`](../results/ppo_multiseed_v1/) ·
Manifest: [`manifest.json`](../results/ppo_multiseed_v1/manifest.json)

## Question

Experiment 2 observed that extending PPO training to 1,280 episodes (iteration 80)
on training seed 0 produced a transient improvement of **−0.229** over the BC + DAgger
warm start, which subsequently decayed by 5,120 episodes (iteration 320) back to
warm-start level (**+0.052**), forming an apparent U-shaped training curve.

The central question of Experiment 3 is:

> **Does the transient PPO improvement around 1,280–2,560 episodes reproduce across
> independent PPO training seeds, and does performance then decay with extended training?**

This is a **pure replication study**. No hyperparameters were tuned, no algorithm components
were altered, and the warm-start model was strictly frozen. The only experimental
variable is the **PPO training seed**.

---

## Setup & Training-Seed Control

- **Frozen warm start**: Every PPO trajectory was initialized from the exact same
  BC + DAgger model (`ml/policy_bc.pt`, `parameter_sha256 = 1f8077f9b982eb31ba1ab45ddaf0d0afa8b296b267ac1587b77fc80d8aa84f1b`).
  Iteration 0 for all seeds reproduces this parameter hash bitwise before any PPO update.
  This isolates **PPO training randomness** from imitation-learning randomness.
- **Frozen algorithm recipe**: Exact Experiment 2 hyperparameters across all seeds:
  - Architecture: Transformer PolicyNet (token dim 16, d_model 64, 4 heads, 2 layers, ff 128)
  - Optimizer: AdamW (lr = 3e-4, weight_decay = 1e-4, grad_clip_norm = 1.0)
  - RL parameters: $\gamma = 0.999$, $\lambda = 0.95$, PPO clip = 0.2, entropy coef = 0.01, value coef = 0.5
  - PPO update: 4 epochs per iteration, minibatch size 256
  - Self-play league: snapshot every 8 iterations, max league size 5
  - Shaping schedule: initial 1.0, `--shaping-horizon 40` (reaches 0.0 at iteration 28 and stays 0.0 for iterations >40)
  - Budget: 16 episodes/iteration, 320 iterations (5,120 total episodes per seed trajectory)
- **Training seeds**: Seeds 0 (historical reference), 1, 2, and 3.
  All training episode seeds ($s \times 1{,}000{,}003 + k$) remain strictly disjoint
  from both DEV (`[10,550,000, 10,599,999]`) and TEST (`[10,250,000, 10,299,999]`) intervals.
- **Evaluation protocol**:
  - Primary: 1000 DEV games vs **7× greedy** on reserved seeds `10,550,000–10,550,999`.
  - Secondary: 500 DEV games vs `greedy4_random3` on reserved seeds `10,550,000–10,550,499`.
  - Diagnostic corpus: 4,440 frozen states from `CORPUS_SEED_BASE = 10,590,000` (DEV sub-range).
  - **TEST set remained locked**: No Benchmark v1 TEST seeds were run or inspected.

---

## Cross-Training-Seed Summary

Primary DEV evaluation (1000 games vs 7× greedy; lower average placement is better; 4.5 is field average):

| Iteration | Episodes | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean | Median | Min | Max | Std Dev |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **0** (warm start) | 0 | 6.554 | 6.573 | 6.573 | 6.573 | **6.568** | 6.573 | 6.554 | 6.573 | 0.010 |
| **40** (baseline) | 640 | 6.761 | 6.610 | 6.584 | 6.635 | **6.647** | 6.623 | 6.584 | 6.761 | 0.078 |
| **80** (2×) | 1,280 | **6.325** | 6.526 | 6.715 | 7.142 | **6.677** | 6.620 | 6.325 | 7.142 | 0.349 |
| **160** (4×) | 2,560 | 6.435 | 6.593 | 6.697 | 7.151 | **6.719** | 6.645 | 6.435 | 7.151 | 0.307 |
| **320** (8×) | 5,120 | 6.606 | 6.597 | 6.602 | 6.558 | **6.591** | 6.600 | 6.558 | 6.606 | 0.022 |

Secondary DEV diagnostic field (500 games vs `greedy4_random3`):

| Iteration | Episodes | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean | Median | Std Dev |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **0** | 0 | 4.370 | 4.362 | 4.362 | 4.362 | **4.364** | 4.362 | 0.004 |
| **40** | 640 | 4.432 | 4.452 | 4.382 | 4.432 | **4.424** | 4.432 | 0.030 |
| **80** | 1,280 | 4.282 | 4.432 | 4.438 | 4.894 | **4.512** | 4.435 | 0.262 |
| **160** | 2,560 | 4.206 | 4.408 | 4.518 | 4.892 | **4.506** | 4.463 | 0.286 |
| **320** | 5,120 | 4.408 | 4.562 | 4.358 | 4.402 | **4.432** | 4.405 | 0.090 |

---

## Within-Seed Paired Comparisons

Deterministic paired bootstrap over identical 1000 DEV games ($B = 10{,}000$, seed 0).
Convention: $\Delta = \text{Target} - \text{Reference}$. Positive = Target places **worse** (higher placement). Negative = Target places **better**.

### Seed 0 (Historical Reference)
*Classification: U-like / transient improvement*
- `iter040 − iter000`: **+0.207** `[+0.093, +0.322]` — PPO initially hurts
- `iter080 − iter000`: **−0.229** `[−0.392, −0.061]` — **PPO significantly improves over warm start**
- `iter160 − iter000`: −0.119 `[−0.245, +0.008]` — no clear difference
- `iter320 − iter000`: +0.052 `[−0.104, +0.210]` — no clear difference
- `iter080 − iter040`: **−0.436** `[−0.593, −0.277]` — iter80 beats iter40
- `iter320 − iter080`: **+0.281** `[+0.138, +0.425]` — **significant regression after iter80**

### Seed 1
*Classification: mostly flat / noisy*
- `iter040 − iter000`: +0.037 `[−0.037, +0.111]` — no clear difference
- `iter080 − iter000`: −0.047 `[−0.201, +0.108]` — no clear difference (slight point improvement, CI spans 0)
- `iter160 − iter000`: +0.020 `[−0.037, +0.076]` — no clear difference
- `iter320 − iter000`: +0.024 `[−0.057, +0.106]` — no clear difference
- `iter080 − iter040`: −0.084 `[−0.235, +0.067]` — no clear difference
- `iter320 − iter080`: +0.071 `[−0.075, +0.221]` — no clear difference

### Seed 2
*Classification: other (mid-training degradation followed by recovery)*
- `iter040 − iter000`: +0.011 `[−0.151, +0.170]` — no clear difference
- `iter080 − iter000`: +0.142 `[−0.010, +0.296]` — degraded (CI touches 0)
- `iter160 − iter000`: **+0.124** `[+0.024, +0.225]` — **iter0 places significantly better**
- `iter320 − iter000`: +0.029 `[−0.080, +0.140]` — no clear difference
- `iter080 − iter040`: +0.131 `[−0.023, +0.285]` — no clear difference
- `iter320 − iter080`: −0.113 `[−0.262, +0.036]` — no clear difference (improves vs iter80)

### Seed 3
*Classification: other (severe transient collapse at 1,280–2,560 episodes, recovered at 5,120)*
- `iter040 − iter000`: +0.062 `[−0.015, +0.139]` — no clear difference
- `iter080 − iter000`: **+0.569** `[+0.435, +0.704]` — **massive degradation at iter 80**
- `iter160 − iter000`: **+0.578** `[+0.437, +0.720]` — **massive degradation at iter 160**
- `iter320 − iter000`: −0.015 `[−0.103, +0.073]` — no clear difference (fully recovered)
- `iter080 − iter040`: **+0.507** `[+0.372, +0.641]` — iter80 much worse than iter40
- `iter320 − iter080`: **−0.584** `[−0.719, −0.450]` — **iter320 significantly improves over iter80**

---

## Pre-Specified Replication Answers

### Question A — Does the ~1,280 episode improvement reproduce?

**Answer: NO.**
- Seed 0 showed a large, statistically significant gain at iteration 80: **−0.229** `[−0.392, −0.061]`.
- Across the three replication seeds:
  - Seed 1 was practically flat: **−0.047** `[−0.201, +0.108]` (CI includes zero).
  - Seed 2 worsened: **+0.142** `[−0.010, +0.296]` (point estimate worse).
  - Seed 3 collapsed: **+0.569** `[+0.435, +0.704]` (statistically significant severe degradation).
- **Count summary**:
  - Seeds improving by point estimate: 2 / 4 (Seed 0, Seed 1)
  - Seeds worsening by point estimate: 2 / 4 (Seed 2, Seed 3)
  - Seeds with 95% CIs excluding zero: 2 / 4 (Seed 0 improved, Seed 3 worsened)
- **Replication outcome**: The iteration-80 gain was an excursion unique to Seed 0's random trajectory, not a general property of PPO budget scaling.

### Question B — Does performance decay after the transient improvement?

**Answer: NO (decay was conditional on Seed 0's initial dip).**
- For Seed 0, performance regressed after iteration 80: `iter320 − iter80 = +0.281 [+0.138, +0.425]`.
- For Seed 1, performance was indistinguishable: `+0.071 [−0.075, +0.221]`.
- For Seeds 2 and 3, which experienced mid-training degradation, performance between iteration 80 and iteration 320 actually **improved** as the policies recovered:
  - Seed 2: `−0.113 [−0.262, +0.036]`
  - Seed 3: `−0.584 [−0.719, −0.450]` (significant recovery)
- **Count summary**:
  - Seeds regressing after iter 80: 1 / 4 (Seed 0)
  - Seeds improving after iter 80: 1 / 4 (Seed 3)
  - Seeds statistically indistinguishable: 2 / 4 (Seed 1, Seed 2)

### U-Shape Summary
- **1 / 4 trajectories** showed the classic transient improvement followed by regression (Seed 0).
- **1 / 4 trajectories** was mostly flat across all budgets (Seed 1).
- **2 / 4 trajectories** showed mid-budget degradation / instability followed by recovery back to warm-start level (Seeds 2 and 3).

---

## Policy-Drift Replication

Tested on the frozen 4,440-state diagnostic corpus (`CORPUS_SEED_BASE = 10,590,000`):

| Metric | Warm Start (Iter 0) | Seed 0 (Iter 320) | Seed 1 (Iter 320) | Seed 2 (Iter 320) | Seed 3 (Iter 320) |
|:---|:---|:---|:---|:---|:---|
| **Expert Agreement** | 84.5% | 42.6% | 47.1% | 42.5% | 45.3% |
| **Warm-Start Agreement** | 100.0% | 40.8% | 54.7% | 50.9% | 46.0% |
| **KL from Warm Start** | 0.000 | 1.171 | 1.052 | 0.922 | 1.412 |
| **Corpus Entropy** | 0.352 | 0.757 | 1.190 | 0.861 | 0.598 |
| **Value Head Mean $\pm$ Std** | −0.092 $\pm$ 0.535 | +0.129 $\pm$ 0.435 | +0.046 $\pm$ 0.334 | +0.098 $\pm$ 0.407 | −0.063 $\pm$ 0.551 |

### Drift Findings:
1. **Massive behavioral drift replicates across 100% of PPO training seeds.** In all 4 seeds, agreement with the expert collapsed from ~85% down to 42.5%–47.1%, warm-start agreement fell to 40.8%–54.7%, and KL divergence from the warm-start policy exceeded 0.92–1.41.
2. **Best checkpoint vs. drift correlation**:
   - Seed 0: Best checkpoint was iter 80 (placement 6.325, expert agreement 80.9%, KL 0.249). Later drift ruined performance.
   - Seed 1: Best checkpoint was iter 80 (placement 6.526, expert agreement 77.2%, KL 0.344).
   - Seed 2: Best checkpoint was iter 0 (placement 6.573, expert agreement 84.5%, KL 0.000).
   - Seed 3: Best checkpoint was iter 320 (placement 6.558, expert agreement 45.3%, KL 1.412).
   While Seed 0, 1, and 2 found their best placements when drift was low (KL $\le 0.35$), unconstrained optimization consistently produced large downstream drift across all seeds.

---

## Action-Category Replication

Action categories over the 28-action space at iteration 320:

| Category | Expert Decisions | Seed 0 Disagree % | Seed 1 Disagree % | Seed 2 Disagree % | Seed 3 Disagree % | Consistent Pattern? |
|:---|:---|:---|:---|:---|:---|:---|
| **Buy** | 823 | 73.4% | 68.7% | 74.4% | 68.8% | Yes (warm-start baseline slot artifact) |
| **Roll** | 1,352 | 70.6% | 72.9% | 68.3% | 81.3% | **Yes (massive tempo drift across all seeds)** |
| **End** | 1,025 | 50.1% | 48.0% | 51.6% | 61.6% | **Yes (massive tempo drift across all seeds)** |
| **Play** | 750 | 40.0% | 20.3% | 34.1% | 23.3% | Yes (moderate drift) |
| **Level** | 422 | 25.8% | 30.6% | 27.0% | 40.8% | Yes (moderate drift) |
| **Sell** | 68 | 100.0% | 100.0% | 98.5% | 100.0% | Yes (small count) |
| **Freeze** | 0 | Appears (3.4%) | 0.0% | Appears (0.2%) | 0.0% | Non-expert behavior emergence |

### Category Findings:
- **Tempo decisions dominate policy drift across every seed.** PPO repeatedly alters **Roll** (68%–81% disagreement) and **End Turn** (48%–62% disagreement) decisions.
- Unconstrained PPO alters the turn-pacing dynamics of the game rather than refining minion valuation.

---

## RL-Signal & Optimization Diagnostics Across Seeds

Means per training block (Blocks 1: iters 1–40; Block 2: iters 41–160; Block 3: iters 161–320):

| Metric | Block | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|:---|:---|:---|:---|:---|:---|
| **Value Explained Variance** | 1 / 2 / 3 | 0.614 / 0.682 / 0.688 | 0.611 / 0.678 / 0.706 | 0.607 / 0.669 / 0.704 | 0.609 / 0.688 / 0.712 |
| **Mean \|Advantage\|** | 1 / 2 / 3 | 0.205 / 0.196 / 0.179 | 0.207 / 0.194 / 0.173 | 0.208 / 0.195 / 0.174 | 0.205 / 0.203 / 0.187 |
| **Approximate KL** | 1 / 2 / 3 | 0.0113 / 0.0082 / 0.0073 | 0.0108 / 0.0094 / 0.0079 | 0.0110 / 0.0091 / 0.0076 | 0.0118 / 0.0102 / 0.0084 |
| **Entropy** | 1 / 2 / 3 | 0.493 / 0.563 / 0.575 | 0.501 / 0.602 / 0.729 | 0.496 / 0.579 / 0.654 | 0.490 / 0.548 / 0.589 |
| **Clip Fraction** | 1 / 2 / 3 | 0.056 / 0.068 / 0.070 | 0.054 / 0.069 / 0.079 | 0.054 / 0.067 / 0.074 | 0.057 / 0.073 / 0.082 |
| **Gradient Norm** | 1 / 2 / 3 | 1.080 / 1.144 / 1.041 | 1.062 / 1.134 / 1.096 | 1.056 / 1.121 / 1.083 | 1.089 / 1.201 / 1.149 |

### Diagnostic Takeaways:
- Internal optimization dynamics (gradient norms, value explained variance $\approx 0.70$, advantage magnitude, clip fractions) are nearly identical across all seeds.
- Despite having healthy internal optimization metrics, the policy trajectories wander in policy space, with some seeds suffering catastrophic mid-training dips (Seed 3) and others remaining flat (Seed 1).

---

## Artifacts & Plots

Generated artifacts:
- [`results/ppo_multiseed_v1/manifest.json`](../results/ppo_multiseed_v1/manifest.json)
- [`results/ppo_multiseed_v1/aggregate/cross_seed_summary.json`](../results/ppo_multiseed_v1/aggregate/cross_seed_summary.json)
- [`results/ppo_multiseed_v1/aggregate/paired_results.json`](../results/ppo_multiseed_v1/aggregate/paired_results.json)
- [`results/ppo_multiseed_v1/aggregate/replication_analysis.json`](../results/ppo_multiseed_v1/aggregate/replication_analysis.json)

Machine-generated plots:
- **Plot A**: Multi-seed DEV learning curves (`results/ppo_multiseed_v1/aggregate/plots/A_multiseed_dev_learning_curves.png`)
- **Plot B**: Cross-seed mean curve with std dev (`results/ppo_multiseed_v1/aggregate/plots/B_cross_seed_mean_curve.png`)
- **Plot C**: Expert agreement curves (`results/ppo_multiseed_v1/aggregate/plots/C_expert_agreement.png`)
- **Plot D**: KL divergence from warm start (`results/ppo_multiseed_v1/aggregate/plots/D_kl_from_warmstart.png`)
- **Plot E**: Warm-start action agreement (`results/ppo_multiseed_v1/aggregate/plots/E_warmstart_agreement.png`)
- **Plot F**: Action-category drift at iteration 320 (`results/ppo_multiseed_v1/aggregate/plots/F_action_category_drift.png`)
- **Plot G**: PPO optimization diagnostics (`results/ppo_multiseed_v1/aggregate/plots/G_ppo_diagnostics.png`)

---

## Limitations

1. **Small Sample of Training Seeds ($N=4$)**: While 4 seeds provide definitive evidence that Seed 0's improvement does not reliably reproduce, $N=4$ is still a small sample for estimating population-level training distributions.
2. **DEV Split Only**: Benchmark v1 TEST set remained locked throughout this study and was not used to select any checkpoint.

---

## Conclusion

The evidence best supports **Outcome C (High Training Trajectory Variance & Seed 0 Excursion)**:
- The transient improvement at iteration 80 observed in Experiment 2 was a stochastic excursion specific to Seed 0 and **did not reproduce** across independent training seeds (Seed 1 was flat, Seed 2 degraded slightly, Seed 3 collapsed mid-training).
- Unconstrained PPO optimization exhibits high variance across random seeds while consistently producing massive policy drift away from the warm start (KL reaching ~1.0–1.4 across all seeds).

---

## Recommendation for Experiment 4

**Recommended Intervention: Experiment 4 — PPO Policy Anchoring (KL Penalty to BC Prior)**

### Hypothesis:
Unconstrained PPO optimization on sparse placement rewards allows the policy to drift uncontrollably away from the competent imitation prior into degenerate tempo dynamics (rolling and ending turns erratically). Adding a **KL divergence regularization penalty toward the frozen BC prior** ($\mathcal{L}_{\text{PPO}} - \beta \text{KL}(\pi_{\text{BC}} \parallel \pi_\theta)$) will anchor the policy near the high-performing imitation manifold while allowing PPO to optimize placement within a bounded trust region.
