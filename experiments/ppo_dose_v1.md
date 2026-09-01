# Replay Experiment 5 — KL Anchoring Dose-Response

Date: 2026-09-01 · Split: **DEV only** (Benchmark v1 TEST never run) ·
Artifacts: [`results/ppo_dose_v1/`](../results/ppo_dose_v1/) ·
Contract: [`contract.json`](../results/ppo_dose_v1/contract.json) ·
Manifest: [`manifest.json`](../results/ppo_dose_v1/manifest.json)

## Question

Experiment 4b showed β = 0.1 stabilizes PPO but does not reliably improve on
the BC warm start (~6.550). With only two fixed anchoring strengths tested, we
do not know whether β = 0.1 is stronger than necessary.

> **What is the weakest KL anchoring strength that preserves PPO stability while
> allowing enough policy movement to beat the BC warm start?**

Four-way comparison: β ∈ {0.00, 0.01, 0.03, 0.10}. New training for
β = 0.01 and β = 0.03 only (8 trajectories); β = 0.0 and β = 0.1 reused from
Experiment 4b.

## Pre-flight: full test suite

Before training, `pytest tests/` on the canonical branch:

```
383 passed in 21.41s
```

## Frozen contract (identical warm start for all arms)

| Field | Value |
|---|---|
| Warm-start `parameter_sha256` | `c85de276471872c8d5c5365bd3550e9a31ae2cbc5046244f2869392dd0cacf1c` |
| Python | 3.12.3 |
| Torch | 2.13.0+cpu (CPU backend) |
| Runtime fingerprint | `b4907d0a1782dc0a66c1bb09a01712c7f97e66bf228e4112861f64874d1e52b0` |
| PPO config hash | `69218321dfaf64c3a95d559905f6e5d4952fa5cab3a84d4f6c43159945339030` |
| DEV primary seeds | 10,550,000–10,550,999 (1000 games vs 7× greedy) |
| BC warm-start baseline | **6.550** (cross-seed iter-0 mean) |

All new-arm gates passed: every `iter_000` checkpoint matched the warm-start
hash; iter-0 DEV spot-checks returned 6.550 for all seeds.

## Commands

```bash
# Step 1 — full test suite (required before training)
pytest tests/

# Step 2 — train β=0.01 and β=0.03 (8 trajectories)
python scripts/train_dose_ab.py

# Step 3 — DEV eval + policy drift (new arms only)
python scripts/eval_dose_ab.py

# Step 4 — four-arm analysis (reuses Experiment 4b β=0 / β=0.1)
python scripts/ppo_dose_report.py
python scripts/ppo_dose_manifest.py
```

β = 0.0 and β = 0.1 results are read from `results/ppo_matched_ab_v1/` — not
retrained.

## Cross-seed summary (primary DEV, 1000 games vs greedy)

Lower average placement is better. Δ vs BC = mean placement − 6.550.

| Iter | β | Mean | Std | Worst | Δ vs BC | Mean KL | Exp% |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 6.550 | 0.000 | 6.550 | +0.000 | 0.000 | 84.4% |
| 0 | 0.01 | 6.550 | 0.000 | 6.550 | +0.000 | 0.000 | 84.4% |
| 0 | 0.03 | 6.550 | 0.000 | 6.550 | +0.000 | 0.000 | 84.4% |
| 0 | 0.10 | 6.550 | 0.000 | 6.550 | +0.000 | 0.000 | 84.4% |
| 40 | 0.00 | 6.563 | 0.077 | 6.633 | +0.013 | 0.768 | 64.9% |
| 40 | 0.01 | 6.625 | 0.068 | 6.732 | +0.075 | 0.175 | 81.4% |
| 40 | 0.03 | 6.602 | 0.021 | 6.622 | +0.052 | 0.192 | 79.7% |
| 40 | 0.10 | 6.655 | 0.065 | 6.766 | +0.105 | 0.063 | 83.0% |
| 80 | 0.00 | 6.757 | **0.177** | **7.061** | +0.207 | 0.962 | 62.5% |
| 80 | 0.01 | 6.549 | 0.027 | 6.579 | −0.001 | 0.476 | 69.0% |
| 80 | 0.03 | 6.602 | 0.032 | 6.651 | +0.052 | 0.318 | 76.6% |
| 80 | 0.10 | 6.584 | **0.043** | 6.653 | +0.034 | **0.058** | 84.1% |
| 160 | 0.00 | 6.771 | **0.228** | **7.042** | +0.221 | 1.250 | 50.9% |
| 160 | 0.01 | 6.578 | 0.093 | 6.689 | +0.028 | 0.891 | 58.0% |
| 160 | 0.03 | 6.570 | **0.018** | 6.584 | +0.020 | 0.291 | 76.0% |
| 160 | 0.10 | 6.582 | 0.035 | 6.610 | +0.032 | **0.055** | 83.8% |
| 320 | 0.00 | 6.718 | 0.050 | 6.792 | +0.168 | 2.012 | 45.7% |
| 320 | 0.01 | **6.531** | **0.234** | **6.894** | **−0.019** | 0.944 | 56.9% |
| 320 | 0.03 | **6.540** | **0.028** | 6.573 | **−0.010** | 0.335 | 80.3% |
| 320 | 0.10 | 6.602 | 0.044 | 6.673 | +0.052 | **0.058** | 83.3% |

## Per-seed iter-320 vs BC (Δ = placement − 6.550; negative = better)

| Seed | β=0.00 | β=0.01 | β=0.03 | β=0.10 |
|---|---|---|---|---|
| 0 | +0.141 | **−0.308** | +0.023 | +0.123 |
| 1 | +0.183 | +0.344 | −0.014 | +0.012 |
| 2 | +0.242 | −0.037 | **−0.052** | +0.053 |
| 3 | +0.107 | −0.076 | +0.003 | +0.019 |

## Key observations

1. **β = 0.03 is the best stability/exploration tradeoff.** At iter 320 it
   beats BC on cross-seed mean (6.540 vs 6.550) with the lowest non-zero-β
   variance (std 0.028) and moderate KL (0.335). Expert agreement stays at
   80.3% — much better than unconstrained (45.7%) while allowing more movement
   than β = 0.1 (KL 0.058).

2. **β = 0.01 beats BC on mean but is not reliable.** Cross-seed mean 6.531
   (−0.019 vs BC) comes with high variance (std 0.234) and a seed-1 excursion
   to +0.344 vs BC. Seed 0 shows a large improvement (−0.308) but seed 1
   regresses (+0.344) — classic high-variance PPO behavior at weak anchoring.

3. **β = 0.1 remains the strongest drift control** (KL 0.058, expert 83.3%) but
   sits above BC at full budget (+0.052). It prevents catastrophic seeds but
   over-constrains policy movement for improvement.

4. **Unconstrained PPO still destroys BC** (+0.168 at iter 320, KL 2.012) —
   replicating Experiment 4b under the four-arm view.

5. **Mid-training stability ordering holds:** at iter 80/160, β = 0.03 matches
   β = 0.1 stability (std 0.032/0.018 vs unconstrained 0.177/0.228) while
   maintaining higher KL and lower expert agreement — i.e., more exploration.

## Does any fixed β reliably beat BC?

**No — not across seeds.**

| β | Mean iter-320 | Beats BC on mean? | Seeds beating BC (Δ < −0.01) | Catastrophic seeds? |
|---|---|---|---|---|
| 0.00 | 6.718 | No (+0.168) | 0/4 | Yes (seed 1 iter 80: 7.06) |
| 0.01 | 6.531 | Yes (−0.019) | 2/4 (seeds 0, 2) | Yes (seed 1: +0.344) |
| 0.03 | 6.540 | Yes (−0.010) | 2/4 (seeds 1, 2) | No (worst +0.023) |
| 0.10 | 6.602 | No (+0.052) | 0/4 | No |

β = 0.03 is the closest to the pre-specified success criterion (stable mean
below BC, moderate KL, reasonable expert agreement) but does not replicate
across ≥3 training seeds.

## Outcome classification: **B**

Intermediate β improves the stability/exploration tradeoff vs β = 0.1 — β = 0.03
achieves near-BC mean with much lower drift constraint — but **no fixed β
reliably beats BC across seeds**.

## Recommended next action (ONE experiment only)

**Consider Experiment 6 — scheduled KL anchoring starting from β = 0.03:**

- Begin with β = 0.03 (or 0.03→0.01 schedule) during the high-drift window
  (iter 1–80) to retain mid-training stability.
- Anneal β toward 0 after iter 80–160 to allow late-training improvement over BC.
- Do NOT run until explicitly approved. Do NOT start LR/entropy/β-grid sweeps.

If Experiment 6 also fails to beat BC reliably across seeds, **stop fixed-PPO
tuning** and pivot to Simulator Fidelity Phase 2.

## Hard stop reminder

This was the last pre-specified fixed-β tuning experiment. No further β grids,
LR sweeps, or entropy sweeps unless Experiment 6 produces convincing BC-beating
evidence.

## Limitations

- Four training seeds — sufficient for stability signals, not for definitive
  multi-seed BC-beating claims.
- β = 0.0 and β = 0.1 reused from Experiment 4b (same warm start, same contract).
- Checkpoint binaries gitignored; fingerprints in manifest.
- TEST not run.
