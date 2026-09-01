# Replay Experiment 4b — Matched Anchored vs Unanchored PPO

Date: 2026-09-01 · Split: **DEV only** (Benchmark v1 TEST never run) ·
Artifacts: [`results/ppo_matched_ab_v1/`](../results/ppo_matched_ab_v1/) ·
Contract: [`contract.json`](../results/ppo_matched_ab_v1/contract.json) ·
Manifest: [`manifest.json`](../results/ppo_matched_ab_v1/manifest.json)

## Question

Experiment 4 showed KL anchoring controls drift, but compared against historical
unconstrained runs with a different BC warm start and runtime. This experiment
asks:

> **Does β = 0.1 reduce cross-seed instability and catastrophic policy drift while
> preserving placement compared with a perfectly matched β = 0.0 control?**

Eight matched trajectories: training seeds {0,1,2,3} × {β=0.0, β=0.1}. The only
within-pair variable is the KL coefficient.

## Frozen contract (identical for all 8 runs)

| Field | Value |
|---|---|
| Warm-start `parameter_sha256` | `c85de276471872c8d5c5365bd3550e9a31ae2cbc5046244f2869392dd0cacf1c` |
| Python | 3.12.3 |
| Torch | 2.13.0+cpu (CPU backend) |
| Training commit | `17881b7a4592a519c4d17b70ecd96ffd24670507` |
| Runtime fingerprint | `b4907d0a1782dc0a66c1bb09a01712c7f97e66bf228e4112861f64874d1e52b0` |
| PPO config hash | `69218321dfaf64c3a95d559905f6e5d4952fa5cab3a84d4f6c43159945339030` |
| DEV primary seeds | 10,550,000–10,550,999 (1000 games vs 7× greedy) |

All reproducibility gates passed: every `iter_000` checkpoint matched the warm
start hash; iter-0 DEV placement sequences were identical between matched arms
for each seed.

## Cross-seed summary (primary DEV, 1000 games vs greedy)

| Iter | β=0.0 mean | β=0.0 std | β=0.0 worst | β=0.0 KL | β=0.1 mean | β=0.1 std | β=0.1 worst | β=0.1 KL | Exp% β=0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6.550 | 0.000 | 6.550 | 0.000 | 6.550 | 0.000 | 6.550 | 0.000 | 84.4% |
| 40 | 6.563 | 0.077 | 6.633 | 0.768 | 6.655 | 0.065 | 6.766 | 0.063 | 83.0% |
| 80 | 6.757 | **0.177** | **7.061** | 0.962 | 6.584 | **0.043** | 6.653 | **0.058** | 84.1% |
| 160 | 6.771 | **0.228** | **7.042** | 1.250 | 6.582 | **0.035** | 6.610 | **0.055** | 83.8% |
| 320 | 6.718 | 0.050 | 6.792 | **2.012** | **6.602** | **0.044** | 6.673 | **0.058** | 83.3% |

Lower average placement is better. Cross-seed std and worst-seed placement
measure stability; KL and expert agreement measure behavioral drift.

## Per-seed paired comparisons (anchored − unconstrained)

Positive Δ = anchored places **worse**. All use paired bootstrap on identical
1000 DEV games.

### Seed 0
| Iter | Δ | 95% CI | Reading |
|---|---|---|---|
| 40 | +0.023 | [−0.134, +0.180] | no clear difference |
| 80 | −0.036 | [−0.183, +0.113] | no clear difference |
| 160 | +0.031 | [−0.125, +0.191] | no clear difference |
| 320 | −0.018 | [−0.171, +0.138] | no clear difference |

### Seed 1
| Iter | Δ | 95% CI | Reading |
|---|---|---|---|
| 40 | +0.022 | [−0.133, +0.182] | no clear difference |
| 80 | **−0.408** | **[−0.550, −0.268]** | **anchored better** |
| 160 | **−0.432** | **[−0.579, −0.287]** | **anchored better** |
| 320 | **−0.171** | **[−0.317, −0.027]** | **anchored better** |

Unconstrained seed 1 collapsed to 7.06 avg at iter 80 and 7.04 at iter 160;
anchored seed 1 stayed at 6.65 / 6.61.

### Seed 2
| Iter | Δ | 95% CI | Reading |
|---|---|---|---|
| 40 | **+0.333** | **[+0.176, +0.489]** | **unconstrained better** |
| 80 | −0.145 | [−0.291, +0.002] | no clear difference (CI touches 0) |
| 160 | +0.071 | [−0.054, +0.194] | no clear difference |
| 320 | **−0.189** | **[−0.306, −0.073]** | **anchored better** |

### Seed 3
| Iter | Δ | 95% CI | Reading |
|---|---|---|---|
| 40 | −0.011 | [−0.166, +0.149] | no clear difference |
| 80 | −0.103 | [−0.222, +0.020] | no clear difference |
| 160 | **−0.427** | **[−0.545, −0.310]** | **anchored better** |
| 320 | −0.088 | [−0.205, +0.031] | no clear difference |

## Observations

1. **Drift control replicates under matched conditions.** At iter 320, anchored
   cross-seed mean KL is 0.058 vs unconstrained 2.012 (~35×). Expert agreement
   stays ~83% vs ~46% for unconstrained.

2. **Anchoring sharply reduces mid-training cross-seed placement variance.**
   At iter 80, cross-seed std is 0.043 (anchored) vs 0.177 (unconstrained) —
   a 4.1× ratio. At iter 160: 0.035 vs 0.228 (6.5×). This is the stability
   benefit Experiment 4 could not establish causally.

3. **Catastrophic seed-specific excursions are blocked.** Seed 1 unconstrained
   reached 7.06 avg at iter 80; anchored seed 1 stayed at 6.65. Seed 3
   unconstrained hit 6.95 at iter 160; anchored stayed at 6.52.

4. **Mean placement at full budget is preserved or slightly improved.** Cross-seed
   mean at iter 320: 6.602 (anchored) vs 6.718 (unconstrained). Paired mean Δ
   across seeds at iter 320: −0.117 (anchored better on average).

5. **Seed 2 iter-40 is the one clear placement cost.** Unconstrained scored
   6.433 vs anchored 6.766 (+0.333, CI excludes zero). This is the main case
   where anchoring blocked early exploration that helped one seed.

6. **Outcome classification: A (stabilizes and preserves performance).**
   Anchoring materially lowers drift and mid/late cross-seed variance without
   clearly hurting mean placement at 5,120 episodes.

## Conclusion

Under a clean matched A/B design — same warm start, same runtime, same DEV
seeds, same PPO recipe — **β = 0.1 turns PPO from a high-variance optimizer
into a repeatable one** while keeping DEV placement flat or slightly better at
full budget. The causal claim from Experiment 4 ("anchoring controls drift") is
confirmed; the previously untrusted placement comparison is now resolved in
favor of anchoring on stability grounds, with seed 2 iter-40 as the main
counterexample.

## Recommended Experiment 5

**Weaker / scheduled anchoring** — start with β = 0.1 (or stronger) during the
high-drift window (iter 1–80), then anneal β → 0 to recover the exploratory
regime that helped unconstrained seed 2 at iter 40 while retaining mid-training
stability.

Do not run Experiment 5 yet.

## Limitations

- Single β value (0.1), four training seeds — not a dose–response curve.
- Warm-start hash differs from Experiment 2 historical BC; this experiment
  defines its own immutable warm start for causal comparison.
- CPU-only Torch 2.13 / Python 3.12 — results are internally matched but not
  directly comparable to Experiment 2's CUDA/Python 3.11 historical runs.
- Checkpoint binaries gitignored; fingerprints in manifest.
- TEST not run.
