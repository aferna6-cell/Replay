# Replay Experiment 1 — PPO Degradation Diagnosis

Date: 2026-08-28 · Split: **DEV** (never Benchmark v1 TEST seeds) ·
Artifacts: [`results/ppo_diagnosis_v1/`](../results/ppo_diagnosis_v1/)

## Question

Baseline Experiment v1 measured that PPO makes its BC + DAgger warm start
**worse by 0.271 average placements** (paired 95% CI [0.152, 0.391], 1000
TEST games). When and how does the shipped PPO recipe destroy that value?

No hyperparameter, algorithm, reward, or network change was made. This is a
measurement of the existing recipe's trajectory through parameter space.

## Setup

- **PPO recipe — frozen, exactly as shipped**: warm start from the baseline
  `policy_bc.pt` (`bd3a4386…`), 40 iterations × 16 episodes, seed 0, shaping
  1.0 annealed, lr 3e-4 AdamW wd 1e-4, γ 0.999, λ 0.95, clip 0.2, entropy
  0.01, value coef 0.5, 4 PPO epochs, league snapshot every 8 iterations
  (max 5), unchanged `mixed_field` opponents.
- **Checkpoints**: iterations 0, 1, 2, 4, 8, 12, 16, 24, 32, 40. Iteration 0
  is the exact warm-start weights before any PPO update. Snapshotting and
  diagnostics draw no RNG; a test trains with and without instrumentation and
  requires bit-identical final weights.
- **DEV split** (`ml/seeds.py`): reserved interval **[10,550,000 –
  10,599,999]**, disjoint from the TEST interval [10,250,000 – 10,299,999]
  and from every training scheme under the documented bounds (for any base
  seed, a PPO run needs ≥549,970 episodes and a midgame run ≥49,686 lobbies
  to reach it). Evaluation via `python -m ml.dev_benchmark` — same metrics as
  Benchmark v1, output labeled "NOT Benchmark v1 test results",
  `evaluation_split: "dev"`, and structurally unpairable with TEST results.
- **DEV evaluation**: 200 games per checkpoint per field, seeds
  10,550,000–10,550,199, identical for every checkpoint (paired game-by-game).
- **Frozen drift corpus**: 4,440 recruit-phase states from 100 greedy lobbies
  at DEV seed 10,590,000 (fingerprint `2ec217b353bd`), each labeled with the
  greedy expert's action. No TEST seeds.
- **Benchmark v1 TEST was not run at any point in this experiment.**

## Reproduction control

**The baseline recipe reproduced exactly.** The diagnostic run's final model
has **bit-identical parameters** to the Baseline Experiment v1 PPO checkpoint,
and re-saving it under the same filename reproduces the baseline's SHA-256
`7d240bbcfdc04b5f…` byte for byte.

The initial SHA comparison appeared to fail, and the cause is worth recording:
**`torch.save` embeds an archive name derived from the output filename**, so
the same model written to `ppo_repro.pt` and to `policy_ppo.pt` produces
different bytes (311,626 vs 311,669 — 43 bytes across 43 zip entries). This
means Benchmark v1's `checkpoint_sha256` is **filename-sensitive**: identical
models saved under different names fingerprint differently. It does not
invalidate any published result (every baseline checkpoint kept its name), but
a parameter-level hash would be the stricter identity. Recorded as a known
limitation, not fixed in this milestone.

## Training trajectory

Per-iteration optimization diagnostics (`train_diag.jsonl`, all 40 rows):

| iter | rollout avg | shaping | league | steps | π loss | v loss | entropy | approx KL | clip frac | grad norm |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.75 | 1.00 | 0 | 492 | −0.0035 | 0.131 | 0.456 | 0.0195 | 0.040 | 0.94 |
| 2 | 5.81 | 0.96 | 0 | 507 | −0.0015 | 0.072 | 0.394 | 0.0022 | 0.023 | 0.58 |
| 4 | 5.62 | 0.89 | 0 | 609 | +0.0158 | 0.077 | 0.427 | 0.0109 | 0.029 | 0.68 |
| 8 | 6.00 | 0.75 | 0 | 498 | −0.0024 | 0.055 | 0.477 | 0.0007 | 0.023 | 1.03 |
| 12 | 5.44 | 0.61 | 1 | 586 | −0.0054 | 0.063 | 0.569 | 0.0064 | 0.044 | 0.95 |
| 16 | 6.31 | 0.46 | 1 | 480 | −0.0032 | 0.057 | 0.564 | 0.0027 | 0.036 | 1.23 |
| 24 | 5.56 | 0.18 | 2 | 630 | −0.0110 | 0.051 | 0.504 | 0.0108 | 0.070 | 1.21 |
| 32 | 6.00 | 0.00 | 3 | 564 | −0.0058 | 0.054 | 0.544 | 0.0129 | 0.091 | 1.48 |
| 40 | 6.50 | 0.00 | 4 | 508 | −0.0056 | 0.067 | 0.515 | 0.0049 | 0.061 | 1.52 |

Means, iterations 1–8 → 9–40: entropy **0.436 → 0.507**, clip fraction
**0.034 → 0.061**, grad norm **0.728 → 1.168**, value loss 0.078 → 0.055,
approx KL 0.0107 → 0.0115. Training-field rollout placement stays in a noisy
4.88–6.50 band with no trend (that field is the mixed league, not the
evaluation field).

## DEV performance curve

200 games per checkpoint, same DEV seeds, vs 7× field
(`learning_curve.json`, plots in `plots/`):

| iter | Greedy avg | Top-4 | Random avg | Expert agree | Warm-start agree | KL from iter 0 | value mean±sd |
|---|---|---|---|---|---|---|---|
| 0 | 6.680 | 13.5% | 1.000 | 84.5% | 100.0% | 0.0000 | −0.092 ± 0.535 |
| 1 | 6.815 | 12.0% | 1.000 | 84.5% | 96.0% | 0.0108 | −0.232 ± 0.145 |
| 2 | 6.765 | 10.0% | 1.000 | 84.2% | 98.1% | 0.0219 | −0.125 ± 0.309 |
| 4 | 6.730 | 10.0% | 1.000 | 82.3% | 94.7% | 0.0615 | −0.051 ± 0.438 |
| 8 | 6.770 | 11.0% | 1.000 | 78.0% | 91.1% | 0.2156 | −0.041 ± 0.471 |
| 12 | 6.710 | 12.0% | 1.000 | 77.9% | 80.3% | 0.2160 | +0.081 ± 0.407 |
| 16 | 6.600 | 12.0% | 1.000 | 79.3% | 81.3% | 0.1975 | +0.077 ± 0.537 |
| 24 | 6.855 | 10.5% | 1.000 | 74.9% | 87.0% | 0.2399 | +0.090 ± 0.496 |
| 32 | 6.775 | 14.5% | 1.000 | 78.3% | 91.8% | 0.2363 | +0.028 ± 0.502 |
| 40 | 6.880 | 12.0% | 1.000 | 77.2% | 90.2% | 0.3714 | +0.030 ± 0.569 |

**The random field is saturated.** Every checkpoint — including iteration 0,
the untouched BC + DAgger clone — wins **200/200** games against 7 random
opponents. Placement is exactly 1.000 everywhere.

## Paired DEV comparisons vs the warm start

Greedy field, paired over identical seeds, deterministic paired bootstrap
(positive = checkpoint worse than the warm start):

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| iter 1 vs 0 | +0.135 | [−0.160, +0.425] | no clear difference |
| iter 4 vs 0 | +0.050 | [−0.085, +0.195] | no clear difference |
| iter 8 vs 0 | +0.090 | [−0.060, +0.245] | no clear difference |
| iter 16 vs 0 | −0.080 | [−0.325, +0.160] | no clear difference |
| iter 24 vs 0 | +0.175 | [−0.100, +0.450] | no clear difference |
| iter 40 vs 0 | +0.200 | [−0.040, +0.445] | no clear difference |

Pooled pre-specified contrasts over the same seeds:

- late (24,32,40) − early (0,1,2): **+0.083**, 95% CI [−0.112, +0.275]
- trend slope: **+0.0027** placements/iteration, 95% CI [−0.0032, +0.0085]
  (**+0.107** over 40 iterations)

**Power.** The paired SD of (iter 40 − iter 0) is 1.776, so at 200 games the
smallest effect DEV can resolve is ≈ **0.246** placements. The published TEST
degradation is **0.271** — right at that edge. The DEV point estimate
(**+0.200**) is entirely consistent with it. A flat DEV curve here is
**underpowered evidence, not a refutation** of the TEST result.

## Policy drift

- **Expert agreement falls 84.5% → 77.2%**: by iteration 40 roughly one
  decision in four differs from the greedy expert the policy was cloned from.
  It is flat through iteration 2 (84.5, 84.5, 84.2), then declines from
  iteration 4 (82.3) and drops most sharply by iteration 8 (78.0).
- **Warm-start agreement falls to 90.2%**, and it moves immediately: **96.0%
  after a single PPO iteration**. It is non-monotonic (80.3% at iter 12,
  91.8% at iter 32) — decisions churn rather than settling.
- **KL from the warm start rises 0 → 0.371**, with the largest single jump
  between iterations 4 and 8 (0.0615 → 0.2156, ~3.5×).
- **Value head drifts from −0.092 to +0.030** and its spread collapses then
  recovers (sd 0.535 → 0.145 at iter 1 → 0.569 at iter 40).

## Key observations

Facts supported directly by the measurements:

1. **There is no iteration at which performance clearly breaks.** Every
   paired DEV comparison against the warm start, and both pooled trend
   contrasts, have confidence intervals containing zero. Degradation is not
   localized to the first updates, to league entry, or to the shaping decay.
2. **Degradation is not caused by league opponents entering.** The first
   league snapshot is taken at iteration 8 and first occupies opponent seats
   at iteration 9, but expert agreement has already fallen 84.5% → 78.0% and
   KL has already risen to 0.216 **before** any league opponent is used.
3. **The policy drifts substantially while its outcome does not change.**
   7.3 points of expert agreement and 0.37 nats of KL bought a DEV placement
   change of +0.200 ± 0.245 — i.e. behavior changes a great deal, results
   barely move.
4. **Entropy rises rather than falls** (0.436 → 0.507 mean), and gradient
   norm rises to the 1.0 clipping threshold and beyond (0.73 → 1.17). The
   policy is becoming *less* decisive over training, not more.
5. **The "PPO dominates random opponents" property is inherited, not
   learned.** The BC + DAgger warm start already wins 200/200 versus a random
   field, as does every later checkpoint. The random field has zero
   diagnostic resolution and cannot show an opponent-specialization tradeoff
   in either direction.
6. **The recipe is exactly reproducible** (bit-identical parameters), so the
   baseline TEST number is not a reproducibility artifact.
7. **200 DEV games per checkpoint is underpowered** for effects of the size
   we care about (~0.25 placements).

## Hypotheses (not established by this data)

- **H1 — the learning signal is too weak at this scale.** 640 episodes of a
  terminal, zero-mean placement reward may carry too little information to
  direct a 28-action policy, so updates diffuse the policy away from the BC
  prior without improving it. *Consistent with:* rising entropy, eroding
  expert agreement, non-monotonic warm-start agreement, flat outcomes,
  near-zero policy losses (|π loss| ≈ 0.005).
- **H2 — the BC prior is a good local optimum and PPO's exploration only
  leaves it.** Any movement away from an 84.5%-agreement clone costs a little
  and gains nothing at this budget.
- **H3 — training/evaluation field mismatch.** The rollout field mixes league
  snapshots and random seats while evaluation is pure greedy. *Weakened by:*
  observation 2 — drift precedes league entry — but the `mixed_field` random
  seats are present from iteration 1.
- **H4 — value-function quality.** The value head's scale and spread move
  considerably (sd 0.535 → 0.145 → 0.569); poor value estimates would make
  GAE advantages noisy, which fits H1.

These need controlled experiments, not argument.

## Decision tree for Experiment 2

The measured pattern is: **flat DEV placement through 40 iterations, with
large, immediate, non-localized policy drift and rising entropy, on an
exactly reproduced recipe.** Applying the pre-agreed branches:

- *"Greedy performance falls immediately while KL jumps"* → **not matched.**
  KL rises early but placement does not fall measurably.
- *"Stable until league opponents enter"* → **not matched.** Drift starts at
  iteration 1 and most of the KL jump precedes league use.
- *"Random improves while Greedy deteriorates"* → **cannot be tested.** The
  random field is saturated at 1.000 from iteration 0.
- *"Flat through 40 iterations → the TEST degradation may reflect
  variance"* → **partially matched, with a correction.** DEV is flat, but DEV
  at 200 games cannot resolve a 0.271 effect. The honest statement is that
  the degradation is *small, diffuse, and not localizable*, not that it is
  noise.
- *"Expert agreement collapses but placement stays stable"* → **matched, and
  this is the dominant signal.**

**Recommended Experiment 2** (do not run yet): treat *"is there any learning
signal at all?"* as the question, not *"which hyperparameter is wrong."* The
best-supported next intervention is a **signal-strength / training-budget
study**: hold the algorithm fixed and vary only the training budget (e.g.
640 → several thousand episodes) with checkpoints evaluated on DEV, to
establish whether placement ever moves in either direction. Pair it with a
methodological fix that this experiment proved necessary: **raise DEV
evaluation to ≥1000 games per compared checkpoint** (SE 0.126 → 0.056), or
the study will again be unable to see the effect size in question.

Two supporting measurements worth taking in the same experiment, both cheap:
replace the saturated random field with a *discriminating* second field (the
random field yields no information), and record which decision categories
(buy / play / sell / roll / level / freeze / end) the drifting 15–25% of
actions fall into — observation 3 says behavior is changing a lot without
changing outcomes, and knowing *which* decisions change is the prerequisite
for any targeted fix.

Nothing was tuned in this experiment, and Benchmark v1 TEST was not touched.
