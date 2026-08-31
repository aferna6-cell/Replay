# Replay Experiment 3 — Multi-Seed PPO Budget Replication

Date: 2026-08-31 · Split: **DEV only** (Benchmark v1 TEST never run) ·
Artifacts: [`results/ppo_multiseed_v1/`](../results/ppo_multiseed_v1/) ·
Manifest: [`manifest.json`](../results/ppo_multiseed_v1/manifest.json) ·
Seed 0 reference (unmodified): [`results/ppo_budget_v1/`](../results/ppo_budget_v1/)

## Question

Experiment 2 ([`experiments/ppo_budget_study_v1.md`](ppo_budget_study_v1.md))
found — on a single training seed (0) — that PPO improves significantly over
its warm start at 1,280 episodes (iteration 80), then that gain decays and is
gone by 5,120 episodes (iteration 320), while behavioral drift from the warm
start grows the entire time. Experiment 2's own stated limitation was that a
single trajectory cannot distinguish a reproducible budget effect from a
lucky excursion of one random seed.

**Does the transient improvement (and the later drift/regression) replicate
across independent training seeds, or was seed 0's iteration-80 gain a
one-off?** One variable changes here: the PPO training seed (1, 2, 3 new;
seed 0 reused as-is). This is a replication study, not a tuning exercise —
nothing about the algorithm, warm start, or evaluation protocol changes.

## Historical observation (Experiment 2, seed 0 — summarized, not rewritten)

At the identical checkpoints and DEV protocol used here:

| iter | episodes | Greedy avg (1000 games) | vs iter 0 | Expert agree | Warm-start agree | KL |
|---|---|---|---|---|---|---|
| 0 | 0 | 6.554 | — | 84.5% | 100.0% | 0.000 |
| 40 | 640 | 6.761 | +0.207 [+0.093, +0.322] worse | 77.2% | 90.2% | 0.371 |
| 80 | 1,280 | **6.325** | **−0.229 [−0.392, −0.061] better** | 80.9% | 77.8% | 0.249 |
| 160 | 2,560 | 6.435 | −0.119 [−0.245, +0.008] | 74.3% | 73.1% | 0.484 |
| 320 | 5,120 | 6.606 | +0.052 [−0.104, +0.210] | 42.6% | 40.8% | 1.171 |

Experiment 2 classified this as a **U-shaped, transient improvement**: a
real, statistically clear gain at 2× budget that does not survive to 8×,
against a backdrop of drift that grows monotonically the whole time
(warm-start agreement 100% → 40.8%, KL 0 → 1.17, and adoption of "freeze" —
an action the greedy expert never takes — at iteration 320). These committed
Experiment 2 artifacts are used here exactly as-is; nothing in
`results/ppo_budget_v1/` was regenerated or altered.

## Setup — exact frozen PPO recipe

Every element of the Experiment 2 recipe is unchanged except the training
seed:

- **Architecture / optimizer / hyperparameters**: identical PolicyNet, AdamW
  (lr 3e-4, weight decay 1e-4), γ=0.999, λ=0.95, PPO clip 0.2, entropy
  coefficient 0.01, value coefficient 0.5, 4 PPO epochs, minibatch 256, grad
  clip norm 1.0, league snapshot every 8 iterations (max 5).
- **Training protocol**: 16 episodes/iteration, 320 iterations, 5,120 total
  training episodes per seed. `--shaping 1.0 --shaping-horizon 40` — the
  identical fixed shaping schedule as Experiment 2: iterations 1–40 reproduce
  the historical schedule (reaching zero at iteration 28) and shaping stays
  exactly 0.0 afterwards, regardless of the 320-iteration total horizon.
- **Checkpoints**: saved at iterations {0, 40, 80, 160, 320} (cumulative
  episodes {0, 640, 1280, 2560, 5120}) for each of seeds 1, 2, 3.
- **The only experimental variable**: PPO training seed. Seeds 1, 2, 3 are
  new; seed 0 is the existing, committed Experiment 2 trajectory, reused
  as-is (not retrained, not altered).
- **Seed reservation check**: every new seed's PPO episode-seed span
  (`ml.seeds.ppo_episode_seed(seed, k)` for k = 1..5120) was verified
  programmatically, before training, to fall entirely outside both the DEV
  interval `[10,550,000, 10,599,999]` and the TEST interval
  `[10,250,000, 10,299,999]` (seed 1: `[1,000,004, 1,005,123]`; seed 2:
  `[2,000,007, 2,005,126]`; seed 3: `[3,000,010, 3,005,129]`).

**Exact training commands** (identical across seeds except `--seed` and
output paths):

```
python -m ml.train_ppo --iters 320 --episodes 16 --seed 1 --shaping 1.0 \
  --shaping-horizon 40 --from-bc ml/policy_bc.pt \
  --out results/ppo_multiseed_v1/seed_1/final.pt \
  --save-iters 0,40,80,160,320 \
  --save-dir results/ppo_multiseed_v1/seed_1/checkpoints \
  --diag-log results/ppo_multiseed_v1/seed_1/train_diag.jsonl

python -m ml.train_ppo --iters 320 --episodes 16 --seed 2 --shaping 1.0 \
  --shaping-horizon 40 --from-bc ml/policy_bc.pt \
  --out results/ppo_multiseed_v1/seed_2/final.pt \
  --save-iters 0,40,80,160,320 \
  --save-dir results/ppo_multiseed_v1/seed_2/checkpoints \
  --diag-log results/ppo_multiseed_v1/seed_2/train_diag.jsonl

python -m ml.train_ppo --iters 320 --episodes 16 --seed 3 --shaping 1.0 \
  --shaping-horizon 40 --from-bc ml/policy_bc.pt \
  --out results/ppo_multiseed_v1/seed_3/final.pt \
  --save-iters 0,40,80,160,320 \
  --save-dir results/ppo_multiseed_v1/seed_3/checkpoints \
  --diag-log results/ppo_multiseed_v1/seed_3/train_diag.jsonl
```

All three seeds ran as independent OS processes (no shared mutable state,
no cross-seed RNG coupling — each process seeds `torch` and Python's `random`
independently via its own `--seed`) to reduce wall-clock time; this does not
change any seed's own trajectory (verified by the iteration-0 reproduction
gate below and by each seed producing the exact same warm start).

## Training-seed control — why the BC warm start is identical across seeds

Every new seed starts from the **same, frozen** BC+DAgger checkpoint used in
Experiments 1 and 2. `ml/policy_bc.pt` is gitignored per repo convention and
was not present locally at the start of this experiment, so it was
**reproduced** by re-running the exact historical command recorded in
`results/benchmark_v1/manifest.json`:

```
python -m ml.bc --lobbies 150 --epochs 6 --dagger-rounds 2 --dagger-lobbies 80 \
  --seed 0 --out ml/policy_bc.pt
```

The reproduced checkpoint's `parameter_sha256` —
**`094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b`** —
matched the historical value recorded in the Experiment 2 manifest
**exactly** (bitwise-identical parameters) before any Experiment 3 training
began; the training log (decisions collected, per-epoch imitation accuracy,
DAgger round sizes, warm-start DEV placement) also matched the committed
`results/benchmark_v1/logs/train_bc_dagger.log` line for line. The frozen
4,440-state policy-drift diagnostic corpus was likewise verified by
fingerprint (`2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e`)
to match the historical value before use, rather than being regenerated.

For every new seed, the reproduction gate (iteration-0 checkpoint
`parameter_sha256` must equal the frozen warm start) **passed**:

| Seed | iter0 `parameter_sha256` | Matches warm start |
|---|---|---|
| 1 | `094417bdcaa7af62…` | ✔ |
| 2 | `094417bdcaa7af62…` | ✔ |
| 3 | `094417bdcaa7af62…` | ✔ |

Because every seed's iteration-0 parameters are bitwise identical, any
divergence measured from iteration 40 onward is attributable purely to the
PPO training seed (which controls episode/rollout stochasticity and league
opponent sampling), not to a different starting point.

## A genuine finding en route: a non-terminating episode at seed 1 / iteration 320

While evaluating seed 1's iteration-320 checkpoint, `ml.benchmark`'s
integrity check (which refuses to silently score an episode that does not
terminate within 400 decisions) raised on DEV seed 10,550,123. Direct
inspection showed the deterministic (argmax) policy repeatedly selecting
**"freeze"** — an idempotent shop toggle — forever: once a fully-drifted
policy's greedy decision boundary settles on "freeze" regardless of the
resulting (alternating) observation, a deterministic policy is mathematically
guaranteed to loop. This happened on 5 of 1000 greedy-field DEV seeds and 2
of 500 mixed-field DEV seeds for this one checkpoint, and on no checkpoint of
any other seed. This is treated as a genuine result about severe drift (seed
1 drifts further than any other trajectory here — see below), not a pipeline
defect. To keep every paired comparison sample-by-sample aligned, those exact
DEV seeds are excluded **uniformly from all five checkpoints of seed 1**
(995/1000 games for the greedy field, 498/500 for the mixed field); every
affected result JSON records the excluded seeds and the reason explicitly
(`excluded_seeds`, `non_termination_note`) — nothing is silently dropped or
scored as an unfinished game. No other seed encountered a non-terminating
episode at any checkpoint.

## Per-seed results — every trajectory shown

**Cross-seed DEV avg placement (7× greedy, ~1000 paired games per seed;
lower is better):**

| seed | iter 0 (0 eps) | iter 40 (640 eps) | iter 80 (1,280 eps) | iter 160 (2,560 eps) | iter 320 (5,120 eps) |
|---|---|---|---|---|---|
| 0 (existing) | 6.554 | 6.761 | 6.325 | 6.435 | 6.606 |
| 1 (new) | 6.564 | 6.794 | 7.101 | 6.874 | 6.893 |
| 2 (new) | 6.554 | 6.560 | 6.569 | 6.523 | 6.571 |
| 3 (new) | 6.554 | 6.877 | 7.074 | 6.869 | 6.691 |
| **mean** | 6.556 | 6.748 | 6.767 | 6.675 | 6.690 |
| **median** | 6.554 | 6.777 | 6.822 | 6.696 | 6.649 |
| **min** | 6.554 | 6.560 | 6.325 | 6.435 | 6.571 |
| **max** | 6.564 | 6.877 | 7.101 | 6.874 | 6.893 |
| **std (n=4)** | 0.005 | 0.134 | 0.383 | 0.230 | 0.144 |

**n = 4 training seeds is a small sample.** These cross-seed statistics are
descriptive summaries, not population estimates — individual seed
trajectories (shown above, never hidden behind the mean) are the primary
evidence. See [`aggregate/cross_seed_summary.json`](../results/ppo_multiseed_v1/aggregate/cross_seed_summary.json).

Seed 0 is the only trajectory that improves at iteration 80; seeds 1 and 3
are markedly *worse* at iteration 80 than at iteration 0 (their worst point
in the whole trajectory), and seed 2 is essentially flat throughout. Two
levels of replication unit matter here and should not be conflated: the
**training seed** is the unit for claims about PPO's stochastic behavior
(n=4), while the **1000 DEV games within each trajectory** give a precise,
low-variance estimate of that one trajectory's placement at each checkpoint.
**4 seeds × 1000 games is not 4,000 independent training experiments** — it
is 4 training experiments, each measured with high precision.

### Per-seed paired comparisons (9 pre-registered comparisons each)

Deterministic paired percentile bootstrap (10,000 resamples, bootstrap seed
0), over the identical DEV seeds within each trajectory. Convention: positive
= first (target) checkpoint places **worse** than the reference; negative =
better. Full data: [`aggregate/paired_results.json`](../results/ppo_multiseed_v1/aggregate/paired_results.json).

**Seed 0 (existing, from `results/ppo_budget_v1/`):**

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| iter40 − iter0 | +0.207 | [+0.093, +0.322] | worse |
| iter80 − iter0 | **−0.229** | [−0.392, −0.061] | **better** |
| iter160 − iter0 | −0.119 | [−0.245, +0.008] | no clear difference |
| iter320 − iter0 | +0.052 | [−0.104, +0.210] | no clear difference |
| iter80 − iter40 | −0.436 | [−0.593, −0.277] | better |
| iter160 − iter40 | −0.326 | [−0.451, −0.205] | better |
| iter320 − iter40 | −0.155 | [−0.303, +0.000] | no clear difference |
| iter160 − iter80 | +0.110 | [−0.044, +0.262] | no clear difference |
| iter320 − iter80 | **+0.281** | [+0.138, +0.425] | **worse (regression)** |

**Seed 1 (new, n=995 aligned games after uniform exclusion):**

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| iter40 − iter0 | +0.230 | [+0.078, +0.380] | worse |
| iter80 − iter0 | **+0.537** | [+0.438, +0.635] | **worse** |
| iter160 − iter0 | +0.311 | [+0.166, +0.454] | worse |
| iter320 − iter0 | +0.330 | [+0.182, +0.476] | worse |
| iter80 − iter40 | +0.307 | [+0.184, +0.428] | worse |
| iter160 − iter40 | +0.080 | [−0.045, +0.210] | no clear difference |
| iter320 − iter40 | +0.099 | [−0.036, +0.240] | no clear difference |
| iter160 − iter80 | −0.226 | [−0.340, −0.112] | better |
| iter320 − iter80 | **−0.207** | [−0.330, −0.085] | **better (partial recovery)** |

**Seed 2 (new, n=1000):**

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| iter40 − iter0 | +0.006 | [−0.116, +0.133] | no clear difference |
| iter80 − iter0 | **+0.015** | [−0.107, +0.141] | **no clear difference** |
| iter160 − iter0 | −0.031 | [−0.186, +0.126] | no clear difference |
| iter320 − iter0 | +0.017 | [−0.109, +0.145] | no clear difference |
| iter80 − iter40 | +0.009 | [−0.044, +0.061] | no clear difference |
| iter160 − iter40 | −0.037 | [−0.196, +0.124] | no clear difference |
| iter320 − iter40 | +0.011 | [−0.142, +0.161] | no clear difference |
| iter160 − iter80 | −0.046 | [−0.205, +0.116] | no clear difference |
| iter320 − iter80 | **+0.002** | [−0.148, +0.151] | **no clear difference** |

**Seed 3 (new, n=1000):**

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| iter40 − iter0 | +0.323 | [+0.181, +0.470] | worse |
| iter80 − iter0 | **+0.520** | [+0.375, +0.668] | **worse** |
| iter160 − iter0 | +0.315 | [+0.175, +0.460] | worse |
| iter320 − iter0 | +0.137 | [−0.017, +0.295] | no clear difference |
| iter80 − iter40 | +0.197 | [+0.093, +0.301] | worse |
| iter160 − iter40 | −0.008 | [−0.113, +0.095] | no clear difference |
| iter320 − iter40 | −0.186 | [−0.323, −0.048] | better |
| iter160 − iter80 | −0.205 | [−0.287, −0.123] | better |
| iter320 − iter80 | **−0.383** | [−0.500, −0.265] | **better (partial recovery)** |

## 1,280-episode replication — does iter80 beat the warm start across seeds?

**Question A** (seed 0's reference value: **−0.229**, a significant
improvement):

| Seed | iter80 − iter0 | 95% CI | CI excludes zero |
|---|---|---|---|
| 0 | −0.229 | [−0.392, −0.061] | yes (**improvement**) |
| 1 | +0.537 | [+0.438, +0.635] | yes (**regression**) |
| 2 | +0.015 | [−0.107, +0.141] | no |
| 3 | +0.520 | [+0.375, +0.668] | yes (**regression**) |

**1 of 4 seeds improves** at iter80 (seed 0 only). **2 of 4 seeds
significantly worsen** (seeds 1, 3), and **1 of 4 is statistically
indistinguishable** from the warm start (seed 2). **0 of the 3 new seeds
replicate the transient improvement**; on the contrary, the two new seeds
whose CI excludes zero move in the *opposite* direction from seed 0. No
significance threshold was invented after seeing the data — the same 95%
paired-bootstrap CI convention as Experiment 2 is used throughout.

## Long-training regression — does iter320 regress vs iter80?

**Question B** (seed 0 showed clear regression: **+0.281**):

| Seed | iter320 − iter80 | 95% CI | Reading |
|---|---|---|---|
| 0 | +0.281 | [+0.138, +0.425] | **regresses** |
| 1 | −0.207 | [−0.330, −0.085] | continues improving (partial recovery from its iter80 low) |
| 2 | +0.002 | [−0.148, +0.151] | statistically indistinguishable |
| 3 | −0.383 | [−0.500, −0.265] | continues improving (partial recovery from its iter80 low) |

**1 of 4 seeds regresses** after iter80 in the direction seed 0 did (seed 0
itself). **2 of 4 seeds "continue improving"** from iter80 to iter320 — but
this is recovery *back toward* the warm start after an iter80 regression, not
sustained new improvement (both seeds 1 and 3 remain significantly or
directionally worse than iter0 at iter320; see the iter320−iter0 rows above).
**1 of 4 is indistinguishable** (seed 2, flat throughout). The seed-0-style
late regression (getting worse after a genuine mid-training improvement) does
**not** appear in any new seed, because no new seed had a genuine mid-training
improvement to regress from.

## U-shape replication

Classification rule (documented, applied identically to all 4 seeds; see
`classify_trajectory` in `scripts/ppo_multiseed_report.py`): using only the
four pre-registered comparisons against iteration 0 and the raw average
placement at each checkpoint, in this fixed order — (1) *mostly flat/noisy*
if no checkpoint differs significantly from iter0; (2) *monotonic
improvement* if placement is non-increasing across all 5 checkpoints **and**
iter320 is significantly better than iter0; (3) *monotonic degradation*,
the mirror of (2); (4) *U-like/transient improvement* if some interior
checkpoint (40, 80, or 160) is significantly better than iter0 but iter320 is
not; (5) *other* for anything the first four don't cover.

| Seed | Classification | Basis |
|---|---|---|
| 0 | **U-like/transient improvement** | iter80 significantly better than iter0; iter320 is not |
| 1 | **other** | significantly *worse* than iter0 at every checkpoint (40, 80, 160, 320) — not flat, not monotonic, and never significantly better, so it fits none of the first four buckets |
| 2 | **mostly flat/noisy** | no checkpoint differs significantly from iter0 |
| 3 | **other** | significantly worse at 40/80/160, recovers to "no clear difference" by 320 — a regression-then-partial-recovery shape, not a transient *improvement* |

**1 / 4 trajectories (seed 0 only) show the transient-improvement-then-
regression U shape.** The other three show: persistent regression that
partially recovers but never turns into improvement (seed 1), flat/no effect
(seed 2), and regression-then-recovery-to-baseline without ever besting the
warm start (seed 3). The seed-0 U shape does **not** replicate as a general
property of this PPO recipe at this budget.

## Drift replication

**At iteration 320** (frozen 4,440-state diagnostic corpus):

| Seed | Expert agreement | Warm-start agreement | KL from warm start | Corpus entropy |
|---|---|---|---|---|
| 0 | 42.6% | 40.8% | 1.171 | 0.757 |
| 1 | 42.0% | 36.4% | 2.021 | 1.074 |
| 2 | 42.8% | 48.6% | 1.255 | 0.990 |
| 3 | 48.2% | 44.5% | 1.356 | 0.974 |

**Large behavioral drift appears consistently across all 4 seeds** — every
trajectory falls from 100% warm-start agreement / 84.5% expert agreement at
iteration 0 to roughly 36–49% / 42–48% by iteration 320, and every seed's KL
from the warm start rises monotonically and substantially (0 → 1.17–2.02).
This is the single most consistent finding across the replication: unlike
the placement U-shape, drift growth is universal.

**Descriptive correlation only (n=20 checkpoint-observations across 4 seeds,
not independent samples — no causal claim):** expert agreement vs. DEV
placement, r ≈ **−0.43** (higher expert agreement associated with better,
i.e. lower, placement); KL from warm start vs. placement, r ≈ **+0.50**
(higher KL associated with worse placement). Both point in the intuitive
direction but are weak-to-moderate and exploratory.

**Does each seed's best-performing checkpoint have higher expert agreement /
lower KL than its later checkpoints?** For the 3 seeds whose best checkpoint
is not the last one (seeds 0, 1, 2 — best at iterations 80, 40, and 160
respectively), **yes in all 3**: the best checkpoint has both higher expert
agreement and lower KL than the mean of the checkpoints that come after it.
(Seed 3's best DEV result is at iteration 320 itself, so there is no "later"
checkpoint to compare.) This descriptive pattern is consistent with — but
does not prove — drift being a real driver of placement decline within a
trajectory, independent of whether any given trajectory ever improves on the
warm start.

## Action-category replication

**At iteration 320, disagreement/contribution to total drift, on the frozen
corpus** (categories from `ml/action_categories.py`, unchanged):

| Seed | Top contributor | roll disagreement share | roll contribution to total drift | freeze appears | freeze selections / rate |
|---|---|---|---|---|---|
| 0 | roll (37.4%) | 70.6% | 37.4% | yes | 153 / 3.45% |
| 1 | roll (52.5%) | 100.0% | 52.5% | yes | 167 / 3.76% |
| 2 | roll (53.3%) | 100.0% | 53.3% | yes* — but freeze never selected | 0 / 0.00% |
| 3 | roll (58.3%) | 99.1% | 58.3% | yes | 100 / 2.25% |

(*"freeze appears" in the table header refers to the category being present
in the mapping; seed 2 is the one trajectory that never actually selects
freeze on the diagnostic corpus at iteration 320.)

**The tempo-decision (roll / end / play) drift pattern from Experiment 2
repeats in all 4 of 4 seeds** — "roll" is the single largest contributor to
total expert-disagreement at iteration 320 in every seed, and roll
disagreement itself is at or near saturation (70.6–100%) everywhere. This is
**not** unique to seed 0. **Freeze adoption (an action the greedy expert
never takes) appears in 3 of 4 seeds** (0, 1, 3) at iteration 320, at
comparable rates (2.25–3.76% of corpus states); seed 2 — the one flat,
low-drift-relative-to-others trajectory in placement terms, though its KL and
expert-agreement drift are comparable to the others — is the only seed that
never adopts freeze. The "buy" disagreement remains large (68–78%) but
consistent with Experiment 2's finding that it is substantially a warm-start
artifact rather than newly PPO-introduced drift, since it does not track
which seeds regress and which stay flat.

## RL diagnostics — do internal PPO metrics distinguish successful vs. unsuccessful trajectories?

Per-iteration policy loss, value loss, entropy, approximate KL, clip
fraction, and gradient norm were recorded for every seed exactly as in
Experiment 2 (`rl_signal.json`, `train_diag.jsonl` per seed; plot G).
Qualitatively, **all four seeds' optimization diagnostics look similar**:
value loss decays from ~0.06–0.08 toward ~0.04–0.05 and stays there; entropy
rises across training for every seed (ending in the 0.6–0.9 range, seed 1
highest); approximate KL per update and clip fraction stay in comparable
bands (roughly 0.005–0.02 and 0.05–0.10 respectively) with occasional shared
spikes across seeds (all four show sporadic policy-loss/grad-norm spikes,
consistent with league-snapshot points at multiples of 8 iterations). **No
internal PPO diagnostic in this set visibly distinguishes seed 0 (the one
trajectory with a genuine transient DEV improvement) from seeds 1–3.** The
value head is healthy in every seed (no collapse, no runaway loss), so a
"weak/degenerate learning signal" explanation for why seeds 1–3 don't
replicate the improvement is not supported by these diagnostics — the
optimizer is doing ordinary, stable PPO updates in all four trajectories;
whatever determines DEV placement outcome is not visible in these
aggregate optimization statistics.

## Limitations

- **Only four total PPO training seeds** (0 existing + 1, 2, 3 new). This is
  a small sample for any population-level claim about PPO's stochastic
  behavior; every cross-seed statistic in this report (mean/median/min/max/
  std, the correlation estimates) is explicitly descriptive/exploratory, not
  inferential. Per-seed DEV estimates (1000 paired games) are precise; the
  number of *independent trajectories* is not.
- Seed 1's iteration-320 checkpoint required excluding 5 (of 1000) /
  2 (of 500) DEV seeds uniformly across all its checkpoints due to a genuine
  non-terminating episode (see above); this slightly reduces that seed's
  effective sample size (995/498 vs. 1000/500) but does not otherwise change
  the evaluation protocol.
- Checkpoint binaries are gitignored per repo convention; only their
  fingerprints are stored in the committed artifacts.
- DEV evaluation is deterministic (argmax actions, seeded env), but bitwise
  reproducibility across different torch versions or hardware is not
  guaranteed.
- No new seed was added beyond the pre-specified 1, 2, 3, and no checkpoint
  was selected or deployed from any trajectory, per the experiment's hard
  constraints.

## Conclusion

Evaluated against the four candidate outcomes:

- **Outcome A (transient improvement replicates)** — **not supported.** Only
  1 of 4 seeds (the original, seed 0) shows the transient improve-then-regress
  pattern; the 3 new seeds do not reproduce a significant iter80 improvement
  in either direction consistent with "most independent trajectories
  improve."
- **Outcome B (seed 0 was a lucky excursion)** — **best supported.** None of
  the 3 new seeds significantly improves over the warm start at iteration 80
  (0 improve, 2 significantly regress, 1 is flat) — in fact, **PPO never
  produces a statistically significant improvement over the warm start at any
  measured checkpoint (40, 80, 160, or 320) in any of the 3 new seeds.**
  Seed 0's −0.229 gain at iter80 stands alone.
- **Outcome C (highly variable, qualitatively different curves)** — **partially
  present but not the dominant story.** There is real qualitative diversity
  (seed 2 flat; seeds 1 and 3 regress-then-partially-recover; seed 0
  improve-then-regress), but the *placement* variability is secondary to a
  clean, consistent negative result: none of the new seeds beat the warm
  start anywhere, and drift grows comparably and substantially in all four.
- **Outcome D** — not needed; B fits the data well without invoking a novel
  pattern.

**Outcome B is the best-supported conclusion: seed 0's iteration-80
improvement does not replicate.** What *does* replicate with unusual
consistency across all 4 seeds is the **drift** side of Experiment 2's
finding — large, monotonically growing divergence from the warm start
(expert agreement roughly halving, KL rising by 1.2–2.0, tempo-decision
(roll/end/play) drift dominating disagreement, and freeze adoption in 3 of 4
seeds) — with no compensating, reproducible placement benefit in the 3 new
seeds.

## Recommendation for Experiment 4 (not run)

Per the pre-registered decision rule, applied to the finding above (**PPO
never produces a statistically significant improvement over the warm start
on any of the 3 new seeds, at any measured checkpoint**):

> **Recommendation: revisit whether seed 0's iteration-80 gain was purely
> stochastic rather than algorithmic. Do not build an intervention around a
> non-replicated result.**

This explicitly rules out both "PPO policy anchoring" (a KL penalty toward
the BC prior would be motivated by a real improvement-then-drift pattern
that only 1 of 4 seeds shows) and a "PPO stability / rollout variance study"
(motivated by wildly inconsistent curves, which is not the dominant pattern
here — the dominant pattern is a consistent absence of improvement).
Experiment 4, if and when it runs, should first characterize what was
specific to seed 0's iteration-80 trajectory (e.g. the particular league
opponents and lobbies it happened to encounter near that point) rather than
committing further training budget to an intervention designed to protect or
extend a gain that has not been shown to be reproducible.
