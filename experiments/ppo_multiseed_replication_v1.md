# Replay Experiment 3 — Multi-Seed PPO Budget Replication

DEV split only. Benchmark v1 TEST (seeds 10,250,000–10,299,999) was never
run and never consulted. All artifacts: `results/ppo_multiseed_v1/`
(manifest: `results/ppo_multiseed_v1/manifest.json`).

## Question

Experiment 2 (training seed 0 only) found a transient DEV improvement around
1,280–2,560 PPO episodes followed by decay back to the warm-start level.
Does that pattern reproduce across independent PPO training seeds, or was it
a property of one stochastic trajectory?

## Historical observation (Experiment 2, seed 0 — committed, not rerun)

1000 paired DEV games vs 7× greedy (seeds 10,550,000–10,550,999):
iter0 = 6.554, iter40 = 6.761, iter80 = **6.325**, iter160 = 6.435,
iter320 = 6.606. Paired: iter80−iter0 = −0.229 [−0.392, −0.061] (clear
improvement); iter320−iter80 = +0.281 [+0.138, +0.425] (clear regression).
Behavioral drift at iter320: expert agreement 42.6%, warm-start agreement
40.8%, KL from warm start 1.171.

## Setup — the frozen recipe

Nothing was tuned. The exact Experiment 2 command was reused verbatim except
for `--seed` and output paths:

```
python -m ml.train_ppo --iters 320 --episodes 16 --seed {S} --shaping 1.0 \
  --shaping-horizon 40 --from-bc ml/policy_bc.pt \
  --out results/ppo_multiseed_v1/seed_{S}/final.pt \
  --save-iters 0,10,20,40,80,120,160,240,320 \
  --save-dir results/ppo_multiseed_v1/seed_{S}/checkpoints \
  --diag-log results/ppo_multiseed_v1/seed_{S}/train_diag.jsonl
```

for S ∈ {1, 2, 3} (seed 0 = the committed Experiment 2 run). Unchanged:
architecture, AdamW lr 3e-4 wd 1e-4, γ 0.999, λ 0.95, clip 0.2, entropy
0.01, value coef 0.5, 4 PPO epochs, minibatch 256, league every 8 (max 5),
reward, `--shaping-horizon 40` (shaping is the original 40-iteration
schedule, zero after iteration 40; it never depends on the 320-iteration
horizon), 16 episodes × 320 iterations = 5,120 episodes per seed,
observation encoder, action space. Two operational notes (neither touches
training): the runs appended `--eval-episodes 1` to shorten the
post-training console printout (a legacy evaluation that runs after the
final weights are saved), and the three runs executed concurrently in
separate single-threaded processes (`OMP_NUM_THREADS=1`); each run is
self-contained and seeded, so concurrency cannot alter any run's RNG stream.

- Training episode seeds: `ml.seeds.ppo_episode_seed(S, k) = S·1,000,003 + k`
  → seed 1: 1,000,004–1,005,123; seed 2: 2,000,007–2,005,126; seed 3:
  3,000,010–3,005,129. All verified outside DEV [10,550,000–10,599,999] and
  TEST [10,250,000–10,299,999] before training.
- Evaluation: identical to Experiment 2 — 1000 DEV games vs 7× greedy on
  seeds 10,550,000–10,550,999 per primary checkpoint (iterations 0, 40, 80,
  160, 320), plus the `greedy4_random3` diagnostic (500 games, same base
  seed, same fixed seats; no 4.5 threshold — relative comparisons only).
- Drift: the frozen 4,440-state Experiment 1/2 corpus, fingerprint verified
  equal to the historical value
  `2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e`
  before any analysis. Same action-category mapping as Experiment 2.

## Training-seed control — the warm start is identical

Every trajectory starts from the same exact BC + DAgger checkpoint from
Experiments 1–2. The checkpoint binary is gitignored, so it was reproduced
with the historical recorded command
(`python -m ml.bc --lobbies 150 --epochs 6 --dagger-rounds 2
--dagger-lobbies 80 --seed 0 --out ml/policy_bc.pt`) and verified BEFORE any
training:

- `parameter_sha256 =
  094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b`
  (matches the Experiment 2 manifest exactly; the raw artifact hash
  `bd3a4386…` also matched bit-for-bit).
- Each seed's `iter_000.pt` snapshot reproduces that parameter hash
  (recorded per seed in `checkpoints.json`; enforced by tests).
- Pipeline control: each seed's iteration-0 DEV evaluation is game-by-game
  identical to Experiment 2's warm-start evaluation (avg 6.554, all 1000
  placements equal) — the eval stack reproduces the historical numbers
  exactly.

## Per-seed results

DEV avg placement vs 7× greedy (1000 paired games; lower is better):

| iter | episodes | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|
| 0 | 0 | 6.554 | 6.554 | 6.554 | 6.554 |
| 40 | 640 | 6.761 | 6.791 | 6.560 | 6.877 |
| 80 | 1,280 | **6.325** | 7.088 | 6.569 | 7.074 |
| 160 | 2,560 | 6.435 | 6.861 | 6.523 | 6.869 |
| 320 | 5,120 | 6.606 | **unscoreable** | 6.571 | 6.691 |

Across scoreable seeds per budget (n = 4 except iter320 where n = 3;
descriptive only): iter0 6.554/σ0.000, iter40 mean 6.747 median 6.776
min 6.560 max 6.877 σ0.134, iter80 mean 6.764 median 6.822 min 6.325
max 7.088 σ0.379, iter160 mean 6.672 median 6.692 min 6.435 max 6.869
σ0.226, iter320 mean 6.623 median 6.606 min 6.571 max 6.691 σ0.062.

**Seed 1's iteration 320 has no defined DEV score.** Its deterministic
argmax play never finishes 5 of the 1000 greedy-field DEV games (and 2 of
the 500 mixed-field games): it locks into an infinite `freeze` loop (action
26 repeated indefinitely; loop probes in
`seed_1/dev/iter320_vs_greedy.protocol_failure.json`). The frozen protocol
refuses to score non-terminating episodes — a silent 8th place would corrupt
paired numbers — so the checkpoint is recorded as a protocol failure, the
strongest observable form of degradation. No score was fabricated.

Mixed diagnostic (`greedy4_random3`, 500 games, relative only) tells the
same story: seed 0 improved at iter80 (4.37 → 4.28), seeds 1 and 3 got worse
(4.37 → 4.87 / 4.73), seed 2 stayed flat (4.44).

### Within-seed paired comparisons

Paired deterministic bootstrap over the identical 1000 DEV seeds, 10,000
resamples; positive = first checkpoint WORSE; `*` = 95% CI excludes zero.

Seed 0 (Experiment 2 data, recomputed from committed per-game placements):

| pair | diff | 95% CI | |
|---|---|---|---|
| iter40−iter0 | +0.207 | [+0.093, +0.322] | * |
| iter80−iter0 | −0.229 | [−0.392, −0.061] | * |
| iter160−iter0 | −0.119 | [−0.245, +0.008] | |
| iter320−iter0 | +0.052 | [−0.104, +0.210] | |
| iter80−iter40 | −0.436 | [−0.593, −0.277] | * |
| iter160−iter40 | −0.326 | [−0.451, −0.205] | * |
| iter320−iter40 | −0.155 | [−0.303, +0.000] | |
| iter160−iter80 | +0.110 | [−0.044, +0.262] | |
| iter320−iter80 | +0.281 | [+0.138, +0.425] | * |

Seed 1:

| pair | diff | 95% CI | |
|---|---|---|---|
| iter40−iter0 | +0.237 | [+0.088, +0.387] | * |
| iter80−iter0 | +0.534 | [+0.436, +0.633] | * |
| iter160−iter0 | +0.307 | [+0.160, +0.453] | * |
| iter320−iter0 | unscoreable (iter320 fails the DEV protocol) | | |
| iter80−iter40 | +0.297 | [+0.174, +0.418] | * |
| iter160−iter40 | +0.070 | [−0.058, +0.196] | |
| iter320−iter40 | unscoreable | | |
| iter160−iter80 | −0.227 | [−0.340, −0.110] | * |
| iter320−iter80 | unscoreable | | |

Seed 2:

| pair | diff | 95% CI | |
|---|---|---|---|
| iter40−iter0 | +0.006 | [−0.116, +0.133] | |
| iter80−iter0 | +0.015 | [−0.107, +0.141] | |
| iter160−iter0 | −0.031 | [−0.186, +0.126] | |
| iter320−iter0 | +0.017 | [−0.109, +0.145] | |
| iter80−iter40 | +0.009 | [−0.044, +0.061] | |
| iter160−iter40 | −0.037 | [−0.196, +0.124] | |
| iter320−iter40 | +0.011 | [−0.142, +0.161] | |
| iter160−iter80 | −0.046 | [−0.205, +0.116] | |
| iter320−iter80 | +0.002 | [−0.148, +0.151] | |

Seed 3:

| pair | diff | 95% CI | |
|---|---|---|---|
| iter40−iter0 | +0.323 | [+0.181, +0.470] | * |
| iter80−iter0 | +0.520 | [+0.375, +0.668] | * |
| iter160−iter0 | +0.315 | [+0.175, +0.460] | * |
| iter320−iter0 | +0.137 | [−0.017, +0.295] | |
| iter80−iter40 | +0.197 | [+0.093, +0.301] | * |
| iter160−iter40 | −0.008 | [−0.113, +0.095] | |
| iter320−iter40 | −0.186 | [−0.323, −0.048] | * |
| iter160−iter80 | −0.205 | [−0.287, −0.123] | * |
| iter320−iter80 | −0.383 | [−0.500, −0.265] | * |

## The 1,280-episode replication (Question A)

iter80 − iter0 per seed (negative = iteration 80 better):

| seed | diff | 95% CI | reading |
|---|---|---|---|
| 0 | −0.229 | [−0.392, −0.061] | improved (CI excludes 0) |
| 1 | +0.534 | [+0.436, +0.633] | clearly WORSE |
| 2 | +0.015 | [−0.107, +0.141] | indistinguishable |
| 3 | +0.520 | [+0.375, +0.668] | clearly WORSE |

**1 / 4 seeds improve at iteration 80; 0 / 3 new seeds improve.** One CI
excludes zero on the improvement side (seed 0 only); two CIs exclude zero on
the worsening side (seeds 1 and 3). The ~1,280-episode improvement is NOT
common across trajectories — at the exact budget where seed 0 gained
−0.229, two of three new seeds lost ~+0.53.

## Long-training regression (Question B)

iter320 − iter80 per seed (positive = the trajectory regressed after
iteration 80):

| seed | diff | 95% CI | reading |
|---|---|---|---|
| 0 | +0.281 | [+0.138, +0.425] | clearly regressed |
| 1 | no CI possible | | regressed to protocol failure (iter320 unscoreable) |
| 2 | +0.002 | [−0.148, +0.151] | indistinguishable |
| 3 | −0.383 | [−0.500, −0.265] | clearly kept improving |

1 seed clearly regresses, 1 regresses in the strongest observable sense
(unscoreable), 1 is flat, and 1 clearly *improves* after iteration 80 —
because seed 3 had degraded so badly by iter80 (7.074) that its later
"improvement" is partial recovery toward (but still worse than) the warm
start (6.691 vs 6.554). Post-iter80 behavior is not consistent across
trajectories.

#### Restricted supplement for seed 1 iteration 320 (NOT a DEV result)

Seed 1's iteration-320 row above is a gap, not a number. So that the
pre-specified question still has a stated answer for that trajectory,
`aggregate/restricted_supplement.json` re-pairs it against the scored
budgets on only the 995 greedy-field lobbies it *did* finish. **This is
biased in the failing checkpoint's favour** — the 5 dropped lobbies are
exactly the ones it could not terminate — and it is deliberately excluded
from every headline table, the cross-seed summary, and the U-shape rule:

| restricted pair (seed 1, greedy, n = 995) | diff | 95% CI | |
|---|---|---|---|
| iter320−iter0 | +0.330 | [+0.182, +0.476] | * |
| iter320−iter40 | +0.099 | [−0.036, +0.240] | |
| iter320−iter80 | −0.207 | [−0.330, −0.085] | * |
| iter320−iter160 | +0.019 | [−0.080, +0.119] | |

Restricted average on the completed subset: 6.893 (mixed field: 4.749,
n = 498). Even on this favourable subset seed 1's final policy is clearly
worse than its own warm start (+0.330), and the apparent −0.207 "gain" over
iter80 is measured on lobbies selected for being ones it could finish. The
honest summary of Question B for seed 1 remains: regression to the strongest
observable form of failure.

## U-shape classification

Pre-specified rule (`ml.multiseed_analysis.classify_ushape`, committed
before the new seeds' evaluations existed): U-like iff min(p80, p160) < p0
AND p320 > that mid-best, on the greedy means; monotone/flat/other
otherwise; CI flags recorded alongside. One documented extension was added
*after* observing that a checkpoint can fail the protocol outright (its
direction is forced, not tuned): an unscoreable checkpoint is ordered
strictly worse than every scoreable one.

| seed | strict label | CI support | extension label |
|---|---|---|---|
| 0 | U-like / transient improvement | mid gain AND late regression both clear | same |
| 1 | other (iter320 unscoreable) | no mid gain; late "regression" = protocol failure | monotonic degradation |
| 2 | U-like / transient improvement | NO clear signals — all nine CIs include zero; the "U" is a ±0.05 wobble | same (rule artifact) |
| 3 | other (degrade → partial recovery, ends worse than iter0) | degradation at 40/80/160 clear | same |

Summary: **2 / 4 trajectories are labeled "transient improvement followed by
regression" by the letter of the rule, but only seed 0's transient is
statistically supported.** Seed 2's label is a rule artifact on a flat curve
(mid gain −0.031 [−0.186, +0.126]; late regression +0.048, both CIs include
zero). Honest reading: 1 / 4 trajectories (seed 0 itself) shows a supported
transient improvement; 0 / 3 new trajectories reproduce it.

## Drift replication

At iteration 320 (frozen 4,440-state corpus):

| seed | expert agreement | warm-start agreement | KL from warm start | entropy |
|---|---|---|---|---|
| 0 | 42.6% | 40.8% | 1.171 | 0.757 |
| 1 | 42.0% | 36.4% | 2.021 | 1.074 |
| 2 | 42.8% | 48.6% | 1.255 | 0.990 |
| 3 | 48.2% | 44.5% | 1.356 | 0.974 |

**Large behavioral drift is fully consistent across trajectories** (expert
agreement mean 43.9% ± 2.9 from 84.5%; KL mean 1.45 ± 0.39) — even though
the performance consequences differ wildly. Descriptively (no causal claim),
every seed's best-performing checkpoint has higher expert agreement AND
lower KL than its final checkpoint (4/4 and 4/4): seed 0 best = iter80
(80.9% vs 42.6%, KL 0.249 vs 1.171); seed 1 best = iter0; seed 2 best =
iter160 (50.3% vs 42.8%, KL 1.002 vs 1.255); seed 3 best = iter0. Note that
for seeds 1 and 3 "best" is the untouched warm start, so this pattern partly
restates that PPO made things worse.

Full curves per seed (expert % / warm-start % / KL at iters 0, 40, 80, 160,
320):

- seed 0: 84.5/100/0 → 77.2/90.2/0.371 → 80.9/77.8/0.249 → 74.3/73.1/0.484
  → 42.6/40.8/1.171
- seed 1: 84.5/100/0 → 71.1/74.1/0.442 → 54.3/64.7/1.208 → 43.3/37.7/2.350
  → 42.0/36.4/2.021
- seed 2: 84.5/100/0 → 61.5/66.6/0.750 → 70.1/75.7/0.552 → 50.3/48.5/1.002
  → 42.8/48.6/1.255
- seed 3: 84.5/100/0 → 68.6/64.8/0.542 → 62.5/58.6/0.728 → 43.0/38.9/1.905
  → 48.2/44.5/1.356

## Action-category replication (iteration 320, vs the greedy expert)

Same category mapping as Experiment 2 (buy/play/sell/roll/level/freeze/end).

| seed | expert agree | roll share changed | end share changed | play share changed | freeze count (rate) | top transition |
|---|---|---|---|---|---|---|
| 0 | 42.6% | 70.6% | 50.1% | 40.0% | 153 (3.5%) | roll→buy ×935 |
| 1 | 42.0% | 100% | 31.8% | 22.4% | 167 (3.8%) | roll→buy ×1352 |
| 2 | 42.8% | 100% | 30.2% | 8.9% | 0 (0%) | roll→buy ×1352 |
| 3 | 48.2% | 99.1% | 15.6% | 0.3% | 100 (2.3%) | roll→buy ×774 |

The **tempo-decision drift pattern is repeatable, not unique to seed 0**:
in every seed the dominant drift is the collapse of the expert's `roll`
decisions (mostly into `buy`), contributing 37–58% of total disagreement,
with `end`-decision churn second. The anomalous `freeze` selections appear
in 3 of 4 seeds (seed 2 selects freeze zero times); in seed 1 the freeze
attraction is strong enough that deterministic play freeze-loops forever in
5/1000 DEV lobbies — the protocol failure above.

`buy` is the one category that must NOT be read as PPO drift, exactly as
Experiment 2 noted. Its disagreement share is already 74.0% at iteration 0 —
the untouched warm start — and stays flat across every budget and every
seed (seed 0: 74.0/74.5/72.2/71.8/73.4%; seed 1: 74.0/76.3/74.1/67.6/67.7%;
seed 2: 74.0/75.9/78.3/79.6/74.6%; seed 3: 74.0/71.8/77.0/77.0/78.4%). It is
an inherited BC artifact — the warm start picks a different minion than the
greedy expert on most buy states — so it is constant background, not
something PPO caused. Its *contribution* to total disagreement falls with
budget (88.6% → 22–28% at iter320) only because the tempo categories add so
much new disagreement around it, not because buy behavior changed.

## RL-signal diagnostics

Per-iteration diagnostics (identical instrumentation to Experiment 2;
RNG-neutral, verified by the existing bit-identical-weights test) are
committed per seed in `train_diag.jsonl` / `rl_signal.json`. Contrast of the
one improved trajectory (seed 0) vs the three that never improved, block
means:

| metric (iters 41–160) | seed 0 (improved) | seeds 1–3 (not improved) |
|---|---|---|
| entropy | 0.563 | 0.636 |
| approx KL / update | 0.0082 | 0.0069 |
| clip fraction | 0.068 | 0.059 |
| value explained variance | 0.682 | 0.626 |
| mean \|raw advantage\| | 0.196 | 0.186 |
| grad norm | 1.144 | 0.920 |

The internal optimization metrics of the successful and unsuccessful
trajectories are **nearly indistinguishable** — no PPO health signal
(entropy, KL, clipping, value fit, advantage scale, gradient norm) separates
the one seed that improved from the ones that degraded. PPO's own
diagnostics gave no warning that three of four trajectories were losing to
their initialization. (Exploratory, n=4: corr(iter80 gain, iter320 expert
agreement) = 0.45 — sign-unstable at this n; no claim made.)

## Plots

All figures are regenerated by `scripts/ppo_multiseed_aggregate.py` from the
committed JSON artifacts (`seed_*/learning_curve.json`,
`seed_*/policy_drift.json`, `seed_*/action_category_drift.json`,
`seed_*/rl_signal.json` and the aggregate files). No numeric value is
hard-coded in the plotting code, and a test renders the learning-curve figure
twice with one placement perturbed in the loaded JSON and asserts the image
changes — a hard-coded series could not.

| | figure | content |
|---|---|---|
| A | `results/ppo_multiseed_v1/aggregate/plots/A_multiseed_dev_curves.png` | one unsmoothed DEV curve per training seed, x = {0, 640, 1280, 2560, 5120} episodes |
| B | `results/ppo_multiseed_v1/aggregate/plots/B_cross_seed_mean.png` | cross-seed mean with every individual seed line/point overlaid |
| C | `results/ppo_multiseed_v1/aggregate/plots/C_expert_agreement.png` | greedy-expert action agreement by seed |
| D | `results/ppo_multiseed_v1/aggregate/plots/D_kl_from_warmstart.png` | KL from the warm start by seed |
| E | `results/ppo_multiseed_v1/aggregate/plots/E_warmstart_agreement.png` | warm-start action agreement by seed |
| F | `results/ppo_multiseed_v1/aggregate/plots/F_category_drift_iter320.png` | action-category drift at iteration 320 across seeds |
| G | `results/ppo_multiseed_v1/aggregate/plots/G_rl_diagnostics.png` | PPO optimization diagnostics across seeds |

Seed 1's iteration-320 point is absent from A and B rather than imputed; the
figures mark it as unscoreable.

## Limitations

- **n = 4 training seeds** (1 historical + 3 new). All cross-seed statistics
  are descriptive; no population-level effect sizes are claimed. Direction
  counts (0/3 new seeds improving) are the strongest statement supported.
- **Software versions differ from Experiment 2's recorded environment.**
  Experiment 3 ran on Python 3.12.3 / torch 2.13.0+cpu / numpy 2.5.2;
  Experiment 2's manifest records Python 3.11.15 / torch 2.13.0+cu130 /
  numpy 2.4.6. Seed 0's numbers are reused from Experiment 2's committed
  artifacts and were NOT recomputed, so any build-dependent difference would
  sit *between* seed 0 and seeds 1–3 — precisely the comparison this
  experiment rests on. The one direct check available is reassuring rather
  than conclusive: the iteration-0 checkpoint (identical weights in all four
  seeds) reproduces Experiment 2's 1000 per-game DEV placements exactly on
  this build, so the evaluation stack is bit-stable across the two
  environments. Whether *training* is equally build-stable was not tested,
  and cannot be without rerunning seed 0, which the protocol forbids.
- Seed 1's iteration 320 has no DEV score by construction (the protocol
  refuses non-terminating play); iter320 cross-seed statistics cover 3
  scoreable seeds and are flagged in every artifact.
- The seed-2 "U-like" label illustrates that the mean-based shape rule has
  no noise floor; the paired CIs recorded alongside are the authoritative
  evidence, and they show seed 2 as flat.
- Within-seed CIs share the identical 1000 DEV lobbies across checkpoints
  and seeds (by design, for pairing); conclusions are about this DEV block,
  not fresh lobbies. TEST remains locked and untouched.
- DEV evaluation is deterministic given weights and seeds; training itself
  was verified reproducible in Experiment 2 (iter-40 hash gate), but no
  bitwise cross-hardware claim is made.

## Conclusion

**Outcome B — seed 0's improvement was a lucky excursion.** The pre-specified
replication test fails: 0 of 3 new trajectories reproduce the ~1,280-episode
improvement (seed 0: −0.229 [−0.392, −0.061]; new seeds: +0.534*, +0.015,
+0.520* — two significantly WORSE, one flat). What replicates is not the
transient gain but the *drift*: every trajectory ends ~42–48% expert
agreement and KL 1.2–2.0 from the warm start, dominated by the same
roll→buy tempo collapse, while DEV performance never ends better than the
untouched warm start (6.55) in any seed. Trajectory shapes are additionally
heterogeneous (one supported U, one flat, one degrade-then-recover, one
degrade-to-protocol-failure) — elements of Outcome C are present — but the
crisp answer to the question this experiment was designed to ask is that the
iteration-80 gain did not replicate, which is Outcome B.

## Experiment 4 recommendation (recommendation only — not run)

**Revisit whether seed 0's iteration-80 gain was stochastic before building
any intervention on it.** Concretely: do NOT proceed to PPO+KL-anchoring on
the strength of the seed-0 excursion — the premise "PPO finds useful signal
but drifts away from it" is not supported when 3 of 3 new trajectories never
found the signal in the first place. The measured facts (universal drift
with indistinguishable internal PPO diagnostics, performance never beating
the warm start, one trajectory collapsing to non-terminating play) point the
follow-up at the variance/reliability question first — e.g., cheap
multi-seed replications at the 1,280-episode budget to estimate the
distribution of iter80−iter0 before any anchoring study is justified.
