# Replay Baseline Experiment v1

Date: 2026-08-28 · Benchmark: **Replay Benchmark v1** · Code commit: `5b7b46f`
· Manifest: [`results/benchmark_v1/manifest.json`](../results/benchmark_v1/manifest.json)

The first real, reproducible baseline table for Replay's agents. This is the
"before" measurement every future experiment must beat. Nothing was tuned:
every training run uses the shipped default configuration, and every number
below comes from committed Benchmark v1 result JSON under
`results/benchmark_v1/`.

## Experimental setup

- **Evaluation**: `python -m ml.benchmark --agent … --games 1000 --field greedy`
  — 1000 games per agent on the reserved evaluation seeds 10,250,000–10,250,999
  (identical for all five agents; tested agent in seat 0 vs 7 scripted greedy
  opponents). Learned checkpoints act by argmax. Training-time eval printouts
  in the `ml.bc` / `ml.train_ppo` logs are legacy diagnostics, **not** official
  numbers.
- **BC (plain)**: `python -m ml.bc --lobbies 150 --epochs 6 --dagger-rounds 0
  --seed 0 --out ml/policy_bc_plain.pt` — 7,319 expert demonstrations, final
  imitation accuracy 84.3%.
- **BC + DAgger**: `python -m ml.bc --lobbies 150 --epochs 6 --dagger-rounds 2
  --dagger-lobbies 80 --seed 0 --out ml/policy_bc.pt` — 7,319 demos + 2,729 +
  2,691 DAgger-labeled states (12,739 total), final imitation accuracy 82.4%
  (on the larger mixed dataset; not directly comparable to plain BC's 84.3%).
- **PPO**: `python -m ml.train_ppo --iters 40 --episodes 16 --seed 0
  --shaping 1.0 --from-bc ml/policy_bc.pt --out ml/policy_ppo.pt` — the
  shipped recipe unchanged (640 episodes total, league self-play, shaping
  annealed to 0 by iter ~28), warm-started from the BC + DAgger checkpoint.
- Software: Python 3.11.15, torch 2.13.0+cu130 (CPU), numpy 2.4.6.
- Checkpoints are gitignored (repo convention); their SHA-256 fingerprints
  are in the manifest and in each result JSON. Without the binaries the runs
  are reproducible only by retraining with the recorded commands.
- Determinism cross-check: the BC and BC + DAgger 1000-game benchmarks were
  each executed twice (under heavy concurrent CPU load, then on an idle
  machine) and produced **byte-identical placement sequences**. Committed
  JSONs and latency figures are from the idle-machine runs.

## Results

1000 games each, seeds 10,250,000–10,250,999, vs 7× greedy. Lower placement
is better; 4.5 = the field's average.

| Agent | Avg Place | 95% CI | Median | Std | Top-4 | Win | p95 latency |
|---|---|---|---|---|---|---|---|
| Greedy | **4.445** | [4.308, 4.600] | 4 | 2.29 | 52.6% | 11.8% | 0.008 ms |
| BC | 6.497 | [6.375, 6.618] | 7 | 1.94 | 17.2% | 3.1% | 1.15 ms |
| BC + DAgger | 6.527 | [6.409, 6.652] | 7 | 1.95 | 15.8% | 2.8% | 1.10 ms |
| PPO | 6.798 | [6.686, 6.910] | 8 | 1.78 | 12.5% | 1.9% | 1.10 ms |
| Random | 7.989 | [7.982, 7.995] | 8 | 0.10 | 0.0% | 0.0% | 0.004 ms |

Placement distributions (1st → 8th):

| Agent | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Greedy | 118 | 141 | 131 | 136 | 112 | 112 | 124 | 126 |
| BC | 31 | 34 | 37 | 70 | 63 | 124 | 180 | 461 |
| BC + DAgger | 28 | 44 | 35 | 51 | 79 | 110 | 177 | 476 |
| PPO | 19 | 23 | 41 | 42 | 67 | 98 | 161 | 549 |
| Random | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 989 |

The Greedy-vs-Greedy control behaves exactly as construction predicts: its CI
contains 4.5 and its placement distribution is nearly flat — no seat-0 bias.

Paired comparisons (same seeds game-by-game; paired percentile bootstrap,
10,000 resamples, bootstrap seed 0 — `results/benchmark_v1/paired_analysis.json`):

| Comparison | Mean paired diff | 95% CI | Reading |
|---|---|---|---|
| BC − (BC + DAgger) | −0.030 | [−0.194, +0.139] | no clear difference |
| (BC + DAgger) − PPO | −0.271 | [−0.391, −0.152] | PPO is clearly worse than its warm start |
| PPO − Greedy | +2.353 | [+2.199, +2.509] | PPO ~2.35 places behind Greedy |
| (BC + DAgger) − Greedy | +2.082 | [+1.934, +2.232] | BC + DAgger ~2.08 places behind Greedy |

## BC → DAgger effect

None measurable. The paired difference is −0.030 with a CI of
[−0.194, +0.139]: this benchmark does not establish any difference between
plain BC and BC + 2 DAgger rounds. The extra 5,420 DAgger-labeled states did
not translate into better placements under the current configuration.

## DAgger → PPO effect

**Negative.** PPO training, run exactly as shipped for 640 episodes, made the
policy 0.271 places *worse* than the BC + DAgger checkpoint it was
warm-started from (CI [0.152, 0.391], zero excluded). Its 8th-place rate rose
from 47.6% to 54.9%. The training log shows no sustained improvement trend in
training-field placement across the 40 iterations (e.g. iter 8: 4.88,
iter 20: 6.06, iter 33: 4.88, iter 39: 6.50 — noisy, vs a *mixed* league
field, not the pure-greedy benchmark field). The historical claim that PPO
crushes a random field still holds: the training-time diagnostic measured
1.00 average placement vs 7 random opponents.

## PPO vs Greedy — the main current gap

PPO finishes 2.353 places behind the scripted greedy baseline
(CI [2.199, 2.509]). No learned agent is anywhere near the 4.5
beat-the-field threshold; the entire learned stack sits in a 6.5–6.8 band,
closer to each other than to either Greedy or Random.

## Failure analysis

**Observed** (supported by this data):

1. Every learned policy clusters at 6.5–6.8 while the expert it imitates
   scores 4.445. A clone with 84.3% per-action imitation accuracy loses
   ~2.05 places to the policy it clones — the residual ~16% of decisions
   (and/or their compounding effects across a game) costs about two full
   placement ranks.
2. The learned policies' most common outcome is elimination in last place
   (46–55% of games finish 8th), while their win rates stay at 2–3%. They
   are not "slightly worse everywhere" — they disproportionately die early.
3. PPO's 640-episode run moved the policy away from the benchmark field's
   optimum, not toward it, while remaining dominant against random
   opponents.
4. DAgger's additional on-policy expert labels changed nothing measurable.

**Hypotheses for future experiments** (NOT established by this data):
the PPO training field (league snapshots + occasional random seats) differs
from the pure-greedy evaluation field, so the recipe may be optimizing
against the wrong opponent distribution; 640 episodes may simply be far too
little RL signal for a 28-action recruit-phase game; the zero-mean placement
reward arrives only at episode end and may be too sparse at this scale; and
the argmax evaluation of a policy trained with entropy regularization may
behave differently from its sampling behavior. These need controlled
experiments, not guesses.

## Next research question

**Why does the shipped PPO recipe make its warm start worse on the benchmark
it is meant to win?** The measured bottleneck is that the RL step currently
*subtracts* value (+0.271 places, CI excludes zero) while the whole learned
stack still trails the scripted expert by ~2.1–2.4 places. Before any
architecture or reward work, the next milestone should instrument and vary
the PPO recipe's most basic levers (training-field composition vs the
evaluation field, and training budget) under Benchmark v1 — one change at a
time, measured against this table.

Nothing was optimized in this experiment. That is the point: these are the
honest "before" numbers.
