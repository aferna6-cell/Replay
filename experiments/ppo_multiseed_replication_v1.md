# Replay Experiment 3 — Multi-seed PPO Budget Replication

DEV only; Benchmark v1 TEST was never used. Seed 0 is the committed Experiment 2 trajectory; seeds 1–3 are new frozen-recipe replications.

## Protocol

- Warm-start parameter SHA256: `094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b`.
- Frozen corpus: 4,440 states, `2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e`.
- PPO: 16 episodes/iteration, 320 iterations, shaping horizon 40; checkpoints 0/40/80/160/320.
- DEV: 1,000 paired games vs 7× greedy and 500 paired games vs fixed `greedy4_random3`, starting at seed 10,550,000.

## Full budget table

| iter | episodes | seed | greedy avg | top-4 | win | mixed avg | unfinished G/M | expert | warm-start | KL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 6.554 | 14.9% | 3.5% | 4.370 | 0/0 | 84.5% | 100.0% | 0.000 |
| 40 | 640 | 0 | 6.761 | 12.8% | 1.6% | 4.432 | 0/0 | 77.2% | 90.2% | 0.371 |
| 80 | 1280 | 0 | 6.325 | 18.0% | 2.8% | 4.282 | 0/0 | 80.9% | 77.8% | 0.249 |
| 160 | 2560 | 0 | 6.435 | 19.1% | 0.6% | 4.206 | 0/0 | 74.3% | 73.1% | 0.484 |
| 320 | 5120 | 0 | 6.606 | 13.2% | 3.5% | 4.408 | 0/0 | 42.6% | 40.8% | 1.171 |
| 0 | 0 | 1 | 6.554 | 14.9% | 3.5% | 4.370 | 0/0 | 84.5% | 100.0% | 0.000 |
| 40 | 640 | 1 | 6.791 | 10.1% | 1.4% | 4.606 | 0/0 | 71.1% | 74.1% | 0.442 |
| 80 | 1280 | 1 | 7.088 | 4.7% | 0.8% | 4.866 | 0/0 | 54.3% | 64.7% | 1.207 |
| 160 | 2560 | 1 | 6.861 | 8.3% | 0.6% | 4.670 | 0/0 | 43.3% | 37.7% | 2.350 |
| 320 | 5120 | 1 | 6.893 | 10.4% | 0.8% | 4.749 | 5/2 | 42.0% | 36.4% | 2.021 |
| 0 | 0 | 2 | 6.554 | 14.9% | 3.5% | 4.370 | 0/0 | 84.5% | 100.0% | 0.000 |
| 40 | 640 | 2 | 6.560 | 14.0% | 3.6% | 4.416 | 0/0 | 61.5% | 66.6% | 0.750 |
| 80 | 1280 | 2 | 6.569 | 14.5% | 3.5% | 4.438 | 0/0 | 70.1% | 75.7% | 0.552 |
| 160 | 2560 | 2 | 6.523 | 15.9% | 0.8% | 4.502 | 0/0 | 50.3% | 48.5% | 1.002 |
| 320 | 5120 | 2 | 6.571 | 14.7% | 0.8% | 4.400 | 0/0 | 42.8% | 48.6% | 1.255 |
| 0 | 0 | 3 | 6.554 | 14.9% | 3.5% | 4.370 | 0/0 | 84.5% | 100.0% | 0.000 |
| 40 | 640 | 3 | 6.877 | 7.8% | 0.2% | 4.776 | 0/0 | 68.6% | 64.8% | 0.542 |
| 80 | 1280 | 3 | 7.074 | 4.3% | 1.7% | 4.728 | 0/0 | 62.5% | 58.6% | 0.728 |
| 160 | 2560 | 3 | 6.869 | 6.4% | 1.1% | 4.680 | 0/0 | 43.0% | 38.9% | 1.905 |
| 320 | 5120 | 3 | 6.691 | 12.3% | 3.4% | 4.448 | 0/0 | 48.2% | 44.5% | 1.356 |

## Paired effects — greedy

Positive is worse placement. Each CI is a paired 10,000-resample bootstrap.

| seed | comparison | effect [95% CI] |
|---:|---|---|
| 0 | iter40 − iter0 | +0.207 [+0.093, +0.322] |
| 0 | iter80 − iter0 | -0.229 [-0.392, -0.061] |
| 0 | iter160 − iter0 | -0.119 [-0.245, +0.008] |
| 0 | iter320 − iter0 | +0.052 [-0.104, +0.210] |
| 0 | iter80 − iter40 | -0.436 [-0.593, -0.277] |
| 0 | iter160 − iter40 | -0.326 [-0.451, -0.205] |
| 0 | iter320 − iter40 | -0.155 [-0.303, +0.000] |
| 0 | iter320 − iter80 | +0.281 [+0.138, +0.425] |
| 1 | iter40 − iter0 | +0.237 [+0.089, +0.387] |
| 1 | iter80 − iter0 | +0.534 [+0.437, +0.632] |
| 1 | iter160 − iter0 | +0.307 [+0.159, +0.453] |
| 1 | iter320 − iter0 | +0.330 [+0.181, +0.475] complete-case (995/1000); sensitivity [+0.310, +0.345] |
| 1 | iter80 − iter40 | +0.297 [+0.174, +0.421] |
| 1 | iter160 − iter40 | +0.070 [-0.057, +0.198] |
| 1 | iter320 − iter40 | +0.099 [-0.037, +0.236] complete-case (995/1000); sensitivity [+0.073, +0.108] |
| 1 | iter320 − iter80 | -0.207 [-0.332, -0.083] complete-case (995/1000); sensitivity [-0.224, -0.189] |
| 2 | iter40 − iter0 | +0.006 [-0.120, +0.128] |
| 2 | iter80 − iter0 | +0.015 [-0.111, +0.139] |
| 2 | iter160 − iter0 | -0.031 [-0.189, +0.125] |
| 2 | iter320 − iter0 | +0.017 [-0.107, +0.142] |
| 2 | iter80 − iter40 | +0.009 [-0.044, +0.063] |
| 2 | iter160 − iter40 | -0.037 [-0.193, +0.121] |
| 2 | iter320 − iter40 | +0.011 [-0.135, +0.163] |
| 2 | iter320 − iter80 | +0.002 [-0.143, +0.151] |
| 3 | iter40 − iter0 | +0.323 [+0.178, +0.465] |
| 3 | iter80 − iter0 | +0.520 [+0.375, +0.667] |
| 3 | iter160 − iter0 | +0.315 [+0.174, +0.458] |
| 3 | iter320 − iter0 | +0.137 [-0.023, +0.295] |
| 3 | iter80 − iter40 | +0.197 [+0.093, +0.304] |
| 3 | iter160 − iter40 | -0.008 [-0.112, +0.095] |
| 3 | iter320 − iter40 | -0.186 [-0.325, -0.046] |
| 3 | iter320 − iter80 | -0.383 [-0.505, -0.264] |

## Paired effects — greedy4_random3

Positive is worse placement. Each CI is a paired 10,000-resample bootstrap.

| seed | comparison | effect [95% CI] |
|---:|---|---|
| 0 | iter40 − iter0 | +0.062 [-0.078, +0.202] |
| 0 | iter80 − iter0 | -0.088 [-0.254, +0.078] |
| 0 | iter160 − iter0 | -0.164 [-0.310, -0.020] |
| 0 | iter320 − iter0 | +0.038 [-0.122, +0.204] |
| 0 | iter80 − iter40 | -0.150 [-0.320, +0.022] |
| 0 | iter160 − iter40 | -0.226 [-0.374, -0.078] |
| 0 | iter320 − iter40 | -0.024 [-0.192, +0.142] |
| 0 | iter320 − iter80 | +0.126 [-0.016, +0.268] |
| 1 | iter40 − iter0 | +0.236 [+0.080, +0.390] |
| 1 | iter80 − iter0 | +0.496 [+0.366, +0.626] |
| 1 | iter160 − iter0 | +0.300 [+0.136, +0.462] |
| 1 | iter320 − iter0 | +0.371 [+0.205, +0.536] complete-case (498/500); sensitivity [+0.364, +0.392] |
| 1 | iter80 − iter40 | +0.260 [+0.132, +0.392] |
| 1 | iter160 − iter40 | +0.064 [-0.080, +0.210] |
| 1 | iter320 − iter40 | +0.143 [+0.000, +0.287] complete-case (498/500); sensitivity [+0.128, +0.156] |
| 1 | iter320 − iter80 | -0.124 [-0.273, +0.022] complete-case (498/500); sensitivity [-0.132, -0.104] |
| 2 | iter40 − iter0 | +0.046 [-0.096, +0.188] |
| 2 | iter80 − iter0 | +0.068 [-0.066, +0.206] |
| 2 | iter160 − iter0 | +0.132 [-0.028, +0.290] |
| 2 | iter320 − iter0 | +0.030 [-0.100, +0.160] |
| 2 | iter80 − iter40 | +0.022 [-0.054, +0.098] |
| 2 | iter160 − iter40 | +0.086 [-0.078, +0.254] |
| 2 | iter320 − iter40 | -0.016 [-0.166, +0.134] |
| 2 | iter320 − iter80 | -0.038 [-0.182, +0.110] |
| 3 | iter40 − iter0 | +0.406 [+0.250, +0.562] |
| 3 | iter80 − iter0 | +0.358 [+0.196, +0.518] |
| 3 | iter160 − iter0 | +0.310 [+0.158, +0.462] |
| 3 | iter320 − iter0 | +0.078 [-0.088, +0.242] |
| 3 | iter80 − iter40 | -0.048 [-0.182, +0.082] |
| 3 | iter160 − iter40 | -0.096 [-0.216, +0.022] |
| 3 | iter320 − iter40 | -0.328 [-0.478, -0.180] |
| 3 | iter320 − iter80 | -0.280 [-0.428, -0.132] |

## Replication result

Among new seeds 1–3, the iter80-vs-iter0 improvement occurs by point estimate in 0/3 (0/3 with CI excluding zero). Iter320 regresses from iter80 in 1/3 by point estimate (0/3 with CI excluding zero). The pre-documented U shape appears in 0/3 by point estimate and 0/3 under the strict paired-CI definition.

Cross-seed descriptive summaries and seed-level bootstrap intervals are in `aggregate.json`; with only four trajectories, these intervals are descriptive and not a population-level asymptotic claim.

## Drift, action categories, and RL signal

The full budget table above is the expert/warm-start/KL curve. Iteration-320 category summaries follow; the committed aggregate keeps the complete confusion matrices.

| seed | expert agreement | warm-start agreement | expert→freeze | warm-start→freeze | largest expert transitions |
|---:|---:|---:|---:|---:|---|
| 0 | 42.6% | 40.8% | 153 | 153 | roll→buy 935, end→roll 251, play→buy 227 |
| 1 | 42.0% | 36.4% | 167 | 167 | roll→buy 1352, end→buy 192, play→buy 135 |
| 2 | 42.8% | 48.6% | 0 | 0 | roll→buy 1352, end→buy 192, level→buy 127 |
| 3 | 48.2% | 44.5% | 100 | 100 | roll→buy 774, roll→end 494, end→freeze 100 |

RL-signal means by training block (raw advantages are measured before normalization):

| seed | block | mean abs adv | positive adv | value EV | return SD | placement SD | entropy | approx KL | clip frac |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | iters_1_40 | 0.205 | 0.509 | 0.614 | 0.433 | 1.767 | 0.493 | 0.0113 | 0.056 |
| 0 | iters_41_160 | 0.196 | 0.491 | 0.682 | 0.433 | 1.792 | 0.563 | 0.0082 | 0.068 |
| 0 | iters_161_320 | 0.179 | 0.496 | 0.687 | 0.403 | 1.709 | 0.575 | 0.0073 | 0.070 |
| 1 | iters_1_40 | 0.198 | 0.504 | 0.607 | 0.398 | 1.653 | 0.560 | 0.0077 | 0.051 |
| 1 | iters_41_160 | 0.182 | 0.500 | 0.615 | 0.363 | 1.546 | 0.706 | 0.0054 | 0.053 |
| 1 | iters_161_320 | 0.179 | 0.488 | 0.673 | 0.398 | 1.731 | 0.695 | 0.0061 | 0.059 |
| 2 | iters_1_40 | 0.202 | 0.542 | 0.565 | 0.386 | 1.665 | 0.559 | 0.0099 | 0.050 |
| 2 | iters_41_160 | 0.204 | 0.494 | 0.645 | 0.429 | 1.770 | 0.593 | 0.0081 | 0.063 |
| 2 | iters_161_320 | 0.194 | 0.502 | 0.625 | 0.397 | 1.675 | 0.545 | 0.0067 | 0.056 |
| 3 | iters_1_40 | 0.205 | 0.504 | 0.614 | 0.411 | 1.719 | 0.553 | 0.0129 | 0.064 |
| 3 | iters_41_160 | 0.173 | 0.496 | 0.616 | 0.354 | 1.536 | 0.608 | 0.0072 | 0.061 |
| 3 | iters_161_320 | 0.183 | 0.490 | 0.669 | 0.402 | 1.732 | 0.592 | 0.0082 | 0.064 |

Cross-seed primary-effect summaries:

| comparison | group | mean | SD | range | seed-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| iter80 − iter0 | seeds 0–3 | +0.210 | 0.379 | [-0.229, +0.534] | [-0.107, +0.527] |
| iter80 − iter0 | seeds 1–3 | +0.356 | 0.296 | [+0.015, +0.534] | [+0.015, +0.534] |
| iter320 − iter80 | seeds 0–3 | -0.077 | 0.286 | [-0.383, +0.281] | [-0.295, +0.159] |
| iter320 − iter80 | seeds 1–3 | -0.196 | 0.193 | [-0.383, +0.002] | [-0.383, +0.002] |

Seed 1 iteration 320 had 5/1,000 unfinished greedy games and 2/500 unfinished mixed games at the unchanged 400-decision integrity cap. They remain null, never imputed; its reported means/CIs are labeled complete-case and paired effects include best/worst-case bounds.

All remaining raw-advantage, return, critic, entropy, KL, clipping, gradient, loss, placement, reward-source, category, and hash fields are machine-recorded in `aggregate.json`.

## Outcome and next experiment

Selected **Outcome B**: the seed-0 U shape does not replicate in a majority of new seeds.

Experiment 4 recommendation: test a single fixed warm-start KL anchor coefficient against this frozen multi-seed protocol.

No PPO tuning, checkpoint selection, Experiment 4 execution, or TEST evaluation was performed.
