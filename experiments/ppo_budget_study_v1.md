# Replay Experiment 2 — PPO Training-Budget Study

Date: 2026-08-28 · Split: **DEV only** (Benchmark v1 TEST never run) ·
Artifacts: [`results/ppo_budget_v1/`](../results/ppo_budget_v1/) ·
Manifest: [`manifest.json`](../results/ppo_budget_v1/manifest.json)

## Question

Experiment 1 found that PPO drifts substantially from its BC + DAgger warm
start, but 200-game DEV evaluation could not localize where placement breaks.
Is the shipped PPO recipe **data-starved**, or does more experience merely
produce more drift?

One variable changed: **training budget**. Architecture, learning rate, γ, λ,
clip, entropy coefficient, value coefficient, optimizer, league logic and the
reward function are all untouched.

## Setup

- **One continuous trajectory** from the exact BC + DAgger warm start
  (`policy_bc.pt`, parameter hash `094417bdcaa7af62…`, which the iteration-0
  checkpoint reproduces exactly), seed 0, 16 episodes/iteration, 320
  iterations.
- **Frozen shaping schedule.** `--shaping-horizon 40` pins the anneal
  reference, so iterations 1–40 see exactly the original shaping values
  (reaching 0 at iteration 28) and everything after 40 sees 0. Without it, a
  320-iteration run would have stretched the schedule and changed the reward
  for the first 40 iterations too — confounding budget with reward.
- **Reproduction gate (hard, passed).** The extended trajectory's
  iteration-40 parameters are **identical** to the historical baseline PPO
  model: `parameter_sha256 = 2ba6d7020b8747ec…` for both. Experiment 2 is
  therefore an extension *of the published run*, not a lookalike.
- **Model identity fixed.** `ml/model_fingerprint.py` adds
  `parameter_sha256` (sorted keys + name + dtype + shape + CPU-contiguous
  bytes), which is filename-independent; the raw `checkpoint_sha256` is
  retained alongside it. This is what makes the gate above checkable at all —
  the two files differ in raw bytes purely because `torch.save` embeds an
  archive name derived from the filename.
- **Primary evaluation**: 1000 DEV games vs **7× greedy**, seeds
  10,550,000–10,550,999, identical for every checkpoint (paired game by game).
- **Secondary diagnostic**: 500 DEV games vs `greedy4_random3` (seats 1–4
  greedy, 5–7 random — fixed positional assignment). It replaces the all-random
  field, which Experiment 1 showed is saturated at 1.000. Calibrated once on
  the warm start (4.55, spread distribution) on a separate DEV sub-range; no
  search over compositions. **The 4.5 threshold does not apply to it** — the
  tested seat is not exchangeable with the opponent seats — so it is used only
  to compare checkpoints against each other.
- **Drift corpus**: the *same frozen corpus as Experiment 1*, not regenerated
  — 4,440 states, fingerprint `2ec217b353bd`.

## Training budget table

| Iteration | Cumulative episodes | Primary checkpoint |
|---|---|---|
| 0 | 0 | ✔ warm start |
| 40 | 640 | ✔ original baseline budget |
| 80 | 1,280 | ✔ 2× |
| 160 | 2,560 | ✔ 4× |
| 320 | 5,120 | ✔ 8× |

## DEV performance curve

| iter | episodes | Greedy avg (1000 games) | 95% CI | Top-4 | Win | Mixed avg (500) | Expert agree | Warm-start agree | KL |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 6.554 | [6.438, 6.675] | 14.9% | 3.5% | 4.370 | 84.5% | 100.0% | 0.000 |
| 40 | 640 | 6.761 | [6.654, 6.868] | 12.8% | 2.7% | 4.432 | 77.2% | 90.2% | 0.371 |
| 80 | 1,280 | **6.325** | [6.203, 6.445] | 18.0% | 4.6% | 4.282 | 80.9% | 77.8% | 0.249 |
| 160 | 2,560 | 6.435 | [6.322, 6.547] | 19.1% | 4.5% | **4.206** | 74.3% | 73.1% | 0.484 |
| 320 | 5,120 | 6.606 | [6.497, 6.714] | 13.2% | 3.4% | 4.408 | **42.6%** | 40.8% | **1.171** |

## Paired comparisons

Deterministic paired bootstrap over the identical 1000 DEV seeds; positive =
the checkpoint places **worse** than the reference.

**vs the warm start (iteration 0), greedy field:**

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| iter 40 − iter 0 | **+0.207** | [+0.093, +0.322] | **worse — the baseline degradation, confirmed** |
| iter 80 − iter 0 | **−0.229** | [−0.392, −0.061] | **better than the warm start** |
| iter 160 − iter 0 | −0.119 | [−0.245, +0.008] | no clear difference |
| iter 320 − iter 0 | +0.052 | [−0.104, +0.210] | no clear difference |

**vs the original budget (iteration 40), greedy field:**

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| iter 80 − iter 40 | **−0.436** | [−0.593, −0.277] | better |
| iter 160 − iter 40 | **−0.326** | [−0.451, −0.205] | better |
| iter 320 − iter 40 | −0.155 | [−0.303, +0.000] | no clear difference (CI touches 0) |

**Intermediate diagnostic field**, vs iteration 0: iter 40 +0.062
[−0.078, +0.202]; iter 80 −0.088 [−0.254, +0.078]; iter 160 **−0.164
[−0.310, −0.020]**; iter 320 +0.038 [−0.122, +0.204].

**Pre-specified trend** across the primary budgets, per doubling of episodes:
**−0.0355** placements, 95% CI [−0.0797, +0.0074] — includes zero, so no
sustained monotone budget effect is established.

## Key observations

Facts supported directly by the measurements:

1. **The published degradation is real and now confirmed on DEV with adequate
   power.** At the original 640-episode budget, PPO places **+0.207
   [+0.093, +0.322]** worse than its warm start — the CI excludes zero, and
   the effect matches the +0.271 measured on 1000 TEST games. Experiment 1's
   flat curve was a power problem, exactly as its report said; at 1000 games
   the effect resolves.
2. **More data does help — but only transiently.** Doubling the budget to
   1,280 episodes produces a checkpoint that is significantly better than
   both the original budget (**−0.436**) and the warm start itself
   (**−0.229**). This is the first measured instance of PPO adding value in
   this project.
3. **The gain does not survive further training.** At 2,560 episodes the
   advantage over the warm start is marginal (CI touches zero); by 5,120
   episodes it is gone (+0.052, CI includes zero), and the checkpoint is back
   to roughly warm-start strength. The curve is **non-monotonic (a U shape)**,
   not a learning curve.
4. **Drift grows without bound the whole time.** Warm-start agreement falls
   monotonically 100% → 90.2% → 77.8% → 73.1% → **40.8%**, and KL rises
   0 → **1.171**. At 5,120 episodes the policy disagrees with its own warm
   start on nearly 3 decisions in 5.
5. **Expert agreement is non-monotonic and tracks performance.** 84.5% →
   77.2% → **80.9%** (the best-placing checkpoint partially *returned* toward
   expert behavior) → 74.3% → **42.6%**. The budget where placement is best
   is also where the policy is closest to the expert since training began.
6. **The intermediate field shows the same shape, not a different story.**
   4.370 → 4.432 → 4.282 → 4.206 → 4.408, best at iteration 160. There is no
   sign of gains against weaker opponents that fail to transfer to greedy.
7. **The learning signal is not degenerate, and the value head is healthy.**
   Across training blocks (1–40 / 41–160 / 161–320): value explained variance
   **0.614 / 0.682 / 0.688**; mean |raw advantage| 0.205 / 0.196 / 0.179;
   fraction of positive advantages 0.509 / 0.491 / 0.496. Returns and
   placements keep real spread (return sd ≈ 0.40–0.43, placement sd ≈ 1.71–1.79).
   Experiment 1's "weak signal" hypothesis is **weakened**: the advantages are
   signed, non-trivial in magnitude, and the critic predicts returns well.
8. **Entropy rises monotonically across blocks** (0.493 / 0.563 / 0.575) and
   clip fraction with it (0.056 / 0.068 / 0.070), while approximate KL per
   update *falls* (0.0113 / 0.0082 / 0.0073). The policy keeps flattening
   through small, individually-conservative updates.

## Action-category drift

Categories are derived from `bg_env`'s real 28-action space (buy / play /
sell / roll / level / freeze / end). Expert category mix over the frozen
corpus: roll 1352, end 1025, buy 823, play 750, level 422, sell 68,
**freeze 0**.

| Budget | Agreement | Largest expert→PPO changes | Drift concentrated in |
|---|---|---|---|
| 0 (warm start) | 84.5% | sell→roll 58 | buy 89%, sell 10% |
| 640 | 77.2% | end→roll 195, play→end 88 | buy 61%, end 20% |
| 1,280 | 80.9% | end→sell 98, sell→roll 63, roll→end 51 | buy 70%, end 15% |
| 2,560 | 74.3% | roll→end 274, end→roll 121 | buy 52%, roll 24% |
| 5,120 | **42.6%** | **roll→buy 935**, end→roll 251, play→buy 227, end→freeze 153 | roll 37%, buy 24%, end 20% |

Two things stand out:

- **The "buy" disagreement is a warm-start artifact, not PPO drift.** 74% of
  expert-buy states already disagree at iteration 0, and that share is flat
  across every budget (0.74, 0.74, 0.72, 0.72, 0.73) — the clone picks a
  different *shop slot*. The drift PPO actually introduces is concentrated in
  the **tempo decisions**: roll, end, and play.
- **At 5,120 episodes the policy leaves the expert's action space.** Roll
  disagreement jumps to 71% (from 4–20%), play to 40%, end to 50%, and the
  policy starts choosing **freeze — an action the greedy expert never takes
  even once** in 4,440 states (`end→freeze` 153).

## Optimization diagnostics

Means per training block (1–40 / 41–160 / 161–320): policy loss stays near
zero throughout, value loss 0.078 → 0.055 → 0.055, entropy 0.493 → 0.563 →
0.575, approximate KL 0.0113 → 0.0082 → 0.0073, clip fraction 0.056 → 0.068 →
0.070, gradient norm 1.080 → 1.144 → 1.041, rollout placement on the mixed
league field 5.788 → 5.503 → 5.524. Shaping contributes only in the first
block (sum −0.109) and is exactly 0 afterwards, as designed. Full
per-iteration series in [`rl_signal.json`](../results/ppo_budget_v1/rl_signal.json).

Near-zero policy loss is **not** read here as proof of failure — alongside a
0.69 explained-variance critic and non-degenerate advantages, it is consistent
with PPO's clipped objective doing small conservative updates.

## Which outcome does the data support?

Against the four pre-specified outcomes:

- **A (more data helps)** — *partially, and only up to 1,280 episodes.* The
  predicted shape was monotone improvement; what happened was a significant
  gain at 2× that decayed to nothing by 8×. The pre-specified trend statistic
  includes zero. A is **not** supported as stated.
- **B (flat while drift keeps rising)** — **best supported overall.** Between
  the two endpoints that matter (0 and 5,120 episodes) placement is
  statistically indistinguishable (+0.052 [−0.104, +0.210]) while drift grows
  without bound (KL 0 → 1.17, warm-start agreement → 40.8%). But B must be
  qualified: the path between those endpoints is *not* flat, and the
  transient gain at 1,280–2,560 episodes is real and reproducible from the
  committed artifacts.
- **C (progressively worse)** — **not supported.** No monotone decline; the
  worst primary checkpoint is the *original* 640-episode budget.
- **D (intermediate field improves, greedy does not)** — **not supported.**
  Both fields follow the same non-monotonic shape.

**Answer to the experiment's question: the current PPO algorithm is not simply
data-starved.** More experience does buy a real improvement at 2× the original
budget, but the effect is transient and gone by 8×, while policy drift
increases the entire time. Training budget alone is not the fix.

Per the protocol, iteration 80 is reported as a measured point, **not**
selected as a "best checkpoint" — its regression at higher budgets is reported
in the same breath, and no checkpoint was chosen using TEST (TEST was not run).

## Limitations

- **Training seed 0 only.** This is a budget-mechanism study on the exact
  historical trajectory, not a claim about PPO across random initializations.
  The U shape could be trajectory-specific; a single seed cannot distinguish a
  reproducible budget effect from a lucky excursion. This is the single
  biggest caveat on observation 2.
- Checkpoint binaries are gitignored per repo convention; only fingerprints
  are stored.
- The mixed diagnostic field is a comparison instrument, not a benchmark, and
  carries no absolute interpretation.

## Recommended Experiment 3 (do not run yet)

**Replicate the budget curve across independent training seeds.** Run the
identical frozen recipe (same shaping horizon, same checkpoints, same DEV
protocol at 1000 games) from training seeds 1, 2 and 3, and test one
pre-specified question: *does the significant improvement at ~1,280 episodes
reproduce, and does it decay by ~5,120 episodes?*

This is the right next step because every interpretation above hinges on
whether the U shape is a property of the algorithm or of this one trajectory —
and because the alternative candidates for Experiment 3 (anchoring the policy
to its BC prior, changing credit assignment, altering the opponent
distribution) all presuppose a stable phenomenon to fix. Observation 7 says
the critic and the advantage signal are healthy, so the "add signal" family of
interventions is not the indicated one; observation 4 says drift is unbounded,
which makes a KL/trust-region or anchoring intervention the natural *follow-up*
— after replication establishes what it must preserve.

No hyperparameters were tuned in this experiment, no checkpoint was selected,
and Benchmark v1 TEST was not touched.
