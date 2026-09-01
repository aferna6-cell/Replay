# Replay Experiment 6 — Scheduled KL Anchoring

Date: 2026-09-01 · Split: **DEV only** · Status: **COMPLETE — STOP** ·
Artifacts: [`results/ppo_schedule_v1/`](../results/ppo_schedule_v1/) ·
Manifest: [`manifest.json`](../results/ppo_schedule_v1/manifest.json)

## Question

> Can delayed relaxation of KL anchoring (β=0.03 through iter 160, then linear
> to β=0.01 by iter 320) preserve fixed-β=0.03 stability while reliably
> beating the BC warm start (~6.550)?

This was the **last PPO experiment on the current simulator**.

## Design

| Arm | KL anchoring | Data |
|---|---|---|
| **Control** | fixed β=0.03 | Reused from `results/ppo_dose_v1/beta003/` |
| **Treatment** | `0.03@160,0.01@320` | 4 new trajectories (seeds 0–3) |

Schedule (1-based iterations):

```text
1–160:   β = 0.030
161–320: linear 0.030 → 0.010
```

Never anneals to zero.

## Pre-flight gates

### Full test suite

```
389 passed, 5 skipped
```

### Control code equivalence

Shadow fixed β=0.03 seed-0 run with PR #15 `train_ppo.py` matched Experiment 5
β=0.03 seed-0 parameter SHA256 at all five checkpoints:

| Iter | Match |
|---|---|
| 0 | OK |
| 40 | OK |
| 80 | OK |
| 160 | OK |
| 320 | OK |

`control_code_equivalence_passed: true` — Experiment 5 control reused.

## Frozen success criteria (all required)

| Criterion | Threshold |
|---|---|
| Mean Δ vs BC | ≤ −0.02 (mean ≤ **6.530**) |
| Seeds beating BC | ≥ 3/4 (Δ < −0.01) |
| Worst seed Δ vs BC | ≤ +0.05 |
| Cross-seed std | ≤ 1.5× control std |

## Iter-320 results (primary DEV, 1000 games vs greedy)

| Arm | Mean | Std | Δ vs BC | Mean KL |
|---|---|---|---|---|
| Control (fixed β=0.03) | **6.540** | **0.028** | **−0.010** | 0.335 |
| Scheduled (β=0.03→0.01) | 6.618 | 0.054 | +0.068 | 0.600 |

### Per-seed iter-320 vs BC (Δ = placement − 6.550)

| Seed | Control Δ | Scheduled Δ |
|---|---|---|
| 0 | +0.023 | −0.015 |
| 1 | −0.014 | **+0.124** |
| 2 | **−0.052** | +0.059 |
| 3 | +0.003 | +0.105 |

### Success criteria evaluation (scheduled arm)

| Criterion | Required | Actual | Pass? |
|---|---|---|---|
| Mean Δ vs BC | ≤ −0.02 | +0.068 | **No** |
| Seeds beating BC | ≥ 3 | 1 | **No** |
| Worst seed Δ | ≤ +0.05 | +0.124 | **No** |
| Std ratio vs control | ≤ 1.5× | 1.94× | **No** |

## Key observations

1. **Through iter 160, scheduled and control policies are identical** (expected:
   both use β=0.03; shadow equivalence gate confirmed fixed-β code path).

2. **Annealing to β=0.01 after iter 160 hurt rather than helped.** Scheduled
   cross-seed mean regressed from 6.570 (iter 160) to 6.618 (iter 320), while
   fixed control improved to 6.540.

3. **Seed 1 catastrophic under schedule** (+0.124 vs BC at iter 320) — the
   failure mode β=0.01 showed in Experiment 5, now triggered by late annealing.

4. **Fixed β=0.03 remains the best known PPO recipe** on this simulator but
   still does not meet the strengthened success bar (mean ≤ 6.530, 3/4 seeds).

## Outcome: **STOP**

Experiment 6 failed all four pre-specified criteria. **PPO tuning on the
current simulator is complete.**

## Recommended next action

**Pivot to Simulator Fidelity Phase 2.** Do not run schedule variants, β grids,
or further PPO coefficient experiments.

Priority areas (from prior simulator research):

1. Fix late-game scaling runaway (~4.3× real board strength by turn 14)
2. Replace abstract multiplicative growth with real card effects
3. Calibrate board growth turn-by-turn against real data
4. Measure composition quality, not only total stats
5. Add hero/trinket/anomaly context
6. Keep fast combat simulator; calibrate specific failures only

Once simulator behavior changes materially, define Replay Benchmark v2 and
retrain BC / DAgger / PPO inside the improved environment.

## ML research arc (closed)

1. Diagnosed PPO instability across seeds
2. Built reproducibility infrastructure after discovering confounds
3. Experiment 4b: causally showed KL anchoring fixes instability
4. Experiment 5: mapped dose-response; β=0.03 best fixed tradeoff
5. Experiment 6: scheduled relaxation did not beat fixed β=0.03 or BC reliably
6. **Hard stop enforced** — no hidden weak results behind endless tuning

## Commands (reproducibility)

```bash
pytest tests/
python scripts/shadow_control_equivalence.py
python scripts/train_schedule_ab.py
python scripts/eval_schedule_ab.py
python scripts/ppo_schedule_report.py
python scripts/ppo_schedule_manifest.py
```

## Limitations

- Four training seeds
- Control reused from Experiment 5 (equivalence-gated)
- TEST not run (by design)
- Checkpoint binaries gitignored; fingerprints in manifest
