# Replay Experiment 6 — Scheduled KL Anchoring (protocol)

Date: 2026-09-01 · Status: **protocol only — not yet run** ·
Branch: `cursor/ppo-schedule-v1-ffbb`

## Context

Experiment 5 closed the fixed-β question. **β=0.03** is the best fixed
tradeoff (mean 6.540 vs BC 6.550, std 0.028) but only 2/4 seeds beat BC.
**β=0.01** shows exploration upside with high variance.

PR #14 is merged. Fixed-β tuning is **finished**.

This is the **last PPO experiment on the current simulator**.

## Question

> Can delayed relaxation of KL anchoring preserve β=0.03's cross-seed stability
> while recovering the exploration benefit seen at β=0.01?

## Design

| Arm | KL anchoring | Training |
|---|---|---|
| **Control** | fixed β=0.03 | Reused from `results/ppo_dose_v1/beta003/` |
| **Treatment** | scheduled (below) | 4 new trajectories (seeds 0–3) |

### Frozen schedule (1-based iterations)

```text
Iterations   β
1–160        0.030
161–320      linear 0.030 → 0.010
```

Encoded as: `0.03@160,0.01@320`

No adaptive rules. No annealing to zero.

## Frozen contract

Same as Experiments 4b/5:

- Warm-start SHA: `c85de276471872c8d5c5365bd3550e9a31ae2cbc5046244f2869392dd0cacf1c`
- Python 3.12.3, Torch 2.13.0+cpu
- 320 iterations × 16 episodes = 5,120 episodes
- DEV seeds 10,550,000–10,550,999 (1000 games vs greedy)
- TEST locked

## Control code equivalence gate (required before training)

Before scheduled runs, one shadow fixed β=0.03 seed-0 trajectory using the
current `train_ppo.py` must match Experiment 5's β=0.03 seed-0 parameter
SHA256 at iter 0, 40, 80, 160, and 320.

```bash
python scripts/shadow_control_equivalence.py
```

If all five hashes match: `control_code_equivalence_passed: true` — reuse the
existing four-seed β=0.03 control.

If any differ: **STOP** and retrain both control and treatment under current code.

## Success criteria (all required)

The scheduled arm at iter 320 must satisfy **all four**:

1. **Mean Δ vs BC ≤ −0.02** (cross-seed mean ≤ **6.530**)
2. **≥3/4 training seeds beat BC** (Δ < −0.01 per seed)
3. **No catastrophic seed** (worst Δ vs BC ≤ +0.05)
4. **Cross-seed std ≤ 1.5×** fixed β=0.03 control std

A marginal mean like 6.549 does **not** pass criterion 1.

## Commands

```bash
pytest tests/
python scripts/shadow_control_equivalence.py
python scripts/train_schedule_ab.py
python scripts/eval_schedule_ab.py
python scripts/ppo_schedule_report.py
python scripts/ppo_schedule_manifest.py
```

## Artifacts (after run)

- `results/ppo_schedule_v1/`
- `experiments/ppo_schedule_v1.md` (results write-up)
- `ml/kl_schedule.py` — schedule parser
- `tests/test_kl_schedule.py`, `tests/test_schedule_study.py`
