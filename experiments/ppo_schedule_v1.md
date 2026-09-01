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

## Success criteria (all required)

The scheduled arm at iter 320 must satisfy **all three**:

1. **Mean placement meaningfully below BC 6.550**
2. **≥3/4 training seeds beat BC** (Δ < −0.01 per seed)
3. **No catastrophic seed** (worst Δ vs BC ≤ +0.05) **and** variance within
   1.5× the fixed β=0.03 control

A mean of 6.540→6.530 alone is **not sufficient**.

## Hard stop

| Outcome | Action |
|---|---|
| **SUCCESS** | Freeze training procedure; eventual TEST confirmation |
| **FAIL** | STOP PPO tuning; pivot to Simulator Fidelity Phase 2 |

## Commands

```bash
pytest tests/
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
