# Simulator Fidelity Phase 2R — replacement-collapse mechanism diagnostic

Date: 2026-09-03 · Status: **`2r_v1` IN PROGRESS — measurement only** ·
Artifacts: [`results/sim_fidelity_phase_2r/`](../results/sim_fidelity_phase_2r/)

## Verdict

**HOLD.** Keep **#29 HOLD** and **#33 HOLD**. Confirm **11500–11699**
untouched. No residual-scaling retune. No Phase 2J α retune.
`PHASE_2Q_RECRUIT_VALUE_STATS` stays default **OFF**.

## Research question

> After 2Q unblocked replacements (scaling-blocked 80.3%→0; replace rate
> 1.55%→28.7%) but collapsed macro (greedy T10 post-scale 0.953→0.471;
> avg game length 15.6→13.1), does **replacement churn / combat-strength
> loss** explain most of the hole, or is the residual/pace coupling doing
> the damage?

## Design (measurement only)

One causal dimension stays the 2Q toggle. Scaling math, α=0.5, pool, shop,
economy, combat, and card effects are unchanged.

Every full-board replacement on **T8–T14** records:

- incumbent combat stats
- incumbent recruit-value stats
- candidate recruit stats
- combat-strength loss of sell→buy→play
- residual scaling added afterward
- next-turn carried strength
- replacement frequency / churn per turn
- death / game-length impact

Required outputs:

1. Per-turn decomposition: combat strength removed by replacement vs
   recruit gain vs residual scaling recovery
2. Distribution of replacement losses
3. Paired post-scale Firestone ratios and alive curve

## Predeclared routing

| Finding | Next design |
|---|---|
| Replacement churn/loss explains most of the post-scale hole (share ≥ 0.50, including compounded start-of-recruit carry) | Preserve legitimate accumulated combat value while using unscaled recruit-value for selection |
| Residual recovery hole dominates | Identify residual/pace coupling; do not jump to a combat-preserve split |
| Mixed / not reproduced | Inspect `gap_decomposition_by_turn` before designing |

## Arms

| Arm | Policy | `PHASE_2Q_RECRUIT_VALUE_STATS` |
|---|---|---|
| Primary control | raw greedy | OFF |
| Primary treatment | raw greedy | ON |
| Secondary control | BoardOpp α=0.5 | OFF |
| Secondary treatment | BoardOpp α=0.5 | ON |

Phase 2J is **report-only**. α is not retuned.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N / 2O / 2P / 2Q DEV | 11700–13699 | consumed |
| **2R DEV** | **13700–14199** | this phase |

## Protocol

```bash
pytest tests/test_phase_2r.py
python -m ml.fidelity_phase_2r          # 13700–14199
```

## Results

*(filled after the clean-tree DEV run)*
