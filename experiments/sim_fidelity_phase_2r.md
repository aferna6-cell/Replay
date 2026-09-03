# Simulator Fidelity Phase 2R — replacement churn / combat-loss diagnostic

Date: 2026-09-03 · Status: **`2r_v1` RUNNING — measurement only** ·
Artifacts: [`results/sim_fidelity_phase_2r/`](../results/sim_fidelity_phase_2r/)

## Intent

Phase 2Q confirmed that recruit-value valuation unblocks scaling-blocked
upgrades (80%→0%) and lifts full-board replace rate (~1.6%→29% greedy), but
post-scale macro collapses (T10 0.95→0.47; games 15.6→13.1).

**Phase 2R does not retune scaling or α.** It instruments every completed
full-board replacement on T8–T14 to isolate whether replacement churn/combat
loss explains the collapse, or whether residual/pace coupling dominates.

## Protocol

- Control: `PHASE_2Q_RECRUIT_VALUE_STATS=False`
- Treatment: `PHASE_2Q_RECRUIT_VALUE_STATS=True` (arm-scoped only; default OFF)
- Primary: greedy A/B
- Secondary: Phase 2J α=0.5 report-only (frozen; no retune)
- Seeds: **13700–14199** (fresh DEV after 13699)
- Confirm **11500–11699** reserved
- Keep **#29 / #33 HOLD**; no merge

Per completed sell→buy→play (or sell→play) replacement:

| Field | Meaning |
|---|---|
| Incumbent combat / recruit | Live `attack+health` vs `recruit_*` |
| Candidate combat / recruit | Shop/hand unit stats |
| `combat_strength_loss` | incumbent combat − candidate combat |
| `recruit_value_gain` | candidate recruit − incumbent recruit |
| Residual scaling added | end-of-recruit residual budget that turn |
| Next-turn carried strength | start-of-recruit strength next turn |

Required outputs:

1. Per-turn decomposition: combat removed vs recruit gain vs residual recovery
2. Distribution of replacement combat losses
3. Paired post-scale Firestone ratios + alive curve

## Route

| Finding | Next design |
|---|---|
| `replacement_churn_loss_explains_macro_collapse` | Preserve legitimate accumulated combat value while using unscaled recruit value for selection |
| `residual_or_pace_coupling_dominates` | Inspect residual/pace coupling (target, clamp, apply timing) |

Threshold: cumulative excess unrecovered replacement combat loss over T8–T10
explains ≥55% of the T10 post-scale combat-stats deficit (same-turn T10 fraction
also reported).

## Commands

```bash
pytest tests/test_phase_2r.py
python -m ml.fidelity_phase_2r          # 13700–14199
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N–2Q DEV | 11700–13699 | consumed |
| **2R DEV** | **13700–14199** | this phase |
