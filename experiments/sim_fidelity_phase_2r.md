# Simulator Fidelity Phase 2R — replacement churn / combat-loss diagnostic

Date: 2026-09-03 · Status: **`2r_v1` HOLD — churn/loss explains macro collapse** ·
Artifacts: [`results/sim_fidelity_phase_2r/`](../results/sim_fidelity_phase_2r/)

## Verdict

**HOLD.** Keep **#29 / #33 HOLD**. Confirm **11500–11699** untouched. No
scaling or α retune. Toggle `PHASE_2Q_RECRUIT_VALUE_STATS` remains default **OFF**.

Primary finding: **`replacement_churn_loss_explains_macro_collapse`**

Cumulative excess unrecovered replacement combat loss over T8–T10 accounts for
**99.4%** of the T10 post-scale combat-stats deficit (threshold 55%). Next design
should **preserve legitimate accumulated combat value while using unscaled
recruit value for selection** — not α retune, not confirm burn.

## DEV 13700–14199 (greedy primary)

| Metric | Control | Treatment |
|---|---:|---:|
| Full-board replace rate | 1.62% | **29.98%** |
| Completed replacements T8–T14 | 1,467 | **24,848** |
| Mean combat loss / replace | −0.8 | **+96.1** |
| Median combat loss | 0 | **56** |
| P90 combat loss | 5 | **226** |
| Share loss ≥50 | 0% | **51.4%** |
| Sum combat removed | −1.1k | **+2.39M** |
| Sum recruit-value gain | +7.4k | +86.7k |
| Sum residual scaling added | 23.7M | **3.7M** |
| Post-scale / Firestone T10 | **0.954** | **0.468** |
| Post-scale / Firestone T14 | 1.828 | **0.115** |
| Avg game length | 15.73 | **13.09** |
| Churn explains frac T10 (cum T8–T10) | — | **0.994** |
| Same-turn T10 frac | — | 0.359 |

### Per-turn decomposition (treatment − control, mean per seat-turn)

| Turn | Δ replaces | Δ combat removed | Δ recruit gain | Δ residual | Δ net after residual | Δ post-scale stats |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | +3.24 | +188 | +11.8 | −306 | **+494** | −495 |
| 10 | ~0 | ~0 | ~0 | −279 | **+279** | **−778** |
| 11 | +3.56 | +465 | +11.8 | −1726 | **+2191** | −2967 |
| 12 | +3.09 | +426 | +9.8 | −3507 | **+3932** | −6948 |
| 13 | +2.65 | +381 | +8.0 | −4146 | **+4527** | −11646 |
| 14 | +2.27 | +251 | +6.5 | −2055 | **+2306** | −14204 |

Interpretation: treatment sells combat-inflated incumbents for printed
candidates (mean loss ~96). Residual, being growth-on-current, then adds far
less on the cratered board — a death spiral, not an independent residual bug.
Carry-forward from T9 cratering dominates the T10 deficit (same-turn T10
replacements alone explain only ~36%).

### Paired alive curve (sim players alive)

| Turn | Control | Treatment | Δ |
|---:|---:|---:|---:|
| 8–9 | equal | equal | 0 |
| 10 | 5.36 | 5.11 | −0.26 |
| 12 | 3.66 | 2.96 | −0.71 |
| 14 | 2.60 | 2.18 | −0.43 |

## Phase 2J α=0.5 (report-only, no retune)

Same route: churn frac **1.004**. Replace rate 6.9%→20.0%; post-scale T10
0.878→0.661; game length 15.1→13.4. Mean combat loss already high under control
(~306; BoardOpp replaces more) and rises further under treatment (~336).

## Route → next design

```text
2Q recruit-value valuation
        ↓
mass sell of scaled combat for printed shop/hand
        ↓
combat crater (mean loss ~96 / replace)
        ↓
residual (ratio-on-current) undershoots
        ↓
post-scale macro collapses; games shorten
```

**Next:** preserve legitimate accumulated combat value on replace while keeping
unscaled recruit-value for selection (e.g. combat carry-over / transfer on
upgrade, or residual budget that does not assume scaled incumbents persist).
Do **not** retune α; do **not** burn confirm seeds.

Independent QA ([`sim_fidelity_phase_2r_qa.md`](sim_fidelity_phase_2r_qa.md)):
the **0.9938** T8–T10 identity recomputes from committed per-turn tables;
event accounting (sell→buy→play, residual recovery, next-turn carry) is clean.
Phase 2S is preregistered only — see [`sim_fidelity_phase_2s.md`](sim_fidelity_phase_2s.md).

## Protocol

```bash
pytest tests/test_phase_2r.py
python -m ml.fidelity_phase_2r          # 13700–14199
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N–2Q DEV | 11700–13699 | consumed |
| **2R DEV** | **13700–14199** | consumed |
