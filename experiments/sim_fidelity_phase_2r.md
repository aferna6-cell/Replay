# Simulator Fidelity Phase 2R — replacement-collapse mechanism diagnostic

Date: 2026-09-03 · Status: **`2r_v1` COMPLETE — replacement churn/loss explains the 2Q collapse** ·
Artifacts: [`results/sim_fidelity_phase_2r/`](../results/sim_fidelity_phase_2r/)

## Verdict

**HOLD.** Keep **#29 HOLD** and **#33 HOLD**. Confirm **11500–11699**
untouched. No residual-scaling retune. No Phase 2J α retune.
`PHASE_2Q_RECRUIT_VALUE_STATS` stays default **OFF**.

Primary finding: **`replacement_churn_loss_explains_macro_collapse`**

The 2Q macro collapse reproduces on fresh DEV **13700–14199**. Selling
combat-inflated incumbents for printed Tavern units is the initiating
cause. Residual under-recovery is mostly the cratered-board budget
(`ratio_add ∝ current`), not an independent pace-formula defect.

**Next design:** preserve legitimate accumulated combat value on
incumbents while using unscaled recruit-value for selection.

## Research question

> After 2Q unblocked replacements (scaling-blocked 80.3%→0; replace rate
> 1.55%→28.7%) but collapsed macro (greedy T10 post-scale 0.953→0.471;
> avg game length 15.6→13.1), does **replacement churn / combat-strength
> loss** explain most of the hole, or is the residual/pace coupling doing
> the damage?

## Design (measurement only)

One causal dimension stays the 2Q toggle. Scaling math, α=0.5, pool, shop,
economy, combat, and card effects are unchanged.

Every full-board replacement on **T8–T14** records incumbent combat,
incumbent recruit-value, candidate recruit, sell→buy→play combat loss,
residual added afterward, next-turn carried strength, churn, and
death / game-length.

Attribution (predeclared, ≥0.50 share):

- **Replacement-initiated** = same-turn replacement net + carried
  start-of-recruit hole + crater-induced residual shrink
- **Independent residual** = residual-recovery hole minus crater shrink

## DEV 13700–14199 (500 lobbies, 457s, clean harness `604e8c6`)

### Greedy causal test (primary)

| Metric | Control | Treatment |
|---|---:|---:|
| Replacements T8–T14 | 1,467 | **24,848** |
| Mean combat loss / replace | −0.8 | **+96.1** |
| Mean incumbent combat / recruit | 10.9 / 6.6 | **107.1 / 7.5** |
| Mean candidate recruit | 11.7 | 11.0 |
| Mean inflation (combat/recruit) | 1.94× | **13.9×** |
| T9–T12 replacement net | +0.1 | **−231** |
| Post-scale / Firestone T10 | **0.954** | **0.468** |
| Post-scale / Firestone T14 | 1.828 | **0.115** |
| Avg game length | 15.73 | **13.09** |

2Q headline reproduced: T10 0.953→0.471 and length 15.6→13.1 on a new seed band.

### Per-turn decomposition (greedy means)

| Turn | Arm | Repl | Combat removed | Recruit gain | Repl net | Residual | Post | Post/FS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | C / T | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 103 / 103 | 352 / 352 | 1.150 / 1.150 |
| 9 | C | 1,467 | 5.3 | 5.7 | **+0.4** | 581 | 944 | 1.308 |
| 9 | T | **11,239** | **224** | 36 | **−188** | **275** | **448** | **0.621** |
| 10 | C | 0 | 0 | 0 | 0 | 559 | 1,527 | 0.954 |
| 10 | T | 0 | 0 | 0 | 0 | 280 | 749 | **0.468** |
| 11 | C / T | 0 / 6,612 | 0 / 505 | 0 / 40 | 0 / −465 | 2,191 / 465 | 3,765 / 799 | 1.183 / 0.251 |
| 12 | C / T | 0 / 4,093 | 0 / 464 | 0 / 38 | 0 / −426 | 3,961 / 454 | 7,844 / 896 | 1.467 / 0.168 |
| 14 | C / T | 0 / 734 | 0 / 283 | 0 / 32 | 0 / −251 | 2,335 / 280 | 15,159 / 955 | 1.828 / 0.115 |

T8 is identical (toggle has nothing to sell yet). **T9 is the initiating
turn:** every treatment seat replaces (3.72 / seat-turn; full-board
replace rate 39.6% vs control 6.0%). Control replacements are printed-for-
printed (mean loss −0.8). Treatment sells 13.9× inflated combat for ~11
printed recruit.

**T10 has zero replacements in both arms.** The T10 post-scale hole
(778 combat; 0.954→0.468× Firestone) is **64% carried start-of-recruit**
from the T9 crater and 36% smaller residual on that weaker current.
Next-turn carried strength matches post-scale for survivors.

### T9 residual is not an independent pace bug

Treatment T9 residual 275 vs control 581 (hole 306). The observational
counterfactual — apply the recorded `ratio_g` / clamp to
`end_recruit + replacement_net_loss` — recovers **574**, i.e. crater
shrink **+298**. Almost the entire residual hole is `ratio_add ∝ current`.

### Replacement-loss distribution (greedy)

| | Control (n=1,467) | Treatment (n=24,848) |
|---|---:|---:|
| Mean | −0.8 | **96.1** |
| Median | 0 | **56** |
| p90 | 5 | **226** |
| p95 | 6 | **413** |
| Share combat_loss > 0 | 45% | **93%** |

Treatment mass sits in `[40,160)` (39%) with a long tail ≥160 (16%).
Control losses never leave `[0,10)`.

### Paired post-scale Firestone and alive curve (greedy)

| Turn | Post/FS C | Post/FS T | Alive C | Alive T |
|---:|---:|---:|---:|---:|
| 8 | 1.150 | 1.150 | 7.20 | 7.20 |
| 9 | 1.308 | **0.621** | 6.20 | 6.20 |
| 10 | **0.954** | **0.468** | 5.36 | **5.11** |
| 11 | 1.183 | 0.251 | 4.42 | 3.96 |
| 12 | 1.467 | 0.168 | 3.66 | 2.96 |
| 14 | 1.828 | 0.115 | 2.60 | 2.18 |

Alive curves first split at T10, after the T9 crater has already been
scaled and fought. Treatment damage after a replacement turn is 7.57 vs
5.26 with no replace; deaths shift earlier (T9–T12). Game length
15.73→13.09.

### Attribution (greedy)

| Window | Replacement-initiated share | Independent residual |
|---|---:|---:|
| T9 (initiating) | **0.98** (188 net + 298 crater shrink) / 495 | 0.02 |
| T10 (2Q headline) | **0.64** (all carried start) | 0.36 |
| T9–T12 mean | **0.60** | 0.40 |

Same-turn replacement net is only ~10% of the T9–T12 *post-scale* hole
because residual then compounds on the crater. Crediting that coupling
to the replacement still clears 0.50. Independent residual does not.

### Phase 2J α=0.5 (report-only, no retune)

| Metric | Control | Treatment |
|---|---:|---:|
| Replacements | 4,353 | 9,689 |
| Mean combat loss / replace | 306 | 335 |
| Post-scale T10 | 0.878 | **0.661** |
| Post-scale T14 | 1.527 | **0.508** |
| Avg game length | 15.14 | **13.41** |

Same direction, smaller amplitude. BoardOpp already replaces some scaled
incumbents under control (mean loss 306), so the toggle adds less
incremental churn than greedy. Mechanism survives; α is not retuned.

## Interpretation

```text
T9: recruit-value selection sells 13.9× inflated combat for printed shop
        ↓  combat removed ≈224 / seat-turn; recruit gain ≈36
end-of-recruit craters (362 → 173)
        ↓  residual ratio_add ∝ current
residual recovers ~275 instead of ~580
        ↓
T10 starts at 469 vs 968; no further replaces
        ↓
post-scale T10 0.47× Firestone; earlier deaths; games shorten
```

This is **not** “residual math broke on the new seed band.” Residual does
what Simulator v1.1 always did: scale the board that recruit left behind.
Naive 2Q selection throws away the combat stock residual is supposed to
grow.

## Next (not done here)

Preserve **legitimate accumulated combat value** on the incumbent while
keeping **unscaled recruit-value** for “is this shop unit an upgrade?”
Do not retune residual budget, do not retune α=0.5, do not burn
11500–11699, do not default the 2Q toggle on.

## Protocol

```bash
pytest tests/test_phase_2r.py
python -m ml.fidelity_phase_2r          # 13700–14199
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N / 2O / 2P / 2Q DEV | 11700–13699 | consumed |
| **2R DEV** | **13700–14199** | **consumed** |
