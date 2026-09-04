# Simulator Fidelity Phase 3G — punch-sample selection decomposition

Date: 2026-09-04 · Status: **`3g_v1` HOLD — `mixture_role_selection_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3g/`](../results/sim_fidelity_phase_3g/)

Stacked on PR #52 (`cursor/phase-3f-carry-divergence-765c`, head `4cd321b`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3F DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3F left the #51 punch-row Δcarry **−196** as **9.1% paired unconditional /
90.9% selection**, with unpaired T1 opposing carry **798 → 65**. This hour
splits that crater with a symmetric Kitagawa / Oaxaca reweight on common
support.

## Verdict

**HOLD.** Route: **`mixture_role_selection_dominates`**.

3D A1, the 3E carry share, the unpaired punch Δcarry **−196.333**, and the
3F paired-uncond / selection shares all reproduce exactly. On the same
3E punch-row sample (54223 / 50116), treatment − control mean carry is
**810 → 614**.

The entire crater is the **turn × winner-start-tier mixture** (who
produces punch rows). Within a matched `(turn, winner-start-tier)` cell,
opponent carry is arm-matched (Δ **+0.20**, share **−0.001**). Role /
alive / elimination selection inside those cells **offsets** the crater
(+36, share **−0.185**). Leftover is **0**. Mixture + role share is
**0.816**.

Treatment T1–T3 rows are **disproportionately early / low-carry
opponents**, not a true within-cell pool deficit. Control T1–T3 still
punch at T12–T14 against 1600–5700 carry boards; treatment T1–T3 mass
collapses after T8 and is almost gone by T12.

Do **not** apply a scaling correction. Do **not** retune constants.
Next hour traces the upstream gameplay / composition process that
changes those winner-tier × turn matchups.

| Component | Δcarry | Share of #51 −196 |
|---|---:|---:|
| (1) turn × winner-start-tier mixture | **−196.53** | **1.001** |
| (2) opponent carry \| matched turn × tier | +0.20 | −0.001 |
| (3) role / alive / elimination (nested) | +36.35 | −0.185 |
| (4) leftover | ~0 | ~0 |
| mixture + role (decision) | — | **0.816** |

## Classification (observational)

Punch rows are the 3E start-minion sample on T7–T14 hit fights. Cells
are `(turn, winner_start_tier)`. Weights are `n / N` per arm. Symmetric
Kitagawa:

```text
mix  = Σ (w_T − w_C) · (μ_C + μ_T) / 2
rate = Σ (w_C + w_T) / 2 · (μ_T − μ_C)
```

Exclusive outer cells (one arm empty) count as mixture — they are the
extreme of who produces punch rows. Role / alive is nested inside
common-support outer cells on `(n_alive bin, winner_tavern low/high)`
and is a refinement of (2), not an extra slice of Δ.

```text
applied        = _hero_damage          unchanged
3E carry term  = unpaired punch Δcarry −196
3G mixture     = 100.1% of −196
3G within-cell = ~0
3G mix+role    = 81.6% of −196
```

## DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#52](https://github.com/aferna6-cell/Replay/pull/52).

Punch-row n **54223 / 50116**. Mean carry **809.97 vs 613.63**.
History-link **49960 / 49960** and **46426 / 46426**. Ghost/bye skipped
(7 / 7). Weight sums = 1. `mix + rate + leftover` recovers −196 exactly.

Within-cell opponent carry is matched at every early cell (T7 W1
**32.0 vs 32.0**, p50 **31 / 31**). The 3F unpaired T1 **798 → 65** is
the unadjusted mix of those early T1 rows with control-only late T1
rows (T13 W1 **3479**, T14 W1 **5710**; treatment T14 W1 **n = 0**).

### T1–T3 early / low-carry diagnostic

Verdict: **`disproportionately_early_low_carry`**.

| Arm | n | p(T7–T9) | mean turn | mean carry | early carry | late carry |
|---|---:|---:|---:|---:|---:|---:|
| Control | 42486 | 0.578 | 9.42 | 798 | 77 | 1786 |
| Treatment | 26170 | **0.837** | **8.17** | **152** | 69 | 577 |

Turn-mixture share of the T1–T3 Δ (**−646**) is **1.012**. Within-turn
rate is **−0.012**. Treatment’s T1–T3 weight shift onto T7–T8 is
**+0.259**.

### Punch-row weights (selected cells)

| Cell | n C / n T | w C / w T | C carry | T carry | Δ |
|---|---:|---:|---:|---:|---:|
| T7 W1 | 2618 / 2730 | 0.048 / 0.054 | 32 | 32 | +0.1 |
| T8 W1 | 2237 / 2283 | 0.041 / 0.046 | 83 | 82 | −1.5 |
| T9 W1 | 1823 / **134** | 0.034 / **0.003** | 132 | 131 | −0.9 |
| T10 W1 | 1472 / **106** | 0.027 / **0.002** | 389 | 377 | −12 |
| T12 W1 | 885 / **2** | 0.016 / 0.000 | 1639 | 1762 | +123 |
| T14 W1 | 579 / **0** | 0.011 / 0 | 5710 | — | — |
| T12 W5 | 64 / 1250 | 0.001 / 0.025 | 1591 | 1642 | +52 |
| T13 W6 | 0 / 877 | 0 / 0.017 | — | 3505 | — |
| T14 W6 | 0 / 694 | 0 / 0.014 | — | 5701 | — |

Treatment keeps producing late punch rows, but they are T5–T6
winner-start bodies, not T1–T3. Exclusive T6 / vanished late T1 cells
are mixture, not a within-cell carry hole.

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Punch-row n (3E sample) | 54223 | 50116 |
| Cell counts sum to n | 54223 / 54223 | 50116 / 50116 |
| Mixture weights sum | 1.0 | 1.0 |
| mix + rate + leftover = Δ | −196.333 = −196.333 | gap 0 |
| Live punch → prior-turn seat history | 49960 / 49960 | 46426 / 46426 |
| Ghost / no-loser skipped | 7 | 7 |
| Carry / scale-add / history-gap mismatches | 0 | 0 |
| 3D A1 reproduced | 0.4216721428553852 | 0.4216721428553852 |
| 3E carry share of A1 reproduced | 0.7236353954551374 | 0.7236353954551374 |
| Punch-row Δcarry reproduced | −196.33317557443002 | −196.33317557443002 |
| 3F uncond / selection shares | 0.09084 / 0.90916 | 0.09084 / 0.90916 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
#51 carry term = unpaired punch Δcarry −196
        ↓
turn × winner-start-tier mixture = −196.53 (100.1%)
within matched turn × tier carry = +0.20 (~0)
role / alive / elimination       = +36 (−18.5%; offsets)
leftover                         = 0
mixture + role                   = 81.6%
        ↓
mixture_role_selection_dominates
        ↓
next: trace the upstream gameplay / composition
      process that changes those winner-tier ×
      turn matchups (who still fields T1–T3
      winner-start bodies late, and which seats
      remain alive to be punched), not a
      within-cell pool deficit and not scaling
      do not apply a scaling correction
      do not rewrite 2Q; do not change constants
      do not burn confirm
```

**Not** a scaling-input audit. **Not** a 2Q rewrite. **Not** a
`_hero_damage` change. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3F / **3G DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3g.py tests/test_phase_3f.py tests/test_phase_3e.py tests/test_phase_3d.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3g          # reused 14200–14699
```

Working tree was clean at contract time (`8071237`, 149.26s). Tracer is
observational.
