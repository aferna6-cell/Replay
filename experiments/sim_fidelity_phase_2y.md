# Simulator Fidelity Phase 2Y — slot / attack-order vs teammate protection

Date: 2026-09-04 · Status: **`2y_v1` HOLD — `unexplained_combat_mechanics`** ·
Artifacts: [`results/sim_fidelity_phase_2y/`](../results/sim_fidelity_phase_2y/)

Stacked on PR #42 (`cursor/phase-2x-synthetic-allocation-e49b`, head `7d88748`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–2X DEV **14200–14699** (no new seeds).

2X left **+1.372 / hit (81.7% of 2V B)** after holding tavern tier +
recruit/raw + synthetic share. This hour splits that residual into
board-slot / attack-opportunity vs teammate-protection vs leftover
combat mechanics.

## Verdict

**HOLD.** Route: **`unexplained_combat_mechanics`**.

2X residual R is reproduced exactly. Holding tier + recruit/raw + synth
share, then slot bin, then teammate-raw quintile: **neither (A) nor (B)
clears ~70% of R**. Slot mix is **negative** (−23.5%). Teammate
protection is the largest *identified* positive term (54.5%) but misses
the bar. Leftover same-slot / same-teammate survival is **68.9%**.

Do **not** audit the recruit/play positioning policy yet. Do **not**
rewrite 2Q. Do **not** jump to a board-composition / effect rewrite.
Next hour: isolate the specific combat mechanic (taunt / targeting /
deathrattle / attack-cursor) before any behavior change. Do not change
`_hero_damage`. Do not burn confirm.

| Component of 2X residual R (+1.372) | Δ / hit | Share of R |
|---|---:|---:|
| (A) Slot / attack opportunity | −0.322 | **−0.235** |
| (B) Teammate / board-size protection | +0.748 | 0.545 |
| (C) Unexplained combat mechanics | **+0.946** | **0.689** |
| Nested residual vs 2X R | ~0 | — |

A + B + C = R. Nested terms add to 2V B (residual ~0).

## Classification (observational)

For every decisive T7–T14 hit, each winner starting minion is tagged by
printed tavern tier, recruit/base raw, synthetic share, combat raw,
board slot, first-attack index, n_attacks, death-before-first-attack,
taunt, n_targeted / was_targeted, teammate combat-raw excluding self,
board size, and survival.

```text
applied        = _hero_damage          unchanged
2X residual R  = Σ_t t · n̄_t · ΔP(survive|t,r,s)
ΔP(survive|t,r,s) = slot-mix + teammate-mix + leftover
```

Exclusive T6 stays in 2V A (composition), not B/R. Tracer does not
consume RNG. Play still appends (`board.append`); no positioning rewrite.

Slot bins are 0 / 1 / 2 / 3 / 4+. Teammate-raw uses per-tier quintiles.
Board size is ~7 on both arms (Δ≈0), so (B) is teammate *strength*, not
count.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#42](https://github.com/aferna6-cell/Replay/pull/42). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

Unconditional T4/T5 still sit earlier and swing more — that 2X
descriptive fact is real. After holding stats it does **not** explain
leftover survival. Earlier slots are front-liners; moving mass there
predicts *less* survival (negative mix). Dying-before-first-attack
drops because those bodies live longer, not because slot composition
accounts for ΔP.

| Tier | Ctrl slot | Treat slot | Ctrl n_atk | Treat n_atk | Ctrl P(die pre-atk) | Treat P(die pre-atk) |
|---|---:|---:|---:|---:|---:|---:|
| T3 | 3.47 | 2.92 | 0.67 | 0.84 | 0.300 | 0.140 |
| T4 | 4.76 | **3.46** | 0.46 | **0.82** | 0.395 | **0.149** |
| T5 | 5.89 | **3.84** | 0.28 | **0.78** | 0.518 | **0.151** |

First-attack index also moves earlier (T4 7.01→5.46; T5 7.40→5.96).
Taunt share stays low (T4 ~0.08; T5 ~0.04). Targeting exposure rises
only slightly (T4 0.615→0.672; T5 0.596→0.650).

| Tier | 2X leftover | (A) slot | (B) teammate | (C) unexplained |
|---|---:|---:|---:|---:|
| T3 | +0.350 | −0.103 | +0.156 | +0.297 |
| T4 | +0.812 | −0.078 | +0.386 | **+0.504** |
| T5 | +0.298 | +0.003 | **+0.239** | +0.056 |

T5 leftover is mostly teammate protection. T4 — the bulk of R — is
mostly unexplained. Weighted overall, neither A nor B dominates.

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then slot bin, then teammate-raw quintile.
`n̄_t` matches 2V so B is the same +1.678. R matches 2X +1.372.

```text
hold tier + recruit/raw + synth share     →  2X leftover R = +1.372
    ↓
hold P(slot_bin | t, r, s)                →  −23.5% of R
    ↓
hold P(teammate-raw quintile | t, r, s, slot) →  54.5% of R
    ↓
leftover P(survive | t, r, s, slot, teammates) →  68.9% of R
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Tier-bucket sums = survivor tier sum | 0 | 0 |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| 2X residual R reproduced | 1.3719447683362298 | 1.3719447683362298 |
| A+B+C − R | ~0 | ~0 |
| Nested mix+synth+A+B+C − B | ~0 | ~0 |

Hooked vs unhooked placements / HP / RNG match. `_hero_damage` unchanged.
Largest-remainder vs painted combat is **not** a conservation check —
same 2X note; the pool identity is Σ shares.

## Decision

```text
2X leftover R = +1.372 after tier + recruit + synth
        ↓
treatment T4/T5 still sit earlier and attack more
        ↓
that slot shift is the wrong sign for leftover survival
(front-line mix −23.5% of R)
        ↓
teammate-raw mix is +54.5% — real, not ≥70%
        ↓
68.9% is same-slot / same-teammate survival
        ↓
unexplained combat mechanics
(taunt / targeting / deathrattle / attack-cursor)
        ↓
next: isolate that mechanic before any behavior change
      do not rewrite 2Q; do not retune total scaling
      do not audit positioning policy yet
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a positioning-policy implementation. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–2X / **2Y DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2y.py tests/test_phase_2x.py tests/test_phase_2w.py tests/test_sim.py -q
python -m ml.fidelity_phase_2y          # reused 14200–14699
```

Working tree was clean at contract time (`356d512`, 51.69s). Tracer is
observational.
