# Simulator Fidelity Phase 2Z — targeting / cursor / represented deathrattle

Date: 2026-09-04 · Status: **`2z_v1` HOLD — `ranked_residual_needs_next_observable`** ·
Artifacts: [`results/sim_fidelity_phase_2z/`](../results/sim_fidelity_phase_2z/)

Stacked on PR #43 (`cursor/phase-2y-slot-attack-order-7573`, head `c836390`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–2Y DEV **14200–14699** (no new seeds).

2Y left **+0.946 / hit (68.9% of 2X R)** after holding tavern tier +
recruit/raw + synth share + slot bin + teammate-raw. This hour splits that
leftover into targeting/taunt, attack-cursor/initiative, represented
generated-body/deathrattle, marked unsupported-effect coverage, and still
unexplained.

## Verdict

**HOLD.** Route: **`ranked_residual_needs_next_observable`**.

2Y leftover C is reproduced exactly. Holding the 2Y cells, then target bin,
then cursor bin, then represented-gen bin, then unsupported bin: **no
represented mechanic clears ~70% of C**. Unsupported coverage is **0**
(placeholder / approximate DRs never appear on winner starting bodies).
Leftover same-everything survival is **84.5%**.

Do **not** preregister a taunt, cursor, or deathrattle correction. Do **not**
audit a missing effect class from this sample (coverage is empty, not
dominant). Next hour: record the smallest extra observable — per-swing
divine-shield / poisonous / cleave lethal cause, or start-of-combat hits —
before any behavior change. Do not rewrite 2Q. Do not retune total scaling.
Do not burn confirm.

| Component of 2Y leftover C (+0.946) | Δ / hit | Share of C |
|---|---:|---:|
| (A) Targeting / taunt | +0.048 | 0.051 |
| (B) Attack-cursor / initiative | +0.032 | 0.034 |
| (C) Represented generated / DR | +0.066 | 0.070 |
| (D) Unsupported-effect coverage | 0 | **0** |
| (E) Still unexplained | **+0.799** | **0.845** |
| Nested residual vs 2Y C | ~0 | — |

A+B+C+D+E = C. Nested terms add to 2V B (residual ~0). Ranked: E ≫ C > A > B > D.

## Classification (observational)

For every decisive T7–T14 hit, each winner starting minion is tagged by the
2Y covariates plus attacker/defender identity, taunt-forced vs open
targeting, pre/post HP, death cause, represented vs placeholder/approximate
effects, and attack-cursor advance / wrap.

```text
applied        = _hero_damage          unchanged
2Y leftover C  = Σ_t t · n̄_t · ΔP(survive|t,r,s,slot,teammates)
ΔP             = targeting + cursor + represented-DR + unsupported + leftover
```

Exclusive T6 stays in 2V A. Tracer does not consume RNG. Play still
appends. No positioning rewrite.

Unsupported stubs (**Kaboom Bot**, **Spawn of N'Zoth**) and the marked
approximate **Rat Pack** DR are tagged, never fitted. They have **zero**
winner-start mass on both arms (`p_unsupported_effect = 0`).

Target bins: never / open-only / taunt-or-forced.
Cursor bins: never reached / side-first no wrap / wrap or second-side.
Gen bins: no faithfully represented DR/generated exposure / has it.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#43](https://github.com/aferna6-cell/Replay/pull/43). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

Taunt share stays low (~0.05). Side-first is ~0.50 on both arms. Represented
generated/DR exposure is uncommon and **falls** on treatment T4/T5
(0.077→0.030 / 0.068→0.005). Cursor wrap-before-first rises slightly.
None of those mix shifts explain leftover T4 survival.

| Tier | 2Y leftover | (A) target | (B) cursor | (C) gen/DR | (E) unexplained |
|---|---:|---:|---:|---:|---:|
| T3 | +0.297 | +0.053 | +0.028 | +0.009 | **+0.207** |
| T4 | +0.504 | −0.020 | +0.009 | +0.048 | **+0.467** |
| T5 | +0.056 | −0.036 | −0.011 | +0.002 | +0.101 |

T4 — the bulk of C — is still unexplained after holding targeting, cursor,
and represented DR. Winner-start death causes are almost all `attack` /
`counterattack`; treatment poison deaths rise 6→211 but were **not** used
as a mix term (next observable).

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then slot bin, then teammate-raw quintile, then
target bin, then cursor bin, then represented-gen bin, then unsupported bin.
`n̄_t` matches 2V so B is the same +1.678. R matches 2X +1.372. C matches
2Y +0.946.

```text
hold tier + recruit/raw + synth + slot + teammates  →  2Y leftover C = +0.946
    ↓
hold P(target_bin | 2Y cells)                       →   5.1% of C
    ↓
hold P(cursor_bin | …)                              →   3.4% of C
    ↓
hold P(represented gen/DR | …)                      →   7.0% of C
    ↓
hold P(unsupported | …)                             →   0% of C (empty)
    ↓
leftover P(survive | all of the above)              →  84.5% of C
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Tier-bucket sums = survivor tier sum | 0 | 0 |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| Event-count (attacks / forced+open / created) | 0 mismatches | 0 mismatches |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| 2X residual R reproduced | 1.3719447683362298 | 1.3719447683362298 |
| 2Y leftover C reproduced | 0.9456715648873479 | 0.9456715648873479 |
| A+B+C+D+E − C | ~0 | ~0 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten.

## Decision

```text
2Y leftover C = +0.946 after slot + teammates
        ↓
targeting / taunt mix is 5%
attack-cursor / initiative mix is 3%
represented generated/DR mix is 7%
unsupported coverage is 0 (placeholders absent, not fitted)
        ↓
84.5% is same-target / same-cursor / same-DR survival
        ↓
ranked residual; smallest extra observable:
per-swing divine-shield / poisonous / cleave lethal cause,
or start-of-combat hits
        ↓
next: record that observable before any behavior change
      do not rewrite 2Q; do not retune total scaling
      do not preregister a targeting/cursor/DR correction
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a combat-behavior implementation. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–2Y / **2Z DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2z.py tests/test_phase_2y.py tests/test_phase_2x.py tests/test_sim.py -q
python -m ml.fidelity_phase_2z          # reused 14200–14699
```

Working tree was clean at contract time (`8ab781c`, 59.34s). Tracer is
observational.
