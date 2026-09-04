# Simulator Fidelity Phase 3A — lethal-cause / keyword attribution

Date: 2026-09-04 · Status: **`3a_v1` HOLD — `ranked_residual_needs_next_observable`** ·
Artifacts: [`results/sim_fidelity_phase_3a/`](../results/sim_fidelity_phase_3a/)

Stacked on PR #44 (`cursor/phase-2z-combat-mechanics-560d`, head `7017dbc`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–2Z DEV **14200–14699** (no new seeds).

2Z left **+0.799 / hit (84.5% of 2Y leftover C)** after holding tavern
tier + recruit/raw + synth share + slot + teammates + target + cursor +
represented-DR + unsupported. This hour splits that leftover into
divine-shield, poisonous/venomous, cleave, represented start-of-combat,
ordinary attack/counterattack, and still unexplained.

## Verdict

**HOLD.** Route: **`ranked_residual_needs_next_observable`**.

2Z leftover E is reproduced exactly. Holding the 2Z cells, then DS bin,
then poison bin, then cleave bin, then SOC bin, then ordinary-hit bin:
**no represented lethal mechanism clears ~70% of E**. Represented
mechanisms jointly explain **−3.5%** (wrong-signed DS/poison mix;
cleave / SOC / ordinary mix are **0**). Leftover same-everything
survival is **103.5%** of E (larger than E because DS/poison mix is
negative).

Do **not** preregister a DS, poison, cleave, SOC, or ordinary-combat
correction. Winner-start cleave and represented SOC coverage are
**empty** (p=0 both arms). Ordinary hit-type mix is absorbed by the 2Z
target/cursor cells. Poison deaths still move **6→211** but as an
exposure mix they do not explain leftover T4 survival. Next hour:
record the smallest extra observable — **per-hit remaining HP /
overkill / hit count** (binary ordinary exposure is already held) —
before any behavior change. Do not rewrite 2Q. Do not retune total
scaling. Do not burn confirm.

| Component of 2Z leftover E (+0.799) | Δ / hit | Share of E |
|---|---:|---:|
| (A) Divine shield | −0.019 | **−0.024** |
| (B) Poisonous / venomous | −0.009 | −0.011 |
| (C) Cleave primary/secondary | 0 | **0** |
| (D) Represented start-of-combat | 0 | **0** |
| (E) Ordinary attack/counterattack | 0 | **0** |
| (F) Still unexplained | **+0.828** | **1.035** |
| Nested residual vs 2Z E | ~0 | — |

A+B+C+D+E+F = E. Nested terms add to 2V B (residual ~0). Ranked: F ≫ |A| > |B| > C=D=E=0.

## Classification (observational)

For every decisive T7–T14 hit, each winner starting minion is tagged by
the 2Z covariates plus per-swing DS before/after and shield-pop cause,
poisonous/venomous hit + lethal flag, cleave primary vs secondary +
lethal, represented SOC hits, ordinary attack/counterattack lethal,
tier/raw/synth/slot/teammate covariates, and survival.

```text
applied        = _hero_damage          unchanged
2Z leftover E  = Σ_t t · n̄_t · ΔP(survive|t,r,s,slot,teammates,tgt,cur,gen,unsup)
ΔP             = DS + poison + cleave + SOC + ordinary + leftover
```

Exclusive T6 stays in 2V A. Tracer does not consume RNG. Play still
appends. Venomous == poisonous in-sim (Deadly Spore grant). Cleave and
SOC are tagged when present, never invented.

DS bins: never / started-with-DS never popped / popped.
Poison bins: never hit by poisonous / hit (including shield-eaten).
Cleave bins: never / primary of a cleave attacker / secondary splash.
SOC bins: never / hit by represented start-of-combat.
Ordinary bins: never / attack-hit as defender / counterattack-hit.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#44](https://github.com/aferna6-cell/Replay/pull/44). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

Cleave and represented SOC have **zero** winner-start mass on both
arms. Unconditional ordinary-hit rates do shift (T4 counter 0.42→0.69)
but that mix is already held by 2Z targeting/cursor cells, so nested
ordinary mix is 0. Treatment poison deaths rise 6→211; poison
*exposure* mix is small and negative. T4 leftover remains the bulk of E.

| Tier | 2Z leftover E | (A) DS | (B) poison | (F) unexplained |
|---|---:|---:|---:|---:|
| T3 | +0.207 | −0.004 | −0.003 | **+0.214** |
| T4 | +0.467 | −0.010 | −0.005 | **+0.482** |
| T5 | +0.101 | −0.004 | −0.000 | **+0.105** |

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then slot bin, then teammate-raw quintile,
then target / cursor / gen / unsupported (2Z cells), then DS, poison,
cleave, SOC, ordinary. `n̄_t` matches 2V so B is the same +1.678. R
matches 2X +1.372. C matches 2Y +0.946. E matches 2Z +0.799.

```text
hold tier + recruit/raw + synth + slot + teammates
    + target + cursor + gen + unsupported          →  2Z leftover E = +0.799
    ↓
hold P(ds_bin | 2Z cells)                          →  −2.4% of E
    ↓
hold P(poison_bin | …)                             →  −1.1% of E
    ↓
hold P(cleave_bin | …)                             →   0% of E (empty)
    ↓
hold P(soc_bin | …)                                →   0% of E (empty)
    ↓
hold P(ordinary_bin | …)                           →   0% of E (absorbed)
    ↓
leftover P(survive | all of the above)             →  103.5% of E
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Tier-bucket sums = survivor tier sum | 0 | 0 |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| Event-count (attacks / hits / DS pops / poison / cleave / SOC / ordinary / deaths) | 0 mismatches | 0 mismatches |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| 2X residual R reproduced | 1.3719447683362298 | 1.3719447683362298 |
| 2Y leftover C reproduced | 0.9456715648873479 | 0.9456715648873479 |
| 2Z leftover E reproduced | 0.7993514476549548 | 0.7993514476549548 |
| A+B+C+D+E+F − E | ~0 | ~0 |

Winner-start death causes: control attack 22004 / counter 14828 / poison **6**;
treatment attack 16670 / counter 14466 / poison **211**. Shield pops are
almost all ordinary attack/counterattack (treatment also 8 poison pops).

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten.

## Decision

```text
2Z leftover E = +0.799 after target + cursor + DR + unsupported
        ↓
DS mix is −2.4% (wrong-signed)
poison/venomous mix is −1.1% (deaths 6→211, not the leftover)
cleave coverage is 0 (absent on winner starts)
represented SOC coverage is 0 (absent on winner starts)
ordinary hit-type mix is 0 (absorbed by 2Z target/cursor)
        ↓
103.5% is same-DS / same-poison / same-cleave / same-SOC / same-ordinary survival
        ↓
ranked residual; smallest extra observable:
per-hit remaining HP / overkill / hit count
(binary ordinary exposure is already held)
        ↓
next: record that observable before any behavior change
      do not rewrite 2Q; do not retune total scaling
      do not preregister a DS/poison/cleave/SOC/ordinary correction
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a combat-behavior implementation. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–2Z / **3A DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3a.py tests/test_phase_2z.py tests/test_phase_2y.py tests/test_phase_2x.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3a          # reused 14200–14699
```

Working tree was clean at contract time (`759e016`, 59.87s). Tracer is
observational.
