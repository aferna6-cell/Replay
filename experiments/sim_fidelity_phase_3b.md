# Simulator Fidelity Phase 3B — HP depletion / overkill / hit-count

Date: 2026-09-04 · Status: **`3b_v1` HOLD — `damage_per_hit_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3b/`](../results/sim_fidelity_phase_3b/)

Stacked on PR #45 (`cursor/phase-3a-lethal-cause-8c4b`, head `5dcdbf8`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3A DEV **14200–14699** (no new seeds).

3A left **+0.828 / hit (103.5% of 2Z leftover E)** after holding tavern
tier + recruit/raw + synth share + slot + teammates + target + cursor +
represented-DR + unsupported + DS + poison + cleave + SOC + ordinary.
This hour splits that leftover into damaging-hit count, damage-per-hit /
HP depletion margin, overkill / death-threshold, and still unexplained.

## Verdict

**HOLD.** Route: **`damage_per_hit_dominates`**.

3A leftover F is reproduced exactly. Holding the 3A cells, then damaging-hit
bin, then HP-to-incoming-damage quintile, then overkill bin: **one HP
mechanism clears ~70% of F**. Damage-per-hit / depletion margin is
**+113%** of F. Hit-count mix is **−13%** (wrong-signed: treatment T4
takes *more* damaging hits). Overkill mix is **−1.2%**. Same-everything
survival leftover is **1.1%**.

Do **not** preregister a hit-count or overkill correction. Next hour:
audit the **upstream represented mechanic that changes incoming punch
size relative to starting HP** on same-keyword / same-target bodies
before any behavior change. Do not rewrite 2Q. Do not retune total
scaling. Do not burn confirm.

| Component of 3A leftover F (+0.828) | Δ / hit | Share of F |
|---|---:|---:|
| (A) Damaging hits / exposure | −0.110 | **−0.133** |
| (B) Damage per hit / HP margin | **+0.939** | **1.134** |
| (C) Overkill / death-threshold | −0.010 | −0.012 |
| (D) Still unexplained | +0.009 | **0.011** |
| Nested residual vs 3A F | ~0 | — |

A+B+C+D = F. Nested terms add to 2V B (residual ~0). Ranked: B ≫ |A| > |C| ≈ D.

## Classification (observational)

For every decisive T7–T14 hit, each winner starting minion is tagged by
the 3A covariates plus starting HP, each incoming ordinary/poison/cleave/SOC
amount, HP immediately before/after the last hit, cumulative incoming /
applied HP loss, damaging-hit count, overkill on death, and survival.

```text
applied        = _hero_damage          unchanged
3A leftover F  = Σ_t t · n̄_t · ΔP(survive|t,r,s,slot,teammates,tgt,cur,gen,unsup,ds,poi,cl,soc,ord)
ΔP             = hit-count + damage-per-hit + overkill + leftover
```

Exclusive T6 stays in 2V A. Tracer does not consume RNG. Play still
appends. HP-flow identity: `start − end = Σ (hp_before − hp_after)`.

Hit-count bins: 0 / 1 / 2+ damaging hits (HP actually reduced; DS-eaten
hits are incoming but not damaging).
Margin bins: within-tier quintiles of `start_HP / max(mean incoming, 1)`.
Overkill bins: 0 (survived or exact) / 1–4 / 5+.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#45](https://github.com/aferna6-cell/Replay/pull/45). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

Unconditional T4 incoming-per-hit falls **228 → 150**; damaging-hit count
rises **1.06 → 1.71**. T4 leftover remains the bulk of F and is almost
entirely (B). Treatment T4 start HP is higher (126 → 250) but recruit/raw
+ synth are already held; the leftover is the residual punch-to-HP ratio
inside those cells.

| Tier | 3A leftover F | (A) hits | (B) dmg/hit | (D) unexplained |
|---|---:|---:|---:|---:|
| T3 | +0.214 | −0.022 | **+0.263** | −0.004 |
| T4 | +0.482 | −0.049 | **+0.526** | +0.002 |
| T5 | +0.105 | −0.022 | **+0.104** | +0.004 |

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then slot bin, then teammate-raw quintile,
then target / cursor / gen / unsupported / DS / poison / cleave / SOC /
ordinary (3A cells), then damaging-hit bin, then HP-margin quintile,
then overkill bin. `n̄_t` matches 2V so B is the same +1.678. R matches
2X +1.372. C matches 2Y +0.946. E matches 2Z +0.799. F matches 3A +0.828.

```text
hold tier + recruit/raw + synth + slot + teammates
    + target + cursor + gen + unsupported
    + DS + poison + cleave + SOC + ordinary        →  3A leftover F = +0.828
    ↓
hold P(hit_count_bin | 3A cells)                   →  −13.3% of F
    ↓
hold P(hp_margin quintile | …)                     →  113.4% of F
    ↓
hold P(overkill_bin | …)                           →  −1.2% of F
    ↓
leftover P(survive | all of the above)             →  1.1% of F
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Tier-bucket sums = survivor tier sum | 0 | 0 |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| Event-count (attacks / hits / incoming / applied / HP-flow / overkill / deaths) | 0 mismatches | 0 mismatches |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| 2X residual R reproduced | 1.3719447683362298 | 1.3719447683362298 |
| 2Y leftover C reproduced | 0.9456715648873479 | 0.9456715648873479 |
| 2Z leftover E reproduced | 0.7993514476549548 | 0.7993514476549548 |
| 3A leftover F reproduced | 0.8275878344476644 | 0.8275878344476644 |
| A+B+C+D − F | ~0 | ~0 |

Winner-start death causes unchanged from 3A: control attack 22004 / counter
14828 / poison **6**; treatment attack 16670 / counter 14466 / poison **211**.

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten.

## Decision

```text
3A leftover F = +0.828 after DS + poison + cleave + SOC + ordinary
        ↓
hit-count mix is −13% (treatment T4 takes more damaging hits)
damage-per-hit / HP margin is +113% (dominates)
overkill / death-threshold mix is −1.2%
        ↓
1.1% is same-hits / same-margin / same-overkill survival
        ↓
damage_per_hit_dominates
        ↓
next: audit the upstream represented mechanic that changes
      incoming punch size relative to starting HP
      before any behavior change
      do not rewrite 2Q; do not retune total scaling
      do not preregister a hit-count or overkill correction
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a combat-behavior implementation. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3A / **3B DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3b.py tests/test_phase_3a.py tests/test_phase_2z.py tests/test_phase_2y.py tests/test_phase_2x.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3b          # reused 14200–14699
```

Working tree was clean at contract time (`ad28089`, 68.33s). Tracer is
observational.
