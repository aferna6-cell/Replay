# Simulator Fidelity Phase 3C — attacker-punch attribution

Date: 2026-09-04 · Status: **`3c_v1` HOLD — `jointly_explained_rank_largest`** ·
Artifacts: [`results/sim_fidelity_phase_3c/`](../results/sim_fidelity_phase_3c/)

Stacked on PR #46 (`cursor/phase-3b-hp-depletion-bf02`, head `a687894`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3B DEV **14200–14699** (no new seeds).

3B left **+0.939 / hit (113.4% of 3A leftover F)** in damage-per-hit /
HP-margin after holding tavern tier + recruit/raw + synth share + slot +
teammates + target + cursor + represented-DR + unsupported + DS + poison
+ cleave + SOC + ordinary + damaging-hit count. This hour splits that
term into attacker attack-strength mix at impact, synthetic-vs-recruit
attack composition, same-strength pairing / attack-order, and still
unexplained.

## Verdict

**HOLD.** Route: **`jointly_explained_rank_largest`**.

3B damage-per-hit B is reproduced exactly. Holding the 3B cells through
hit-count, then attacker-attack quintile, then synth-attack-share
quintile, then pairing/order quintile: **neither (A) nor (C) clears
~70% of +0.939**. Attack-strength mix is **54.6%**. Synth-vs-recruit
composition is **21.4%**. Same-strength pairing/order is **10.6%**.
Represented punch jointly explains **86.6%**. Leftover is **13.4%**.

Do **not** preregister an attacker-strength, synth-attack, or
pairing/order correction. Per the hour's forks: (A) is the largest
piece but not ≥70%; (C) is small, so this is **not** a targeting /
initiative hour. Next hour: isolate the largest upstream cause first —
the **board-strength / allocation source of the attacker-attack
distribution** — without tuning. Do not rewrite 2Q. Do not retune total
scaling. Do not burn confirm.

| Component of 3B damage-per-hit B (+0.939) | Δ / hit | Share of B |
|---|---:|---:|
| (A) Attacker attack-strength mix | +0.512 | **0.546** |
| (B) Attacker synth-vs-recruit attack | +0.201 | 0.214 |
| (C) Pairing / attack-order \| same strength | +0.100 | 0.106 |
| (D) Still unexplained | +0.125 | **0.134** |
| Nested residual vs 3B remainder after hits | 0 | — |

A+B+C+D = 3B (damage_per_hit + overkill + leftover) = F − hits = 0.938070.
Shares are of reproduced B = +0.938553. Ranked: A ≫ B > D > C.

## Classification (observational)

For every damaging hit on a winner-start T7–T14 body, the tracer records
attacker identity / tier / slot, attack immediately before the swing,
recruit/base attack, synthetic-attack share, golden / keyword / generated
status, defender pre-hit HP, actual HP loss, and whether the attacker
survived the swing or the hit was a counterattack. 3B covariates stay.

```text
applied        = _hero_damage          unchanged
3B B           = +0.939 damage-per-hit / HP margin
ΔP after hits  = attack-strength + synth-vs-recruit + pairing/order + leftover
ordinary HP    = min(pre-hit HP, effective incoming attack)
```

Exclusive T6 stays in 2V A. Tracer does not consume RNG. Play still
appends. Poison / shield / cleave / SOC / death-burst are tagged
separately and excluded from the ordinary identity.

Attack-strength bins: within-tier quintiles of mean dealer attack at
damaging-hit impact.
Synth-attack bins: within-tier quintiles of mean
`(attack − recruit_attack) / max(attack, 1)`.
Pairing/order bins: within-tier quintiles of
`first_attack_index + 0.05 · (attacker_slot − defender_slot)`
(−1 when the body took no damaging hit).

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#46](https://github.com/aferna6-cell/Replay/pull/46). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

Unconditional T4 incoming attacker-attack at impact falls **262 → 161**;
T3 **239 → 68**; T1 **225 → 20**. Recruit attack stays ~3–4 on both
arms; the punch gap is almost entirely painted synthetic on the
*opposing* board. T4 leftover remains the bulk of B and is mostly (A).

| Tier | 3B remainder after hits | (A) atk | (B) synth | (C) pair | (D) unexplained |
|---|---:|---:|---:|---:|---:|
| T3 | +0.236 | **+0.140** | +0.043 | +0.042 | +0.011 |
| T4 | +0.531 | **+0.316** | +0.079 | +0.044 | +0.093 |
| T5 | +0.128 | +0.013 | **+0.065** | +0.017 | +0.033 |

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then slot bin, then teammate-raw quintile,
then target / cursor / gen / unsupported / DS / poison / cleave / SOC /
ordinary / hit-count (3B cells through A), then attacker-attack
quintile, then synth-attack-share quintile, then pairing/order
quintile. `n̄_t` matches 2V so B is the same +1.678. R matches 2X
+1.372. C matches 2Y +0.946. E matches 2Z +0.799. F matches 3A +0.828.
3B damage-per-hit matches +0.9385531501941458.

```text
hold tier + recruit/raw + synth + slot + teammates
    + target + cursor + gen + unsupported
    + DS + poison + cleave + SOC + ordinary
    + hit_count                                    →  3A F − hits = +0.938
    ↓
hold P(attacker_attack quintile | …)               →  54.6% of +0.939
    ↓
hold P(synth_attack_share quintile | …)            →  21.4% of +0.939
    ↓
hold P(pairing_order quintile | …)                 →  10.6% of +0.939
    ↓
leftover P(survive | all of the above)             →  13.4% of +0.939
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Tier-bucket sums = survivor tier sum | 0 | 0 |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| Event-count (attacks / hits / incoming / applied / HP-flow / ordinary HP-loss) | 0 mismatches | 0 mismatches |
| Ordinary `applied = min(pre-hit HP, incoming)` | 0 mismatches | 0 mismatches |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| 2X residual R reproduced | 1.3719447683362298 | 1.3719447683362298 |
| 2Y leftover C reproduced | 0.9456715648873479 | 0.9456715648873479 |
| 2Z leftover E reproduced | 0.7993514476549548 | 0.7993514476549548 |
| 3A leftover F reproduced | 0.8275878344476644 | 0.8275878344476644 |
| 3B damage-per-hit B reproduced | 0.9385531501941458 | 0.9385531501941458 |
| A+B+C+D − (B+overkill+leftover)_3B | 0 | 0 |

Winner-start death causes unchanged from 3A/3B: control attack 22004 /
counter 14828 / poison **6**; treatment attack 16670 / counter 14466 /
poison **211**.

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten.

## Decision

```text
3B damage-per-hit B = +0.939 after hit-count
        ↓
attacker attack-strength mix is 54.6% (largest, not ≥70%)
synth-vs-recruit attack composition is 21.4%
same-strength pairing/order is 10.6% (not a targeting hour)
        ↓
represented jointly 86.6%; leftover 13.4%
        ↓
jointly_explained_rank_largest
        ↓
next: isolate the largest upstream cause first —
      board-strength / allocation producing the
      attacker-attack distribution (without tuning)
      do not rewrite 2Q; do not retune total scaling
      do not preregister an A/B/C correction
      leftover 13% is windfury / reborn / death-burst HP
      if that audit does not move B
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a combat-behavior implementation. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3B / **3C DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3c.py tests/test_phase_3b.py tests/test_phase_3a.py tests/test_phase_2z.py tests/test_phase_2y.py tests/test_phase_2x.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3c          # reused 14200–14699
```

Working tree was clean at contract time (`64a8d59`, 84.66s). Tracer is
observational.
