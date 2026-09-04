# Simulator Fidelity Phase 3D — upstream attacker-punch source attribution

Date: 2026-09-04 · Status: **`3d_v1` HOLD — `board_pool_magnitude_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3d/`](../results/sim_fidelity_phase_3d/)

Stacked on PR #47 (`cursor/phase-3c-attacker-punch-6180`, head `71093e3`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3C DEV **14200–14699** (no new seeds).

3C left **+0.512 / hit (54.6% of 3B B)** in attacker attack-strength mix
at impact after holding tavern tier + recruit/raw + synth share + slot +
teammates + target + cursor + represented-DR + unsupported + DS + poison
+ cleave + SOC + ordinary + damaging-hit count. This hour splits that
term into opposing board-pool magnitude, allocation concentration onto
bodies that actually attack, in-combat attack mutation, and leftover.

## Verdict

**HOLD.** Route: **`board_pool_magnitude_dominates`**.

3C attack-strength A is reproduced exactly. Holding the 3B cells through
hit-count, then opposing board-pool quintile, then pool-on-attackers
quintile, then combat-delta quintile, then residual attacker-attack
quintile: **board-pool magnitude clears ~70% of +0.512 (82.4%)**.
Allocation concentration is **50.5%**. In-combat attack growth is
**0%**. Residual attack-strength mix is **15.7%**.

Do **not** preregister a pool-magnitude, allocation, or combat-delta
correction. Per the hour's forks: (A1) is ≥70%, so the next hour audits
**why opposing board-level synthetic strength differs so sharply across
arms**, without retuning total scaling. Do not rewrite 2Q. Do not burn
confirm. Combat mutation is identically zero (attack never changes after
combat start), so this is **not** a represented-effect hour.

| Component of 3C attack-strength A (+0.512) | Δ / hit | Share of A |
|---|---:|---:|
| (A1) Board-pool magnitude | +0.422 | **0.824** |
| (A2) Allocation onto attacking bodies | +0.259 | 0.505 |
| (A3) In-combat attack growth / effects | 0 | 0 |
| (A4) Leftover residual attack mix | +0.080 | 0.157 |

A1+A2+A3+A4 = 0.761 (path-dependent; inserting pool/concentration before
the 3C attack quintile also absorbs some of 3C B/C). Shares are of
reproduced A = +0.5120447786800975. Ranked: A1 ≫ A2 > A4 > A3.

## Classification (observational)

For every damaging hit on a winner-start T7–T14 body, the tracer records
the dealer's combat-start recruit attack, combat-start abstract-pool
attack, in-combat attack delta, opposing board total recruit attack,
total abstract-pool attack, board size / mean tier / tier mix, and the
attacker's share and rank of that pool. 3C covariates stay.

```text
applied        = _hero_damage          unchanged
3C A           = +0.512 attacker attack-strength mix
ΔA             = pool magnitude + concentration + combat Δ + leftover
identity       = impact = start_recruit + start_pool_share + combat_delta
```

Exclusive T6 stays in 2V A. Tracer does not consume RNG. Play still
appends. Attack is stamped at combat start / token spawn; combat never
reads `start_attack`.

Pool-magnitude bins: within-tier quintiles of opposing board total
abstract-pool attack at combat start.
Concentration bins: within-tier quintiles of (pool sitting on start
bodies that actually attacked) / (opposing board pool).
Combat-delta bins: within-tier quintiles of mean
`impact_attack − start_attack`.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#47](https://github.com/aferna6-cell/Replay/pull/47). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

Unconditional T4 incoming attacker-attack at impact falls **262 → 161**
and reconciles as recruit **3.16 → 4.12** + start pool **259 → 157** +
combat Δ **0**. T4 opposing board pool **1317 → 1062**; T1 **1258 → 118**;
T3 **1270 → 442**. Recruit attack stays ~3–4 on both arms. Pool-on-attackers
share **0.77 → 0.69**. Combat delta is **0** on every punched body.

| Tier | 3C A analogue | (A1) pool | (A2) conc | (A3) Δ | (A4) leftover |
|---|---:|---:|---:|---:|---:|
| T3 | +0.208 | **+0.131** | +0.085 | 0 | −0.008 |
| T4 | +0.434 | **+0.247** | +0.108 | 0 | +0.079 |
| T5 | +0.087 | +0.010 | +0.050 | 0 | +0.027 |

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then slot bin, then teammate-raw quintile,
then target / cursor / gen / unsupported / DS / poison / cleave / SOC /
ordinary / hit-count (3B cells through A), then opposing board-pool
quintile, then pool-on-attackers quintile, then combat-delta quintile,
then residual attacker-attack quintile (plus 3C synth / pairing).
`n̄_t` matches 2V so B is the same +1.678. R matches 2X +1.372. C matches
2Y +0.946. E matches 2Z +0.799. F matches 3A +0.828. 3B damage-per-hit
matches +0.9385531501941458. **3C A matches +0.5120447786800975**.

```text
hold tier + recruit/raw + synth + slot + teammates
    + target + cursor + gen + unsupported
    + DS + poison + cleave + SOC + ordinary
    + hit_count                                    →  3A F − hits = +0.938
    ↓
hold P(opp board-pool quintile | …)                →  82.4% of +0.512
    ↓
hold P(pool-on-attackers quintile | …)             →  50.5% of +0.512
    ↓
hold P(combat-delta quintile | …)                  →  0% of +0.512
    ↓
hold P(attacker_attack quintile | …)               →  15.7% of +0.512
    ↓
3C synth / pairing / leftover survival
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Tier-bucket sums = survivor tier sum | 0 | 0 |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| Event-count (attacks / hits / incoming / applied / HP-flow / ordinary HP-loss) | 0 mismatches | 0 mismatches |
| Ordinary `applied = min(pre-hit HP, incoming)` | 0 mismatches | 0 mismatches |
| `impact = start_recruit + start_pool + combat_delta` | 0 mismatches | 0 mismatches |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| 2X residual R reproduced | 1.3719447683362298 | 1.3719447683362298 |
| 2Y leftover C reproduced | 0.9456715648873479 | 0.9456715648873479 |
| 2Z leftover E reproduced | 0.7993514476549548 | 0.7993514476549548 |
| 3A leftover F reproduced | 0.8275878344476644 | 0.8275878344476644 |
| 3B damage-per-hit B reproduced | 0.9385531501941458 | 0.9385531501941458 |
| 3C attack-strength A reproduced | 0.5120447786800975 | 0.5120447786800975 |
| punch_sum − (B+overkill+leftover)_3B | 0 | 0 |

Winner-start death causes unchanged from 3C: control attack 22004 /
counter 14828 / poison **6**; treatment attack 16670 / counter 14466 /
poison **211**.

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten.

## Decision

```text
3C attack-strength A = +0.512 after hit-count
        ↓
board-pool magnitude is 82.4% (≥70%)
allocation onto attacking bodies is 50.5%
in-combat attack growth is 0%
leftover residual attack mix is 15.7%
        ↓
board_pool_magnitude_dominates
        ↓
next: audit why opposing board-level synthetic
      strength differs so sharply across arms
      (without retuning total scaling)
      do not rewrite 2Q; do not preregister an A1/A2/A3 correction
      combat Δ is identically 0; not an effect-fidelity hour
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a combat-behavior implementation. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3C / **3D DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3d.py tests/test_phase_3c.py tests/test_phase_3b.py tests/test_phase_3a.py tests/test_phase_2z.py tests/test_phase_2y.py tests/test_phase_2x.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3d          # reused 14200–14699
```

Working tree was clean at contract time (`d4d3812`, 116.4s). Tracer is
observational.
