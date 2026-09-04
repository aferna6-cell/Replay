# Simulator Fidelity Phase 3E — board-pool lifecycle attribution

Date: 2026-09-04 · Status: **`3e_v1` HOLD — `carry_history_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3e/`](../results/sim_fidelity_phase_3e/)

Stacked on PR #50 (`cursor/phase-3d-attacker-source-f28d`, head `e8f55ba`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3D DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3D left **A1 = +0.422 / hit (82.4% of 3C A)** in opposing board-pool
magnitude at combat start. This hour traces the abstract/synthetic attack
pool through the turn and splits that arm gap into inherited carry,
current-turn scaling add, replacement retention/loss, and lifecycle
selection + leftover.

## Verdict

**HOLD.** Route: **`carry_history_dominates`**.

3D A1 is reproduced exactly (`0.4216721428553852`, share **0.8235**).
Holding the 3B cells through hit-count, then carry quintile, then
scaling-add quintile, then replacement-loss quintile, then board-size
quintile, then remaining pool quintile: **inherited carry clears ~70% of
+0.422 (72.4%)**. Current-turn scaling-add is **55.8%**. Replacement
churn is **44.6%**. Lifecycle selection (board size) + leftover pool is
**6.3%**.

Do **not** preregister a carry, scaling-input, or 2S-lifecycle
correction. Per the hour's forks: carry is ≥70%, so the next hour traces
**when the board-pool divergence first appears** (history / earlier
turns), without retuning total scaling. Do not rewrite 2Q. Do not audit
scaling constants. Do not burn confirm.

Additive punch-row flow agrees: of the −275 treatment−control gap in
mean opposing combat-start attack pool, **carry is 71.4%**, scale-add
28.7%, replacement **−0.4%** (treatment slightly *gains* attack-pool
during recruit), leftover 0.2%. Snapshot identity closes on every
instrumented seat-turn.

| Component of 3D A1 (+0.422) | Δ / hit | Share of A1 |
|---|---:|---:|
| (1) Inherited / carry pool | +0.305 | **0.724** |
| (2) Current-turn scaling add | +0.235 | 0.558 |
| (3) Replacement / churn | +0.188 | 0.446 |
| (4) Lifecycle selection + leftover | +0.026 | 0.063 |

(1)+(2)+(3)+(4) = 0.755 (path-dependent; inserting carry/scale/replace
before the 3D pool quintile also absorbs some of 3D A2–A4). Shares are
of reproduced A1 = +0.4216721428553852. Ranked: carry ≫ scale-add >
replace ≫ leftover. Board-size selection is identically 0 (both arms
field ~7).

## Classification (observational)

For each alive player-turn T7–T14 the tracer records attack-pool and
stats-pool at prior combat end, recruit start, immediately pre-scale,
current-turn residual/ratio add, post-scale, and combat start; sell→buy→play
replacement events with carried/preserved/lost synthetic; alive /
board-size / tier mix; and the scaling budget already computed by
`_residual_scaling_budget` (Firestone target, pace target, growth
factor, `ratio_g`, residual add, clamp, just-leveled, tavern tier).
Punch rows join the opposing (loser) seat's lifecycle. 3D covariates stay.

```text
applied        = _hero_damage          unchanged
3D A1          = +0.422 opposing board-pool magnitude
ΔA1            = carry + scale-add + replace + leftover
identity       = post = carry + add − represented loss/transfer
```

Exclusive T6 stays in 2V A. Tracer does not consume RNG. Play still
appends. Residual/ratio math is unchanged; the hook only copies the
budget dict after RNG.

Carry bins: within-tier quintiles of opposing attack-pool at recruit
start. Scale-add bins: within-tier quintiles of post-scale − pre-scale
attack-pool. Replace bins: within-tier quintiles of recruit-start −
pre-scale (positive = lost). Selection bins: within-tier quintiles of
opposing combat-start board size.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#50](https://github.com/aferna6-cell/Replay/pull/50). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

Punch-row opposing attack-pool **1283 → 1008** (Δ **−275**). Carry
**810 → 614** (Δ **−196**, 71.4% of the gap). Scale-add **472 → 393**
(Δ **−79**, 28.7%). Replace-loss **+0.28 → −0.78** (treatment retains;
realloc can raise the attack share). Snapshot flow residual **0**;
`p_flow_ok = 1`. Combat-start matches post-scale (0 mismatches).

Unconditional alive seat-turns (not the punch sample) stay close across
arms through T14 — T7 carry **35.7 vs 35.8**, T10 **421 vs 412**, T14
**5907 vs 5891**. The punch-sample crater is concentrated on low-tier
winner-start bodies (T1 opposing pool **1258 → 118**; T1 carry
**797 → 65**). Next hour should date that history, not retune this
turn's scaling constants.

| Tier | 3D A1 analogue | (1) carry | (2) add | (3) replace | (4) leftover |
|---|---:|---:|---:|---:|---:|
| T3 | +0.131 | +0.060 | +0.066 | +0.045 | +0.005 |
| T4 | +0.247 | **+0.212** | +0.137 | +0.062 | +0.009 |
| T5 | +0.010 | +0.007 | +0.014 | +0.064 | +0.010 |

T4 opposing board pool **1317 → 1062**; carry **833 → 617**; scale-add
**484 → 446**. Treatment replacement rate on T9/T11–T14 is ~2–4 per
seat-turn vs ~0 on control; snapshot pool loss stays ~0 (2S conserves).
Event-sold vs snapshot-net disagrees under treatment (10358 seat-turns)
because sold-body synthetic is the conserved pool, not a loss — that is
the 2S mechanism, not an A1 driver.

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then slot bin, then teammate-raw quintile,
then target / cursor / gen / unsupported / DS / poison / cleave / SOC /
ordinary / hit-count (3B cells through A), then opposing carry quintile,
then scaling-add quintile, then replace-loss quintile, then board-size
quintile, then remaining board-pool quintile (plus 3D conc / delta /
attack). `n̄_t` matches 2V so B is the same +1.678. **3D A1 matches
+0.4216721428553852**. 3C A matches +0.5120447786800975.

```text
hold tier + recruit/raw + synth + slot + teammates
    + target + cursor + gen + unsupported
    + DS + poison + cleave + SOC + ordinary
    + hit_count                                    →  3A F − hits = +0.938
    ↓
hold P(opp carry quintile | …)                     →  72.4% of +0.422
    ↓
hold P(opp scaling-add quintile | …)               →  55.8% of +0.422
    ↓
hold P(opp replace-loss quintile | …)              →  44.6% of +0.422
    ↓
hold P(opp board-size quintile | …)                →  0% of +0.422
    ↓
hold P(opp board-pool quintile | …)                →  6.3% leftover of A1
    ↓
3D concentration / combat Δ / residual attack mix
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| `post = carry + add − loss` (snapshot) | 0 mismatches | 0 mismatches |
| Combat-start attack-pool = post-scale | 0 | 0 |
| Punch-row flow residual | 0 | 0 |
| Event-count / attack-identity / ordinary HP-loss | 0 | 0 |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| 2X residual R reproduced | 1.3719447683362298 | 1.3719447683362298 |
| 2Y leftover C reproduced | 0.9456715648873479 | 0.9456715648873479 |
| 2Z leftover E reproduced | 0.7993514476549548 | 0.7993514476549548 |
| 3A leftover F reproduced | 0.8275878344476644 | 0.8275878344476644 |
| 3B damage-per-hit B reproduced | 0.9385531501941458 | 0.9385531501941458 |
| 3C attack-strength A reproduced | 0.5120447786800975 | 0.5120447786800975 |
| 3D A1 board-pool magnitude reproduced | 0.4216721428553852 | 0.4216721428553852 |

Winner-start death causes unchanged from 3D: control attack 22004 /
counter 14828 / poison **6**; treatment attack 16670 / counter 14466 /
poison **211**.

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
3D A1 = +0.422 opposing board-pool magnitude
        ↓
inherited/carry pool is 72.4% (≥70%)
current-turn scaling-add is 55.8%
replacement retention/loss is 44.6%
lifecycle selection + leftover is 6.3%
        ↓
carry_history_dominates
        ↓
next: trace when the board-pool divergence first appears
      (history / earlier turns), without retuning total scaling
      do not rewrite 2Q; do not change scaling constants
      do not preregister a carry/scale/2S correction
      additive punch-row replace share is ~0; not a 2S-fidelity hour
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a scaling-constant retune. **Not** a 2S lifecycle rewrite.
**Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3D / **3E DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3e.py tests/test_phase_3d.py tests/test_phase_2s.py tests/test_phase_2r.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3e          # reused 14200–14699
```

Working tree was clean at contract time (`b5dd8c6`, 130.83s). Tracer is
observational.
