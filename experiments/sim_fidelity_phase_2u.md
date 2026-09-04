# Simulator Fidelity Phase 2U — survivor-tier damage fidelity

Date: 2026-09-04 · Status: **`2u_v1` HOLD — `isolate_survivor_composition`** ·
Artifacts: [`results/sim_fidelity_phase_2u/`](../results/sim_fidelity_phase_2u/)

Stacked on PR #38 (`cursor/phase-2t-game-length-a550`, head `9013bb4`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 HOLD**. Do **not** merge.
Confirm **11500–11699** untouched. No α / residual / `_hero_damage` / gate /
behavior / default changes. Reused consumed 2S/2T DEV **14200–14699**
(no new seeds).

## Verdict

**HOLD.** Route: **`isolate_survivor_composition`**.

Actual combat-survivor tiers do **not** remove the treatment−control
`_hero_damage` amplification. They **increase** it.

| Amp when hit (applied − count-only) | Control | Treatment | Δ |
|---|---:|---:|---:|
| Board-mean proxy (2T / applied) | 3.41 | 6.19 | **+2.78** |
| Rules-faithful CF (tavern + Σ survivor tiers) | 3.09 | 6.72 | **+3.63** |
| Share of +2.78 removed | — | — | **−0.31** (grew 31%) |

The proxy is slightly **high** on control (error +0.33) and **low** on
treatment (error −0.53). Treatment survivors are higher-tier than the
winner's current board mean (3.56 vs 3.33); control survivors are
slightly lower (2.37 vs 2.51). High-tier bodies live, chaff dies — that
composition gap is the remaining asymmetry.

Do **not** preregister a `_hero_damage` formula change this next hour.
Isolate who survives (tier histogram / token share / board-vs-survivor
mix) first. Do not burn confirm.

## Counterfactuals (observational; applied HP unchanged)

```text
applied        = _hero_damage  = tavern + round(n_surv × mean(board.tier))
count_only     = sim.py raw    = tavern + n_surv
counterfactual = rules-faithful = tavern + sum(actual survivor tavern tiers)
error          = applied − counterfactual
```

`simulate_once(..., trace=)` fills survivor identities/tiers after the
fight. Hooked vs unhooked placements / HP / RNG match. Survivor count
matched recovered raw on every T7–T14 hit (both arms).

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#38](https://github.com/aferna6-cell/Replay/pull/38). Game
length **15.692 → 13.510** reproduced.

| Metric (T7–T14 hits) | Control | Treatment | Δ |
|---|---:|---:|---:|
| n hits | 7747 | 7162 | −585 |
| Applied / hit | 10.87 | 13.95 | +3.08 |
| Count-only / hit | 7.46 | 7.76 | +0.30 |
| Counterfactual / hit | 10.54 | **14.48** | **+3.94** |
| Survivor count | 2.25 | 2.63 | +0.38 |
| Survivor tier sum | 5.34 | **9.35** | **+4.01** |
| Survivor tier mean | 2.37 | **3.56** | +1.19 |
| Winner board tier mean | 2.51 | 3.33 | +0.83 |
| Proxy − CF (mean) | +0.33 | **−0.53** | −0.86 |
| Lethal rate (applied) | 0.385 | 0.465 | +0.080 |
| Lethal rate (CF) | 0.376 | **0.480** | **+0.104** |
| Lethal flips | 260 | 191 | — |
| Proxy overkill (lethal, CF not) | 163 | 41 | — |
| Proxy underkill (not lethal, CF is) | 97 | 150 | — |

## Proxy − CF error by arm

| Arm | n | mean | median | std | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 7747 | +0.33 | 0 | 1.25 | −1 | +2 | −4 | +4 |
| Treatment | 7162 | −0.53 | −1 | 1.16 | −2 | +1 | −4 | +4 |

Per-turn treatment error stays negative from T7 (−0.66) through T14
(−0.50). The CF amp delta is already +1.37 at T7 and +1.46 at T8, then
jumps with the board-tier step (+4.23 at T9, +6.05 at T11).

## Decision

```text
2S / 2Q puts higher-tier bodies on the board
        ↓
combat selects those bodies as survivors (tier mean 3.33 → 3.56)
        ↓
rules-faithful damage = tavern + Σ survivor tiers
        ↓
treatment−control amp +3.63 (larger than the +2.78 proxy gap)
        ↓
board-mean proxy is not the source of the asymmetry
```

**Next:** isolate survivor composition / tier distribution (who lives:
printed tier vs tokens vs leftover chaff). **Not** a default-OFF
`_hero_damage` formula treatment until that mix is measured. **Not** a
recruit/scaling retune. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S / 2T / **2U DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2u.py tests/test_phase_2t.py tests/test_sim.py -q
python -m ml.fidelity_phase_2u          # reused 14200–14699
```

Working tree was clean at contract time (`ffd8194`, 29.53s). Tracer is
observational (same-seed hooked vs unhooked placements/HP/RNG match).
