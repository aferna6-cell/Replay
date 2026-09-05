# Simulator Fidelity Phase 3O — matched-board survivor-mechanic attribution

Date: 2026-09-05 · Status: **`3o_v1` HOLD — `start_stats_synth_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3o/`](../results/sim_fidelity_phase_3o/)

Stacked on PR #60 (`cursor/phase-3n-damage-attribution-2d61`, head `f61a0a90`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3N DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3N left 1059 same-outcome first splits board-matched, with rules-faithful
CF **+0.688 / split** assigned entirely to within-tier survival
(B = 1.0). This hour restricts primary analysis to T5/T6 class-(3)
fights and attributes that leftover survival to represented combat
mechanics.

## Verdict

**HOLD.** Route: **`start_stats_synth_dominates`**.

3N class-(3) **1059 / 1059** and within-tier B **+0.6883852691218131**
reproduce. Primary T5/T6 window is **1033 / 1033** (T5 884 + T6 149).
T7 (26) is held out of the primary walk; those 26 fights are the only
class-(3) rows that field T4 starts.

On T5/T6, body-level within-tier B is **+0.617 / split**. Holding
printed tier, then recruit/synth/start-HP, then slot, target, keywords,
then teammate-raw: **start-stats clears 121% of B**. Synthetic
allocation is the entire start-stats term (118% of B). Recruit-raw mix
is **0**. Start-HP mix is 2.5%. Residual same-everything survival is
**0**. Nested parts add to B (gap ~0).

Do **not** audit positioning, targeting, or a keyword correction.
Those terms are small or wrong-signed. Next hour: trace **why
matched-tier boards allocate synthetic stats differently** onto the
same printed-tier / same-recruit / same-slot bodies. Do not rewrite 2Q.
Do not change `_hero_damage`. Do not burn confirm.

| Component of T5/T6 B (+0.617) | Δ / split | Share of B |
|---|---:|---:|
| (1) Start-body combat strength / HP / synth | +0.746 | **1.210** |
| — recruit-raw mix | 0 | 0 |
| — synthetic allocation | +0.731 | **1.185** |
| — start-HP mix | +0.015 | 0.025 |
| (2) Attack opportunity / slot | −0.111 | −0.180 |
| (3) Target exposure / taunt | +0.105 | 0.170 |
| (4) Represented keywords | −0.022 | −0.036 |
| (5) Teammate / board protection | −0.101 | −0.165 |
| (6) Residual | 0 | **0** |
| Nested residual vs T5/T6 B | ~0 | — |

## Classification (observational)

Exclusive per 3N class-(3) first-split punch row. Primary window is
first-split turns **T5 / T6**. For every winner starting body record
printed tier, recruit atk/hp, synthetic atk/hp share, total combat
stats, slot / attack order, attacks made, death-before-first-attack,
incoming target count, taunt-forced / open targeting, DS / poison /
cleave / SOC / generated flags where represented, killer attack/tier,
and survived.

```text
applied        = _hero_damage          unchanged
3N B           = +0.6883852691218131   reproduced
T5/T6 B        = Σ_t t · n̄_t · ΔP(survive|t)
ΔP             = start_stats + attack_opp + target
               + keywords + teammate + residual
```

Boards are already matched on printed-tier histogram **and** on
recruit-raw and slot. The treatment difference is how 2S paints the
abstract pool onto those same bodies.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#60](https://github.com/aferna6-cell/Replay/pull/60).

3N T1 / T3 / T4 survival on the full 1059 still matches
(0.659→0.406 / 0.375→0.612 / 0.309→0.618). T5/T6 primary (no T4
starts) is T1 0.673→0.413 and T3 0.378→0.614.

| Tier | Ctrl synth | Treat synth | Ctrl combat | Treat combat | Ctrl P | Treat P | ΔP |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 22.2 | **14.6** | 27.9 | 20.3 | 0.673 | 0.413 | −0.260 |
| T2 | 19.7 | 17.9 | 26.6 | 24.8 | 0.496 | 0.555 | +0.059 |
| T3 | 7.1 | **19.2** | 14.7 | **26.8** | 0.378 | **0.614** | **+0.237** |

Recruit-raw is identical within tier (T1 5.67 / T2 6.85 / T3 7.61).
Slot is identical (T1 0.45 / T2 1.73 / T3 2.39). 2S strips synthetic
off T1 (22→15) and piles it onto T3 (7→19). T3 is **+0.746** of the
T5/T6 B and is 100% start-stats.

| Tier | T5/T6 B | (1) start | (2) slot | (3) target | (4) keywords | (5) teammate | (6) residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | −0.329 | −0.273 | −0.030 | +0.037 | −0.013 | −0.050 | 0 |
| T2 | +0.199 | +0.272 | −0.081 | +0.068 | −0.009 | −0.051 | 0 |
| T3 | **+0.746** | **+0.746** | 0 | 0 | 0 | 0 | 0 |

Keywords (DS / poison / cleave / SOC) have **zero** mix on T3.
Generated mix is small and negative. Residual after holding start
stats is 0 — same-stat bodies do not keep a leftover survival gap.

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles, then start-HP deciles, then slot bin, then
target bin, then DS / poison / cleave / SOC / gen, then teammate-raw
quintile. Full 1059 reproduces 3N B. Primary shares use the T5/T6 B.

```text
hold printed tier                              →  3N B = +0.688 (1059 / 1059)
    ↓
T5/T6 body-level B                             →  +0.617
    ↓
hold P(recruit / synth / start-HP | t)         →  121% of B (synth 118%)
    ↓
hold P(slot_bin | t, stats)                    →  −18% of B
    ↓
hold P(target_bin | …)                         →   17% of B
    ↓
hold P(DS / poison / cleave / SOC / gen)       →  −4% of B
    ↓
hold P(teammate-raw quintile | …)              →  −16% of B
    ↓
leftover P(survive | all of the above)         →   0% of B
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| 3N class-(3) | 1059 / 1059 | gap 0 |
| 3N within-tier B | 0.6883852691218131 | 0.6883852691218131 |
| T5/T6 class-(3) | 1033 / 1033 | — |
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| Event-count flags | 0 mismatches | 0 mismatches |
| Nested parts − T5/T6 B | ~0 | ~0 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
3N B = +0.688 is 100% within-tier survival
        ↓
T5/T6 primary = 1033 / 1059
T5/T6 B = +0.617
        ↓
same printed-tier, same recruit-raw, same slot
        ↓
2S paints synthetic off T1 (22→15)
        onto T3 (7→19)
        ↓
start-stats / synth = 121% of T5/T6 B
slot / target / keywords / teammate do not clear 70%
residual after holding start stats = 0
        ↓
start_stats_synth_dominates
        ↓
next: trace why matched-tier boards
      allocate synthetic stats differently
      do not rewrite 2Q; do not retune total scaling
      do not change `_hero_damage`
      do not burn confirm
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a positioning / targeting / keyword correction. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3N / **3O DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3o.py tests/test_phase_3n.py tests/test_phase_3b.py tests/test_phase_3c.py tests/test_phase_2x.py tests/test_phase_2y.py tests/test_phase_2z.py tests/test_phase_3a.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3o          # reused 14200–14699
```

Working tree was clean at contract time (`5ae2216`, 172.13s). Tracer is
observational.
