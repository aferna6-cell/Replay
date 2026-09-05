# Simulator Fidelity Phase 3O — matched-board survivor-mechanic attribution

Date: 2026-09-05 · Status: **`3o_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_3o/`](../results/sim_fidelity_phase_3o/)

Stacked on PR #60 (`cursor/phase-3n-damage-attribution-2d61`, head `f61a0a90`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3N DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3N left same-outcome first splits board-matched (1059 / 1059) with
rules-faithful CF **+0.688 / split** assigned entirely to within-tier
survival (B = 1.0). This hour restricts primary analysis to T5/T6
class-(3) fights and attributes that leftover survival to represented
combat mechanics.

## Classification (observational)

Exclusive per 3N class-(3) first-split punch row. Primary window is
first-split turns **T5 / T6**. For every winner starting body record
printed tier, recruit atk/hp, synthetic atk/hp share, total combat
stats, slot / attack order, attacks made, death-before-first-attack,
incoming target count, taunt-forced / open targeting, DS / poison /
cleave / SOC / generated flags where represented, killer attack/tier,
and survived.

```text
applied        = _hero_damage                    unchanged
3N B           = Σ_t t · n̄_t · ΔP(survive|t)
ΔP(survive|t)  = start_stats + attack_opp + target
               + keywords + teammate + residual
```

Sequential nested Kitagawa on T5/T6 class-(3) winner starts:

```text
hold printed tier                         →  3N B reproduced on 1059
    ↓
hold P(recruit / synth / start-HP | t)    →  (1) start-body strength
    ↓
hold P(slot_bin | t, stats)               →  (2) attack opportunity
    ↓
hold P(target_bin | …)                    →  (3) target / taunt
    ↓
hold P(DS / poison / cleave / SOC / gen)  →  (4) represented keywords
    ↓
hold P(teammate-raw quintile | …)         →  (5) board protection
    ↓
leftover P(survive | all of the above)    →  (6) residual
```

## Decision rule

```text
if (1) start stats/synth  > ~70%  → trace matched-tier stat allocation
if (2) attack opportunity > ~70%  → audit positioning / initiative
if (3) target exposure    > ~70%  → audit targeting / taunt
if (4) represented keywords > ~70% → isolate that mechanic
else rank components and pursue the largest residual observable
```

Do **not** change `_hero_damage`. Do **not** rewrite 2Q. Do **not**
burn confirm.

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

Tracer is observational. Working-tree contract is recorded after DEV.
