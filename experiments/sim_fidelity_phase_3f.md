# Simulator Fidelity Phase 3F — carry divergence timing + outcome-conditioning audit

Date: 2026-09-04 · Status: **`3f_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_3f/`](../results/sim_fidelity_phase_3f/)

Stacked on PR #51 (`cursor/phase-3e-board-pool-lifecycle-f156`, head `2f93efb`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3E DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3E left **inherited carry = 72.4% of 3D A1 (+0.305 / hit)** with punch-row
Δcarry **−196**. Unconditional alive-seat carry stayed arm-matched; the
punch-sample crater sat on low-tier winner-start bodies. This hour dates
that gap: paired (seed, seat) carry from T7 through each seat’s T7–T14
punch-row appearance, then the same Δ conditioned on later punch
inclusion / winner-start tier / eventual outcome.

## Protocol

```bash
python -m pytest tests/test_phase_3f.py tests/test_phase_3e.py tests/test_phase_3d.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3f          # reused 14200–14699
```

Decision forks (share of the #51 punch-row Δcarry −196):

```text
unconditional paired Δcarry by turn
        ↓
+ later punch inclusion
        ↓
+ low winner-start tier (T1)
        ↓
+ eventual fight outcome
        ↓
if pre-conditioning paired |Δ| / 196 > ~70%
    → paired_divergence_precedes_conditioning
      next: audit upstream scaling *inputs* at first divergence turn
if the gap appears mainly after punch / winner / low-tier / outcome
    → selection_outcome_conditioning_dominates
      next: isolate that selection mechanism; do not change scaling
if mixed
    → quantify both and route to the larger
```

Tracer is observational. History-link identity:
`punch.opp_carry = seat_history[turn].attack_pool_recruit_start`.
Hooked vs unhooked placements / HP / RNG / outcome must match.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3E / **3F DEV** | **14200–14699** | **reused; no new seeds** |
