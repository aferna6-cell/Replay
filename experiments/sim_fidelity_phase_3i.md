# Simulator Fidelity Phase 3I — T1–T3 pairing / who-wins attribution

Date: 2026-09-04 · Status: **`3i_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_3i/`](../results/sim_fidelity_phase_3i/)

Stacked on PR #54 (`cursor/phase-3h-board-retention-f0c7`, head `de26f9e`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3H DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3H left late T1–T3 punch rows **17924 → 4273** (collapse **13651**) as
leftover pairing / who-wins **52.4%** (7155 rows), elimination **48.0%**,
offer-shift **30.9%**. This hour restricts to those leftover control
rows whose paired treatment seat is alive and still fields ≥1 T1–T3
body, then decomposes the missing treatment low-tier winner-start
punches.

## Classification (observational)

Exclusive per leftover punch row:

```text
different live opponent / ghost / bye / missing fight → (1) pairing_schedule
same pairing, treatment loses or ties                 → (2) outcome_flip
same pairing, treatment wins, this row not covered
  by a treatment T1–T3 winner-start punch             → (3) survivor_substitution
same pairing, treatment wins, covered                 → (4) residual
```

Each matched fight records opponent identity/seat, both boards’ T1–T3
count/share, tavern tier, recruit raw, abstract-pool raw, total combat
raw, survivor count/tier sum, pre-fight HP, fight outcome/margin,
whether the low-tier body attacks/survives, and next-turn alive state.

## Protocol

```bash
python -m pytest tests/test_phase_3i.py tests/test_phase_3h.py tests/test_phase_3g.py tests/test_phase_3f.py tests/test_phase_3d.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3i          # reused 14200–14699
```

Tracer is observational. Do **not** apply a scaling correction. Do
**not** rewrite 2Q. Do **not** burn confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3H / **3I DEV** | **14200–14699** | **reused; no new seeds** |
