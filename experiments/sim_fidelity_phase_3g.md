# Simulator Fidelity Phase 3G — punch-sample selection decomposition

Date: 2026-09-04 · Status: **`3g_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_3g/`](../results/sim_fidelity_phase_3g/)

Stacked on PR #52 (`cursor/phase-3f-carry-divergence-765c`, head `4cd321b`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3F DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3F left the #51 punch-row Δcarry **−196** as **9.1% paired unconditional /
90.9% selection / outcome-conditioning**, with unpaired T1 opposing carry
**798 → 65**. This hour decomposes that crater with a symmetric
Kitagawa / Oaxaca reweight on common support.

## Protocol

```bash
python -m pytest tests/test_phase_3g.py tests/test_phase_3f.py tests/test_phase_3e.py tests/test_phase_3d.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3g          # reused 14200–14699
```

Measurement only. Tracer is observational.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3F / **3G DEV** | **14200–14699** | **reused; no new seeds** |
