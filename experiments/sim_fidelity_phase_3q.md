# Simulator Fidelity Phase 3Q — play-lifecycle sticky-vs-repaint causal audit

Date: 2026-09-05 · Status: **`3q_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_3q/`](../results/sim_fidelity_phase_3q/)

Stacked on PR #62 (`cursor/phase-3p-allocation-input-81c3`, head `79a46587`).
Keep **#29 / #33–#47 / #50–#62 HOLD**. Do **not** merge. Confirm
**11500–11699** untouched. No α / residual / `_hero_damage` / gate /
behavior / default / recruit / scaling / 2Q / damage changes. Reused
consumed 2S–3P DEV **14200–14699** (no new seeds). PR #49 CI remains
separate.

3P left T5/T6 class-(3) timing/membership at **98.6%** of within-tier
synth mass (last event = play in both arms). This hour snapshots every
play on those trajectories and computes same-state counterfactuals:

* (A) control board under would-be 2S recruit-raw-proportional repaint
* (B) treatment board under sticky incumbent synth / no repaint

DEV 14200–14699 results will fill the verdict.

## Protocol

```bash
python -m pytest tests/test_phase_3q.py tests/test_phase_3p.py tests/test_phase_3o.py tests/test_phase_2s.py tests/test_phase_2r.py tests/test_phase_2q.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3q          # reused 14200–14699
```

Tracer is observational.
