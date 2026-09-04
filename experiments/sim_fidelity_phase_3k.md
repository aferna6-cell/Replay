# Simulator Fidelity Phase 3K — elimination-timing attribution

Date: 2026-09-04 · Status: **`3k_v1` in progress** ·
Artifacts: [`results/sim_fidelity_phase_3k/`](../results/sim_fidelity_phase_3k/)

Stacked on PR #56 (`cursor/phase-3j-matchmaking-attribution-2f0d`, head `5916ea07`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3J DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3J left pairing-schedule leftover as alive/ghost eligibility **94.9%**
(5648 of 5952). This hour traces T7 through first eligibility
divergence for those rows and splits them by *why* the live set
already differs.

## Protocol

```bash
python -m pytest tests/test_phase_3k.py tests/test_phase_3j.py tests/test_phase_3i.py tests/test_phase_3h.py tests/test_phase_2t.py tests/test_phase_2u.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3k          # reused 14200–14699
```

Tracer is observational. Verdict and exact counts land after the DEV run.
