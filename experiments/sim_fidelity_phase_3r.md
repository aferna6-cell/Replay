# Simulator Fidelity Phase 3R — post-play scale-sync input/timing attribution

Date: 2026-09-05 · Status: **`3r_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_3r/`](../results/sim_fidelity_phase_3r/)

Stacked on PR #63 (`cursor/phase-3q-play-lifecycle-c28e`, head `729d1715`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 / #61 / #62 / #63 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3Q DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3Q left subsequent scaling at **48.0%** of T5/T6 within-tier |mass|
(4132 paired bodies / 1033 primary fights). This hour snapshots both
arms immediately after the last open-slot play and at every subsequent
residual/ratio scale-sync through combat start, then splits that 48.0%
term.

## Verdict

Pending 500-lobby DEV on reused 14200–14699. Offline identities and
hooked/unhooked invariance are locked in `tests/test_phase_3r.py`.

Decision rule (after DEV):

- if input-state share > ~70% → trace which upstream input first diverges
- if sync timing/count share > ~70% → audit lifecycle timing fidelity
- if membership-conditioned allocation share > ~70% → open-slot lifecycle lane
- otherwise rank components and pursue the largest

## Classification (observational)

Exclusive per 3N class-(3) first-split punch row. Primary window is
first-split turns **T5 / T6**. For every paired body record last-play
synth (control sticky / treatment paint) and each later scale-sync:
board recruit raw, abstract/synth pool entering scale, Firestone/target
raw, residual/ratio budget inputs, computed scale increment, body-level
allocation before/after sync, board membership/tier mix, and sync
count/order. Offline CFs only.

```text
Δ_scale = (S_t_start − S_t_paint) − (S_c_start − S_c_sticky)
        = pre_sync_input_state
        + sync_timing_count
        + membership_allocation
        + rounding_residue
        + residual
```

On common turns:

```text
input_state = share_c · (R_t − R_c)
membership  = R_t · (share_t − share_c)
rounding    = (actual_t − exact_t) − (actual_c − exact_c)
```

Extra or missing sync turns are timing/count. Decision shares use
`n · |component|` per printed tier so T1↓ and T3↑ do not cancel.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3Q / **3R DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3r.py tests/test_phase_3q.py tests/test_phase_3p.py tests/test_phase_3o.py tests/test_phase_2s.py tests/test_phase_2r.py tests/test_phase_2q.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3r
```

Tracer is observational. Do not retune scaling. Do not rewrite 2Q. Do
not change `_hero_damage`. Do not burn confirm.
