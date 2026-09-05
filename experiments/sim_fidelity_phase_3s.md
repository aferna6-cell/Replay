# Simulator Fidelity Phase 3S — open-slot board-formation attribution

Date: 2026-09-05 · Status: **`3s_v1` implemented, not yet evaluated** ·
Artifacts: [`results/sim_fidelity_phase_3s/`](../results/sim_fidelity_phase_3s/)

Stacked on PR #64 (`cursor/phase-3r-scale-sync-a549`, head `f39565a2`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 / #61 / #62 / #63 / #64 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3R DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3R left subsequent scaling at **48.0%** of T5/T6 within-tier |mass|, of
which **97.0%** is membership-conditioned allocation. 3Q leftover
**44.8%** is the open-slot sticky-board difference, now confirmed as
the scale-sync amplifier. This hour traces each last open-slot play
backward to the event that created that slot.

## Classification (observational)

Exclusive per 3N class-(3) first-split punch row. Primary window is
first-split turns **T5 / T6**. For every paired body record last
open-slot play: prior board size/tier IDs, incumbent synth/recruit raw,
slot-opening cause (normal under-fill, prior sell, death/generated
cleanup, triple/transform), incoming minion ID/tier/recruit raw, shop
offer set, gold, buy/play order, and board immediately before/after
the play. Offline exclusive first-difference only.

```text
replacement_lifecycle
        = pre_play_membership
        + incoming_identity
        + slot_opening_cause
        + buy_play_order
        + residual

Each pair assigns its full lifecycle term to the first differing
field. The same tag rides on that pair's 3R membership_allocation.
```

Decision shares use `n · |component|` per printed tier so T1↓ and T3↑
do not cancel.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3R / **3S DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3s.py tests/test_phase_3r.py tests/test_phase_3q.py tests/test_phase_3p.py tests/test_phase_3o.py tests/test_phase_2s.py tests/test_phase_2q.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3s          # reused 14200–14699
```

Tracer is observational. No behavior change.
