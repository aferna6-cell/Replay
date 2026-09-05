# Simulator Fidelity Phase 3R — post-play scale-sync input/timing attribution

Date: 2026-09-05 · Status: **`3r_v1` HOLD — `membership_allocation_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3r/`](../results/sim_fidelity_phase_3r/)

Stacked on PR #63 (`cursor/phase-3q-play-lifecycle-c28e`, head `729d1715`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 / #61 / #62 / #63 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3Q DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3Q left subsequent scaling at **48.0%** of T5/T6 within-tier |mass|.
This hour snapshots both arms immediately after the last open-slot play
and at every subsequent residual/ratio scale-sync through combat start.

## Verdict

**HOLD.** Route: **`membership_allocation_dominates`**.

3Q subsequent scaling **0.47994509684306846** reproduces on the T5/T6
3N class-(3) sample (4132 paired bodies, snapshots complete, 1033
primary fights). T1 22.2→14.6 / T3 7.1→19.2 reproduce. Nested parts
sum to Δ_scale (residual ~0). Board- and body-level flow gaps are 0.

The 48.0% term is **not** a different Firestone/residual budget and
**not** a different number or order of scale-syncs. Sync count and
turn sequence match at 1.0 / 1.0 (mean 1.144 syncs both arms). Same
inputs allocate differently because board membership / combat-weight
shares differ: **97.0%**. Rounding 1.9%. Pre-sync input state 1.2%.
Timing/count 0. Residual 0.

`abstract_pool_entering` is the first recorded input difference on
every pair (control 2S pool is 0). That does **not** clear ~70% of
Δ_scale; the body increment difference is membership-conditioned
allocation of treatment's residual budget.

Do **not** retune scaling. Do not rewrite 2Q. Do not change
`_hero_damage`. Next hour: route to the open-slot lifecycle lane
(3Q leftover 44.8% sticky-board difference, now confirmed as the
scale-sync amplifier).

| Component of 3Q subsequent scaling | Share |
|---|---:|
| (1) Pre-sync target/input state | 1.2% |
| (2) Sync timing / count / order | 0 |
| (3) Membership-conditioned allocation | **97.0%** |
| (4) Rounding / numeric residue | 1.9% |
| (5) Residual | 0 |

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

On common turns:
  input_state = share_c · (R_t − R_c)
  membership  = R_t · (share_t − share_c)
  rounding    = (actual_t − exact_t) − (actual_c − exact_c)
```

Extra or missing sync turns are timing/count. Decision shares use
`n · |component|` per printed tier so T1↓ and T3↑ do not cancel.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#63](https://github.com/aferna6-cell/Replay/pull/63).

| Tier | n | 3Q Δ_scale | Input | Timing | Membership | Rounding |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 1307 | **−3.662** | −0.059 | 0 | **−3.608** | +0.005 |
| T2 | 1739 | −0.840 | −0.025 | 0 | −0.923 | +0.107 |
| T3 | 1086 | **+5.839** | −0.027 | 0 | **+5.820** | +0.047 |

Last-play subtype remains open-slot on the paired bodies. Sync
count/turn match rate 1.0. Sample-trajectory post-play syncs:
control 768 (mean residual_add 108.1, board recruit raw 21.9, entering
pool 0); treatment 798 (residual_add 157.4, recruit raw 24.9, entering
pool 271.5). Those board-level budget differences do not move the
within-tier |mass| once allocated at control share.

## Reweighting

```text
hold printed tier, same last play (open-slot, both arms)
    ↓
3Q subsequent scaling reproduced at 48.0% of |mass|
    ↓
same sync count/turn (1.0 / 1.0)
    ↓
same residual_add × control share          →   1.2% of Δ_scale
    ↓
treatment residual_add × (share_t−share_c) →  97.0% of Δ_scale
    ↓
integer rounding of residual apply         →   1.9%
    ↓
residual                                   →   0
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| 3N class-(3) | 1059 / 1059 | gap 0 |
| 3N within-tier B | 0.6883852691218131 | 0.6883852691218131 |
| 3O T5/T6 B | 0.6166505324298197 | 0.6166505324298197 |
| 3P / 3Q paired bodies | 4132 / 4132 | snapshots complete |
| 3Q subsequent scaling | 0.47994509684306846 | reproduced |
| T1 / T3 synth | 22.238→14.632 / 7.066→19.185 | reproduced |
| Nested parts − signed Δ_scale | ~0 | ~0 |
| Board / body flow mismatches | 0 | 0 |
| Same sync count / turns | 1.0 | 1.0 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
3Q subsequent scaling = 48.0% of |mass|
        ↓
snapshot last play + every later scale-sync
        ↓
input state     = 1.2%
sync timing     = 0
membership alloc = 97.0%
rounding        = 1.9%
residual        = 0
        ↓
membership_allocation ≥ ~70%
        ↓
membership_allocation_dominates
        ↓
next: open-slot lifecycle lane
      (do not retune scaling; do not rewrite 2Q;
      do not change `_hero_damage`; do not burn confirm)
```

**Not** a Firestone / residual-budget retune. **Not** a scale-sync
timing-fidelity audit. **Not** a 2Q rewrite. **Not** a `_hero_damage`
change. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3Q / **3R DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3r.py tests/test_phase_3q.py tests/test_phase_3p.py tests/test_phase_3o.py tests/test_phase_2s.py tests/test_phase_2r.py tests/test_phase_2q.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3r          # reused 14200–14699
```

Working tree was clean at contract time (`cdc3d01`, 235.36s sim).
Tracer is observational.
