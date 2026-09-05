# Simulator Fidelity Phase 3S — open-slot board-formation attribution

Date: 2026-09-05 · Status: **`3s_v1` HOLD — `pre_play_membership_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3s/`](../results/sim_fidelity_phase_3s/)

Stacked on PR #64 (`cursor/phase-3r-scale-sync-a549`, head `f39565a2`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 / #61 / #62 / #63 / #64 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3R DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3R left subsequent scaling at **48.0%** of T5/T6 within-tier |mass|, of
which **97.0%** is membership-conditioned allocation. 3Q leftover
**44.8%** is the open-slot sticky-board difference. This hour traces
each last open-slot play backward to the event that created that slot.

## Verdict

**HOLD.** Route: **`pre_play_membership_dominates`**.

3Q open-slot lifecycle **0.4481470184535611** reproduces on the T5/T6
3N class-(3) sample (4132 paired bodies, snapshots complete, 1033
primary fights). 3R membership allocation **0.969763531093672**
reproduces. T1 22.2→14.6 / T3 7.1→19.2 reproduce. Nested parts sum to
the lifecycle term (residual 0). Event→board→synth flow gaps are 0.

Last play is open-slot at 1.0 / 1.0. Slot-opening cause is
**normal under-fill** on every pair (both arms). Last-play **identities**
match at 1.0; last-play **incumbent-synth state** matches at 0. The
44.8% term is not a different incoming minion, not a different
vacancy rule, and not a different buy/play order. Same pieces already
carry different sticky synth before that last play: **100%**.

That same exclusive tag carries the entire 3R membership-allocation
increment: **100%** of the 97.0% membership |mass| is pre-play
composition. Modal earliest board-composition diverge turn is **T5**
(4132 / 4132).

Do **not** retune scaling. Do not rewrite 2Q. Do not change
`_hero_damage`. Next hour: walk the earliest board-composition split
(T5 pre-play incumbent synth).

| Component of 3Q open-slot lifecycle | Share |
|---|---:|
| (1) Pre-play membership / composition | **100%** |
| (2) Incoming identity / tier / raw | 0 |
| (3) Slot-opening cause / timing | 0 |
| (4) Buy/play order / affordability | 0 |
| (5) Residual | 0 |

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
field (identity or incumbent synth counts as pre-play composition).
The same tag rides on that pair's 3R membership_allocation.
```

Decision shares use `n · |component|` per printed tier so T1↓ and T3↑
do not cancel.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#64](https://github.com/aferna6-cell/Replay/pull/64).

| Tier | n | 3Q lifecycle | Pre-play | Incoming | Opening | Order | 3R membership |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 1307 | **−3.325** | **−3.325** | 0 | 0 | 0 | **−3.608** |
| T2 | 1739 | −0.880 | −0.880 | 0 | 0 | 0 | −0.923 |
| T3 | 1086 | **+5.412** | **+5.412** | 0 | 0 | 0 | **+5.820** |

Last-play subtype remains open-slot. Opening cause is normal
under-fill at 1.0 / 1.0. Same last-play identity rate 1.0; same
incumbent-synth state rate 0.

## Reweighting

```text
hold printed tier, same last play (open-slot, both arms)
    ↓
3Q open-slot lifecycle reproduced at 44.8% of |mass|
    ↓
same last-play identities (1.0) / same opening (under-fill)
    ↓
different incumbent synth already on that board  →  100% of lifecycle
    ↓
incoming / opening / order / residual            →  0
    ↓
same tag on 3R membership_allocation             →  100% of 97.0%
    ↓
earliest composition diverge                     →  T5 (all 4132)
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| 3N class-(3) | 1059 / 1059 | gap 0 |
| 3N within-tier B | 0.6883852691218131 | 0.6883852691218131 |
| 3O T5/T6 B | 0.6166505324298197 | 0.6166505324298197 |
| 3P / 3Q / 3R paired bodies | 4132 / 4132 | snapshots complete |
| 3Q open-slot lifecycle | 0.4481470184535611 | reproduced |
| 3R membership allocation | 0.969763531093672 | reproduced |
| T1 / T3 synth | 22.238→14.632 / 7.066→19.185 | reproduced |
| Nested parts − lifecycle | 0 | 0 |
| Event / board flow mismatches | 0 | 0 |
| Last play subtype | open_slot 1.0 | open_slot 1.0 |
| Slot-opening cause | under-fill 1.0 | under-fill 1.0 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
3Q open-slot lifecycle = 44.8% of |mass|
        ↓
trace last open-slot play → vacancy event
        ↓
pre-play composition = 100%
incoming identity    = 0
slot-opening cause   = 0
buy/play order       = 0
residual             = 0
        ↓
pre_play_membership ≥ ~70%
        ↓
pre_play_membership_dominates
        ↓
3R membership increment also 100% this tag
        ↓
next: earliest board-composition split (T5)
      (do not retune scaling; do not rewrite 2Q;
      do not change `_hero_damage`; do not burn confirm)
```

**Not** a last-play incoming-identity audit. **Not** a vacancy-rule
audit (under-fill is universal). **Not** a buy/play-order audit.
**Not** a scaling retune. **Not** a 2Q rewrite. **Not** a
`_hero_damage` change. **Not** confirm.

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

Working tree was clean at contract time (`c422cd5`, 278.89s sim).
116 focused tests passed. Tracer is observational.
