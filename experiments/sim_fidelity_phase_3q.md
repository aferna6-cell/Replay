# Simulator Fidelity Phase 3Q — play-lifecycle sticky-vs-repaint causal audit

Date: 2026-09-05 · Status: **`3q_v1` HOLD — `ranked_residual_needs_next_observable`** ·
Artifacts: [`results/sim_fidelity_phase_3q/`](../results/sim_fidelity_phase_3q/)

Stacked on PR #62 (`cursor/phase-3p-allocation-input-81c3`, head `79a46587`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 / #61 / #62 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3P DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3P left T5/T6 class-(3) timing/membership at **98.6%** of within-tier
synth mass, with last membership = play in both arms. This hour
snapshots every play on those trajectories and computes same-state
counterfactuals: (A) control board under would-be 2S
recruit-raw-proportional repaint; (B) treatment board under sticky
incumbent synth / no repaint.

## Verdict

**HOLD.** Route: **`ranked_residual_needs_next_observable`**.

3N class-(3) **1059 / 1059** and within-tier B **+0.6883852691218131**
reproduce. 3O T5/T6 **1033 / 1033**, B **+0.6166505324298197**, T1
**22.238→14.632**, T3 **7.066→19.185** reproduce. 3P **4132 / 4132**
paired bodies, 0 unpaired, last-play snapshots complete on every pair.
Paint / CF-A / sticky pool identities hold (0 share mismatches). Nested
parts sum to ΔS (residual 0).

The T1↓ / T3↑ combat-start move does **not** appear immediately as a
same-state sticky-vs-repaint on the last play (**7.2%**). Subsequent
scaling after that play is the largest term (**48.0%**). Replacement /
open-slot sticky-board difference is close behind (**44.8%**). Neither
clears ~70%. Last play subtype is **open_slot** at 1.0 on the paired
bodies (sell→buy→play 0; sell→play 0; triple 0). Signed pooled ΔS ≈
+0.023 sits entirely in subsequent scaling and is not the decision
denominator.

Do **not** retune scaling. Do not rewrite 2Q. Do not change
`_hero_damage`. Next hour: rank the leftover and pursue the largest
still-observable term (subsequent scale-sync inputs/timing, then
open-slot sticky membership).

| Component of T5/T6 within-tier synth mass | Share |
|---|---:|
| (1) Same-state repaint vs sticky | 7.2% |
| (2) Replacement / open-slot lifecycle | 44.8% |
| (3) Subsequent scaling | **48.0%** |
| (4) Residual | 0 |

## Classification (observational)

Exclusive per 3N class-(3) first-split punch row. Primary window is
first-split turns **T5 / T6**. For every play on a feeding trajectory
record pre-play board/body synth, incoming recruit raw/tier, open-slot
vs sell→buy→play, post-play pre-reallocation, post-reallocation, and
post-scale combat-start. Offline CFs only.

```text
ΔS = (S_t_paint − S_t_sticky_cf)          # same-state repaint
   + (S_t_sticky_cf − S_c_sticky)         # lifecycle (sticky boards)
   + (S_t_start − S_t_paint)
   − (S_c_start − S_c_sticky)             # subsequent scaling
   + residual
```

Decision shares use `n · |component|` per printed tier so T1↓ and T3↑
do not cancel.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#62](https://github.com/aferna6-cell/Replay/pull/62).

| Tier | n | Ctrl synth | Treat synth | ΔS | Same-state | Lifecycle | Scaling |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 1307 | 22.238 | **14.632** | **−7.606** | −0.619 | −3.325 | **−3.662** |
| T2 | 1739 | 19.711 | 17.913 | −1.798 | −0.077 | −0.880 | −0.840 |
| T3 | 1086 | 7.066 | **19.185** | **+12.119** | +0.868 | +5.412 | **+5.839** |

Last play subtype is **open_slot** at 1.0 / 1.0 on every paired T1–T3
body. Kind mismatch rate 0. Sample-trajectory plays: control 1393
(96.3% open-slot, 1.5% sell→buy→play, 2.2% sell→play); treatment 2401
(56.7% open-slot, 30.2% sell→buy→play, 13.1% sell→play). Incoming mean
tier 2.44 vs 3.30. Those earlier replacements are **not** the last play
that sets combat-start membership.

## Reweighting

```text
hold printed tier, same last play (open-slot, both arms)
    ↓
same-state 2S paint vs sticky at that play   →   7.2% of |mass|
    ↓
sticky treatment board ≠ sticky control      →  44.8% of |mass|
    ↓
post-play → combat-start residual/ratio      →  48.0% of |mass|
    ↓
residual                                     →   0
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| 3N class-(3) | 1059 / 1059 | gap 0 |
| 3N within-tier B | 0.6883852691218131 | 0.6883852691218131 |
| 3O T5/T6 B | 0.6166505324298197 | 0.6166505324298197 |
| 3P paired bodies | 4132 / 4132 | snapshots complete |
| T1 / T3 synth | 22.238→14.632 / 7.066→19.185 | reproduced |
| CF-A / actual Σ shares = painted pool | 0 mismatches | 0 mismatches |
| Nested parts − signed ΔS | 0 | 0 |
| Last play subtype | open_slot 1.0 | open_slot 1.0 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
3P timing / membership = 98.6% of |mass|
        ↓
last event = play (both arms)
        ↓
snapshot every play; CF-A paint / CF-B sticky
        ↓
same-state paint at last play = 7.2%
open-slot sticky-board diff     = 44.8%
subsequent scaling              = 48.0%
residual                        = 0
        ↓
neither term ≥ ~70%
        ↓
ranked_residual_needs_next_observable
        ↓
next: largest leftover is subsequent
      scale-sync inputs/timing
      (then open-slot sticky membership;
      do not retune scaling; do not rewrite 2Q;
      do not change `_hero_damage`; do not burn confirm)
```

**Not** a same-state representation audit. **Not** a sell→buy→play
subtype trace on the last play (that last play is open-slot). **Not** a
scaling retune. **Not** a 2Q rewrite. **Not** a `_hero_damage` change.
**Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3P / **3Q DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3q.py tests/test_phase_3p.py tests/test_phase_3o.py tests/test_phase_2s.py tests/test_phase_2r.py tests/test_phase_2q.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3q          # reused 14200–14699
```

Working tree was clean at contract time (`4ce91bc`, 235.37s sim).
Tracer is observational.
