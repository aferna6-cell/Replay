# Simulator Fidelity Phase 3P — synthetic-pool allocation-input attribution

Date: 2026-09-05 · Status: **`3p_v1` HOLD — `timing_membership_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3p/`](../results/sim_fidelity_phase_3p/)

Stacked on PR #61 (`cursor/phase-3o-survivor-mechanic-f896`, head `00c33ee5`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 / #60 / #61 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3O DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3O left T5/T6 class-(3) start-stats / synth at **121% / 118%** of B
(+0.617), with T1 synth **22.2→14.6** and T3 **7.1→19.2** on matched
printed-tier / same-recruit / same-slot boards. This hour reconstructs
the exact 2S paint equation at combat start and attributes that
body-synth move.

## Verdict

**HOLD.** Route: **`timing_membership_dominates`**.

3N class-(3) **1059 / 1059** and within-tier B **+0.6883852691218131**
reproduce. 3O T5/T6 **1033 / 1033**, B **+0.6166505324298197**,
start-stats **121%**, synth **118%**, T1 **22.238→14.632**, T3
**7.066→19.185** reproduce. Painted pool `= round(abstract_pool)` and
Σ body shares reconcile (0 mismatches). Nested parts sum to ΔS
(residual ~0).

On 4132 paired T5/T6 starting bodies, recruit-raw weights and board
denominators are identical within each slot pair (composition **0**).
Player painted pools are almost the same (T1 67.3 vs 67.4; T3 69.5 vs
69.4), so pool magnitude is **0.6%** of within-tier |mass|. Rounding /
largest-remainder is **0.8%** — not material, not a bug.

The T1↓ / T3↑ move is **98.6%** reallocation timing / membership:
control never paints (sticky per-minion synthetic), treatment paints
on the last board-membership-change. That event is **play** on
**100%** of T1/T2/T3 bodies in both arms (sell 0, triple 0; kind
mismatch 0). Signed pooled ΔS ≈ +0.023 cancels the opposite-signed
tier moves and is not the decision denominator.

Do **not** retune scaling. Do not rewrite 2Q. Do not change
`_hero_damage`. Next hour: trace the exact play lifecycle that leaves
control sticky and treatment recruit-raw-proportional.

| Component of T5/T6 within-tier synth mass | Share |
|---|---:|
| (1) Pool magnitude | 0.6% |
| (2) Weight / board-denominator | **0** |
| (3) Reallocation timing / membership | **98.6%** |
| (4) Integer rounding / remainder | 0.8% |
| (5) Residual | ~0 |

## Classification (observational)

Exclusive per 3N class-(3) first-split punch row. Primary window is
first-split turns **T5 / T6**. For every winner starting body record
player `abstract_pool`, recruit-raw weight, board recruit-raw
denominator, board size / tier histogram, largest-remainder vs exact
proportional share, last membership event (sell / play / triple), and
pre/post-reallocation synthetic share.

```text
painted_pool = round(abstract_pool)
share_i      = largest_remainder(w_i / W, painted_pool)
ΔS           = (pool_t − pool_c)·share_t
             + pool_c·(share_t − share_c)
             + (LR_c − S_c) + (S_t − LR_t)
             + (LR_t − exact_t) − (LR_c − exact_c)
```

`(LR_c − S_c)` is control sticky vs the would-be 2S paint.
`(S_t − LR_t)` is treatment combat-start vs paint (≈0 after scale
sync). Decision shares use `n · |component|` per printed tier so T1↓
and T3↑ do not cancel.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#61](https://github.com/aferna6-cell/Replay/pull/61).

| Tier | n | Ctrl synth | Treat synth | ΔS | Pool | Weight | Timing | Rounding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 1307 | 22.238 | **14.632** | **−7.606** | +0.014 | 0 | **−7.599** | −0.020 |
| T2 | 1739 | 19.711 | 17.913 | −1.798 | +0.066 | 0 | −1.925 | +0.061 |
| T3 | 1086 | 7.066 | **19.185** | **+12.119** | −0.035 | 0 | **+12.227** | −0.073 |

Last membership event is **play** at 1.0 / 1.0 (control / treatment)
on every T1–T3 starting body. Kind mismatch rate 0.

## Reweighting

```text
hold printed tier, same recruit-raw, same slot
    ↓
pool magnitude                         →   0.6% of |mass|
    ↓
weight / board-denominator             →   0
    ↓
last membership = play (both arms)
control sticky vs LR paint             →  98.6% of |mass|
    ↓
largest-remainder vs exact             →   0.8% of |mass|
    ↓
residual                               →   ~0
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| 3N class-(3) | 1059 / 1059 | gap 0 |
| 3N within-tier B | 0.6883852691218131 | 0.6883852691218131 |
| 3O T5/T6 B | 0.6166505324298197 | 0.6166505324298197 |
| T1 / T3 synth | 22.238→14.632 / 7.066→19.185 | reproduced |
| Σ minion shares = painted pool | 0 mismatches | 0 mismatches |
| Nested parts − signed ΔS | ~0 | ~0 |
| Last event | play 1.0 | play 1.0 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
3O synth = 118% of T5/T6 B
        ↓
T1 22.2→14.6 and T3 7.1→19.2
        ↓
same pool, same recruit-raw weights, same last event (play)
        ↓
control keeps sticky per-minion synth
treatment paints on play
        ↓
timing / membership = 98.6% of within-tier |mass|
pool 0.6% · weight 0 · rounding 0.8% · residual ~0
        ↓
timing_membership_dominates
        ↓
next: trace the play lifecycle that creates
      sticky-vs-paint (do not retune scaling;
      do not rewrite 2Q; do not change `_hero_damage`;
      do not burn confirm)
```

**Not** a pool-magnitude / scaling retune. **Not** a recruit-raw
proportional-paint audit. **Not** a rounding bug. **Not** a 2Q
rewrite. **Not** a `_hero_damage` change. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3O / **3P DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3p.py tests/test_phase_3o.py tests/test_phase_3e.py tests/test_phase_2s.py tests/test_phase_2r.py tests/test_phase_2q.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3p          # reused 14200–14699
```

Working tree was clean at contract time (`c01b973`, 225.14s sim).
Tracer is observational.
