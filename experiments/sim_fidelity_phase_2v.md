# Simulator Fidelity Phase 2V — survivor-composition attribution

Date: 2026-09-04 · Status: **`2v_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_2v/`](../results/sim_fidelity_phase_2v/)

Stacked on PR #39 (`cursor/phase-2u-survivor-tier-e84b`, head `81151da`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 HOLD**. Do **not** merge.
Confirm **11500–11699** untouched. No α / residual / `_hero_damage` / gate /
behavior / default / recruit / scaling changes. Reused consumed 2S/2T/2U DEV
**14200–14699** (no new seeds).

## Question

2U found that actual-survivor (rules-faithful) damage **widens** the
treatment−control amp gap (+3.63 vs proxy +2.78). Survivor tier mean is
3.56 (treatment) vs 2.37 (control); survivor tier sum Δ **+4.01 / hit**.
This hour attributes that +4.01: who is on the winner board at fight start,
who actually lives, and how much of the gap is tokens.

Do **not** change `_hero_damage`.

## Classification (per decisive T7–T14 hit)

Winner board-at-start (env minion) and actual combat survivors, by:

* printed Tavern tier
* golden / non-golden
* token / generated status (traced combat bodies)
* recruit-value raw stats vs combat raw stats
* tribe / archetype
* board slot

## Decomposition

```text
Δ survivor_tier_sum = (A) fielded composition
                    + (B) within-tier survival
                    + (C) token / generated bodies
                    + residual
```

Kitagawa mid-point on starting-origin bodies so (A)+(B) equals the
starting-origin gap. (C) is generated-survivor tier sum.

## Decision (after DEV)

* Most of +4.01 from **(A)** → next measures whether 2Q replacement
  over-selects tier/raw stats vs real Firestone board composition.
* Most from **(B)** → diagnose combat/scaling allocation by tier.
* Most from **(C)** → audit token tier / creation fidelity.

Dominant-share threshold: 0.55. Do not burn confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S / 2T / 2U / **2V DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2v.py tests/test_phase_2u.py tests/test_phase_2t.py tests/test_sim.py -q
python -m ml.fidelity_phase_2v          # reused 14200–14699
```

Tracer is observational (same-seed hooked vs unhooked placements/HP/RNG match).
Reconciliation: tier-bucket sums = survivor tier sum; survivors ⊆ starting ∪
created combat bodies.
