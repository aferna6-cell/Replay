# Simulator Fidelity Phase 2U — survivor-tier damage fidelity

Date: 2026-09-04 · Status: **`2u_v1` implemented, DEV pending** ·
Artifacts: [`results/sim_fidelity_phase_2u/`](../results/sim_fidelity_phase_2u/)

Stacked on PR #38 (`cursor/phase-2t-game-length-a550`, head `9013bb4`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 HOLD**. Do **not** merge.
Confirm **11500–11699** untouched. No α / residual / `_hero_damage` / gate /
behavior / default changes. Reused consumed 2S/2T DEV **14200–14699**
(no new seeds).

## Question

Phase 2T attributed ~79% of the −2.182-turn 2S game-length miss to
`_hero_damage`: it recovers survivor *count* from sim.py raw and reweights
by **mean winner-board tier**, not actual combat-survivor tiers. Treatment
amplification is +2.78 HP/hit vs control.

This hour is **measurement only**. Does using *actual* surviving minion
tavern tiers remove most of that asymmetric amplification?

## Counterfactuals (observational; applied HP unchanged)

```text
applied        = _hero_damage  = tavern + round(n_surv × mean(board.tier))
count_only     = sim.py raw    = tavern + n_surv
counterfactual = rules-faithful = tavern + sum(actual survivor tavern tiers)
error          = applied − counterfactual
```

Combat instrumentation is observational: `simulate_once(..., trace=)` fills
survivor identities/tiers after the fight. RNG, outcomes, placements, HP,
and `_hero_damage` are unchanged.

## Decision rule (preregistered)

```text
share_removed = 1 − (Δ(CF − count_only) / Δ(applied − count_only))
                treatment − control, T7–T14 hits
```

| Result | Route |
|---|---|
| `share_removed ≥ 0.55` | **`preregister_default_off_damage_formula`** — next hour preregister a default-OFF damage-formula treatment |
| else | **`isolate_survivor_composition`** — survivor mix/tier distribution still carries the asymmetry; do not change `_hero_damage` yet |

Do not burn confirm. Do not retune recruit/scaling.

## Protocol

```bash
python -m pytest tests/test_phase_2u.py tests/test_phase_2t.py tests/test_sim.py -q
python -m ml.fidelity_phase_2u          # reused 14200–14699
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S / 2T / **2U DEV** | **14200–14699** | **reused; no new seeds** |

## Verdict

Pending 500-lobby DEV on clean tree.
