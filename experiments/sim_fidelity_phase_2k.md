# Simulator Fidelity Phase 2K — post-assembly residual composition-gap diagnostic

Date: 2026-09-02 · Status: **measurement-only (2k_v1)** ·
Artifacts: [`results/sim_fidelity_phase_2k/`](../results/sim_fidelity_phase_2k/)

## Research question

> After a build starts coherently forming under accepted Phase 2J, where exactly
> is the remaining weighted core-coverage mass lost?

## Frozen policy

- `BoardOpportunityCostPolicy`, **α = 0.5**
- `prior_hash_sha256 = 9b31c93a8d89…` (Phase 2J canonical prior)
- No policy/simulator changes — observational only

## Seeds

| Range | Role |
|---|---|
| **9000–9499** | DEV diagnostic (500 lobbies) |
| 9500–9999 | Adaptive expansion if &lt;40 post-assembly states |
| 8000–8199 | **Forbidden** (Phase 2J confirmation) |
| 10000–10199 | Reserved future intervention confirmation |

## Cohort

`(lobby, winner_seat, archetype_key)` enters at the first end-of-recruit where:

```text
inferred target == archetype_key
core_count >= 2
```

Archetype is frozen from that moment; later `infer_target` changes are events
(`F_TARGET_SWITCH`), not silent retargets.

## Missing-mass taxonomy

Each unit of missing final weighted coverage (`arch.core` normalized) gets exactly
one primary cause: never-available / available-not-bought / bought-not-deployed /
deployed-then-lost / existing-core-lost / target-switch / triple-discover /
unresolved.

Headline: **% of missing weighted coverage mass by cause**.

## Commands

```bash
pytest tests/test_post_assembly_gap_diagnostic.py
python -m ml.fidelity_phase_2k
```

## Decision tree (precommitted)

```text
>50% NEVER_AVAILABLE     → shop/pool availability fidelity
>50% AVAILABLE_NOT_BOUGHT → post-commit valuation/policy
>50% buy/deploy/existing loss → retention/sell policy
>50% TARGET_SWITCH       → target commitment/hysteresis
triple/discover dominates → triple/discover fidelity
available+acquired+retained but coverage stalls
                         → representation/core-set before card effects
mixed                    → expand DEV; do not implement
```

Card effects are **not** justified by low core-coverage alone.

## Frozen

No acceptance gate, no new policy, no card effects, no BC/DAgger/PPO.
