# Simulator Fidelity Phase 2L — post-assembly availability decomposition

Date: 2026-09-02 · Status: **measurement-only complete (2l_v1)** ·
Artifacts: [`results/sim_fidelity_phase_2l/`](../results/sim_fidelity_phase_2l/)

## Research question

> Of the missing weighted core mass that is never *legally* buyable after first-2,
> how much is tier-locked, never raw-offered, raw-offered-but-illegal, or excluded
> from the lobby pool?

## Frozen policy

Phase 2J α=0.5, prior `9b31c93a…`. Seeds **10200–10699** (44 states; no expand).

## DEV results

**44** post-assembly states. Never-legal mass ≈ **100%** of total missing in this cohort
(consistent with 2K: available cores are bought).

### Subfate share of never-legal missing mass

| Subfate | Share |
|---|---:|
| **A3_TIER_ELIGIBLE_ZERO_RAW** | **62.8%** |
| A1_NOT_IN_LOBBY_POOL | 37.2% |
| A2_NEVER_TIER_ELIGIBLE | 0% |
| **A4_RAW_BUT_ZERO_LEGAL** | **0%** |
| A5_OTHER | 0% |

### Headlines

```text
tier-eligible but ZERO RAW:     62.8% of never-legal mass
RAW but ZERO LEGAL:              0.0% of never-legal mass
```

Static sampler expectation for never-legal cards predicted ~48 raw appearances;
observed **0**. Economy/action-mask is **not** the bottleneck.

## Decision

`a3_tier_eligible_zero_raw` → **Phase 2M: shop/pool generation**

Do **not** touch legality/economy first (A4 = 0). Secondary A1 mass (37%)
suggests also auditing lobby pool / tribe filtering / card-data inclusion when
implementing generation fixes.

## Commands

```bash
pytest tests/test_availability_decomposition.py
python -m ml.fidelity_phase_2l
```

## Frozen

No policy changes, no card effects, no BC/DAgger/PPO.
