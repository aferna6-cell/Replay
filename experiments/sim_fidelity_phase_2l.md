# Simulator Fidelity Phase 2L — post-assembly availability decomposition

Date: 2026-09-02 · Status: **2l_v2 methodology cleanup (re-run pending / in PR)** ·
Artifacts: [`results/sim_fidelity_phase_2l/`](../results/sim_fidelity_phase_2l/)

## Research question

> Of the missing weighted core mass that is never *legally* buyable after first-2,
> how much is absent from the **exact** simulator catalogue, tier-locked, never
> raw-offered while catalogue+tier-eligible, or raw-offered-but-illegal?

## Methodology `2l_v2` (vs `2l_v1`)

1. **A1** uses exact `BGEnv.build_pool(lobby_tribes=…)` membership (not a simplified
   tribe∩KB check). Only exact-catalogue cards enter A2/A3/A4.
2. A1 subtypes: `MISSING_KB_OR_TIER_OR_STATS` / `TRIBE_EXCLUDED` /
   `BUILD_POOL_EXCLUDED` (+ reconciliation: KB tribe-ok but catalogue excludes).
3. Sampler calibration is **unconditioned**: missing-final ∩ exact catalogue ∩
   tier-eligible — expected vs observed raw; expected/observed cards with ≥1
   appearance; **observed zero-offer rate vs expected zero-offer rate**
   (`P_zero` product under static pool weights).
4. Do **not** treat “expected ~N on never-legal cards, observed 0” as confirmatory
   sampler failure (selection-biased by A3 definition).

## Frozen policy

Phase 2J α=0.5, prior `9b31c93a…`. Seeds **10200–10699** (reuse; diagnostic DEV).

## `2l_v1` DEV results (superseded framing; keep for audit trail)

**44** post-assembly states. Never-legal mass ≈ **100%** of total missing.

| Subfate (2l_v1 naming) | Share |
|---|---:|
| A3_TIER_ELIGIBLE_ZERO_RAW | **62.8%** |
| A1_NOT_IN_LOBBY_POOL (simplified) | 37.2% |
| A4_RAW_BUT_ZERO_LEGAL | **0%** |

A4=0 remains strong: do not touch buy legality/economy. The 63/37 split and the
biased “~48 expected vs 0 observed” claim required `2l_v2` before making
shop-generation canonical.

## Decision gate

If after `2l_v2` re-run **A3 still >50%** of never-legal mass under exact catalogue
membership → approve Phase 2L and open **Phase 2M as a shop/pool rules audit**
(copy counts, catalogue sync, shared-pool depletion, freeze/return, etc.) — not an
immediate generation rewrite. Known concrete mismatch already: sim `POOL_COPIES[6]=6`
vs current BG reference **7**.

## Commands

```bash
pytest tests/test_availability_decomposition.py
python -m ml.fidelity_phase_2l
```

## Frozen

No policy changes, no card effects, no BC/DAgger/PPO.
