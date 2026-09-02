# Simulator Fidelity Phase 2L — post-assembly availability decomposition

Date: 2026-09-02 · Status: **`2l_v2` complete — approve merge (A3 still >50%)** ·
Artifacts: [`results/sim_fidelity_phase_2l/`](../results/sim_fidelity_phase_2l/)

## Research question

> Of the missing weighted core mass that is never *legally* buyable after first-2,
> how much is absent from the **exact** simulator catalogue, tier-locked, never
> raw-offered while catalogue+tier-eligible, or raw-offered-but-illegal?

## Methodology `2l_v2`

1. **A1** = not in exact `BGEnv.build_pool(lobby_tribes=…)` (not simplified tribe∩KB).
2. A1 subtypes: `MISSING_KB_OR_TIER_OR_STATS` / `TRIBE_EXCLUDED` / `BUILD_POOL_EXCLUDED`.
3. Sampler calibration unconditioned on never-legal/A3: missing-final ∩ exact
   catalogue ∩ tier-eligible — expected vs observed raw; cards ≥1 appearance;
   observed vs expected zero-offer rate.
4. Do not treat A3-conditioned “expected N vs 0” as confirmatory sampler failure.

## Frozen policy

Phase 2J α=0.5, prior `9b31c93a…`. Seeds **10200–10699** (44 states; no expand).
Commit `f20a8be` (code) + artifact commit; clean tree for DEV run.

## DEV results (`2l_v2`)

**44** post-assembly states. Never-legal mass ≈ **100%** of total missing.

### Headlines (never-legal missing mass)

```text
X% not in exact simulator catalogue:              37.2%
Y% exact-catalogue + tier eligible but zero raw:  62.8%
Z% raw but never legal:                            0.0%
```

| Subfate | Share |
|---|---:|
| **A3_TIER_ELIGIBLE_ZERO_RAW** | **62.8%** |
| A1_NOT_IN_EXACT_CATALOGUE | 37.2% |
| A2 / A4 / A5 | 0% |

### A1 exclusion breakdown

| Reason | Share of A1 |
|---|---:|
| **MISSING_KB_OR_TIER_OR_STATS** | **100%** |
| TRIBE_EXCLUDED | 0% |
| BUILD_POOL_EXCLUDED | 0% |

`kb_says_tribe_ok_but_build_pool_excludes` mass = **0**. The entire A1 bucket is
null/missing KB tier or stats (catalogue sync / core-reference), not tribe filter
or skin/dedup exclusion. Exact `build_pool` membership did **not** shrink A3 vs
`2l_v1` for this cohort (same 63/37 split).

### Unconditioned sampler calibration

Cohort: 256 card-windows (missing-final ∩ exact catalogue ∩ ≥1 tier-eligible obs).

| Metric | Value |
|---|---:|
| Sum expected raw appearances | ≈48.1 |
| Sum observed raw appearances | **0** |
| Expected cards with ≥1 appearance | ≈42.3 |
| Observed cards with ≥1 appearance | **0** |
| **Expected zero-offer rate** | **≈0.835** |
| **Observed zero-offer rate** | **1.000** |

Observed zeros exceed the static model, but expected zero is already high in short
windows. This is legitimate Phase 2M input — not the selection-biased A3-only
comparison from `2l_v1`.

## Decision

`a3_tier_eligible_zero_raw` (62.8% > 50%) → **approve Phase 2L merge**.

**Phase 2M** = full **shop/pool rules audit** before any generation rewrite:

- per-tier copy counts (sim T6=`6` vs current BG ref **7**)
- current minion catalogue / removed / missing cards
- tribe / All / dual-tribe / lobby filtering
- shared-pool depletion, shop slots, freeze/return, triples, death return
- card/KB sync (`card_tier = null` cores — primary A1 mass)

Question for 2M:

> Is post-assembly scarcity caused by incorrect simulator catalogue/pool
> parameters, or the expected result of the current finite shared-pool model?

**A4=0:** do **not** touch buy legality/economy.

## Commands

```bash
pytest tests/test_availability_decomposition.py
python -m ml.fidelity_phase_2l
```

## Frozen

No policy changes, no card effects, no BC/DAgger/PPO.
