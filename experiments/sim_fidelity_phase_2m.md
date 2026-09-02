# Simulator Fidelity Phase 2M — shop/pool rules audit

Date: 2026-09-02 · Status: **measurement-only (in progress)** ·
Artifacts: [`results/sim_fidelity_phase_2m/`](../results/sim_fidelity_phase_2m/)

## Research question

> Is post-assembly scarcity caused by incorrect simulator catalogue/pool rules,
> incorrect live-pool accounting, or expected scarcity under a correctly
> implemented finite shared pool?

## Scope (audit first — no implementation patches)

| Output | Content |
|---|---|
| Catalogue sync | Every archetype core → KB? tier? stats? exact `build_pool`? why missing? |
| Pool contract | Copy counts, init, dual/All tribe, banned-tribe filter, dedup |
| Lifecycle | Deal/roll/freeze/buy/sell/triple/elimination return behavior |
| Shop generation | Slot counts, eligible-tier weighting, without-replacement within deal |
| Live probability | Pre-deal remaining copies + eligible total → exact `P_zero` / E[raw] |
| Calibration | Observed vs expected zeros/hits; by tier / archetype / entry turn |
| Rule comparison | Documented mismatches vs current BG — **no fixes in this phase** |

## Frozen

- Policy: Phase 2J α=0.5, prior `9b31c93a…`
- Diagnostic DEV: **10200–10699** (reuse 2L; continuity)
- Reserved (not consumed): intervention **11000–11499**, confirm **11500–11699**
- Forbidden: 8000–8199, 9000–9999, 10000–10199, plus reserved ranges above
- No buy-legality/economy changes, no card effects, no BC/DAgger/PPO

## Decision tree → Phase 2N

```text
catalogue/KB mismatch explains most gap
→ Phase 2N: synchronize card/core data

live-pool accounting bug
→ Phase 2N: fix shared-pool lifecycle

shop draw probabilities/rules mismatch
→ Phase 2N: fix generation model

implementation matches rules and observed scarcity
is statistically consistent with exact live-pool expectation
→ shop generation is not actually broken;
  investigate roll/opportunity horizon / core-set assumptions

multiple substantial mismatches
→ fix independently in scoped interventions,
  don't bundle them into one "better shops" patch
```

## Already-known mismatch (document only)

```text
Simulator T6 copies: 6
Current BG reference: 7
```

Plus: elimination does not return copies; freeze does not top up incomplete shops.

## Commands

```bash
pytest tests/test_shop_pool_audit.py tests/test_bg_env.py
python -m ml.fidelity_phase_2m
```
