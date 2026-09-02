# Simulator Fidelity Phase 2M — shop/pool rules audit

Date: 2026-09-02 · Status: **`2m_v2` complete — multiple substantial mismatches** ·
Artifacts: [`results/sim_fidelity_phase_2m/`](../results/sim_fidelity_phase_2m/)

## Research question

> Is post-assembly scarcity caused by incorrect simulator catalogue/pool rules,
> incorrect live-pool accounting, or expected scarcity under a correctly
> implemented finite shared pool?

## Verdict

**`multiple_substantial_mismatches`** — scoped Phase 2N fixes, not a bundled
shop rewrite:

1. Catalogue/KB sync (classify missing cores vs current active pool first)
2. Lifecycle (death return; freeze top-up)
3. T6 copy count (6 → 7)
4. Re-measure deal-level live calib on reserved intervention seeds

**`_draw()` is not the primary defect.** After correcting the post-assembly
boundary and using deal-level calibration, observed hits are a **mild
undershoot** (~80% of exact pre-deal expectation; lobby CI for obs−exp just
excludes 0). That is not a substantial undershoot warranting a draw-path
rewrite before the known catalogue/lifecycle/copy fixes.

Contextual (not Phase 2N bugs): spell-era shop sizes, Tier 7.

## Methodology `2m_v2`

1. Post-assembly deals: **`turn > entry_turn`** (249 entry-turn deals excluded)
2. Primary: deal-level `ΣE(raw)` / `ΣP(hit)` vs observed + lobby bootstrap
3. Adaptive window `product(P_zero)` demoted to descriptive
4. Actionable vs contextual mismatch split

## DEV results (10200–10699, 44 states)

### Catalogue

| Status | Share |
|---|---:|
| IN_EXACT_CATALOGUE | 74.1% |
| MISSING_FROM_KB | **24.6%** |
| MISSING_OR_INVALID_TIER | 1.3% |

### Deal-level live calibration (primary)

| Metric | Value |
|---|---:|
| Deal×card observations | 10,925 |
| Σ expected raw | ≈74.9 |
| Σ observed raw | **60** |
| ratio obs/exp | **0.801** |
| Σ P(hit) | ≈74.7 |
| Σ observed hit deals | **60** |
| Lobby mean(obs−exp) | ≈−0.39 |
| Lobby CI95 | [≈−0.79, ≈−0.003] |

### Actionable mismatches

- `pool_copies_tier_6` (6 vs 7)
- `elimination_no_return_to_pool`
- `freeze_no_topup`

## Frozen

Phase 2J α=0.5. DEV 10200–10699. Reserved 11000–11499 / 11500–11699 unused.
No buy-legality/economy, card effects, BC/DAgger/PPO; no sim patches in 2M.

## Commands

```bash
pytest tests/test_shop_pool_audit.py tests/test_bg_env.py
python -m ml.fidelity_phase_2m
```
