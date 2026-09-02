# Simulator Fidelity Phase 2M — shop/pool rules audit

Date: 2026-09-02 · Status: **`2m_v1` complete — multiple substantial mismatches** ·
Artifacts: [`results/sim_fidelity_phase_2m/`](../results/sim_fidelity_phase_2m/)

## Research question

> Is post-assembly scarcity caused by incorrect simulator catalogue/pool rules,
> incorrect live-pool accounting, or expected scarcity under a correctly
> implemented finite shared pool?

## Verdict

**Multiple substantial mismatches** — do **not** ship a bundled “better shops”
rewrite. Phase 2N should fix independently, ordered by mass impact:

1. **Catalogue/KB sync** (explains Phase 2L’s 37% A1 mass)
2. **Lifecycle accounting** (death return; freeze top-up)
3. **Copy counts** (T6: sim 6 vs ref 7)
4. Re-measure live calibration

**Live draw itself is not the smoking gun.** On the unconditioned post-assembly
cohort, observed zero-offer rate **0.745** vs live expected **0.799**, and
observed raw appearances **111** vs expected **≈89**. Scarcity among *missing*
cores is largely the buy/keep selection effect plus rare-card base rates in a
large shared pool — not a defective `_draw()`.

## Scope delivered

| Output | Result |
|---|---|
| Catalogue sync | **24.6%** of core slots `MISSING_FROM_KB` (56/228); 74.1% in exact catalogue |
| Pool contract | Documented (see `rule_mismatches.json`) |
| Lifecycle | Death = no return; freeze = no top-up; buy/sell/roll/triple as coded |
| Live probability | Pre-deal remaining + eligible total → exact without-replacement `P_zero` |
| Calibration | Unconditioned primary + A3-specific + missing-final (biased) sidecar |
| Rule comparison | **5** demonstrated mismatches (document only — no patches) |

## Frozen

- Policy: Phase 2J α=0.5, prior `9b31c93a…`
- Diagnostic DEV: **10200–10699** (reuse 2L)
- Reserved (not consumed): intervention **11000–11499**, confirm **11500–11699**
- No buy-legality/economy, card effects, BC/DAgger/PPO

## DEV results (10200–10699, 44 post-assembly states)

### Catalogue synchronization

| Status | Core slots | Share |
|---|---:|---:|
| IN_EXACT_CATALOGUE | 169 | 74.1% |
| **MISSING_FROM_KB** | **56** | **24.6%** |
| MISSING_OR_INVALID_TIER | 3 | 1.3% |

This directly attacks Phase 2L’s A1 mass (37.2% never-legal missing = exact-
catalogue absent, almost entirely missing KB/tier/stats).

### Demonstrated rule mismatches (no fixes in 2M)

| ID | Area |
|---|---|
| `pool_copies_tier_6` | sim **6** vs current BG **7** |
| `elimination_no_return_to_pool` | dead players’ cards stay out |
| `freeze_no_topup` | incomplete freeze not topped up |
| `shop_slots_vs_spell_era` | sim matches classic minion sizes; spell-era larger |
| `no_tier_7` | sim MAX_TIER=6 |

### Live calibration (primary = unconditioned)

Cohort: all exact-catalogue cores on post-assembly states with ≥1 tier-eligible
deal (**not** conditioned on missing-final).

| Metric | Value |
|---|---:|
| Card-windows | 385 |
| Expected zero-offer rate | **0.799** |
| Observed zero-offer rate | **0.745** |
| Sum expected raw | ≈89.3 |
| Sum observed raw | **111** |
| Expected windows ≥1 hit | ≈77 |
| Observed windows ≥1 hit | (see report) |

**A3 cohort** (definitionally zero-raw): observed zero ≡ 1.0 vs live expected
zero ≈ **0.809**. That gap is mostly selection into the zero-raw bucket, not
proof `_draw()` under-samples once you uncondition.

### Phase 2L continuity

```text
A3 exact-catalogue + tier-eligible zero raw: 62.8%
A1 not in exact catalogue:                   37.2%
A4 raw but never legal:                       0.0%
```

## Decision → Phase 2N

`multiple_substantial_mismatches`

Scoped interventions only — do not bundle into one generation patch. After
catalogue sync + lifecycle/copy fixes, re-run live calib on reserved
intervention seeds **11000–11499**.

## Commands

```bash
pytest tests/test_shop_pool_audit.py tests/test_bg_env.py
python -m ml.fidelity_phase_2m
```
