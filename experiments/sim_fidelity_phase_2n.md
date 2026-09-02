# Simulator Fidelity Phase 2N — shop/pool fidelity interventions

Date: 2026-09-02 · Status: **`2n_v1` measure complete — accept Simulator v1.x candidate** ·
Artifacts: [`results/sim_fidelity_phase_2n/`](../results/sim_fidelity_phase_2n/)

## Verdict

**`accept_simulator_v1_x_candidate`**

All three Phase 2M actionable mismatches are cleared. Deal-level live calib on
intervention seeds **11000–11499** is within the acceptance band (obs/exp ≈
**0.759**). Mild undershoot remains (lobby CI excludes 0) but is not a
substantial `_draw()` defect. Next: freeze Simulator v1.x candidate and confirm
on **11500–11699**.

## Interventions (independent commits / toggles)

| Step | Change | Result |
|---|---|---|
| **2N-A** | HearthstoneJSON KB refresh; classify cores; remove T7 Polarizing Beatboxer | **226/226** cores in exact catalogue |
| **2N-B** | `PHASE_2N_DEATH_RETURN` + `PHASE_2N_FREEZE_TOPUP` | actionable lifecycle mismatches cleared |
| **2N-C** | `POOL_COPIES[6] = 7` | matches current BG |

No `_draw()` rewrite, buy/economy, card effects, or BC/DAgger/PPO.

## 2N-A classification

| Class | Count | Action |
|---|---:|---|
| ACTIVE_MISSING_FROM_KB_FIXED_BY_REFRESH | 38 | KB refresh only |
| TIER_REFRESH_FIXED | 1 (Sanguine Champion) | KB refresh |
| TIER_OUT_OF_SIM_SCOPE | 2 (Polarizing Beatboxer) | removed from cores |

## Measurement (11000–11499, 500 lobbies)

| Metric | 2M DEV (10200) | 2N intervention |
|---|---:|---:|
| Deal×card obs | 10,925 | 18,373 |
| Σ expected raw | ≈74.9 | ≈122.5 |
| Σ observed raw | 60 | **93** |
| obs/exp ratio | 0.801 | **0.759** |
| Actionable mismatches | 3 | **0** |
| A1 never-legal share | 37.2% | **0.6%** |
| A3 zero-raw share | 62.8% | **99.2%** of remaining never-legal |

Catalogue sync removed nearly all A1 mass; remaining scarcity is still mostly
tier-eligible zero-raw under the finite shared pool — consistent with 2M’s
conclusion that `_draw()` is not catastrophically wrong.

## Seeds

| Role | Range | Status |
|---|---|---|
| Intervention measure | **11000–11499** | **consumed** |
| Confirmation | **11500–11699** | reserved |

## Commands

```bash
pytest tests/test_phase_2n.py tests/test_bg_env.py tests/test_shop_pool_audit.py
python -m ml.fidelity_phase_2n
```
