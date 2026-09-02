# Simulator Fidelity Phase 2N — shop/pool fidelity interventions

Date: 2026-09-02 · Status: **`2n_v2` — accept Simulator v1.x candidate; confirmation reserved** ·
Artifacts: [`results/sim_fidelity_phase_2n/`](../results/sim_fidelity_phase_2n/)

## Verdict

**`accept_simulator_v1_x_candidate`** after **2N-D** (active Tavern-pool filter).

Do **not** consume confirmation seeds **11500–11699** until an explicit freeze
step. `2n_v1` acceptance was **withdrawn**: the refreshed KB was still not Bob’s
current Tavern (historical/token pollution; ~8763 initial pool copies).

## Interventions

| Step | Change | Result |
|---|---|---|
| **2N-A** | HSJSON KB refresh; classify cores; remove T7 Polarizing Beatboxer | kept |
| **2N-B** | Death return + freeze top-up | kept |
| **2N-C** | `POOL_COPIES[6]=7` | kept |
| **2N-D** | Frozen `active_tavern_pool.json`; `build_pool` ∩ active; Fish of N'Zoth core hygiene | **required for accept** |

Also: freeze audit `kept_names` / `newly_dealt_names`; finalize returns survivor
holdings; conservation invariant `pool + holdings + 3×golden == initialized`.

## Gates (`2n_v2`)

```text
active-pool recall = 100%
active-pool precision = 100%
token / removed / generated-only / Duos-only / T7 in build_pool = 0
conservation balanced on all lobbies
0.70 ≤ deal-level obs/exp ≤ 1.30
```

## Measurement (11700–12199, 500 lobbies)

| Metric | 2n_v1 (11000) | 2n_v2 (11700) |
|---|---:|---:|
| Solo active catalogue | ~1235 (KB) | **235** |
| Mean initial pool copies | ≈8763 | **≈1504** |
| Active-pool recall/precision | n/a (recall-only) | **1.0 / 1.0** |
| Foraging Bat in build_pool | yes | **no** |
| Conservation OK lobbies | n/a | **500/500** |
| Deal×card obs | 18,373 | 48,620 |
| Σ expected raw | ≈122.5 | ≈1834.3 |
| Σ observed raw | 93 | **1781** |
| obs/exp ratio | 0.759 | **0.971** |
| A1 never-legal share | 0.6% | **0.5%** |
| A4 raw-never-legal share | ~0% | **0.3%** |

Reinterpretation: `2n_v1`’s 0.759 showed `_draw()` matched an **oversized** sim
pool. Under the active Tavern manifest, deal-level calib is near unity.

## Seeds

| Role | Range | Status |
|---|---|---|
| 2n_v1 measure | 11000–11499 | consumed (informed 2N-D) |
| Confirmation | **11500–11699** | **reserved — next after explicit freeze** |
| 2n_v2 remasure | **11700–12199** | **consumed** |

## Commands

```bash
pytest tests/test_phase_2n.py tests/test_bg_env.py tests/test_shop_pool_audit.py
python -m ml.fidelity_phase_2n   # defaults to 11700–12199
```
