# Simulator Fidelity Phase 2N — shop/pool fidelity interventions

Date: 2026-09-02 · Status: **`2n_v2` HOLD — active Tavern-pool filter added; remasure 11700–12199** ·
Artifacts: [`results/sim_fidelity_phase_2n/`](../results/sim_fidelity_phase_2n/)

## Verdict

**HOLD Simulator v1.x** — do **not** consume confirmation seeds **11500–11699**.

`2n_v1` cleared Phase 2M actionable lifecycle/copy mismatches, but acceptance was
**withdrawn**: `build_pool()` admitted the historical KB (≈1,306 BG minions) as
Bob's Tavern, producing false-positive catalogues (e.g. token **Foraging Bat**).
Average initial shared pool ≈8,763 copies implied ≥585 distinct shop minions in a
five-tribe lobby vs ≈261 current BG minions.

## Interventions

| Step | Change | Status |
|---|---|---|
| **2N-A** | HearthstoneJSON KB refresh; classify cores; remove T7 Polarizing Beatboxer | kept |
| **2N-B** | `PHASE_2N_DEATH_RETURN` + `PHASE_2N_FREEZE_TOPUP` | kept |
| **2N-C** | `POOL_COPIES[6] = 7` | kept |
| **2N-D** | Frozen `data/cards/active_tavern_pool.json`; `build_pool` ∩ active manifest | **new (`2n_v2`)** |

Also in `2n_v2`: freeze audit splits `kept_names` / `newly_dealt_names`; pool
conservation invariant `pool + live holdings + 3×golden == initialized`.

Acceptance now requires **recall and precision**:

```text
active-pool recall = 100%
active-pool precision = 100%
token / removed / generated-only / Duos-only / T7 in build_pool = 0
```

## Seeds

| Role | Range | Status |
|---|---|---|
| 2n_v1 combined measure | **11000–11499** | consumed (informed 2N-D; do not reuse) |
| Confirmation | **11500–11699** | **reserved — do not touch** |
| 2n_v2 remasure DEV | **11700–12199** | active |

## Commands

```bash
pytest tests/test_phase_2n.py tests/test_bg_env.py tests/test_shop_pool_audit.py
python -m ml.fidelity_phase_2n   # defaults to 11700–12199
```
