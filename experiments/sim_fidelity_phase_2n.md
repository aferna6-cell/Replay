# Simulator Fidelity Phase 2N — shop/pool fidelity interventions

Date: 2026-09-02 · Status: **interventions applied; measurement pending/in PR** ·
Artifacts: [`results/sim_fidelity_phase_2n/`](../results/sim_fidelity_phase_2n/)

## Scope

Sequential, independently attributable fixes from Phase 2M actionable list.
**No** `_draw()` rewrite, buy/economy changes, card effects, or BC/DAgger/PPO.

| Step | Change | Commit theme |
|---|---|---|
| **2N-A** | Refresh `bg_cards.json`; classify 59 problematic cores; remove T7 Polarizing Beatboxer from mech cores | catalogue sync |
| **2N-B** | `PHASE_2N_DEATH_RETURN` + `PHASE_2N_FREEZE_TOPUP` | lifecycle |
| **2N-C** | `POOL_COPIES[6] = 7` | copy counts |
| **Measure** | Deal-level live calib on **11000–11499** (once, combined) | measurement |
| **Confirm** | **11500–11699** reserved after freeze | not consumed yet |

## 2N-A classification (do not invent cards)

| Class | Count | Action taken |
|---|---:|---|
| ACTIVE_MISSING_FROM_KB_FIXED_BY_REFRESH | 38 unique | KB refresh from HearthstoneJSON |
| TIER_REFRESH_FIXED | 1 (Sanguine Champion) | same refresh (7→6) |
| TIER_OUT_OF_SIM_SCOPE | 2 slots (Polarizing Beatboxer T7) | removed from archetype cores |

Post-2N-A catalogue audit: **226/226** core slots `IN_EXACT_CATALOGUE`.

## Seeds

| Role | Range | Status |
|---|---|---|
| Intervention measure | **11000–11499** | consume once after A/B/C |
| Confirmation | **11500–11699** | reserved |
| Prior DEV | 10200–10699 | do not reuse for 2N measure |

## Commands

```bash
pytest tests/test_phase_2n.py tests/test_bg_env.py tests/test_shop_pool_audit.py
python -m ml.fidelity_phase_2n
```
