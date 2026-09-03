# Simulator Fidelity Phase 2N — shop/pool fidelity interventions

Date: 2026-09-03 · Status: **`2n_v3` validation panel — candidate acceptance HOLD until panel passes** ·
Artifacts: [`results/sim_fidelity_phase_2n/`](../results/sim_fidelity_phase_2n/)

## Verdict

**2N-D catalogue/lifecycle work is accepted.** Candidate freeze/confirmation remains
**HOLD** until the `2n_v3` paired behavioral + macro regression panel passes.

`2n_v2` cleared active-pool precision/recall, conservation, and deal-level
calibration on **11700–12199**. The active-pool change is material
(~8763 → ~1504 initial copies), so acceptance also requires the previously
requested Phase 2J mechanism regression + macro fidelity panel.

## Interventions

| Step | Change | Status |
|---|---|---|
| **2N-A/B/C** | KB refresh; death return + freeze top-up; T6=7 | kept |
| **2N-D** | Active Tavern-pool manifest ∩ `build_pool` | **accepted** |
| **2n_v3** | Validation-only paired greedy vs BoardOpp α=0.5 + macro fidelity | **required** |

No simulator changes in `2n_v3`.

## Candidate gate (`2n_v3`)

```text
active-pool recall/precision = 100%
conservation balanced
0.70 ≤ deal-level obs/exp ≤ 1.30
Phase 2J mechanism regression pass
  (macro_regression_ok, persistent 2+ Δ≥5, committed Δ>0,
   fulfilled→played Δ≥0.25, coverage Δ≥0.003, board-sacrifice ok)
macro fidelity vs Firestone reference envelope pass
        ↓
accept_simulator_v1_x_candidate
→ explicit freeze fingerprints
→ confirm once on 11500–11699
```

## Seeds

| Role | Range | Status |
|---|---|---|
| 2n_v1 measure | 11000–11499 | consumed |
| Confirmation | **11500–11699** | **reserved** |
| 2n_v2/2n_v3 DEV | **11700–12199** | reused for validation panel |

## Commands

```bash
pytest tests/test_phase_2n.py tests/test_bg_env.py tests/test_shop_pool_audit.py
python -m ml.fidelity_phase_2n \
  --reuse-pool-audit-report results/sim_fidelity_phase_2n/phase_2n_report.json
```
