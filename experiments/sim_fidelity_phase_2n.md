# Simulator Fidelity Phase 2N — shop/pool fidelity interventions

Date: 2026-09-03 · Status: **`2n_v3` HOLD — mechanism lifts good; macro gates fail** ·
Artifacts: [`results/sim_fidelity_phase_2n/`](../results/sim_fidelity_phase_2n/)

## Verdict

**HOLD Simulator v1.x freeze/confirmation.**

- **2N-D catalogue/lifecycle work remains accepted** (precision/recall, conservation, draw calib).
- **`2n_v3` paired panel does not pass the full candidate gate.**

Do **not** merge for freeze, and do **not** consume **11500–11699**.

## What passed

| Gate | Result |
|---|---|
| Active-pool recall / precision | **1.0 / 1.0** |
| Conservation | **500/500** |
| Deal-level obs/exp | **0.971** |
| Persistent 2+ (BoardOpp − greedy) | **+90** (≥5) |
| Committed states Δ | **+75** (>0) |
| Fulfilled→played Δ | **+0.283** (≥0.25) |
| Coverage Δ | **+0.100** (≥0.003) |
| Board-sacrifice | mean **0.107**, p95 **0.375** (ok) |
| Phase 2J `mechanism_up` / `coverage_up` | **true** |

## What failed

| Gate | Result |
|---|---|
| Macro regression vs greedy | **FAIL** — turn-14 stats-ratio Δ = **−0.314** (max abs 0.25) |
| Macro fidelity vs Firestone envelope | **FAIL** — turn-10 stats ratio **0.544** vs center 1.033±0.057 |

Game length ≈15.0 (BoardOpp) / 15.9 (greedy). Nontermination 2.6%. Turn-12/14 absolute ratios are inside the Phase 2B max caps; the turn-10 undershoot is the clear envelope miss after the active-pool shrink (~8763 → ~1504 copies).

No thresholds were retuned (report-only).

## Interventions

| Step | Change | Status |
|---|---|---|
| 2N-A/B/C | KB refresh; death return + freeze top-up; T6=7 | kept |
| **2N-D** | Active Tavern-pool ∩ `build_pool` | **accepted** |
| **2n_v3** | Paired greedy vs BoardOpp α=0.5 + macro fidelity | **HOLD** |

## Seeds

| Role | Range | Status |
|---|---|---|
| 2n_v1 | 11000–11499 | consumed |
| Confirmation | **11500–11699** | **reserved** |
| 2n_v2/v3 DEV | **11700–12199** | consumed for validation |

## Commands

```bash
pytest tests/test_phase_2n.py
python -m ml.fidelity_phase_2n \
  --reuse-pool-audit-report results/sim_fidelity_phase_2n/phase_2n_report.json
```
