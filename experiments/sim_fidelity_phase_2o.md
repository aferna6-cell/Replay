# Simulator Fidelity Phase 2O — midgame scaling-budget diagnostic

Date: 2026-09-03 · Status: **`2o_v1` measurement running / pending results** ·
Artifacts: [`results/sim_fidelity_phase_2o/`](../results/sim_fidelity_phase_2o/)

## Verdict (predeclared)

Measurement-only. **Do not merge #29 for freeze. Do not touch Phase 2J α=0.5.
Do not consume 11500–11699.** Keep 2N-D catalogue / death-return / freeze-topup / T6=7.

`2n_v3` remains **FAIL as run** (no retroactive threshold rewrite). The T14
treatment−greedy Δ=−0.314 is interpreted as a **semantic failure of a symmetric
control-difference guard** after the pool distribution changed: treatment moved
**toward** Firestone (1.205× vs greedy 1.520×). Prospective directional
policy-harm metric is recorded for the *next* fresh evaluation only.

## Hypothesis

After 2N-D shrank the Tavern pool (~8763 → ~1504 copies), both greedy and Phase
2J show the same midgame shape: badly under-strength T9–T12, then rapid catch-up
T13–T14, despite Tavern tier on/ahead of pace. Residual scaling’s budget is
`ratio_add ∝ current`, so when `current ≪ Firestone` it does **not** fill the
target gap (`over=0` → `residual_add = ratio_add`).

## Protocol

```bash
pytest tests/test_phase_2o.py tests/test_scaling_residual.py
python -m ml.fidelity_phase_2o   # seeds 12200–12699, 500 lobbies × 2 arms
```

Arms: raw greedy · frozen BoardOpp α=0.5 + frozen prior.
Instrument turns **7–14** for every seat: start-recruit → end-recruit →
scaling intermediates → post-scale → remaining Firestone gap.
Also report **symmetric absolute fidelity** for turns 8–14 (undershoot +
overshoot). Historical Phase 2B upper bounds are preserved, not retuned.

## Routing (predeclared)

| Finding | Next step |
|---|---|
| Pre-scale near Firestone, scaling wrong | scaling formula defect |
| Pre-scale far below, post still far below | **target-gap bridge defect** |
| Recruit contribution collapsed | recruit/effect-value fidelity |
| Just-leveled 0.6× explains deficit | leveling-growth penalty |
| Greedy healthy, Phase 2J low | policy issue |
| Both arms low | simulator-level issue |
| T13–14 suddenly overcompensates | growth timing/budget redistribution |

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2n_v2/v3 DEV | 11700–12199 | consumed (HOLD) |
| **2O DEV** | **12200–12699** | this phase |

## Results

_Pending full DEV run._
