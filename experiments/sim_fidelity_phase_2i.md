# Simulator Fidelity Phase 2I — seeded opportunity decision-margin diagnostic

Date: 2026-09-02 · Status: **measurement-only (2i_v2)** ·
Artifacts: [`results/sim_fidelity_phase_2i/`](../results/sim_fidelity_phase_2i/)

## Research question

> Why does the frozen Phase 2H λ=12 policy reject essentially every legally-buyable
> seeded core opportunity despite the Phase 2E/2G oracle proving those opportunities
> can produce assembly?

## Methodology (2i_v2)

- **Exposure unit:** 2c_v3 — core name × shop generation × seeded current target
- **Policy:** Phase 2H v3 `TempoBoardGreedyPolicy`, λ_build = 12 (frozen)
- **Seeds:** DEV **3000–3499** (500 lobbies) — not held-out 6000–6199
- **Simulator:** v1.1 residual scaling
- **No policy behavior changes** — observational audit hook only

### 2i_v2 corrections (vs 2i_v1)

1. **Compound chosen-transition attribution** — decode initial compound sell from
   `policy.pending` (candidate, replacement slot, net/build), not first matching
   sell `action_id`.
2. **Directional break-even λ buckets** — distinguish higher-λ vs lower-λ help;
   do not treat λ≤12 ties as “needs λ≤12” when lowering λ is required.
3. **Decisive rejection** — close exposure on **first loss of buyability** within
   a shop generation, not at generation change / roll / end.
4. **Reporting** — rank with/without build, core-frequency quartiles, renamed raw-gap
   metrics (`mean_chosen_minus_core_raw_gap`, `mean_core_raw_advantage`).

## Instrumentation

At each decision while a seeded exposure is live, record:

- Core transition score decomposition (raw, build gain, replacement cost, net)
- Chosen action score decomposition
- Decision margin = core_net − chosen_net
- Counterfactuals: `core_free_slot_value`, `core_actual_replacement_value`
- Break-even λ (diagnostic only; not run)

## Failure taxonomy

| Code | Meaning |
|---|---|
| `A_REPLACEMENT_COST_DOMINATES` | Attractive with free slot; board replacement kills it |
| `B_RAW_STAT_COMPETITOR_DOMINATES` | Core viable but outscored on immediate stats |
| `C_BUILD_SIGNAL_TOO_SMALL` | Build bonus too small to close margin |
| `D_BUILD_SIGNAL_NONDISCRIMINATIVE` | Chosen alternative has similar/greater build value |
| `E_ALTERNATE_CORE_SELECTED` | Different target-core transition selected (not a composition failure) |
| `F_ECONOMY_LEGALITY_LOSS` | Gold/hand/roll/end before buy |
| `G_TARGET_CHANGED` | Target shifted before resolution |
| `H_OTHER` | Residual |

## Commands

```bash
pytest tests/test_seeded_margin_diagnostic.py
python -m ml.fidelity_phase_2i
```

## Decision tree (Phase 2J routing)

Dominant cause (>50% of composition-progress failures) routes to Phase 2J branch.
Mixed/sample-insufficient → expand diagnostic before implementing fixes.

## DEV results (seeds 3000–3499, λ=12, clean tree, 2i_v2)

```text
N seeded legally-buyable exposures: 159
├─ fulfilled: 2
└─ rejected: 157
   ├─ A replacement cost dominates: 137 (87%)
   └─ F economy/legality loss: 20 (13%)
```

2c_v3 reconciliation: **pass** (159 = 159 exposures).

Close reasons (rejected): roll 134, end 19, first_loss_of_buyability 4.

Headline metrics:

- **87%** board-full-only replacement-cost losses (`pct_exposures_lost_board_full_only`)
- **0%** core full-board transition net > 0 at λ=12
- Mean core raw advantage: **+8.6** (core not weak on raw stats)
- Mean replacement raw-stat cost: **296** vs mean λ×build bonus: **5.1**
- Rank with λ=12: 38% first, 22% second; without build: 8% first, 66% third+
- Directional break-even: **68%** need higher λ>24; **18%** lower-λ-only; **1%** higher λ≤24

**Recommended Phase 2J branch:** multi-turn / board-slot opportunity-cost model
(replacement cost dominates >50% of composition-progress failures).

## Frozen

No card effects, BC, PPO, DAgger, or λ sweeps in this phase.
