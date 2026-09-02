# Simulator Fidelity Phase 2J — board-relative opportunity-cost policy

Date: 2026-09-02 · Status: **in progress (2j_v1)** ·
Artifacts: [`results/sim_fidelity_phase_2j/`](../results/sim_fidelity_phase_2j/)

## Research question

> Can a scale-normalized, forward-looking board-slot cost recover meaningful
> acquisition and assembly without the hard Phase 2E/2G oracle?

## Motivation (from Phase 2I)

Phase 2H compared ~10–30 candidate raw stats + ~5 build bonus against the
incumbent's **entire accumulated raw-stat stock** (~296 mean). That made late
full-board transitions structurally impossible. Phase 2I diagnosed
`A_REPLACEMENT_COST_DOMINATES` at 87%.

## Design (one causal dimension)

Only the transition-cost formulation changes. Build signal, compound mechanics,
shop/pool/scaling/combat unchanged. **No λ.**

```text
raw_loss = max(0, replacement_raw - candidate_raw)
relative_tempo_loss = raw_loss / max(board_total_raw, 1)
opportunity_cost = relative_tempo_loss * persistence_weight
build_delta = candidate_build_gain - replacement_build_value
transition_score = build_delta - α * opportunity_cost
```

Commit iff `transition_score > 0`. Free-slot `opportunity_cost = 0`.

### Persistence prior (frozen before α sweep)

Fitted from **raw greedy** DEV `7000–7299` only.

Features at decision time: tier band (`le4`/`5`/`6plus`), board raw-stat tertile,
target-core vs non-core. No card-name memorization.

```text
persistence_weight = 0.5 * P(survive 1 recruit turn)
                   + 0.5 * P(survive 2 recruit turns)
```

### One tuning parameter

```text
α ∈ {0.5, 1.0, 2.0}
```

Horizon weights and features frozen before experiments.

## Experimental ranges

| Stage | Seeds | Purpose |
|---|---|---|
| Fit prior | 7000–7299 | Greedy-only persistence table |
| Screen α | 7300–7399 | Rank α=0.5/1/2 vs greedy |
| Replication | 7400–7799 | Top-two α; freeze one |
| Confirm | 8000–8199 | Once: greedy / 2J / oracle |

## Commands

```bash
pytest tests/test_board_opportunity_policy.py
python -m ml.fidelity_phase_2j fit-prior
python -m ml.fidelity_phase_2j calibrate
# commit artifacts, clean tree
python -m ml.fidelity_phase_2j confirm
```

## Confirmation gates

Same core logic as Phase 2H, plus seeded fulfillment > greedy and board-sacrifice
safety (`mean_relative_tempo_loss`, `p95_relative_tempo_loss`).

Report outcomes separately for tier ≤4 / 5 / 6.

## Decision tree

```text
mechanism ↑ + coverage ↑ + macro clean + sacrifice ok
    → ACCEPT candidate board-management policy

mechanism ↑ strongly + coverage flat
    → card-effect fidelity becomes leading bottleneck

fulfillment/deploy ↑ but persistence low
    → retention still insufficient

policy barely changes core conversion
    → replacement cost wasn't enough; revisit build-value

macro regression OR large board-strength sacrifice
    → reject
```

## Frozen

No card effects, BC, DAgger, PPO, or λ sweeps in this phase.
