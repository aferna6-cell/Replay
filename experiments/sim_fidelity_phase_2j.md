# Simulator Fidelity Phase 2J — board-relative opportunity-cost policy

Date: 2026-09-02 · Status: **ACCEPT (2j_v1, α=0.5)** ·
Artifacts: [`results/sim_fidelity_phase_2j/`](../results/sim_fidelity_phase_2j/)

## Research question

> Can a scale-normalized, forward-looking board-slot cost recover meaningful
> acquisition and assembly without the hard Phase 2E/2G oracle?

## Motivation (from Phase 2I)

Phase 2H compared ~10–30 candidate raw stats + ~5 build bonus against the
incumbent's **entire accumulated raw-stat stock** (~296 mean). Phase 2I diagnosed
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

### Persistence prior

Fitted from raw greedy DEV `7000–7299`. Features: tier band, board raw-stat
tertile, target-core vs non-core. Identity matching uses `(name, golden)` (not
scaled ATK/HP). Horizon weights 50/50 over 1- and 2-turn survival.

### Frozen α

`α = 0.5` after screen `{0.5, 1.0, 2.0}` and replication of top-two.

## Experimental ranges

| Stage | Seeds | Result |
|---|---|---|
| Fit prior | 7000–7299 | 18 cells; weak slots cheaper |
| Screen | 7300–7399 | All α macro-ok; 13× 2-core, 23 fulfilled vs greedy 0 |
| Replication | 7400–7799 | Freeze **α=0.5** |
| Confirm | 8000–8199 | **ACCEPT** |

## Confirmation (seeds 8000–8199, clean tree)

| Arm | Persistent 2+ | Committed | Fulfilled→played | Coverage mean |
|---|---:|---:|---:|---:|
| Raw greedy | 0 | 0 | 0 | 0.0057 |
| Phase 2J α=0.5 | **20** | **14** | **34/34** | **0.0199** |
| Phase 2E+2G oracle | 10 | 11 | 33/33 | 0.0103 |

Deltas vs greedy (all gates pass):

- persistent 2+ core Δ = **+20** (≥5)
- committed states Δ = **+14** (>0)
- seeded fulfillment Δ = **+34**
- played rate Δ = **+1.0** (≥0.25)
- coverage Δ = **+0.014** (≥0.003)
- macro_regression_ok = true
- board_sacrifice_ok = true (mean rel loss **0.056**, p95 **0.104**)

**Decision:** `accept_board_management_policy`

## Commands

```bash
pytest tests/test_board_opportunity_policy.py
python -m ml.fidelity_phase_2j fit-prior
python -m ml.fidelity_phase_2j calibrate
python -m ml.fidelity_phase_2j confirm --alpha 0.5
```

## Frozen

No card effects, BC, DAgger, PPO, or λ sweeps in this phase.
