# Simulator Fidelity Phase 2J — board-relative opportunity-cost policy

Date: 2026-09-02 · Status: **scientifically ACCEPT (2j_v1, α=0.5); PR cleanup** ·
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
tertile, target-core vs non-core. Identity matching uses `(name, golden)`.

**Canonical fingerprint:** `prior_hash_sha256` over the full cell table (not just
fit seeds / globals). Clean-tree refit reproduced the identical hash
`9b31c93a…`; confirmation `8000–8199` remains valid.

Note: global P(survive 1/2) ≈ 0.989 / 0.976 — the prior is nearly flat. Do **not**
claim the persistence prior alone solved assembly. The causal story is
**board-relative normalization** of replacement cost (mean relative tempo loss
≈ 5.6% of board strength vs hundreds of absolute stats). Persistence discounting
was included but not independently ablated.

### Frozen α

`α = 0.5` after screen `{0.5, 1.0, 2.0}` and replication of top-two.

## Experimental ranges

| Stage | Seeds | Result |
|---|---|---|
| Fit prior | 7000–7299 | 18 cells; hash `9b31c93a…` |
| Screen | 7300–7399 | All α macro-ok |
| Replication | 7400–7799 | Freeze **α=0.5** |
| Confirm | 8000–8199 | **ACCEPT** (preserved after identical prior refit) |

## Confirmation (seeds 8000–8199, clean tree)

| Arm | Persistent 2+ | Committed | Fulfilled→played | Coverage mean |
|---|---:|---:|---:|---:|
| Raw greedy | 0 | 0 | 0 | 0.0057 |
| Phase 2J α=0.5 | **20** | **14** | **34/34** | **0.0199** |
| Phase 2E+2G stress reference | 10 | 11 | 33/33 | 0.0103 |

The 2E+2G policy is a **causal stress-test reference**, not an upper bound
(2J beats it on persistent 2+, committed, and coverage).

Deltas vs greedy (all gates pass):

- persistent 2+ core Δ = **+20** (≥5)
- committed states Δ = **+14** (>0)
- seeded fulfillment Δ = **+34**
- played rate Δ = **+1.0** (≥0.25)
- coverage Δ = **+0.014** (≥0.003)
- macro_regression_ok = true
- board_sacrifice_ok = true (mean rel loss **0.056**, p95 **0.104**)

**Decision:** `accept_board_management_policy`

Tier-band breakdown (confirm treatment, reconciles to 41/34/20):

| Tier band | Seeded exposures | Fulfilled | Persistent 2+ | Committed |
| --------- | ---------------: | --------: | ------------: | --------: |
| ≤4        |                0 |         0 |             0 |         0 |
| 5         |                9 |         7 |             5 |         3 |
| 6         |               32 |        27 |            15 |        11 |

Most assembly still occurs at tier 6, but tier 5 is no longer empty — the
policy is not exclusively a late-game patch.

## Commands

```bash
pytest tests/test_board_opportunity_policy.py
python -m ml.fidelity_phase_2j fit-prior
python -m ml.fidelity_phase_2j calibrate
python -m ml.fidelity_phase_2j confirm --alpha 0.5
```

## Next

Phase 2K — post-assembly residual composition-gap diagnostic (measurement-only).
Coverage remains ~0.02 vs real ~0.77; ask why coherent 2+ boards still fail to
resemble real winners before touching card effects.

## Frozen

No card effects, BC, DAgger, PPO, or λ sweeps in this phase.
