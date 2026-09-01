# Simulator Fidelity Phase 2C — composition assembly diagnostic

Date: 2026-09-01 · Status: **measurement only (draft PR #18)** ·
Artifacts: [`results/sim_fidelity_phase_2c/`](../results/sim_fidelity_phase_2c/)

## Question

> Why do Simulator v1.1 winners achieve ~0.008 final composition coverage when
> real Firestone winners achieve ~0.766?

No sim mechanics, policies, card effects, shop, scaling, or combat changes.

## Methodology v2 (`2c_v2`)

The first Phase 2C run (**invalidated**) counted opponent shop offers against
the winner and reused a stale turn-start shop snapshot. v2 fixes:

1. **Live pre-action shop** — each event records `pre_shop` and `legal_buy_slots`
   from the pre-action legal mask (not turn-start snapshot).
2. **Shop generation deduplication** — exposure keyed by
   `(lobby, seat, turn, shop_generation)`; rolls increment generation.
3. **Separate scopes**
   - `global_availability` — all eight seats (shop/pool starvation diagnosis)
   - `winner_decision_funnel` — winner seat only
4. **Legally buyable** — core counts only when its exact buy slot is legal
   (gold **and** hand room), not merely `gold ≥ 3`.
5. **Rejection at shop exit** — a core is “rejected” only if it was legally
   buyable in a shop generation and never purchased before roll/end.
6. **Archetype relevance**
   - **current-target view** — cores for `infer_target(board_before)` at offer time
   - **final-target hindsight view** — cores for winner’s eventual archetype (labeled hindsight)
7. **Tribe eligibility** — ineligible archetypes excluded from decision denominators.
8. **Trace equivalence** — traced rollouts match ordinary `play_scripted` for same seeds
   (placements + final-board fingerprints).

### Invalidated prior headline

| Class | Prior count | Status |
|---|---|---|
| B_AVAILABLE_NOT_BOUGHT | 2,516 | **Invalidated** (seat/shop scope bugs) |
| A_IMPOSSIBLE | 1,185 | **Invalidated** |
| C_BOUGHT_NOT_RETAINED | 99 | **Invalidated** |

Do **not** merge PR #18 or start Phase 2D until reviewing corrected v2 results.

## Corrected funnel (winner, relevant archetype)

```text
relevant core in winner's shop (deduped by shop_generation)
        ↓
legally buyable (exact A_BUY slot legal)
        ↓
purchased before shop refresh/end
        ↓
played
        ↓
retained ≥ 2 turns
        ↓
2 distinct core pieces legally available across game
        ↓
4 distinct core pieces legally available across game
        ↓
final composition coverage
```

Separately: `global_availability` tracks all-seat shop exposure for pool diagnosis.

## Commands

```bash
pytest tests/test_composition_diagnostic.py
python -m ml.fidelity_phase_2c --lobbies 200 --seed 0
```

See [`results/sim_fidelity_phase_2c/phase_2c_report.json`](../results/sim_fidelity_phase_2c/phase_2c_report.json)
for v2 per-archetype funnels, rejection patterns, and both relevance views.

## Frozen for Phase 2C

Residual scaling (v1.1), combat, greedy policy, card effects, shop distribution,
triples/discovers, heroes, trinkets, anomalies.
