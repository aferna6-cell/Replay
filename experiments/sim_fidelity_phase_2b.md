# Simulator Fidelity Phase 2B — residual scaling correction

Date: 2026-09-01 · Status: **Simulator v1.1 candidate accepted** ·
Artifacts: [`results/sim_fidelity_v1_1/`](../results/sim_fidelity_v1_1/)

## Intervention (one variable only)

Changed only `_end_of_turn_scaling()` in `hsbg_coach/bg_env.py`:

| Mode | When | Behavior |
|---|---|---|
| `ratio` | Simulator v1 (frozen) | Multiply entire board by Firestone turn-to-turn ratio |
| `residual` | Simulator v1.1 (default) | Turns 1–9: same as ratio. Turn 10+: apply `ratio_add - over` where `over = max(0, current − pace_target)` |

This implements the Phase 2B hypothesis: once recruit has already pushed a board
past the Firestone pace target, stop applying the full abstract growth multiplier
and supply only the residual budget below that overage.

No combat, composition, hero, trinket, or agent changes.

## Experiment protocol

```bash
python -m ml.fidelity_phase_2b --lobbies 200 --seed 0
```

1. Re-run Simulator v1 (`scaling_mode=ratio`) on seeds `0…199`
2. Bootstrap per-lobby variability → freeze gates in
   [`results/sim_fidelity_v1/success_thresholds.json`](../results/sim_fidelity_v1/success_thresholds.json)
3. Run Simulator v1.1 (`scaling_mode=residual`) on the **same seeds**
4. Paired per-lobby comparison + gate evaluation

## Pre-specified gates (derived before v1.1 results)

| Gate | Criterion |
|---|---|
| Turn 14 primary | v1.1 ratio ≤ bootstrap-derived max (2.033×) |
| Turn 12 secondary | v1.1 ratio < v1 ratio and ≤ 1.37× |
| Turn 10 regression | \|v1.1 − 1.033×\| ≤ 0.057 |
| Tavern tier | tier error ≤ 0.75 on measured turns |
| Alive curve | alive error vs prior ≤ 1.5 |
| Composition | Reported only — not optimized |

## Results (200 greedy lobbies, seed base 0)

| Turn | v1 ratio | v1.1 ratio | Real stats |
|---|---|---|---|
| 10 | 1.03× | **1.00×** | 1,601 |
| 12 | 1.40× | **1.01×** | 5,347 |
| 14 | 2.91× | **1.87×** | 8,293 |

Paired turn-14: mean stats −8,551 per lobby (113/133 lobbies improved ratio).

**Gate evaluation: ACCEPT Simulator v1.1** (all core gates passed).

Composition (reported, not tuned): final winner coverage sim **0.008** vs real **0.766** — unchanged problem for Phase 2C.

Avg game length: ~14.8 turns (no material new bias).

## Next: Phase 2C

Stage-matched composition realism — why simulated winners score ~0.008 coverage
vs ~0.766 real. Requires card effects, synergy/build-path dynamics, triples/discovers,
and eventually heroes/trinkets/anomalies — not more abstract scaling tweaks.
