# Simulator Fidelity Phase 2D — build-aware recruit scoring

Date: 2026-09-01 · Status: **negative result (merged)** ·
Artifacts: [`results/sim_fidelity_phase_2d/`](../results/sim_fidelity_phase_2d/)

## Research question

> Does adding existing build-path value to recruit decisions convert Phase 2C's
> seeded composition opportunities into coherent boards without damaging Simulator
> v1.1 macro fidelity?

## Arms

| Arm | Policy | Buy valuation |
|---|---|---|
| Control | `greedy_policy` | `attack + health` |
| Treatment | `build_aware_greedy_policy` | `raw_stats - path_adj / 5.0` |

Frozen pre-specified buy score (not tuned on evaluation seeds 0–199):

```text
buy_score = raw_stats - path_adj / BUILD_PATH_BUY_DIVISOR
BUILD_PATH_BUY_DIVISOR = 5.0
```

**Scale note:** divisor 5.0 is numeric reuse from `draft.rank_discover` (equity
ranking, lower-is-better). It is **not** a raw-stat calibration. `path_value()`
is bounded to ~±1.3 placement units, so max buy bonus ≈ **0.26 stats** — far
below typical +2–+5 shop gaps. Phase 2D tests this exact mapping as a negative
control, not “build-aware recruiting” in general.

## Design

- Paired **200-lobby A/B**, seeds `0–199`
- Phase 2C `2c_v3` tracing on both arms
- Macro fidelity rollouts on both arms
- Control reproduces Phase 2C seeded **0/82** fulfilled

## Result (summary)

| Metric | Control | Treatment |
|---|---:|---:|
| Seeded fulfilled | 0/82 | 0/67 |
| Final winner coverage | ~0.0085 | ~0.0090 |
| Macro regression | — | passed |

**Conclusion:** `path_value()` as `-path_adj/5` in raw-stat space is **insufficient**.
Macro fidelity preserved. Does not implicate card effects (mechanism unchanged).

## Placement strength

**Not evaluated.** Homogeneous eight-seat lobbies (same policy every seat) are
for simulator fidelity, not agent-vs-agent strength. Per-turn placement averaging
over player-turn records is invalid.

## Commands

```bash
pytest tests/test_build_aware_policy.py
python -m ml.fidelity_phase_2d --lobbies 200 --seed 0
```

## Frozen

No changes to scaling, shop, combat, card effects, triples, leveling policy,
heroes, trinkets, anomalies, Firestone references, BC/PPO/TEST.
