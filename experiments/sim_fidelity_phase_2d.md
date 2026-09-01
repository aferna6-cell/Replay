# Simulator Fidelity Phase 2D — build-aware recruit scoring

Date: 2026-09-01 · Status: **controlled A/B** ·
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

Frozen pre-specified buy score (not tuned on eval seeds 0–199):

```text
buy_score = raw_stats - path_adj / BUILD_PATH_BUY_DIVISOR
BUILD_PATH_BUY_DIVISOR = 5.0   # matches draft.rank_discover path term
```

Everything else identical: leveling, rolling, selling, economy, RNG, Simulator v1.1
residual scaling, shop/pool, combat, card effects.

## Design

- Paired **200-lobby A/B**, seeds `0–199`
- Phase 2C `2c_v3` tracing on both arms
- Macro fidelity rollouts on both arms
- Phase 2C control baseline: seeded **0/82** fulfilled (implementation `0781ddf`)

## Primary metrics

**Mechanism (seeded current-target):**

- legally_buyable_exposures, fulfilled_exposures, rejected_exposures, fulfillment_rate
- 2+ / 4+ core assembly, mean max core pieces

**Outcome:**

- sim final-winner composition coverage (real reference ~0.766)

## Macro regression guards (treatment − control)

- Board stats ratio at turns 10/14
- Tavern tier error
- Alive-player curve
- Game length

## Acceptance (pre-specified)

**Mechanism win:** seeded fulfillment clearly above 0/82; 2+ core assembly material.

**Outcome win:** coverage meaningfully above control (~0.009) without macro regression.

## Commands

```bash
pytest tests/test_build_aware_policy.py
python -m ml.fidelity_phase_2d --lobbies 200 --seed 0
```

## Frozen

No changes to scaling, shop, combat, card effects, triples, leveling policy,
heroes, trinkets, anomalies, Firestone references, BC/PPO/TEST.
