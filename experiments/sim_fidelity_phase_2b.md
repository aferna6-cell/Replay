# Simulator Fidelity Phase 2B — residual scaling correction

Date: 2026-09-01 · Status: **Simulator v1.1 candidate** ·
Artifacts: [`results/sim_fidelity_v1_1/`](../results/sim_fidelity_v1_1/)

## Intervention (one variable only)

Changed only `_end_of_turn_scaling()` in `hsbg_coach/bg_env.py`:

| Mode | When | Behavior |
|---|---|---|
| `ratio` | Simulator v1 (frozen) | Multiply entire board by Firestone turn-to-turn ratio |
| `residual` | Simulator v1.1 (default) | Turns 1–9: same as ratio. Turn 10+: apply `ratio_add - over` where `over = max(0, current − pace_target)` |

`scaling_mode` must be exactly `"ratio"` or `"residual"` — invalid values raise at construction.

No combat, composition, hero, trinket, or agent changes.

## Experiment protocol

```bash
pytest tests/
python -m ml.fidelity_phase_2b --lobbies 200 --seed 0
```

1. Re-run Simulator v1 (`scaling_mode=ratio`) on seeds `0…199` for paired baseline rows
2. **Load** (do not overwrite) frozen gates from
   [`results/sim_fidelity_v1/success_thresholds.json`](../results/sim_fidelity_v1/success_thresholds.json)
3. Run Simulator v1.1 (`scaling_mode=residual`) on the **same seeds**
4. Paired per-lobby comparison (identical lobby set per turn) + gate evaluation

The v1.1 contract records `code_commit`, `working_tree_clean`, and
`success_thresholds_sha256` identifying the exact code and frozen gates used.

Use `--freeze-thresholds` only for the one-time initial threshold generation.

## Pre-specified acceptance gates

| Gate | Criterion |
|---|---|
| Turn 14 primary | v1.1 paired ratio ≤ bootstrap-derived max (2.033×) |
| Turn 12 secondary | v1.1 paired ratio < v1 paired ratio and ≤ 1.37× |
| Turn 10 regression | \|v1.1 paired − 1.033×\| ≤ 0.057 |
| Tavern tier | tier error ≤ 0.75 on measured turns |
| Alive curve | alive error vs prior ≤ 1.5 |
| Composition | Reported only — not optimized |
| Game length | **Monitored only** — not a pre-specified acceptance gate |

## Results (200 greedy lobbies, seed base 0, commit `0d34919`)

| Turn | v1 paired | v1.1 paired | n paired | Real stats |
|---|---|---|---|---|
| 10 | 1.03× | **1.00×** | 200 | 1,601 |
| 12 | 1.40× | **1.05×** | 197 | 5,347 |
| 14 | 2.87× | **1.84×** | 133 | 8,293 |

Clean rerun contract: `code_commit=0d34919…`, `working_tree_clean=true`,
`success_thresholds_sha256=a62e66d0…` (unchanged frozen gates).

Paired turn-14: mean stats −8,551/lobby; 122/133 lobbies improved ratio.

**Gate evaluation: ACCEPT Simulator v1.1** (all pre-specified acceptance gates passed).

## Next: Phase 2C (diagnosis before implementation)

Break the composition failure into a measurable funnel by archetype:

```text
card available → purchased → played → retained → 2+ core → 4+ core → final coverage
```

Diagnose before choosing among card-effect fidelity, purchase/build incentives,
triples/discover mechanics, or shop/pool fidelity.
