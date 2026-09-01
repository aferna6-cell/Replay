# Simulator Fidelity Phase 2C — composition assembly diagnostic

Date: 2026-09-01 · Status: **measurement only (draft PR #18)** ·
Artifacts: [`results/sim_fidelity_phase_2c/`](../results/sim_fidelity_phase_2c/)

## Question

> Why do Simulator v1.1 winners achieve ~0.008 final composition coverage when
> real Firestone winners achieve ~0.766?

No sim mechanics, policies, card effects, shop, scaling, or combat changes.

## Methodology v3 (`2c_v3`)

Building on v2 seat/shop fixes, v3 adds **target-confidence strata** and **exact
exposure accounting**.

### Target views

| View | Relevance rule | Use |
|---|---|---|
| `broad_current_target` | `infer_target` chose archetype | Exploratory only |
| `seeded_current_target` | + `core_have >= 1` | **Primary causal diagnostic** |
| `committed_current_target` | + `core_have >= 2` and tier ≥ 4 | Strong confirmation |
| `final_target_hindsight` | Winner's eventual archetype | Labeled hindsight |

Thresholds align with `build_note()` (`have > 0`) and `path_value()` off-path
penalty (`have >= 2` at tier 4+).

### Exposure accounting

Each legally-buyable exposure is **core name × shop generation**. Purchases
**latch** to the active generation regardless of later `infer_target` changes.

```text
fulfilled_exposures + rejected_exposures == legally_buyable_exposures
```

### Invalidated prior headlines

- v1: B=2,516 / A=1,185 / C=99 (seat/shop scope bugs)
- v2 broad: 515 exposures / 6 fulfilled / 509 rejected (no commitment thresholds)

### Reproducibility (Phase 2B standard)

```text
commit implementation → clean tree → run 200 lobbies → artifact-only commit
```

Report records `implementation_commit` and `working_tree_clean: true`.

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
for v3 per-archetype funnels, rejection patterns, and all four relevance views.

## Frozen for Phase 2C

Residual scaling (v1.1), combat, greedy policy, card effects, shop distribution,
triples/discovers, heroes, trinkets, anomalies.
