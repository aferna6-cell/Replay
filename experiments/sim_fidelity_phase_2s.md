# Simulator Fidelity Phase 2S — board-level abstract scaling

Date: 2026-09-04 · Status: **`2s_v1` HOLD — `inconclusive` (game length)** ·
Artifacts: [`results/sim_fidelity_phase_2s/`](../results/sim_fidelity_phase_2s/)

Stacked on PR #36 (`cursor/phase-2s-impl-4b72`, head `39e78bd`). Keep
**#29 / #33 / #34 / #35 / #36 HOLD**. Do **not** merge. Confirm **11500–11699**
untouched. No α / residual retune. Both toggles stay default **OFF**.

## Verdict

**HOLD.** Route: **`inconclusive`**.

Four of five locked greedy gates pass. Replacement is held and T10 post-scale
recovers (treatment 1.007, Δ vs control +0.055). Board-level replacement loss
(`-net_board_combat_delta`) is **−3.46** (a small recruit-side gain; share
loss ≥20 is 0). Game length is the miss: **15.692 → 13.510** (Δ **−2.182**,
floor −0.50).

Not `board_level_scaling_recovers_macro` (not all five). Not
`representation_insufficient` (T10 held). Not `selection_regressed` (replace
rate 0.298 ≥ 0.10). Independent QA next; do not retune α/scaling; do not burn
confirm.

Pool accounting on all 500 treatment lobbies: worst abs drift **0.0**.

## Representation (unchanged)

```text
minion.recruit_*     printed + golden + modeled real buffs   ← selection, sell loss
player.abstract_pool residual/ratio budget already applied   ← held across replaces
minion.combat        recruit_* + share of abstract_pool      ← combat only
```

## Locked greedy gates (DEV 14200–14699, 500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value ON **and** 2S pool ON.
Combat-loss gate uses **`net_board_combat_delta`** (post-realloc board change),
not 2R sold-body `combat_strength_loss`.

| Gate | Control | Treatment | Threshold | Result |
|---|---:|---:|---|---|
| Full-board replace rate | 0.0163 | **0.2982** | ≥ 0.10 | **PASS** |
| T10 post-scale / Firestone | 0.952 | **1.007** | ≥ 0.85 | **PASS** |
| T10 treatment − control | — | **+0.055** | ≥ −0.10 | **PASS** |
| Game length Δ | 15.692 | **13.510** | Δ ≥ −0.50 | **FAIL** |
| Mean replacement combat loss | — | **−3.46** | ≤ 20 | **PASS** |

Harness `decision.json` still scores sold-body loss (296.8) and reports 3/5;
that is the 2R identity, not the 2S gate. Assignment lock is
[`gates_net_board.json`](../results/sim_fidelity_phase_2s/gates_net_board.json)
(**4/5**, same route).

Sold-body vs board-level on the same 24,910 treatment events:

| Metric | Mean | Median | P90 | Share ≥20 |
|---|---:|---:|---:|---:|
| Sold-body `combat_strength_loss` (2R) | 296.8 | 151 | 934 | 0.942 |
| **`-net_board_combat_delta` (2S)** | **−3.46** | −3 | −1 | **0.000** |

`net_board_combat_delta` matches `recruit_value_gain` on inspected events: the
abstract pool is conserved, so the board-level combat change is the recruit
swap. Sold-body still sees painted incumbent combat vs printed candidate.

## T12 / T14 post-scale and alive curve

| Turn | Post-scale C | Post-scale T | Δ | Alive C | Alive T | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.154 | 1.158 | +0.004 | 7.15 | 7.15 | −0.00 |
| 9 | 1.300 | 1.363 | +0.063 | 6.21 | 6.10 | −0.11 |
| 10 | 0.952 | **1.007** | +0.055 | 5.28 | 4.84 | −0.44 |
| 11 | 1.179 | 1.260 | +0.081 | 4.38 | 3.87 | −0.51 |
| 12 | **1.462** | **1.574** | +0.112 | 3.65 | 2.95 | −0.70 |
| 13 | 1.748 | 1.855 | +0.108 | 3.03 | 2.49 | −0.54 |
| 14 | **1.835** | **1.883** | +0.048 | 2.65 | 2.26 | −0.39 |

Post-scale is at or above control from T8–T14 (2R treatment T10 was 0.468 /
T14 0.115). Alive count still falls faster under treatment; that is the same
shortening that fails the game-length gate.

## Phase 2J α=0.5 (report-only, no retune)

Same seeds, frozen α=0.5, no coefficient change. Not a gate input.
Treatment = BoardOpp + 2Q recruit-value + 2S pool.

| Metric | Control | Treatment |
|---|---:|---:|
| Full-board replace rate | 6.70% | **20.48%** |
| Replacement transitions | 3829 | **7064** |
| Mean relative tempo loss | 0.110 | **0.013** |
| P95 relative tempo loss | 0.376 | **0.074** |
| Post-scale / Firestone T10 | 0.887 | **0.929** |
| Post-scale / Firestone T12 | 1.189 | **1.405** |
| Post-scale / Firestone T14 | 1.537 | **1.782** |
| Avg game length | 15.06 | **13.97** |
| Net-board replace loss | — | **−2.23** |

Alive Δ (treatment − control): T10 **−0.20**, T12 **−0.44**, T14 **−0.20**.
Direction matches greedy: post-scale recovers vs 2R’s BoardOpp collapse
(2R T10 0.878→0.661); games still shorten (Δ **−1.09**). Tempo-loss drop and
replace-rate rise are the 2Q selection mechanism surviving under 2S.

Artifacts:
[`phase_2j_report_only.json`](../results/sim_fidelity_phase_2s/phase_2j_report_only.json),
[`phase_2j_comparison.json`](../results/sim_fidelity_phase_2s/phase_2j_comparison.json).

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2N–2R DEV | 11700–14199 | consumed |
| 2S smoke | 14200–14207 | non-evaluative (prior hour) |
| **2S DEV** | **14200–14699** | **consumed this hour** |

## Protocol

```bash
python -m ml.fidelity_phase_2s          # 14200–14699 evaluative DEV
```

Working tree was clean at contract time (`39e78bd`, 34.97s). No code, α,
scaling math, gates, or seed-range edits.
