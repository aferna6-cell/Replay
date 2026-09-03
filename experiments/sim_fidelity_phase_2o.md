# Simulator Fidelity Phase 2O — midgame scaling-budget diagnostic

Date: 2026-09-03 · Status: **`2o_v1` COMPLETE — hypothesis revised** ·
Artifacts: [`results/sim_fidelity_phase_2o/`](../results/sim_fidelity_phase_2o/)

## Verdict

**Keep PR #29 HOLD. Do not touch Phase 2J α=0.5. Do not burn 11500–11699.**

`2n_v3` remains **FAIL as run** (no retroactive rewrite). Measurement-only Phase
2O on **12200–12699** revises the strong “target-gap bridge” prior.

### Headline

| Lens | T10 greedy | T10 Phase 2J |
|---|---:|---:|
| **Pre-scale / Firestone** (what 2n_v3 gates see) | **0.600×** | **0.544×** |
| **Post-scale / Firestone** | **0.952×** | **0.886×** |
| Unfilled gap after scaling | +77 | +183 |
| Recruit Δ | **0** | **−6.5** |

Residual scaling **does** fill most of the T10 deficit when measured after it
runs. The deep midgame hole in 2n_v3 is the **end-of-recruit (pre-scale)**
snapshot — the same timing fidelity macros have always used.

## T10 decomposition (means)

### Greedy

```text
Firestone target             1601
start-recruit                 960
end-recruit                   960
recruit contribution           +0
scaling contribution         +564   (ratio_add ≈579, residual_add ≈564, over ≈15)
post-scale                   1524
unfilled target gap            77
```

### Phase 2J (α=0.5)

```text
Firestone target             1601
start-recruit                 877
end-recruit                   871
recruit contribution           −7   (board sacrifice)
scaling contribution         +547
post-scale                   1418
unfilled target gap           183
```

## Post-scale curve (symmetric absolute fidelity)

| Turn | Greedy post/FS | Phase 2J post/FS |
|---:|---:|---:|
| 8 | 1.149 | 1.028 |
| 9 | 1.297 | 1.175 |
| 10 | **0.952** | **0.886** |
| 11 | 1.180 | 1.009 |
| 12 | **1.456** | **1.194** |
| 13 | 1.739 | 1.408 |
| 14 | **1.822** | **1.551** |

Late game **overshoots** after scale (especially greedy). A naive midgame
target-gap add would risk making T12–T14 worse.

## Predeclared routing (automated)

| Finding | Fired? |
|---|---|
| Pre-scale near FS, scaling wrong | no |
| Pre far below **and** post still far below (**target-gap**) | **no** |
| Recruit contribution collapsed | **yes (primary)** |
| Just-leveled 0.6× explains deficit | yes (secondary; Phase 2J T10 split) |
| Greedy healthy / Phase 2J low | no |
| Both arms low (post-scale midgame) | no (post-scale recovered) |
| T13–14 suddenly overcompensates | borderline / visible in post-scale curve |

**Automated primary:** `recruit_contribution_collapsed` → next step
**recruit/effect-value fidelity**.

**Hypothesis revision:** the residual bridge is not leaving a large *post-scale*
T10 hole. After the active-pool correction, greedy midgame strength growth is
almost entirely abstract scaling (`recruit Δ ≈ 0` from T10+); Phase 2J
intentionally sheds stats. Pre-scale fidelity gates will keep looking “broken”
even when post-scale is near the Firestone curve.

## Prospective directional policy-harm (not applied to 2n_v3)

| Turn | Greedy \|1−r\| | Treatment \|1−r\| | harm (t−g) | Treatment closer? |
|---:|---:|---:|---:|---|
| 10 | 0.048 | 0.114 | +0.066 | no |
| 12 | 0.456 | 0.194 | **−0.261** | **yes** |
| 14 | 0.822 | 0.551 | **−0.271** | **yes** |

Matches the call on T14: Phase 2J moves **toward** Firestone vs greedy. Do not
weaken α=0.5.

## What to keep

- 2N-D catalogue / death-return / freeze-topup / T6=7
- Phase 2J α=0.5 + frozen prior
- `2n_v3` FAIL record + Phase 2B historical upper bounds (untouched)

## Suggested Phase 2P framing (not done here)

1. Decide whether Simulator v1.x acceptance should score **post-scale** board
   stats against Firestone (residual’s actual target), vs continuing to gate on
   pre-scale snapshots.
2. Investigate **recruit/effect-value ≈ 0** under the corrected pool (why boards
   stop changing in midgame recruit).
3. Only if still needed: a **bounded** midgame scaling correction that does not
   amplify T12–T14 post-scale overshoot.

## Protocol

```bash
pytest tests/test_phase_2o.py tests/test_scaling_residual.py
python -m ml.fidelity_phase_2o   # 12200–12699, 500×2 arms
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2n_v2/v3 | 11700–12199 | consumed (HOLD) |
| **2O DEV** | **12200–12699** | **consumed** |
