# Phase 2S smoke — NON-EVALUATIVE

**Do not route the 500-lobby Phase 2S DEV from this measurement.**

| | |
|---|---|
| Role | runtime / pool-accounting check only |
| Lobbies | 8 |
| Seeds | **14200–14207** (inside the reserved 14200–14699 band) |
| Control | 2Q OFF, 2S OFF |
| Treatment | 2Q recruit-value ON **and** 2S board-level pool ON |
| Phase 2J | skipped |
| Evaluative | **false** |
| Commit | `ec252a2` (working tree clean when written) |

## Accounting

- Mid-replace painted-vs-pool assert: **no raise**
- Reallocate conservation (`painted == round(pool)`): **no raise**
- Worst end-of-lobby abs drift: **0.0** (8 lobbies)

Pool conservation / accounting is **clean**. Unit tests (2Q / 2R / 2S + residual
+ `bg_env`) passed. Decision criterion for the next hour: run the full
preregistered 500-lobby DEV (`python -m ml.fidelity_phase_2s`, 14200–14699).

## Informational only (not a route)

These 8-lobby numbers are **not** gates. They exist so a later reader does not
mistake this folder for the 500-lobby DEV.

| Metric | Control | Treatment |
|---|---:|---:|
| Full-board replace rate | 0.014 | 0.303 |
| Post-scale / Firestone T10 | 0.912 | 0.999 |
| Mean game length | 15.125 | 13.500 |
| 2R `mean_combat_loss_per_replacement` (sold-body combat − candidate) | −1.3 | 286.6 |

The 2R loss instrument still snapshots the sold body's **pre-sell combat**
(includes synthetic). That is the 2Q crater definition. Under 2S the pool keeps
that synthetic, so this number is **not** post-realloc board-combat loss.
`net_board_combat_delta` on each replacement event is the post-realloc board
change. Next hour should not treat the raw 2R body-loss as the 2S ≤20 gate
without deciding which instrument to use.

`decision.json` `primary_finding` = `implementation_smoke_non_evaluative`.
Keep #29 / #33 / #34 / #35 HOLD. No merge. No confirm. α / residual untouched.
