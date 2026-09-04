# Phase 2R independent QA

Date: 2026-09-03 · Status: **clean — 2R attribution survives** ·
Source artifacts: [`results/sim_fidelity_phase_2r/`](../results/sim_fidelity_phase_2r/)
(head `fbb2125`, PR #34 HOLD)

Independent recompute and event-accounting review. No simulator change. No
merge. Confirm **11500–11699** untouched.

## Attribution (greedy, committed tables)

From `per_turn_decomposition_greedy.json` only — not from `decision.json`:

| Turn | Control net after residual | Treatment net | Excess net | Replaces (C / T) |
|---:|---:|---:|---:|---:|
| 8 | −103.188 | −103.188 | **0.000** | 0 / 0 |
| 9 | −581.548 | −87.050 | **+494.498** | 1467 / 11239 |
| 10 | −559.173 | −280.118 | **+279.055** | 0 / 0 |

```text
T10 post-scale deficit  = 1527.401 − 749.039 = 778.362
cumulative excess T8–T10 = 0 + 494.498 + 279.055 = 773.553
churn_explains_fraction  = 773.553 / 778.362 = 0.9938216477777939
```

Matches published **99.38%** to 1e-12. Threshold 0.55. Same-turn T10 slice is
**0.359** because greedy T10 has **zero** replacements; that slice is residual
undershoot on the T9 crater, which 2R already reported.

Cross-checks that also match `decision.json`:

- Replace rate 0.01618 → 0.29984
- Mean combat loss / replace −0.776 → 96.099
- Post-scale / Firestone T10 0.954 → 0.468
- Mean game length 15.734 → 13.088

Phase 2J report-only frac **1.0038** recomputes the same way from
`per_turn_decomposition_phase_2j.json` (not a merge input).

## Event accounting

`ReplacementChurnTracer` completes **once**, on play:

- `sell → play` (hand) — greedy 2R events are all this path
- `sell → buy → play` (shop) — supported; buy does not emit an event
- `sell → roll/end/level` or leftover pending at end-of-recruit — abandon, not a replacement

Known-loss identity on every committed example event:

```text
combat_strength_loss  = incumbent_combat_raw − candidate_combat_raw
recruit_value_gain    = candidate_recruit_raw − incumbent_recruit_raw
net_board_combat_delta = −combat_strength_loss
```

No double-count: a second buy after a shop candidate is ignored; two completed
sell→play pairs in one turn emit two events and sum combat removed once each.

Residual recovery: `net_after_residual = combat_removed − residual_add`.
Positive net means residual did not replace the combat sold that seat-turn.

Next-turn carry: `begin_seat_recruit` on turn `t` writes
`next_turn_carried_strength = player.strength()` onto the seat's turn `t−1`
post-scale row. T9 control carry 968 vs post-scale 944 (survivors slightly
stronger than the post-scale mean); treatment 469 vs 448. Direction matches.

## Caveats (not blockers)

1. **T10 same-turn replacements are zero** under greedy. The 99.38% identity is
   carry-forward (T9 crater + T10 residual gap), not same-turn T10 churn.
2. T10 seat-turn counts differ (2577 vs 2450). Means are per alive seat-turn.
3. Residual add is whole-board, not replacement-attributed. That is why T10
   excess net equals the residual delta.

## Verdict

**2R attribution survives independent recomputation.** Primary finding
`replacement_churn_loss_explains_macro_collapse` stands. Keep **#29 / #33 / #34
HOLD**. Do not retune scaling or α. Do not burn confirm. Phase 2S may
preregister a default-OFF board-level abstract-scaling representation.
