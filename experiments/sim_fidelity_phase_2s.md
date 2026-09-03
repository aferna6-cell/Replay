# Simulator Fidelity Phase 2S — board-level abstract scaling (preregistration)

Date: 2026-09-03 · Status: **`2s_v0_prereg` — not run** ·
Depends on: [Phase 2R QA](sim_fidelity_phase_2r_qa.md) (attribution survives)

**No simulator behavior in this commit.** Spec + gate locks only.

## Hold / freeze

Keep **#29 / #33 / #34 HOLD**. Do **not** merge. Confirm **11500–11699**
reserved. Do **not** retune residual/ratio scaling or Phase 2J **α=0.5**.
`PHASE_2Q_RECRUIT_VALUE_STATS` stays default **OFF**. Proposed
`PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING` stays default **OFF** until a later
implementation PR.

## Why this representation

2R (independent recompute **0.9938**): recruit-value selection unblocks
replaces, then selling a combat-inflated minion **destroys the synthetic
scaling that was sitting on that body**. Residual, being growth-on-current,
undershoots the crater. T10 post-scale / Firestone 0.954 → 0.468.

2S proposes: **unscaled recruit value still drives selection**; the
**synthetic abstract scaling component is a board/player pool**, redistributed
after sells, not deleted with the sold minion.

```text
minion.recruit_*     printed + golden + modeled real buffs   ← selection, sell loss
player.abstract_pool residual/ratio budget already applied   ← held across replaces
minion.combat        recruit_* + share of abstract_pool      ← combat only
```

On **sell**, lose that minion's **real / base / golden** recruit stats.
Do **not** subtract its combat−recruit gap from `abstract_pool`. Remaining
(and newly played) minions are re-allocated the same pool.

On **play**, the recruit enters at printed/golden only; combat display picks
up a share of the surviving pool. Residual **budget math is unchanged** —
only the storage of already-applied abstract stats moves from "on the minion"
to "on the player/board".

## Out of scope

- No α search, no Firestone-curve retune, no residual-clamp retune
- No confirm burn, no 11500–11699 touch
- No default-ON
- Implementation + DEV measurement land in a later PR, not here

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N–2R DEV | 11700–14199 | consumed |
| **2S DEV** | **14200–14699** | **predeclared** (>14199) |

## Predeclared gates (greedy, 500 lobbies, 14200–14699)

Control: 2Q toggle OFF, 2S toggle OFF (today's default).
Treatment: 2Q recruit-value selection ON **and** 2S board-level pool ON.
Phase 2J α=0.5 report-only, no retune.

| Gate | Pass if |
|---|---|
| Replacement held | treatment full-board replace rate **≥ 0.10** |
| Post-scale Firestone T10 | treatment ratio **≥ 0.85** |
| T10 not materially worse | treatment − control **≥ −0.10** |
| Game length | treatment − control **≥ −0.50** |
| Replace combat loss contained | treatment mean loss / replace **≤ 20** |

T12/T14 post-scale ratios and alive-curve Δ are report-only (directional
macro-harm vs Firestone, not a 2n_v3 rewrite).

## Routing

| Outcome | Route |
|---|---|
| All 5 gates pass | `board_level_scaling_recovers_macro` — still **HOLD** #29/#33/#34; no merge; no confirm |
| Replace held, T10 still collapsed | `representation_insufficient` — HOLD; do not retune α/scaling |
| Replace rate collapses | `selection_regressed` — HOLD; 2S must not re-block 2Q |
| Mixed / missing | `inconclusive` |

## Protocol (when implemented)

```bash
pytest tests/test_phase_2r.py tests/test_phase_2s.py
# later: python -m ml.fidelity_phase_2s   # 14200–14699 — not in this PR
```
