# Simulator Fidelity Phase 2S — board-level abstract scaling

Date: 2026-09-04 · Status: **`2s_v1` implemented, default OFF — smoke only, not
the 500-lobby DEV** ·
Depends on: [Phase 2S prereg](sim_fidelity_phase_2s.md) / [Phase 2R QA](sim_fidelity_phase_2r_qa.md)

Stacked on PR #35 (`cursor/phase-2s-prereg-3cad`). Keep **#29 / #33 / #34 / #35
HOLD**. Do **not** merge. Confirm **11500–11699** reserved. No α / residual
retune. `PHASE_2Q_RECRUIT_VALUE_STATS` stays the treatment selector and default
**OFF**. `PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING` is wired and default **OFF**.

## Representation

```text
minion.recruit_*     printed + golden + modeled real buffs   ← selection, sell loss
player.abstract_pool residual/ratio budget already applied   ← held across replaces
minion.combat        recruit_* + share of abstract_pool      ← combat only
```

On **sell**, lose that minion's **real / base / golden** recruit stats. Do
**not** subtract its combat−recruit gap from `abstract_pool`. Remaining (and
newly played) minions are re-allocated the same pool.

On **play**, the recruit enters at printed/golden only; combat display picks up
a share of the surviving pool. Residual **budget math is unchanged** — only
the storage of already-applied abstract stats moves from "on the minion" to
"on the player/board". No-replacement boards are not re-painted after scale,
so 2S OFF and 2S ON stay bit-identical until a board membership change.

## This hour (implementation + unit/smoke)

- Focused conservation tests in `tests/test_phase_2s.py`
- Existing 2Q / 2R / 2S suites
- Tiny **non-evaluative** smoke (8 lobbies, 14200–14207) for runtime /
  accounting only — **not** the preregistered 500-lobby DEV

Do **not** treat smoke gates as a route. Next hour, if pool accounting and
tests are clean: `python -m ml.fidelity_phase_2s` (500 lobbies, 14200–14699).

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N–2R DEV | 11700–14199 | consumed |
| **2S smoke** | 14200–14207 | non-evaluative runtime check |
| **2S DEV** | **14200–14699** | **predeclared, not run this hour** |

## Predeclared gates (greedy, 500 lobbies, 14200–14699)

Control: 2Q toggle OFF, 2S toggle OFF.
Treatment: 2Q recruit-value selection ON **and** 2S board-level pool ON.
Phase 2J α=0.5 report-only, no retune.

| Gate | Pass if |
|---|---|
| Replacement held | treatment full-board replace rate **≥ 0.10** |
| Post-scale Firestone T10 | treatment ratio **≥ 0.85** |
| T10 not materially worse | treatment − control **≥ −0.10** |
| Game length | treatment − control **≥ −0.50** |
| Replace combat loss contained | treatment mean loss / replace **≤ 20** |

## Routing (evaluative DEV only)

| Outcome | Route |
|---|---|
| All 5 gates pass | `board_level_scaling_recovers_macro` — still **HOLD** #29/#33/#34/#35; no merge; no confirm |
| Replace held, T10 still collapsed | `representation_insufficient` — HOLD; do not retune α/scaling |
| Replace rate collapses | `selection_regressed` — HOLD; 2S must not re-block 2Q |
| Mixed / missing | `inconclusive` |

## Protocol

```bash
pytest tests/test_phase_2q.py tests/test_phase_2r.py tests/test_phase_2s.py
python -m ml.fidelity_phase_2s --non-evaluative --skip-phase-2j   # smoke only
# later: python -m ml.fidelity_phase_2s   # 14200–14699 evaluative DEV
```
