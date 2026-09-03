# Simulator Fidelity Phase 2Q — recruit vs combat representation split

Date: 2026-09-03 · Status: **`2q_v1` IN PROGRESS** ·
Artifacts: [`results/sim_fidelity_phase_2q/`](../results/sim_fidelity_phase_2q/)

## Verdict

Pending DEV **13200–13699**. Keep **#29 HOLD**. Confirm **11500–11699**
untouched. No scaling retune. No α retune.

## Representation

```text
                 explicit game effects
                ┌─────────────────────┐
printed stats ──┤                     ├──► recruit stats  (valuation)
                │                     │
                └─────────┬───────────┘
                          │
                          │ + synthetic scaling bridge
                          ▼
                    combat stats
```

Toggle: `PHASE_2Q_RECRUIT_VALUE_STATS`

| Arm | Toggle | Replacement valuation |
|---|---|---|
| Control | `False` | live combat `attack+health` (contaminated) |
| Treatment | `True` | `recruit_attack+recruit_health` (excl. synthetic scaling) |

Combat continues to use live `attack`/`health` after residual/ratio scaling.

## Protocol

```bash
pytest tests/test_phase_2q.py
python -m ml.fidelity_phase_2q          # 13200–13699, greedy then Phase 2J
python -m ml.fidelity_phase_2q --skip-phase-2j   # greedy causal only
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N / 2O / 2P DEV | 11700–13199 | consumed |
| **2Q DEV** | **13200–13699** | this phase |

## Primary gates (greedy)

- recruit Δ T9–T12 increases
- full-board replacement rate increases
- scaling-blocked upgrades collapse substantially
- post-scale macro fidelity does not worsen materially
- game length / alive remain acceptable

## Phase 2J (α=0.5 frozen, no retune)

Report mechanism survival only: persistent 2+, committed states, played rate,
coverage, replacement transitions, relative tempo loss, directional macro
policy harm.
