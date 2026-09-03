# Simulator Fidelity Phase 2P — replacement-value contamination diagnostic

Date: 2026-09-03 · Status: **`2p_v1` running / pending results** ·
Artifacts: [`results/sim_fidelity_phase_2p/`](../results/sim_fidelity_phase_2p/)

## Goal

Measure whether abstract scaling contaminates recruit-phase valuation by
comparing fresh Tavern minions on their printed stats against incumbents whose
attack/health have already been inflated by prior turns of synthetic scaling.

No behavior changes. No α change. No pool/economy/effect/combat/PPO changes.

## Protocol

Fresh DEV **12700–13199**, 500 lobbies:

- Arm A: raw greedy
- Arm B: frozen BoardOpportunityCostPolicy α=0.5 + frozen prior

Instrument every **full-board** recruit decision on turns **7–14** with legal
buy slots:

- weakest incumbent current raw and printed/base raw
- best shop candidate current raw and printed/base raw
- weakest-incumbent inflation ratio
- `best_shop > weakest_scaled`
- `best_shop > weakest_printed`
- `SCALING_BLOCKED_UPGRADE = base_scale_accepts and not current_rule_accepts`
- gold, legal buy slots, rolls/buys/sells so far, actual action

For each offered card that the current raw-stat replacement rule rejects, record:

- printed stats, tribes, keywords, rules text present
- target/core status
- build signal (`_shop_build_gain`, `path_value`)
- card2vec-vocab presence

Reject buckets:

- **A** — scaling-blocked upgrade
- **B** — not a printed-stat upgrade, but has build/core value
- **C** — neither

## Key statistic

Per turn T7–T14:

```text
% of full-board recruit states where abstract scaling alone flips
"replace" -> "don't replace"
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2O DEV | 12200–12699 | consumed |
| **2P DEV** | **12700–13199** | this phase |

## Results

_Pending full DEV run._
