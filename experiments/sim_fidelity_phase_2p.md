# Simulator Fidelity Phase 2P — replacement-value contamination diagnostic

Date: 2026-09-03 · Status: **`2p_v1` COMPLETE — contamination dominant** ·
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

**Primary finding:** `scaling_contamination_dominant`

The diagnostic supports the user’s hypothesis directly: by the time boards are
full, abstract scaling has pushed incumbent board copies and fresh shop cards
onto incompatible raw-stat scales.

### Headline

Across all full-board states T8–T14:

| Arm | Full-board states | `% scaling-blocked` | `P(best shop > weakest scaled)` | `P(best shop > weakest printed)` |
|---|---:|---:|---:|---:|
| Greedy | 83,595 | **80.8%** | **4.7%** | **85.5%** |
| Phase 2J | 56,823 | **63.7%** | **20.6%** | **84.3%** |

So in most full-board states, the best Tavern offer would beat the weakest
incumbent if that incumbent were compared on its printed/base raw stats — but
fails once the incumbent’s synthetic scaling is included.

### T10 closes the case

| Metric | Greedy | Phase 2J |
|---|---:|---:|
| Full-board states | 2,569 | 2,221 |
| `% scaling-blocked` | **77.6%** | **57.6%** |
| `P(best shop > weakest scaled)` | **0.0%** | **21.4%** |
| `P(best shop > weakest printed)` | **77.6%** | **79.0%** |
| Median weakest board scaled raw | **45** | **25** |
| Median weakest board printed raw | **7** | **7** |
| Median best shop printed raw | **10** | **10** |
| Median inflation ratio | **7.0×** | **2.82×** |

Greedy’s T10 full-board states are especially stark: the best shop card never
beats the weakest **scaled** incumbent, but beats the weakest **printed**
incumbent in 77.6% of states.

### Later turns get worse for greedy

Greedy T11–T14:

| Turn | `% scaling-blocked` | `P(best shop > weakest scaled)` | `P(best shop > weakest printed)` | Median inflation |
|---:|---:|---:|---:|---:|
| 11 | 84.7% | 0.0% | 84.7% | 11.0× |
| 12 | 85.2% | 0.0% | 85.2% | 26.75× |
| 13 | 86.0% | 0.0% | 86.0% | 55.0× |
| 14 | 87.0% | 0.0% | 87.0% | 86.25× |

Median weakest incumbent raw grows from **7 printed** to **579 scaled** by T14,
while the median best shop printed raw is still only **12**.

### Rejected-offer A/B/C split

Among current-rule rejected shop candidates, the A-bucket (scaling-blocked
upgrade) is the single largest bucket in both arms from T9 onward:

- **Greedy:** A ≈ **38–43%**, B ≈ **20–23%**, C ≈ **36–39%**
- **Phase 2J:** A ≈ **36–40%**, B ≈ **21–24%**, C ≈ **38–41%**

This means there is still residual “neither” mass (C) and a meaningful
build/core-value bucket (B), so missing synergy/effect value is not ruled out.
But the dominant recruit-collapse mechanism is already visible without invoking
missing card effects: **synthetic scaling alone blocks raw-stat upgrades**.

## Interpretation

Phase 2O showed that the apparent T10 macro failure was mostly a pre-scale vs
post-scale timing issue. Phase 2P identifies the recruit-phase degeneration
underneath it:

```text
abstract scaling inflates incumbents
        ↓
fresh shop cards stay on printed/base raw stats
        ↓
full-board replacement compares incompatible scales
        ↓
best shop rarely beats weakest incumbent under current rule
        ↓
recruit contribution collapses
```

Phase 2J is affected too because its opportunity-cost model uses the same scaled
board raw stats, but the arm still shows lower inflation and a nonzero
replacement rate. Nothing here justifies changing α=0.5 before the value-space
representation is fixed.

## Call

Keep **#29 HOLD**. Do **not** consume **11500–11699**.

Next step is **Phase 2Q**:

> separate recruit valuation from scaled combat strength so abstract scaling can
> represent missing combat/buff growth without making incumbents permanently
> incomparable to fresh Tavern minions.
