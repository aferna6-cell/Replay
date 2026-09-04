# Simulator Fidelity Phase 2Q — recruit vs combat representation split

Date: 2026-09-03 · Status: **`2q_v1` HOLD — replacement unblocked; post-scale macro collapses** ·
Artifacts: [`results/sim_fidelity_phase_2q/`](../results/sim_fidelity_phase_2q/)

## Verdict

**HOLD.** Keep **#29 HOLD**. Confirm **11500–11699** untouched. No α retune.
Do **not** enable `PHASE_2Q_RECRUIT_VALUE_STATS` as the default yet.

Primary finding: **`replacement_unblocked_but_post_scale_macro_collapses`**

Phase 2P’s contamination mechanism is **causally confirmed**: switching
replacement valuation to recruit-value stats collapses scaling-blocked upgrades
and restores full-board replaces. But naive treatment sells combat-inflated
incumbents for printed Tavern units; residual scaling does not fully re-bridge
the hole, so post-scale macro undershoots hard and games shorten.

## Representation (implemented)

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

Toggle: `PHASE_2Q_RECRUIT_VALUE_STATS` (default **OFF**).

## DEV 13200–13699 (head after gate fix)

### Greedy causal test

| Metric | Control | Treatment | Gate |
|---|---:|---:|---|
| `% scaling-blocked` (policy) | **80.3%** | **0.0%** | PASS |
| Full-board replace rate | **1.55%** | **28.7%** | PASS |
| Recruit Δ mean T9–T12 (combat) | **+0.08** | **−272** | FAIL |
| Post-scale / Firestone T10 | **0.953** | **0.471** | FAIL |
| Post-scale / Firestone T14 | **1.825** | **0.109** | — |
| Avg game length | **15.6** | **13.1** | FAIL |

Gates passed: **2 / 5**.

Combat recruit Δ going deeply negative under treatment is expected under the
current probe: policies sell **combat-scaled** incumbents for **printed** shop
minions, so `p.strength()` drops mid-recruit before residual re-applies.

### Phase 2J α=0.5 (frozen, no retune)

| Metric | Control | Treatment |
|---|---:|---:|
| `% scaling-blocked` | 64.9% | **0.0%** |
| Full-board replace rate | 6.8% | **19.6%** |
| Replacement transitions | 4125 | **6445** (+2320) |
| Mean relative tempo loss | 0.106 | **0.009** |
| Post-scale T10 | 0.889 | **0.652** |
| Post-scale T14 | 1.540 | **0.476** |
| Avg game length | 15.1 | **13.5** |

Directional macro harm (treatment farther from 1 than control): T10 **+0.24**,
T12 **+0.27**, T14 **−0.016**. Mechanism (replacements ↑, tempo loss ↓) survives
directionally; absolute macro does not.

Composition-trace persistent-2+/coverage omitted this pass (memory); policy_stats
cover replacement/tempo.

## Interpretation

```text
scaled-incumbent valuation contamination
        ↓  (2P, confirmed)
recruit-value split unblocks replaces
        ↓  (2Q treatment)
sell inflated combat stats for printed shop
        ↓
pre-scale board crater → residual undershoot
        ↓
post-scale macro collapses; games shorten
```

Next is **not** “always compare printed,” not α retune, not confirm burn.
Need a follow-on that keeps recruit-value valuation **without** letting residual
pacing assume scaled incumbents persist through the recruit phase (e.g. residual
budget / apply timing co-design, or recruit-value-aware pace target).

## Protocol

```bash
pytest tests/test_phase_2q.py
python -m ml.fidelity_phase_2q          # 13200–13699
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2N / 2O / 2P DEV | 11700–13199 | consumed |
| **2Q DEV** | **13200–13699** | consumed |
