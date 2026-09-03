# Simulator Fidelity Phase 2P — replacement-value contamination diagnostic

Date: 2026-09-03 · Status: **`2p_v2` COMPLETE — contamination confirmed** ·
Artifacts: [`results/sim_fidelity_phase_2p/`](../results/sim_fidelity_phase_2p/)

## Verdict

**Mechanism closed.** Keep **#29 HOLD**. Do **not** consume **11500–11699**.

`2p_v2` applies the golden-aware natural printed baseline and reuses DEV
**12700–13199**. Contamination remains dominant on **both**:

1. all full-board states
2. non-golden-weakest states only

Primary finding: **`scaling_contamination_dominant`**  
`survives_nongolden_weakest_filter`: **true**

## 2p_v2 golden correction

Incumbent natural printed = KB printed × `(2 if PREMIUM/golden else 1)`.

Empirically, **weakest-by-current-raw never selected a golden** in this panel
(`share_weakest_golden = 0.0` on every turn T8–T14, both arms). So the dual
headlines are identical to each other, and match the superseded `2p_v1`
percentages — but those percentages are now **canonical** under the corrected
methodology.

## Headline (T8–T14, golden-aware)

| Arm | Full-board states | `% scaling-blocked` | `P(best shop > weakest scaled)` | `P(best shop > weakest printed)` |
|---|---:|---:|---:|---:|
| Greedy (all / non-golden weakest) | 83,595 | **80.8%** | **4.7%** | **85.5%** |
| Phase 2J (all / non-golden weakest) | 56,823 | **63.7%** | **20.6%** | **84.3%** |

### T10

| Metric | Greedy | Phase 2J |
|---|---:|---:|
| Full-board states | 2,569 | 2,221 |
| `% scaling-blocked` | **77.6%** | **57.6%** |
| `P(best shop > weakest scaled)` | **0.0%** | **21.4%** |
| `P(best shop > weakest printed)` | **77.6%** | **79.0%** |
| Median weakest scaled raw | **45** | **25** |
| Median weakest natural printed raw | **7** | **7** |
| Median best shop printed raw | **10** | **10** |
| Median inflation | **7.0×** | **2.82×** |
| Share weakest golden | **0.0** | **0.0** |

## Interpretation

Recruit collapse is driven by **synthetic scaling contaminating replacement
valuation**, not by miscounting golden doubling on the weakest incumbent.

Causal chain:

```text
repeated synthetic scaling
        ↓
incumbent valuation inflation grows every turn
        ↓
fresh Tavern minions become incomparable
        ↓
replacement shuts down
        ↓
rolling replaces recruiting
        ↓
recruit contribution collapses
```

## Next: Phase 2Q (see PR / `sim_fidelity_phase_2q.md`)

`2q_v1` DEV **13200–13699**: replacement unblocked (scaling-blocked → 0;
replace rate ↑) but **post-scale macro collapses** under naive recruit-value
replacement. HOLD — do not default the toggle on; do not freeze.

## Protocol

```bash
pytest tests/test_phase_2p.py
python -m ml.fidelity_phase_2p   # 12700–13199 reuse
```

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved** |
| 2O DEV | 12200–12699 | consumed |
| **2P DEV** | **12700–13199** | reused for 2p_v2 |
