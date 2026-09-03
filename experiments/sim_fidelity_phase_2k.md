# Simulator Fidelity Phase 2K — post-assembly residual composition-gap diagnostic

Date: 2026-09-02 · Status: **measurement-only complete (2k_v1)** ·
Artifacts: [`results/sim_fidelity_phase_2k/`](../results/sim_fidelity_phase_2k/)

## Research question

> After a build starts coherently forming under accepted Phase 2J, where exactly
> is the remaining weighted core-coverage mass lost?

## Frozen policy

- `BoardOpportunityCostPolicy`, **α = 0.5**
- `prior_hash_sha256 = 9b31c93a8d89…`
- Observational only

## Seeds

| Range | Role |
|---|---|
| 9000–9499 | DEV screen (31 states &lt; 40) |
| **9500–9999** | Adaptive expansion → **78 states total / 1000 lobbies** |
| 8000–8199 | Forbidden (Phase 2J confirmation) |

## DEV results

**78** post-assembly states. Mass reconciliation: **pass**.

### Missing weighted coverage mass by cause

| Cause | Share of missing mass |
|---|---:|
| **A_NEVER_AVAILABLE_POST_ASSEMBLY** | **92.1%** |
| F_TARGET_SWITCH | 7.8% |
| D_DEPLOYED_THEN_LOST | ~0.05% |
| E_EXISTING_CORE_LOST | ~0.03% |
| B / C / G / H | 0% |

### Funnel (post first-2)

```text
78 first persistent 2+
 ├─ 52 reached 3+
 ├─ 19 reached 4+
 ├─ 72 survived +1 turn
 └─ 64 survived +2 turns
mean final core count ≈ 2.97
mean final coverage   ≈ 0.250
coverage_peak − final ≈ 0.0   # no peak→final retention collapse
```

### Weighted opportunity funnel (means)

```text
remaining at first-2 ≈ 0.82
  → legally available after ≈ 0.074
  → purchased ≈ 0.074
  → deployed  ≈ 0.074
  → retained  ≈ 0.072
  → final present ≈ 0.250  # includes cores already held at first-2
```

Once a missing core is offered, it is almost always bought and kept. The gap is
**upstream availability**, not post-commit rejection or sell pressure.

## Decision

`a_never_available_post_assembly` → **Phase 2L: shop/pool/core-availability fidelity**

Not card effects: low overall lobby coverage still reflects rare assembly plus
missing post-assembly offers of the remaining weighted core mass.

## Commands

```bash
pytest tests/test_post_assembly_gap_diagnostic.py
python -m ml.fidelity_phase_2k
```

## Frozen

No acceptance gate, no new policy, no card effects, no BC/DAgger/PPO.
