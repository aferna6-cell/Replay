# Simulator Fidelity Phase 2L — post-assembly availability decomposition

Date: 2026-09-02 · Status: **measurement-only (2l_v1)** ·
Artifacts: [`results/sim_fidelity_phase_2l/`](../results/sim_fidelity_phase_2l/)

## Research question

> Of the missing weighted core mass that is never *legally* buyable after first-2
> assembly (Phase 2K: 92.1%), how much is tier-locked, never raw-offered,
> raw-offered-but-illegal, or excluded from the lobby pool?

## Frozen policy

Phase 2J `BoardOpportunityCostPolicy` α=0.5, prior `9b31c93a…`. Observational only.

## Seeds

| Range | Role |
|---|---|
| **10200–10699** | DEV diagnostic (500) |
| 10700–10999 | Adaptive expand if &lt;40 states |
| 8000–8199 / 9000–9999 / 10000–10199 | Forbidden |

## Subfate taxonomy (of never-legal missing mass)

| Code | Meaning |
|---|---|
| `A1_NOT_IN_LOBBY_POOL` | Card cannot enter catalogue for lobby tribes / KB |
| `A2_NEVER_TIER_ELIGIBLE` | Player never reaches card tech level after first-2 |
| `A3_TIER_ELIGIBLE_ZERO_RAW` | Tier OK sometime; **zero** raw `pre_shop` appearances |
| `A4_RAW_BUT_ZERO_LEGAL` | Raw appearance; **zero** legal-buy slots (gold/hand mask) |
| `A5_OTHER` | Residual |

## Headline metrics

```text
% never-legal missing mass: tier-eligible but ZERO RAW appearances
% never-legal missing mass: RAW appearance but ZERO LEGAL-buy appearances
```

If the first dominates → shop/pool generation (Phase 2M).
If the second dominates → **do not touch the pool**; legality/economy.

## Commands

```bash
pytest tests/test_availability_decomposition.py
python -m ml.fidelity_phase_2l
```

## Frozen

No policy changes, no card effects, no BC/DAgger/PPO.
