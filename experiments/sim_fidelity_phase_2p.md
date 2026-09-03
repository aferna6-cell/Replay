# Simulator Fidelity Phase 2P — replacement-value contamination diagnostic

Date: 2026-09-03 · Status: **`2p_v2` golden correction running / pending** ·
Artifacts: [`results/sim_fidelity_phase_2p/`](../results/sim_fidelity_phase_2p/)

## Goal

Measure whether abstract scaling contaminates recruit-phase valuation by
comparing fresh Tavern minions against incumbents whose `attack/health` have
already been inflated by prior turns of synthetic scaling.

No behavior changes. No α change. No pool/economy/effect/combat/PPO changes.

## 2p_v2 correction

`2p_v1` reconstructed every incumbent against the non-golden KB printed stats.
Golden copies must use the **natural** printed baseline (`2×` KB attack/health),
detected via observe() `tags.PREMIUM == "1"`.

Report contamination for:

1. all full-board states
2. non-golden-weakest states only

Reuse DEV **12700–13199**. Confirm **11500–11699** reserved.

## Protocol

```bash
pytest tests/test_phase_2p.py
python -m ml.fidelity_phase_2p   # seeds 12700–13199, 500 lobbies × 2 arms
```

## Results

_Pending 2p_v2 full DEV re-run._

## Prior 2p_v1 (superseded for golden baseline)

`2p_v1` found scaling-blocked upgrades dominate (~80.8% greedy / ~63.7% Phase 2J
across T8–T14). Those exact headline percentages are **not** canonical until
`2p_v2` confirms after the golden correction.
