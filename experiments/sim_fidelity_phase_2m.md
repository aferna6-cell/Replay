# Simulator Fidelity Phase 2M — shop/pool rules audit

Date: 2026-09-02 · Status: **`2m_v2` methodology cleanup (re-run pending)** ·
Artifacts: [`results/sim_fidelity_phase_2m/`](../results/sim_fidelity_phase_2m/)

## Research question

> Is post-assembly scarcity caused by incorrect simulator catalogue/pool rules,
> incorrect live-pool accounting, or expected scarcity under a correctly
> implemented finite shared pool?

## Methodology `2m_v2` (vs `2m_v1`)

1. **Post-assembly deal boundary:** `turn > entry_turn` (not `>=`). Entry is
   end-of-recruit with first 2+ cores; entry-turn shops are pre-assembly and
   positively selected.
2. **Primary calibration is deal-level:** for each eligible deal, compare
   `E[raw | pre-deal pool]` and `P(hit | pre-deal pool)` to observed count/hit.
   Aggregate `ΣE(raw)` vs `Σobs(raw)` and `ΣP(hit)` vs `Σobs(hits)`.
3. **Lobby-clustered bootstrap** on per-lobby `(obs − exp)` so correlated
   card×deal rows don't masquerade as independent samples.
4. **Adaptive whole-window `product(P_zero)` demoted** to descriptive only
   (later pool state adapts after hits).
5. Rule mismatches split into **Phase 2N-actionable** vs **contextual**
   (`shop_slots_vs_spell_era`, `no_tier_7`).

## Already-valid findings (independent of live calib)

| Finding | Phase 2N? |
|---|---|
| 24.6% core slots missing from KB | **Yes** — classify vs current active pool first |
| Death does not return copies to pool | **Yes** |
| Freeze does not top up incomplete shops | **Yes** |
| T6 copies 6 vs current BG **7** | **Yes** |
| Spell-era shop sizes / Tier 7 | Contextual / out of scope |

## Frozen

- Policy: Phase 2J α=0.5, prior `9b31c93a…`
- Diagnostic DEV: **10200–10699** (reuse; no reserved burn)
- Reserved: intervention **11000–11499**, confirm **11500–11699**
- No buy-legality/economy, card effects, BC/DAgger/PPO; no sim patches in 2M

## Decision gate after re-run

`multiple_substantial_mismatches` is still expected from catalogue + lifecycle +
T6. Separately: if deal-level observed ≈ expected → `_draw()` not implicated;
if observed substantially undershoots → add scoped draw-path investigation.

## Commands

```bash
pytest tests/test_shop_pool_audit.py tests/test_bg_env.py
python -m ml.fidelity_phase_2m
```
