# Simulator Fidelity Phase 2T — game-length / damage attribution

Date: 2026-09-04 · Status: **`2t_v1` HOLD — `damage_model_fidelity`** ·
Artifacts: [`results/sim_fidelity_phase_2t/`](../results/sim_fidelity_phase_2t/)

Stacked on PR #37 (`cursor/phase-2s-dev-eval-75c7`). Keep
**#29 / #33 / #34 / #35 / #36 / #37 HOLD**. Do **not** merge. Confirm
**11500–11699** untouched. No α / residual / `_hero_damage` / gate changes.
Reused consumed 2S DEV **14200–14699** (no new seeds).

## Verdict

**HOLD.** Route: **`damage_model_fidelity`**.

The −2.182-turn shortening (15.692 → 13.510) is the same 2S game-length miss.
Paired combat outcomes T7–T14 are **not** materially more one-sided. Extra HP
drain is **`_hero_damage` survivor-tier amplification** (79% of extra
per-alive-seat-turn damage). Combat-strength fidelity stays healthy: 2S
post-scale / Firestone is ≥ control on T8–T14.

Next step is a **separate damage-model fidelity phase**. Do not retune
recruit/scaling or α. Do not burn confirm.

## `_hero_damage` path (inspected, unchanged)

`sim.py._damage_to_hero` returns `survivor_count + max(tavern_tier, 1)` as
signed `raw`. `BGEnv._hero_damage` recovers that count and reweights it:

```text
survivors = max(1, |raw| − winner_tavern_tier)
applied   = winner_tavern_tier + max(1, round(survivors × mean(board.tier)))
count_only = winner_tavern_tier + survivors          # sim.py
amplification = applied − count_only
```

The formula uses **mean tier of the winner's current board**, not the
combat-survivor identities (those are not returned). Applied HP loss on this
DEV reconciled to both the formula and the HP delta (max err **0**).

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same seeds as
[#37](https://github.com/aferna6-cell/Replay/pull/37).

| Metric | Control | Treatment | Δ |
|---|---:|---:|---:|
| Avg game length | 15.692 | **13.510** | **−2.182** |
| Mean turns-to-elimination | 10.65 | 9.95 | −0.70 |
| HP at T7 (alive) | 18.59 | 18.33 | −0.27 |
| Tie rate | 0.096 | 0.082 | −0.014 |
| Decisive rate | 0.904 | 0.918 | +0.014 |
| Lethal rate | 0.284 | 0.356 | +0.072 |
| Hit rate / alive seat-turn | 0.403 | 0.415 | +0.012 |
| Applied when hit | 10.87 | **13.95** | **+3.08** |
| Count-only when hit | 7.46 | 7.76 | +0.30 |
| Amplification when hit | 3.41 | **6.19** | **+2.78** |
| Survivor count when hit | 2.25 | 2.63 | +0.38 |
| Winner tavern tier | 5.21 | 5.13 | −0.07 |
| Winner minion-tier mean | 2.51 | **3.33** | **+0.83** |
| Winner minion-tier sum | 17.54 | **23.31** | **+5.77** |
| \|raw\| margin (live) | 6.60 | 7.01 | +0.41 |
| \|strength\| margin (live) | 552 | 437 | −115 |
| Applied HP / alive seat-turn | 4.38 | **5.79** | **+1.41** |

Keywords on winning boards are essentially unchanged (DS / Taunt / Windfury
flat; golden 0.007 → 0.041). Strength margins are **less** one-sided.

## Decomposition of the −2.18 turns

HP-flow identity held on both arms (`applied = count_only + amplification`).

| Source of extra dpt | Share of Δdpt | Implied turns |
|---|---:|---:|
| (a) Combat outcome (hit rate + count-only / survivors) | **0.180** | 0.56 |
| (b) `_hero_damage` tier amplification | **0.794** | 2.47 |
| First-order extra-dpt total | 1.000 | 3.12 |
| Actual shortening | — | **2.18** |
| Lifecycle residual (actual − implied) | — | −0.94 |

(a) is small: ties barely drop, survivor count only +0.38, raw margin +0.41.
Lethal rate rises because **hits are bigger**, not because combats become
more decisive.

(b) dominates: winner **board minion-tier mean** 2.51 → 3.33 with tavern
tier flat/down. The formula multiplies recovered survivor count by that
mean. Per-lobby T7–T14 HP: count-only **115.5 → 111.1** (fewer fights in a
shorter game) while amplification **52.9 → 88.7**.

(c) does **not** create the shortening. T7 HP is already almost matched
(−0.27). Extra dpt over-predicts (3.12 vs 2.18) because shorter lobbies
simply stop taking late fights. Ghost share is slightly down.

2S post-scale T8–T14 remains treatment ≥ control. Absolute winner
attack+health is lower under treatment (3684 → 3004); that is a mix /
pace-ratio effect, not a combat-outcome collapse.

## Decision

```text
2S board-level pool recovers post-scale
        ↓
higher-tier bodies on the board (recruit-value + pool)
        ↓
_hero_damage weights survivors × mean(board.tier)
        ↓
+2.78 applied HP / hit, lethal +7.2pp
        ↓
games end 2.18 turns sooner
```

**Next:** damage-model fidelity (count-only vs real BG survivor-tier sum vs
this board-mean proxy). **Not** a recruit/scaling retune. **Not** a
composition/effect combat rewrite unless a later phase shows (a) growing.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S / **2T DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2t.py -q
python -m ml.fidelity_phase_2t          # reused 14200–14699
```

Working tree was clean at contract time (`9219444`, 28.9s). Tracer is
observational (same-seed hooked vs unhooked placements/HP/length match).
