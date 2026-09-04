# Simulator Fidelity Phase 3F — carry divergence timing + outcome-conditioning audit

Date: 2026-09-04 · Status: **`3f_v1` HOLD — `selection_outcome_conditioning_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3f/`](../results/sim_fidelity_phase_3f/)

Stacked on PR #51 (`cursor/phase-3e-board-pool-lifecycle-f156`, head `2f93efb`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3E DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3E left **inherited carry = 72.4% of 3D A1 (+0.305 / hit)** with punch-row
Δcarry **−196**. Unconditional alive-seat carry stayed arm-matched; the
punch-sample crater sat on low-tier winner-start bodies. This hour dates
that gap on paired `(seed, seat)` paths.

## Verdict

**HOLD.** Route: **`selection_outcome_conditioning_dominates`**.

3D A1 and the 3E carry share reproduce exactly (`0.4216721428553852`,
**0.7236353954551374**). Unpaired punch-row Δcarry reproduces **−196.333**.
Pairing the same 3893 `(seed, seat)` keys across arms, T7–T14 pooled
Δcarry is only **−17.8 (9.1% of −196)**. At each seat’s first punch-row
appearance the paired Δ shrinks further to **−6.1 (3.1%)**. Restricting
to later punch inclusion + T1 winner-start + loss leaves **−2.0 (1.0%)**.

The #51 carry crater is therefore **not** a real paired path split that
exists before outcome conditioning. **90.9%** of −196 is the unpaired
punch-sample composition — who enters the 3D punch rows as the opposing
seat, especially against low-tier winner-start bodies.

Do **not** audit scaling inputs at a first-divergence turn. Do **not**
retune constants. Next hour isolates the punch / winner-start / low-tier
**selection mechanism**.

| Frame | Δcarry (T−C) | Share of #51 −196 |
|---|---:|---:|
| Unpaired punch rows (3E term) | **−196.333** | 1.000 |
| Unconditional paired seats (pooled T7–T14) | −17.835 | **0.091** |
| Paired Δ at first punch appearance | −6.093 | 0.031 |
| + low winner-start (T1) at appearance | −2.001 | 0.010 |
| + eventual loss at appearance | −2.001 | 0.010 |
| Selection residual (1 − uncond share) | — | **0.909** |

## Classification (observational)

For every `(seed, seat)` present in both arms the tracer walks T7–T14
and records recruit-start carry, current-turn add, tavern/board mix,
alive status, that turn’s fight outcome, whether/when the seat is the
opposing (loser) seat of a 3D punch-row fight, the winner-start tier of
that fight, and the first turn `|Δcarry| ≥ max(8, 0.10 × scale)`.

Punch rows join the opposing seat’s exact prior-turn history:

```text
punch.opp_carry = seat_history[turn].attack_pool_recruit_start
```

Ghost / bye rows are skipped (no live loser). Live history-link closes
on every punch row.

```text
applied        = _hero_damage          unchanged
3E carry term  = unpaired punch Δcarry −196
paired uncond  = 9.1% of −196
selection      = 90.9% of −196
```

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#51](https://github.com/aferna6-cell/Replay/pull/51). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

3893 paired seats. 3859 later appear as punch-opp; 3752 of those against
a T1 winner-start body. Unconditional T7 carry **35.82 vs 35.87**. T10
**433 vs 417**. Punch-appearance paired Δ is **−6**, not −196.

Unpaired punch-row carry by winner-start tier reproduces the 3E crater:
T1 **798 → 65** (Δ **−734**); T2 **802 → 106**; T3 **801 → 225**. T5
treatment carry is *higher* (1354 → 1470). The #51 mean −196 is a
mixture that over-weights low-tier winner-start bodies in treatment.

| Turn | n pairs | C carry | T carry | paired Δ | share of −196 |
|---|---:|---:|---:|---:|---:|
| T7 | 3893 | 35.82 | 35.87 | +0.05 | 0.000 |
| T8 | 3305 | 92.77 | 91.13 | −1.64 | 0.008 |
| T9 | 2581 | 146.76 | 144.58 | −2.19 | 0.011 |
| T10 | 1819 | 433.11 | 417.02 | −16.09 | 0.082 |
| T11 | 1262 | 723.56 | 699.63 | −23.92 | 0.122 |
| T12 | 813 | 1821.70 | 1746.44 | −75.26 | 0.383 |
| T13 | 464 | 3816.12 | 3685.34 | −130.78 | 0.666 |
| T14 | 215 | 6116.47 | 5819.83 | −296.65 | 1.511 |

Late-turn paired Δ grows, but those cells are a shrinking both-alive
survivorship slice (treatment games end earlier). Evaluated at the
punch-row appearance turn — the #51 sampling frame — paired Δ stays
~0. Per-pair material separation is common (76% of pairs eventually
trip `|Δ| ≥ max(8, 10%)`) but **19%** separate before the punch turn
and the *mean* paired gap at appearance is only −6.

## Unpaired punch-row carry by winner-start tier

| Winner-start tier | n C / n T | C carry | T carry | Δ |
|---|---:|---:|---:|---:|
| T1 | 11124 / 5038 | 798 | **65** | **−734** |
| T2 | 15245 / 8202 | 802 | 106 | −696 |
| T3 | 13685 / 11306 | 801 | 225 | −576 |
| T4 | 9513 / 13685 | 836 | 633 | −203 |
| T5 | 400 / 5523 | 1354 | 1470 | +116 |
| T6 | 0 / 2679 | — | 2974 | — |

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Live punch → prior-turn seat history | 49960 / 49960 | 46426 / 46426 |
| Ghost / no-loser skipped | 7 | 7 |
| Carry / scale-add / history-gap mismatches | 0 | 0 |
| Snapshot `post = carry + add − loss` | 0 mismatches | 0 mismatches |
| 3D A1 reproduced | 0.4216721428553852 | 0.4216721428553852 |
| 3E carry share of A1 reproduced | 0.7236353954551374 | 0.7236353954551374 |
| Punch-row Δcarry reproduced | −196.33317557443002 | −196.33317557443002 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
#51 carry term = unpaired punch Δcarry −196
        ↓
unconditional paired Δ = −17.8 (9.1%)
paired Δ at punch appearance = −6.1 (3.1%)
+ low-tier winner-start / loss = −2.0 (1.0%)
selection residual = 90.9%
        ↓
selection_outcome_conditioning_dominates
        ↓
next: isolate the punch / winner-start / low-tier
      selection mechanism (who enters the 3D punch
      sample as opposing seat vs which winner-start
      bodies remain), not scaling inputs
      do not rewrite 2Q; do not change scaling constants
      do not preregister a carry/scale correction
      do not burn confirm
```

**Not** a scaling-input audit. **Not** a 2Q rewrite. **Not** a
`_hero_damage` change. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3E / **3F DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3f.py tests/test_phase_3e.py tests/test_phase_3d.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3f          # reused 14200–14699
```

Working tree was clean at contract time (`fbfa93f`, 145.57s). Tracer is
observational.
