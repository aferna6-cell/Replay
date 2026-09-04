# Simulator Fidelity Phase 3H — low-tier board-retention lifecycle attribution

Date: 2026-09-04 · Status: **`3h_v1` HOLD — `mixed_route_to_larger`** ·
Artifacts: [`results/sim_fidelity_phase_3h/`](../results/sim_fidelity_phase_3h/)

Stacked on PR #53 (`cursor/phase-3g-punch-selection-af92`, head `5223b20`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3G DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3G left the #51 punch-row Δcarry **−196** as turn × winner-start-tier
mixture (**−196.53, 100.1%**); within matched turn × tier carry ~0.
This hour traces paired T7–T14 boards’ T1–T3 count/share and every
low-tier body’s persist/remove, and classifies the transition away from
T1–T3.

## Verdict

**HOLD.** Route: **`mixed_route_to_larger`**.

3D A1, the 3E carry share, the unpaired punch Δcarry **−196.333**, the
3F uncond / selection shares, and the 3G mixture / within-cell shares
all reproduce exactly. Punch-row n **54223 / 50116**. History-link
**49960 / 49960** and **46426 / 46426**. Ghost/bye skipped (7 / 7).

Late (T10–T14) T1–T3 punch rows **17924 → 4273** (collapse **13651**).
No exclusive lifecycle class clears ~70% of that collapse:

| Component | Punch-row n | Share of collapse |
|---|---:|---:|
| leftover (treatment still has T1–T3) | 7155 | **0.524** |
| (5) alive / elimination | 6550 | **0.480** |
| (3) tavern / offer availability | 4219 | **0.309** |
| (1) full-board 2Q replacement | 0 | 0 |
| (2) open-slot fill | 0 | 0 |
| (4) generated / triple | 0 | 0 |

Punch-row leftover is **not** a body-exit: the treatment counterpart
still fields T1–T3 at that turn, so the missing T1–T3 *winner-start*
punches are pairing / who-wins (the 3G role/alive piece). Among the
five exclusive lifecycle classes the largest is elimination (**48%**),
then offer-shift (**31%**). Neither clears ~70%.

Very-late (T12–T14) leftover shrinks to **11.8%**; elimination **46.8%**
and offer-shift **45.6%** almost tie. Still neither ≥70%.

Do **not** apply a scaling correction. Do **not** rewrite 2Q. Do **not**
retune constants. Rank and pursue the largest piece: leftover pairing /
who-wins first, then elimination mediation, then tavern/offer
availability (the only recorded last-T1–T3-loss class on paired seats).

## Classification (observational)

For every paired `(seed, seat)` the tracer walks T7–T14 and records
T1–T3 count/share at recruit start / pre-scale / combat start, every
low-tier body’s persist/remove, and the first turn combat-start T1–T3
count hits 0. Each exit is exclusive:

```text
triple / generated → (4)
died still holding T1–T3 → (5)
full-board replace + shop T1–T3 = 0 → (3)
full-board replace + shop still offers T1–T3 → (1)
open-slot T4+ play or non-full sell → (2)
else tavern ≥ 4 and shop empty of T1–T3 → (3)
```

Incumbent tier/raw, candidate tier/raw, player tavern tier, gold, shop
offer tiers, and replacement flag are stamped on every recorded event.

```text
applied        = _hero_damage          unchanged
3G mixture     = −196.53 (100.1%)      reproduced
3G within-cell = +0.20 (~0)            reproduced
late T1–T3 n   = 17924 → 4273
collapse       = leftover 52% / elim 48% / offer 31%
```

Exclusive (3) **includes** completed sell→buy→play when the shop had
zero T1–T3 offers — 2Q is the mechanic, availability is why they could
not stay T1–T3. That is why class (1) is 0 on this sample.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#53](https://github.com/aferna6-cell/Replay/pull/53).

**3893** paired seats. All 3893 had T1–T3 at T7 on both arms.

Control **never** loses its last T1–T3 body in T7–T14 (`n_first_loss = 0`;
`p(has T1–T3) = 1.0` on every both-alive turn; mean count **~5.4**
through T14).

Treatment: **1311** seats lose their last T1–T3 (mean first-loss turn
**11.77**). Every recorded last-loss class is **`tavern_offer_shift`**.
**2582** treatment seats still retain ≥1 T1–T3 at last observed
instrumented turn (many of those are early deaths).

| Turn | n both-alive | C mean T1–T3 | T mean T1–T3 | C p(has) | T p(has) |
|---|---:|---:|---:|---:|---:|
| T7 | 3893 | 5.44 | 5.44 | 1.00 | 1.00 |
| T8 | 3305 | 5.63 | 5.64 | 1.00 | 1.00 |
| T9 | 2581 | 5.43 | 3.51 | 1.00 | 0.994 |
| T10 | 1819 | 5.40 | 3.19 | 1.00 | 0.993 |
| T11 | 1262 | 5.41 | 1.59 | 1.00 | 0.805 |
| T12 | 813 | 5.41 | 0.58 | 1.00 | 0.413 |
| T13 | 464 | 5.46 | 0.20 | 1.00 | 0.179 |
| T14 | 215 | 5.45 | 0.06 | 1.00 | 0.060 |

Treatment T1–T3 *count* starts falling at T9 while *presence* stays
near 1 through T10. Presence collapses T11–T14 — the same window 3G
saw late T1–T3 punch mass vanish.

### Very-late (T12–T14) punch-row attribution

| Component | n | Share of 7912 collapse |
|---|---:|---:|
| (5) alive / elimination | 3703 | 0.468 |
| (3) tavern / offer | 3609 | 0.456 |
| leftover (still has T1–T3) | 936 | 0.118 |

Control T12–T14 T1–T3 punch n **8248 → 336**.

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Punch-row n (3E sample) | 54223 | 50116 |
| 3G mixture reproduced | −196.52943934946725 | −196.52943934946725 |
| 3G mixture share | 1.0009996465165045 | 1.0009996465165045 |
| 3G within-cell share | −0.0009996465165047867 | −0.0009996465165047867 |
| Live punch → prior-turn seat history | 49960 / 49960 | 46426 / 46426 |
| Ghost / no-loser skipped | 7 | 7 |
| Carry / scale-add / history-gap mismatches | 0 | 0 |
| Lineage `end = start + add − remove` | 518 / 19235 (p=0.973) | 1182 / 17263 (p=0.932) |
| Late T1–T3 punch-row reconstruction | 17924 / 17924 | gap 0 |
| 3D A1 reproduced | 0.4216721428553852 | 0.4216721428553852 |
| 3E carry share of A1 reproduced | 0.7236353954551374 | 0.7236353954551374 |
| Paired seats | 3893 | 3893 |

Lineage residual is sequential-event chaining (triples / discover
rewards between snapshots), not a flow leak. History-link and 3G
weight identities close.

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
#51 carry term = unpaired punch Δcarry −196
        ↓
3G turn × winner-tier mixture = −196.53 (100.1%)   reproduced
within matched turn × tier     = +0.20 (~0)        reproduced
        ↓
late T1–T3 punch n 17924 → 4273 (collapse 13651)
leftover (still has T1–T3) = 52.4%
alive / elimination        = 48.0%
tavern / offer shift       = 30.9%
2Q replace / open-slot / triple = 0
        ↓
mixed_route_to_larger
        ↓
next: rank and pursue the largest
      1. leftover pairing / who-wins among seats
         that still field T1–T3 (3G role/alive)
      2. damage / elimination mediation
         (treatment games end earlier; dead seats
         cannot produce late T1–T3 punches)
      3. tavern / shop-offer availability
         (every paired last-T1–T3-loss is this
         class; mean first-loss turn 11.77;
         shop had no T1–T3 at exit)
      do not apply a scaling correction
      do not rewrite 2Q; do not change constants
      do not burn confirm
```

**Not** a scaling-input audit. **Not** a 2Q rewrite. **Not** a
`_hero_damage` change. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3G / **3H DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3h.py tests/test_phase_3g.py tests/test_phase_3f.py tests/test_phase_2q.py tests/test_phase_2r.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3h          # reused 14200–14699
```

Working tree was clean at contract time (`999de47`, 182.08s). Tracer is
observational.
