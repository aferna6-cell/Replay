# Simulator Fidelity Phase 3I — T1–T3 pairing / who-wins attribution

Date: 2026-09-04 · Status: **`3i_v1` HOLD — `opponent_schedule_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3i/`](../results/sim_fidelity_phase_3i/)

Stacked on PR #54 (`cursor/phase-3h-board-retention-f0c7`, head `de26f9e`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3H DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3H left late T1–T3 punch rows **17924 → 4273** (collapse **13651**) as
leftover pairing / who-wins **52.4%** (7155 rows), elimination **48.0%**,
offer-shift **30.9%**. This hour restricts to those leftover control
rows whose paired treatment seat is alive and still fields ≥1 T1–T3
body, then decomposes the missing treatment low-tier winner-start
punches.

## Verdict

**HOLD.** Route: **`opponent_schedule_dominates`**.

3D A1, the 3E carry share, the unpaired punch Δcarry **−196.333**, the
3F uncond / selection shares, and the 3G mixture / within-cell shares
all reproduce exactly. Punch-row n **54223 / 50116**. History-link
**49960 / 49960** and **46426 / 46426**. Ghost/bye skipped (7 / 7).
Late T1–T3 **17924 → 4273**. 3H leftover **7155** reproduced exactly
(7148 still field T1–T3; 7 unparseable leftover-class rows).

| Component | Punch-row n | Share of leftover |
|---|---:|---:|
| (1) pairing schedule | 5952 | **0.832** |
| (2) same-pairing outcome flip | 668 | 0.093 |
| (3) survivor substitution | 292 | 0.041 |
| (4) residual | 243 | 0.034 |

Pairing schedule clears ~70%. Of those 5952 rows, **5009** are live vs
a different opponent seat and **943** are kind mismatch (ghost/bye vs
live). Same-pairing keys are only **223 / 1323** leftover fights.

Very-late T12–T14 leftover **936** (3H 936): pairing **80.7%**, outcome
**8.5%**, survivor **8.2%**, residual **2.6%**. Same route.

Do **not** apply a scaling correction. Do **not** rewrite 2Q. Do **not**
retune constants. Next hour: audit matchmaking / pairing fidelity.

## Classification (observational)

Exclusive per leftover punch row:

```text
different live opponent / ghost / bye / missing fight → (1) pairing_schedule
same pairing, treatment loses or ties                 → (2) outcome_flip
same pairing, treatment wins, this row not covered
  by a treatment T1–T3 winner-start punch             → (3) survivor_substitution
same pairing, treatment wins, covered                 → (4) residual
```

Each matched fight records opponent identity/seat, both boards’ T1–T3
count/share, tavern tier, recruit raw, abstract-pool raw, total combat
raw, survivor count/tier sum, pre-fight HP, fight outcome/margin,
whether the low-tier body attacks/survives, and next-turn alive state.

```text
applied        = _hero_damage          unchanged
3G mixture     = −196.53 (100.1%)      reproduced
3H leftover    = 7155                  reproduced
leftover split = pairing 83% / flip 9% / survivor 4% / residual 3%
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Punch-row n (3E sample) | 54223 | 50116 |
| 3G mixture reproduced | −196.52943934946725 | −196.52943934946725 |
| 3G mixture share | 1.0009996465165045 | 1.0009996465165045 |
| Live punch → prior-turn seat history | 49960 / 49960 | 46426 / 46426 |
| Ghost / no-loser skipped | 7 | 7 |
| Carry / scale-add / history-gap mismatches | 0 | 0 |
| Late T1–T3 punch n | 17924 | 4273 |
| 3H leftover | 7155 / 7155 | gap 0 |
| Pairing + flip + survivor + residual | 7155 / 7155 | gap 0 |
| 3D A1 reproduced | 0.4216721428553852 | 0.4216721428553852 |
| 3E carry share of A1 reproduced | 0.7236353954551374 | 0.7236353954551374 |
| Paired seats | 3893 | 3893 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

## Decision

```text
#51 carry term = unpaired punch Δcarry −196
        ↓
3G turn × winner-tier mixture = −196.53 (100.1%)   reproduced
        ↓
3H late T1–T3 leftover = 7155 (52.4% of collapse)  reproduced
        ↓
pairing schedule           = 83.2%
  live different opponent  = 5009
  kind mismatch            = 943
same-pairing outcome flip  = 9.3%
survivor substitution      = 4.1%
residual                   = 3.4%
        ↓
opponent_schedule_dominates
        ↓
next: audit matchmaking / pairing fidelity
      (why treatment leftover seats fight a
       different opponent than control at the
       same late turn). Do not apply a scaling
      correction; do not rewrite 2Q; do not
      change constants; do not burn confirm.
```

**Not** a scaling-input audit. **Not** a 2Q rewrite. **Not** a
`_hero_damage` change. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3H / **3I DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3i.py tests/test_phase_3h.py tests/test_phase_3g.py tests/test_phase_3f.py tests/test_phase_3d.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3i          # reused 14200–14699
```

Working tree was clean at contract time (`c1b935e`, 171.85s). Tracer is
observational.
