# Simulator Fidelity Phase 3J — matchmaking divergence attribution

Date: 2026-09-04 · Status: **`3j_v1` HOLD — `eligibility_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3j/`](../results/sim_fidelity_phase_3j/)

Stacked on PR #55 (`cursor/phase-3i-pairing-who-wins-596b`, head `5954eefd`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3I DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3I left late leftover pairing / who-wins as pairing schedule **83.2%**
(5952 rows: 5009 different live opponent + 943 kind mismatch). This
hour hooks every T10–T14 pairing decision for those leftover seats and
splits the 5952 rows by *why* the opponent differs.

## Verdict

**HOLD.** Route: **`eligibility_dominates`**.

3D A1, the 3E carry share, the unpaired punch Δcarry **−196.333**, the
3F uncond / selection shares, the 3G mixture / within-cell shares, and
the 3I leftover / pairing-schedule split all reproduce exactly.
Punch-row n **54223 / 50116**. History-link **49960 / 49960** and
**46426 / 46426**. Ghost/bye skipped (7 / 7). Late T1–T3 **17924 →
4273**. 3H leftover **7155**. 3I pairing-schedule **5952** reproduced
exactly (5009 / 943). Candidate→choice **5952 / 5952**.

| Component | Punch-row n | Share of pairing-schedule |
|---|---:|---:|
| (1) alive / ghost eligibility | 5648 | **0.949** |
| (2) history / legal candidates | 0 | 0.000 |
| (3) same-candidate RNG / order | 304 | 0.051 |
| (4) missing / unreconciled | 0 | 0.000 |

Eligibility clears ~70%. Of those 5648 rows, **4771** are live vs a
different opponent because the pre-pair alive set (or ghost/bye
eligibility) already differs, and **877** are kind mismatch under the
same eligibility gap. Same-candidate RNG/order is only **304** (238
different live opponent + 66 kind mismatch). History/legal-candidate
divergence is **0**: the pairing algorithm shuffles alive seats and
does **not** apply a no-repeat / prior-opponent filter, so legal
candidates are identical whenever the alive set and ghost/bye flags
match.

Very-late T12–T14 pairing-schedule **755** (3I leftover 936):
eligibility **96.3%**, RNG/order **3.7%**, history/legal **0**,
unreconciled **0**. Same route.

Do **not** apply a scaling correction. Do **not** rewrite 2Q. Do **not**
retune constants. Next hour: trace elimination timing upstream of
T10–T14 pairing.

## Classification (observational)

Exclusive per 3I pairing-schedule leftover punch row. The pairing
algorithm is `alive = [p if p.alive]; rng.shuffle(alive); pair adjacent;
odd seat fights ghost if a dead board exists else bye`. Legal
candidates for leftover seat S are the other alive seats, plus
`ghost`/`bye` iff the lobby is odd.

```text
missing decision / leftover seat absent / chosen ∉ candidates
                                              → (4) unreconciled
alive-set or ghost/bye eligibility differs    → (1) eligibility
same eligibility, legal candidate sets differ → (2) history_legal
same candidates, chosen opponent differs      → (3) rng_order
same choice                                   → (4) unreconciled
```

Each T10–T14 pairing decision records the pre-pair alive-seat set,
ghost/bye eligibility, prior live-opponent history (logged, not
applied), legal candidate set, pairing RNG digest / MT head / index,
shuffled order, chosen opponent, and whether control/treatment
candidate sets are identical.

```text
applied        = _hero_damage          unchanged
3G mixture     = −196.53 (100.1%)      reproduced
3H leftover    = 7155                  reproduced
3I schedule    = 5952                  reproduced
schedule split = eligibility 95% / RNG 5% / history 0 / unrec 0
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
| 3I pairing-schedule | 5952 / 5952 | gap 0 |
| Eligibility + history + RNG + unreconciled | 5952 / 5952 | gap 0 |
| Candidate → choice | 5952 / 5952 | ok |
| 3D A1 reproduced | 0.4216721428553852 | 0.4216721428553852 |
| 3E carry share of A1 reproduced | 0.7236353954551374 | 0.7236353954551374 |
| Paired seats | 3893 | 3893 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched. Pairing hook
does not consume extra RNG.

## Decision

```text
#51 carry term = unpaired punch Δcarry −196
        ↓
3G turn × winner-tier mixture = −196.53 (100.1%)   reproduced
        ↓
3H late T1–T3 leftover = 7155 (52.4% of collapse)  reproduced
        ↓
3I pairing schedule           = 83.2% (5952)
  live different opponent     = 5009
  kind mismatch               = 943
        ↓
eligibility                   = 94.9% (5648)
  different opponent          = 4771
  kind mismatch               = 877
same-candidate RNG / order    = 5.1% (304)
history / legal candidates    = 0
unreconciled                  = 0
        ↓
eligibility_dominates
        ↓
next: trace elimination timing
      upstream of T10–T14 pairing
      (why leftover seats already see
       a different alive / ghost-bye
       set). Do not apply a scaling
      correction; do not rewrite 2Q;
      do not change constants; do not
      burn confirm.
```

**Not** a pairing-RNG rewrite. **Not** a no-repeat rule change.
**Not** a scaling-input audit. **Not** a 2Q rewrite. **Not** a
`_hero_damage` change. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3I / **3J DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3j.py tests/test_phase_3i.py tests/test_phase_3h.py tests/test_phase_3g.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3j          # reused 14200–14699
```

Working tree was clean at contract time (`71635b1`, 205.77s). Tracer is
observational.
