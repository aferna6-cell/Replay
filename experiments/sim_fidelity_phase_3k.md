# Simulator Fidelity Phase 3K — elimination-timing attribution

Date: 2026-09-04 · Status: **`3k_v1` HOLD — `mixed_route_to_larger`** ·
Artifacts: [`results/sim_fidelity_phase_3k/`](../results/sim_fidelity_phase_3k/)

Stacked on PR #56 (`cursor/phase-3j-matchmaking-attribution-2f0d`, head `5916ea07`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3J DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3J left pairing-schedule leftover as alive/ghost eligibility **94.9%**
(5648 of 5952). This hour traces T7 through first eligibility
divergence for those rows and splits them by *why* the live set
already differs.

## Verdict

**HOLD.** Route: **`mixed_route_to_larger`** (top = `ghost_bye_third_party`).

3D A1, the 3E carry share, the unpaired punch Δcarry **−196.333**, the
3F uncond / selection shares, the 3G mixture / within-cell shares, the
3I leftover / pairing-schedule split, and the 3J eligibility /
RNG/order split all reproduce exactly.
Punch-row n **54223 / 50116**. History-link **49960 / 49960** and
**46426 / 46426**. Ghost/bye skipped (7 / 7). Late T1–T3 **17924 →
4273**. 3H leftover **7155**. 3I pairing-schedule **5952**. 3J
eligibility **5648** reproduced exactly (4771 / 877). Candidate→choice
**5952 / 5952**. HP-flow **10482 / 10482** and **9338 / 9338**.
Elimination census **4000 / 4000** both arms; every combat elimination
links to a fight with `post_hp <= 0`.

| Component | Punch-row n | Share of eligibility |
|---|---:|---:|
| (1) treatment seat eliminated earlier | 1108 | 0.196 |
| (2) control comparison/opponent eliminated earlier | 839 | 0.149 |
| (3) ghost/bye third-party elimination | 3701 | **0.655** |
| (4) missing / unreconciled | 0 | 0.000 |

Third-party is the largest class but does **not** clear ~70%. First
divergence is early: T8 **2922**, T9 **1655**, T7 **722**, T10 **342**,
T11 **7**. Of the 3701 third-party rows, **3046** are different live
opponent and **655** are kind mismatch.

Of the 1947 (1)+(2) rows, the decisive HP gap is:

| HP-gap subclass | n | Share of (1)+(2) |
|---|---:|---:|
| accumulated prior HP | 1818 | **0.934** |
| current-fight hit / no-hit | 57 | 0.029 |
| current-fight damage magnitude | 72 | 0.037 |
| HP unreconciled | 0 | 0.000 |

Prior HP would dominate *if* the next hour restricted to named
eliminations. It does not override the timing split: third-party is
still the largest first-divergence class.

Very-late T12–T14 eligibility **727** (3J pairing-schedule 755):
third-party **69.6%**, treatment-earlier **23.1%**, control-opponent
**7.3%**, unreconciled **0**. Same mixed route; prior HP **92.3%** of
the 221 named rows.

Do **not** apply a scaling correction. Do **not** rewrite 2Q. Do **not**
retune constants. Next hour: pursue the largest class — trace the
third-party elimination chain one hop upstream of the first T7–T11
eligibility divergence.

## Classification (observational)

Exclusive per 3J eligibility leftover punch row. Walk pairing
decisions from T7 through the leftover turn; classify the *first*
alive-set / ghost-bye divergence.

```text
missing pairing / leftover seat absent     → (4) unreconciled
leftover or leftover's pairing opponent
  died earlier in treatment                → (1) treatment_eliminated_earlier
leftover's comparison / opponent
  died earlier in control                  → (2) control_opponent_eliminated_earlier
only a third-party death, or same named
  live set with ghost/bye flipped          → (3) ghost_bye_third_party
otherwise                                  → (4) unreconciled
```

For (1)/(2), the decisive seat's elimination-turn fight:

```text
pre-combat HP already differs              → accumulated_prior_hp
same pre-HP, one arm hit / one did not     → current_fight_hit
both hit (or both missed), applied differs → current_fight_damage_magnitude
missing fight / same HP and damage         → hp_unreconciled
```

Each leftover seat records T7→first-divergence: pre/post-combat HP,
opponent, outcome, applied damage, survivor count/tier sum, tavern
tier, board recruit raw, abstract-pool raw, total combat raw, and
exact elimination turn.

```text
applied        = _hero_damage          unchanged
3G mixture     = −196.53 (100.1%)      reproduced
3H leftover    = 7155                  reproduced
3I schedule    = 5952                  reproduced
3J eligibility = 5648                  reproduced
timing split   = third-party 65.5% / treat 19.6% / ctrl 14.9% / unrec 0
HP of (1)+(2)  = prior 93.4% / hit 2.9% / mag 3.7% / unrec 0
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
| 3J eligibility | 5648 / 5648 | gap 0 |
| Timing + unreconciled | 5648 / 5648 | gap 0 |
| HP-gap of (1)+(2) | 1947 / 1947 | gap 0 |
| Instrumented HP-flow | 10482 / 10482 | 9338 / 9338 |
| Combat elim ↔ fight `post_hp<=0` | 3479 / 3479 | 3499 / 3499 |
| Lobby census (elim + survived) | 4000 / 4000 | 4000 / 4000 |
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
        ↓
3J eligibility                = 94.9% (5648)
        ↓
third-party ghost/bye         = 65.5% (3701)
treatment eliminated earlier  = 19.6% (1108)
control opponent earlier      = 14.9% (839)
unreconciled                  = 0
        ↓
of (1)+(2)=1947: prior HP 93.4%
        ↓
mixed_route_to_larger
  (top = ghost_bye_third_party)
        ↓
next: trace the third-party
      elimination chain one hop
      upstream of the first T7–T11
      eligibility divergence.
      Do not apply a scaling
      correction; do not rewrite 2Q;
      do not change constants; do not
      burn confirm.
```

**Not** a pairing-RNG rewrite. **Not** a no-repeat rule change.
**Not** a `_hero_damage` change. **Not** a 2Q rewrite. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3J / **3K DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3k.py tests/test_phase_3j.py tests/test_phase_3i.py tests/test_phase_3h.py tests/test_phase_2t.py tests/test_phase_2u.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3k          # reused 14200–14699
```

Working tree was clean at contract time (`dd2c28c`, 193.25s). Tracer is
observational.
