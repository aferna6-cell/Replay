# Simulator Fidelity Phase 3M — earliest same-seat HP divergence attribution

Date: 2026-09-04 · Status: **`3m_v1` HOLD — `mixed_route_to_larger`** ·
Artifacts: [`results/sim_fidelity_phase_3m/`](../results/sim_fidelity_phase_3m/)

Stacked on PR #58 (`cursor/phase-3l-elimination-chain-79fe`, head `45d21df4`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3L DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3L left same-seat earlier elimination as the largest third-party class
at **56.3%** (2082 of 3701), and within that class prior HP explained
**89.2%**. This hour walks each of those 2082 causal seats backward
from the earlier elimination to the first paired pre-combat or
post-combat HP split.

## Verdict

**HOLD.** Route: **`mixed_route_to_larger`** (top = `same_outcome_damage`).

3D A1, the 3E carry share, the unpaired punch Δcarry **−196.333**, the
3F uncond / selection shares, the 3G mixture / within-cell shares, the
3I leftover / pairing-schedule split, the 3J eligibility / RNG/order
split, the 3K timing / HP-gap split, and the 3L chain / prior-HP split
all reproduce exactly.
Punch-row n **54223 / 50116**. History-link **49960 / 49960** and
**46426 / 46426**. Ghost/bye skipped (7 / 7). Late T1–T3 **17924 →
4273**. 3H leftover **7155**. 3I pairing-schedule **5952**. 3J
eligibility **5648**. 3K third-party **3701**. 3L same-seat earlier
**2082** reproduced exactly. Candidate→choice **5952 / 5952**. HP-flow
**10482 / 10482** and **9338 / 9338**. Elimination census **4000 / 4000**
both arms. Every class-(1) row maps to one first-divergence event
(**2082 / 2082**). Inherited HP carry **0**. Unreconciled **0**.

| Component | Punch-row n | Share of class-(1) |
|---|---:|---:|
| (3) same-outcome damage | 1059 | **0.509** |
| (1) prior alive-set / pairing | 597 | 0.287 |
| (2) same-pairing outcome flip | 426 | 0.205 |
| (4) inherited HP carry | 0 | 0.000 |
| (5) missing / unreconciled | 0 | 0.000 |

Same-outcome damage is the largest class but does **not** clear ~70%.
Walking backward dissolves 3L's 89.2% prior-HP pile: every row has an
originating fight in the recorded window. First-divergence turns sit
early:

| First-divergence turn | n | Share of class-(1) |
|---|---:|---:|
| T5 | 1239 | **0.595** |
| T6 | 657 | 0.316 |
| T7 | 167 | 0.080 |
| T8 | 12 | 0.006 |
| T9 | 7 | 0.003 |

Very-late T12–T14 class-(1) **287** (3L earlier 287): same-outcome
damage **57.1%**, pairing **22.3%**, outcome flip **20.6%**, inherited
**0**, unreconciled **0**. Same mixed route; first splits at T5 **183**,
T6 **82**, T7 **22**.

Do **not** apply a scaling correction. Do **not** rewrite 2Q. Do **not**
retune constants. Do **not** change `_hero_damage`. Next hour: pursue
the largest class — matched-state damage fidelity on the first
same-outcome HP split — while keeping pairing (28.7%) and outcome-flip
(20.5%) as ranked residuals.

## Classification (observational)

Exclusive per 3L same-seat earlier punch row. Identify the causal
third-party seat, walk from turn 1 through the earlier elimination,
and stop at the first turn where paired pre-HP or post-HP differs.

```text
missing event / missing arm observation    → (5) unreconciled
pre-combat HP already differs              → (4) inherited_hp_carry
pairing or alive-set differs               → (1) prior_alive_set_or_pairing
no paired fight at the first observed split→ (4) inherited_hp_carry
same pairing, outcomes differ              → (2) same_pairing_outcome_flip
same pairing, same outcome, applied differs→ (3) same_outcome_damage
otherwise                                  → (5) unreconciled
```

Each class-(1) row records the first-divergence turn, opponent /
pairing kind, pre-HP, outcome, applied damage, winner survivor
count / tier sum, tavern tier, board recruit raw, abstract-pool raw,
total combat raw, and the exclusive cause.

```text
applied        = _hero_damage          unchanged
3G mixture     = −196.53 (100.1%)      reproduced
3H leftover    = 7155                  reproduced
3I schedule    = 5952                  reproduced
3J eligibility = 5648                  reproduced
3K third-party = 3701                  reproduced
3L earlier     = 2082                  reproduced
first-split    = damage 50.9% / pairing 28.7% / flip 20.5% / inherit 0 / unrec 0
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
| 3K third-party | 3701 / 3701 | gap 0 |
| 3L same-seat earlier | 2082 / 2082 | gap 0 |
| First-divergence + unreconciled | 2082 / 2082 | gap 0 |
| Row → history → first-divergence | 2082 / 2082 | gap 0 |
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
3K third-party ghost/bye      = 65.5% (3701)
        ↓
3L same-seat earlier          = 56.3% (2082)
        ↓
same-outcome damage           = 50.9% (1059)
prior alive-set / pairing     = 28.7% (597)
same-pairing outcome flip     = 20.5% (426)
inherited HP carry            = 0
unreconciled                  = 0
        ↓
mixed_route_to_larger
  (top = same_outcome_damage)
        ↓
next: matched-state damage
      fidelity on the first
      same-outcome HP split.
      Keep pairing and outcome
      flip as ranked residuals.
      Do not apply a scaling
      correction; do not rewrite 2Q;
      do not change `_hero_damage`;
      do not burn confirm.
```

**Not** a `_hero_damage` change. **Not** a pairing-RNG rewrite.
**Not** a 2Q rewrite. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3L / **3M DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3m.py tests/test_phase_3l.py tests/test_phase_3k.py tests/test_phase_3j.py tests/test_phase_2t.py tests/test_phase_2u.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3m          # reused 14200–14699
```

Working tree was clean at contract time (`96dbde6`, 228.79s). Tracer is
observational.
