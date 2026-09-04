# Simulator Fidelity Phase 3L — third-party elimination-chain attribution

Date: 2026-09-04 · Status: **`3l_v1` HOLD — `mixed_route_to_larger`** ·
Artifacts: [`results/sim_fidelity_phase_3l/`](../results/sim_fidelity_phase_3l/)

Stacked on PR #57 (`cursor/phase-3k-elimination-timing-026e`, head `3d3fdaab`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3K DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3K left ghost/bye third-party as the largest first-eligibility class
at **65.5%** (3701 of 5648). This hour walks each of those rows to the
exact third-party seat whose elimination first changes ghost/bye
eligibility and splits that causal event.

## Verdict

**HOLD.** Route: **`mixed_route_to_larger`** (top = `same_seat_earlier_elimination`).

3D A1, the 3E carry share, the unpaired punch Δcarry **−196.333**, the
3F uncond / selection shares, the 3G mixture / within-cell shares, the
3I leftover / pairing-schedule split, the 3J eligibility / RNG/order
split, and the 3K timing / HP-gap split all reproduce exactly.
Punch-row n **54223 / 50116**. History-link **49960 / 49960** and
**46426 / 46426**. Ghost/bye skipped (7 / 7). Late T1–T3 **17924 →
4273**. 3H leftover **7155**. 3I pairing-schedule **5952**. 3J
eligibility **5648**. 3K third-party **3701** reproduced exactly.
Candidate→choice **5952 / 5952**. HP-flow **10482 / 10482** and
**9338 / 9338**. Elimination census **4000 / 4000** both arms. Every
third-party row maps to one causal elimination with `post_hp <= 0`
(**3701 / 3701**).

| Component | Punch-row n | Share of third-party |
|---|---:|---:|
| (1) same-seat earlier elimination | 2082 | **0.563** |
| (2) different-seat / alive-set cascade | 561 | 0.152 |
| (3) same-fight outcome flip | 334 | 0.090 |
| (4) same-outcome damage threshold | 724 | 0.196 |
| (5) missing / unreconciled | 0 | 0.000 |

Same-seat earlier is the largest class but does **not** clear ~70%.
Causal eliminations sit one combat before first pairing divergence:
T7 **1987**, T8 **954**, T6 **567**, T9 **186**, T10 **7**.

Of the 2082 class-(1) rows, the decisive HP gap is:

| HP-gap subclass | n | Share of (1) |
|---|---:|---:|
| accumulated prior HP | 1858 | **0.892** |
| current-fight hit / no-hit | 120 | 0.058 |
| current-fight damage magnitude | 104 | 0.050 |
| HP unreconciled | 0 | 0.000 |

Prior HP would dominate *if* the next hour restricted to same-seat
earlier eliminations. It does not override the chain split: same-seat
earlier is still the largest causal class, not a 70% lock.

Very-late T12–T14 third-party **506** (3K third-party 506): same-seat
earlier **56.7%**, damage threshold **19.4%**, cascade **15.2%**,
outcome flip **8.7%**, unreconciled **0**. Same mixed route; prior HP
**96.5%** of the 287 class-(1) rows.

Do **not** apply a scaling correction. Do **not** rewrite 2Q. Do **not**
retune constants. Next hour: pursue the largest class — trace the
earliest turn that same-seat third-party HP paths separate.

## Classification (observational)

Exclusive per 3K third-party leftover punch row. Identify the
third-party seat whose elimination first changes ghost/bye /
alive-set eligibility, then classify that event.

```text
missing pairing / no causal seat           → (5) unreconciled
different causal seats in the two arms     → (2) different_seat_alive_set_cascade
same seat, same fight pairing:
  outcomes differ                          → (3) same_fight_outcome_flip
  same outcome, lethal / applied differs   → (4) same_outcome_damage_threshold
same seat, earlier elimination
  (different fight or only one arm dead)   → (1) same_seat_earlier_elimination
otherwise                                  → (5) unreconciled
```

For (1), the decisive seat's earlier-elimination-turn fight:

```text
pre-combat HP already differs              → accumulated_prior_hp
same pre-HP, one arm hit / one did not     → current_fight_hit
both hit (or both missed), applied differs → current_fight_damage_magnitude
missing fight / same HP and damage         → hp_unreconciled
```

Each third-party row records the causal seat's elimination turn,
opponent, pre/post-fight HP, prior-turn HP, outcome, applied damage,
winner survivor count/tier sum, tavern tier, board recruit raw,
abstract-pool raw, total combat raw, and whether the same seat is
alive/eliminated at that turn in the paired arm.

```text
applied        = _hero_damage          unchanged
3G mixture     = −196.53 (100.1%)      reproduced
3H leftover    = 7155                  reproduced
3I schedule    = 5952                  reproduced
3J eligibility = 5648                  reproduced
3K third-party = 3701                  reproduced
chain split    = earlier 56.3% / thresh 19.6% / cascade 15.2% / flip 9.0% / unrec 0
HP of (1)      = prior 89.2% / hit 5.8% / mag 5.0% / unrec 0
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
| Chain + unreconciled | 3701 / 3701 | gap 0 |
| HP-gap of (1) | 2082 / 2082 | gap 0 |
| Row → causal elim `post_hp<=0` | 3701 / 3701 | gap 0 |
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
same-seat earlier             = 56.3% (2082)
damage threshold              = 19.6% (724)
different-seat cascade        = 15.2% (561)
same-fight outcome flip       = 9.0% (334)
unreconciled                  = 0
        ↓
of (1)=2082: prior HP 89.2%
        ↓
mixed_route_to_larger
  (top = same_seat_earlier_elimination)
        ↓
next: trace the earliest turn
      that same-seat third-party
      HP paths separate.
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
| 2S–3K / **3L DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3l.py tests/test_phase_3k.py tests/test_phase_3j.py tests/test_phase_3i.py tests/test_phase_2t.py tests/test_phase_2u.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3l          # reused 14200–14699
```

Working tree was clean at contract time (`c26f80f`, 216.45s). Tracer is
observational.
