# Simulator Fidelity Phase 3N — first-split matched-state damage attribution

Date: 2026-09-04 · Status: **`3n_v1` HOLD — `within_fight_survival_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_3n/`](../results/sim_fidelity_phase_3n/)

Stacked on PR #59 (`cursor/phase-3m-hp-divergence-11b2`, head `e2f4f95b`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 / #43 / #44 / #45 / #46 / #47 / #50 / #51 / #52 / #53 / #54 / #55 / #56 / #57 / #58 / #59 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–3M DEV **14200–14699** (no new seeds).
PR #49 CI remains separate.

3M left same-outcome damage as the largest first-split class at
**50.9%** (1059 of 2082). This hour restricts to those 1059 paired
fights and attributes the treatment−control applied `_hero_damage` gap
without changing the proxy.

## Verdict

**HOLD.** Route: **`within_fight_survival_dominates`**.

3M class-(1) **2082 / 2082** and class-(3) **1059 / 1059** reproduce.
Every class-(3) row maps to one paired first-split fight. Five-way
terms sum to the applied difference (**residual 0**). Winner tavern
tiers match exactly. All **1059 / 1059** pairs already share the same
pre-fight winner-board printed-tier histogram — the damage gap is
**not** a fielded-composition difference.

| View | Control | Treatment | Δ |
|---|---:|---:|---:|
| Applied / split (`_hero_damage`) | 7.315 | 7.363 | **+0.047** (+50 total) |
| Rules-faithful CF | 6.933 | 7.621 | **+0.688** (+729 total) |
| Proxy − CF | +0.382 | −0.259 | −0.641 (−679 total) |
| Survivor count | 2.086 | 2.126 | +0.040 |
| Survivor tier sum | 3.768 | 4.456 | **+0.688** |
| Winner tavern | 3.165 | 3.165 | 0 |
| Winner start-board tier sum | 8.053 | 8.053 | 0 |

Kitagawa on the matched start boards assigns the entire +0.688
survivor-tier-sum gap to **within-tier survival (B = 1.0)**. Fielded
composition A = 0. Tokens C = 0. Treatment kills more T1 (P(survive)
0.659 → 0.406) and keeps more T3 / T4 (0.375 → 0.612; 0.309 → 0.618).

The tiny applied Δ is what remains after the board-mean proxy
**cancels** most of the CF gap (composition +787 vs proxy −679 of the
+50 applied total). Proxy error is large but **wrong-signed** relative
to the applied gap — it is not the source. Do **not** change
`_hero_damage`. Independently validating 2U is not the next step.

Next hour: isolate the **combat mechanic** that selects higher-tier
survivors on already-matched T5/T6 boards (targeting / punch / DS /
position). Keep pairing (28.7%) and outcome-flip (20.5%) as ranked
3M residuals. Do not burn confirm.

## Classification (observational)

Exclusive per 3M class-(3) first-split punch row. Record both
combat-start boards, winner tavern tier, actual survivor
identities/tiers/count, survivor tier sum, applied `_hero_damage`,
rules-faithful CF, board recruit / abstract-pool / total combat raw,
and combat margin.

```text
applied        = _hero_damage                    unchanged
counterfactual = winner tavern + Σ survivor tiers
Δ applied      = Δ tavern + Δ count + Δ composition|count
               + Δ proxy_error + residual
Δ CF           = Δ tavern + A_fielded + B_survival + C_tokens
```

Sequential count-first Kitagawa on actual survivor tier sum. Source
view standardizes on matched start-board printed-tier mix (2V
Kitagawa; exclusive support → fielded).

```text
applied        = _hero_damage          unchanged
3G mixture     = −196.53 (100.1%)      reproduced
3H leftover    = 7155                  reproduced
3I schedule    = 5952                  reproduced
3J eligibility = 5648                  reproduced
3K third-party = 3701                  reproduced
3L earlier     = 2082                  reproduced
3M class-(3)   = 1059                  reproduced
first-split CF = 100% within-tier survival; fielded A = 0
```

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Punch-row n (3E sample) | 54223 | 50116 |
| 3G mixture reproduced | −196.52943934946725 | −196.52943934946725 |
| Live punch → prior-turn seat history | 49960 / 49960 | 46426 / 46426 |
| Ghost / no-loser skipped | 7 | 7 |
| Carry / scale-add / history-gap mismatches | 0 | 0 |
| 3H leftover | 7155 / 7155 | gap 0 |
| 3I pairing-schedule | 5952 / 5952 | gap 0 |
| 3J eligibility | 5648 / 5648 | gap 0 |
| 3K third-party | 3701 / 3701 | gap 0 |
| 3L same-seat earlier | 2082 / 2082 | gap 0 |
| 3M first-divergence | 2082 / 2082 | gap 0 |
| Class-(3) rows | 1059 / 1059 | gap 0 |
| Row → CF → five-way | 1059 / 1059 | gap 0 |
| Matched start-tier hist + tavern | 1059 / 1059 | — |
| Instrumented HP-flow | 10482 / 10482 | 9338 / 9338 |
| Lobby census (elim + survived) | 4000 / 4000 | 4000 / 4000 |

Hooked vs unhooked placements / HP / RNG / outcome match. `_hero_damage`
unchanged. 2Q not rewritten. Scaling constants untouched.

Class-(3) first-split turns: T5 **884**, T6 **149**, T7 **26**.

Very-late T12–T14 class-(3) **164 / 164**. Applied Δ **−102**. Same
matched-board pattern (tavern 0); not the primary route.

## Decision

```text
#51 carry term = unpaired punch Δcarry −196
        ↓
3G–3M locks reproduced
        ↓
same-outcome first splits = 1059
        ↓
start-board printed-tier mix already matched (1059 / 1059)
winner tavern Δ = 0
        ↓
CF survivor-tier-sum Δ = +0.688 / split
  Kitagawa A fielded     = 0
  Kitagawa B survival    = 1.0
  tokens                 = 0
        ↓
applied Δ = +0.047 / split
  composition|count      = +787 of +50
  proxy error            = −679 of +50 (wrong-signed)
  count                  = −58
  tavern / residual      = 0
        ↓
within_fight_survival_dominates
        ↓
next: isolate the combat mechanic
      that selects T3/T4 survivors
      (and kills T1) on matched
      T5/T6 boards.
      Do not change `_hero_damage`;
      do not rewrite 2Q;
      do not burn confirm.
```

**Not** a `_hero_damage` change. **Not** a 2Q rewrite. **Not** a
recruit / fielded-state walk (boards already match). **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–3M / **3N DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_3n.py tests/test_phase_3m.py tests/test_phase_2u.py tests/test_phase_2v.py tests/test_phase_3b.py tests/test_sim.py tests/test_sim_effects.py -q
python -m ml.fidelity_phase_3n          # reused 14200–14699
```

Working tree was clean at contract time (`743119b`, 220.19s). Tracer is
observational.
