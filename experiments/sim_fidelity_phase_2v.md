# Simulator Fidelity Phase 2V — survivor-composition attribution

Date: 2026-09-04 · Status: **`2v_v1` HOLD — `fielded_composition_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_2v/`](../results/sim_fidelity_phase_2v/)

Stacked on PR #39 (`cursor/phase-2u-survivor-tier-e84b`, head `81151da`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 HOLD**. Do **not** merge.
Confirm **11500–11699** untouched. No α / residual / `_hero_damage` / gate /
behavior / default / recruit / scaling changes. Reused consumed 2S/2T/2U DEV
**14200–14699** (no new seeds).

## Verdict

**HOLD.** Route: **`fielded_composition_dominates`**.

The 2U survivor-tier-sum gap **+4.010 / hit** is reproduced exactly.
Most of it is **(A) higher-tier cards being fielded**, not leftover chaff
or tokens.

| Component of +4.010 | Δ tier-sum / hit | Share |
|---|---:|---:|
| (A) fielded composition (Kitagawa on common support; exclusive-support tiers → A) | +2.334 | **0.582** |
| (B) within-tier survival | +1.678 | 0.419 |
| (C) token / generated bodies | −0.003 | −0.001 |
| Residual (A+B+C − observed) | ~0 | — |

Tokens are absent (`token_share = 0` both arms). Generated/reborn share is
~0.2%. Do **not** audit token tier next. Do **not** change `_hero_damage`.
Do not burn confirm.

Next hour: measure whether the **2Q replacement policy over-selects tavern
tier / raw stats** versus real Firestone board composition. Within-tier
survival is a real secondary (42%) — keep it on the stack after the
composition check.

## Classification (observational)

For every decisive T7–T14 hit, winner board-at-start (env minion) and
actual combat survivors are tagged by printed tavern tier, golden,
token/generated (traced body id), recruit-value raw vs combat raw,
tribe/archetype, and board slot.

```text
applied        = _hero_damage          unchanged
counterfactual = tavern + Σ survivor tiers
Δ survivor_tier_sum = (A) + (B) + (C)
```

Same-tier survival is only defined when **both** arms field that tier.
Control never fields T6 (n=0); that exclusive +1.123 is composition, not
survival. Naive Kitagawa would split T6 50/50 and flip the call to B
(0.559). Sequential (control-rate, same exclusive rule) is A 0.43 / B 0.57.
Primary is the exclusive-to-A Kitagawa because (B) is “same-tier cards
surviving more often.”

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#39](https://github.com/aferna6-cell/Replay/pull/39). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

| Metric (T7–T14 hits) | Control | Treatment | Δ |
|---|---:|---:|---:|
| Survivor tier sum | 5.336 | **9.346** | **+4.010** |
| Start-board tier sum | 17.54 | 23.31 | +5.77 |
| High-tier (T4+) start share | 0.216 | **0.478** | +0.261 |
| High-tier (T4+) survivor share | 0.187 | **0.549** | +0.362 |
| Chaff (T1–T2) survivor share | 0.572 | 0.211 | −0.361 |
| Token survivor share | 0 | 0 | 0 |
| Generated survivor share | 0.0026 | 0.0017 | −0.001 |
| Golden survivor share | 0.001 | 0.009 | +0.008 |
| Survivor recruit raw | 7.31 | 9.58 | +2.26 |
| Survivor combat raw | 384 | 297 | −87 |

Treatment fields fewer T1–T2 and more T4–T6. Survivors are even more
high-tier than the start board (T4+ share 0.478 → 0.549). Control
survivors are slightly *more* chaff than the start board (T4+ 0.216 →
0.187).

### Survival P(survive \| printed tier), starting bodies

| Tier | Control P | Treatment P | ΔP |
|---|---:|---:|---:|
| T1 | 0.380 | 0.221 | −0.159 |
| T2 | 0.348 | 0.318 | −0.030 |
| T3 | 0.279 | 0.371 | +0.091 |
| T4 | 0.279 | 0.422 | +0.143 |
| T5 | 0.246 | 0.442 | +0.196 |
| T6 | — (n=0) | 0.452 | exclusive → A |

Treatment T1s die more; T3–T5 live more. That is the (B) term. It does
not clear 0.55 of the pooled +4.010 once T6 fielding is credited to (A).

### Tier contribution to rules-faithful damage (Σ survivor tiers)

| Tier | Control | Treatment | Δ |
|---|---:|---:|---:|
| T1 | 0.56 | 0.16 | −0.40 |
| T2 | 1.45 | 0.78 | −0.67 |
| T3 | 1.63 | 1.89 | +0.27 |
| T4 | 1.62 | **3.53** | **+1.91** |
| T5 | 0.08 | **1.86** | **+1.78** |
| T6 | 0 | **1.12** | **+1.12** |

T4–T6 account for +4.81 of the gap; T1–T2 give back −1.07.

## By turn

T7–T8 are almost pure (B) (shares ~0.99) while start mixes are still
close. From T9 the gap jumps (+4.68, +4.44, +6.45, …) and (A) takes over
as treatment T5/T6 appear. T4+ survivor share goes 0.28 → 0.63 at T9 and
is ≥0.94 from T12. Pooled +4.010 is the late-game composition wave, not
the early survival tilt.

## Reconciliation

| Check | Control | Treatment |
|---|---:|---:|
| Tier-bucket sums = survivor tier sum | 0 mismatches | 0 |
| Survivors ⊆ starting ∪ created bodies | 0 mismatches | 0 |
| Start env n = start combat n | 0 mismatches | 0 |
| A+B+C residual | ~0 | ~0 |

Hooked vs unhooked placements / HP / RNG match. `_hero_damage` unchanged.

## Decision

```text
2Q / 2S fields higher-tier starting boards (T4+ share 0.22 → 0.48;
T6 exclusive to treatment)
        ↓
combat keeps those bodies (T4+ survivor share 0.19 → 0.55)
        ↓
rules-faithful Σ survivor tiers +4.010
        ↓
(A) 0.58  /  (B) 0.42  /  (C) ~0
        ↓
next: 2Q replacement vs Firestone board composition
```

**Not** a `_hero_damage` formula change. **Not** a token-fidelity audit.
**Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S / 2T / 2U / **2V DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2v.py tests/test_phase_2u.py tests/test_phase_2t.py tests/test_sim.py -q
python -m ml.fidelity_phase_2v          # reused 14200–14699
```

Working tree was clean at contract time (`2595f14`, 39.42s). Tracer is
observational.
