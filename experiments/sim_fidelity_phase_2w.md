# Simulator Fidelity Phase 2W — Firestone final-board vs 2Q selection

Date: 2026-09-04 · Status: **`2w_v1` HOLD — `mixed_or_undershoots_firestone`** ·
Artifacts: [`results/sim_fidelity_phase_2w/`](../results/sim_fidelity_phase_2w/)

Stacked on PR #40 (`cursor/phase-2v-survivor-composition-5ecf`, head `3378c81`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 HOLD**. Do **not** merge.
Confirm **11500–11699** untouched. No α / residual / `_hero_damage` / gate /
behavior / default / recruit / scaling changes. Reused consumed 2S–2V DEV
**14200–14699** (no new seeds).

Firestone `firestone_final_boards.json` is **final-board** data (top-10% MMR,
past-seven, 19 archetypes × 3 examples). Compare to each simulated player's
last/alive **late** board (reached T12+), plus T12–T14 snapshots — not early
turns.

## Verdict

**HOLD.** Route: **`mixed_or_undershoots_firestone`**.

2Q does **not** over-select high tavern tier / printed raw vs Firestone.
Treatment last-alive late boards sit on Firestone's T5+ / T6 / mean-tier
band; printed raw is **below** Firestone (Firestone is 53% golden). The
huge treatment−control T4+ gap (0.23 → 0.97) is control undershooting
Firestone, not treatment overshooting it.

| vs Firestone (last-alive T12+) | Treatment Δ | Material? |
|---|---:|:---|
| T4+ share | **+0.089** | yes (floor 0.08) — leftover T1–T3 deleted |
| Mean printed tier | +0.149 | no (floor 0.25) |
| Mean printed raw | **−1.27** | no (floor +2.0; undershoots) |

Pre-registered overshoot needs **2 of 3**. Only T4+ clears. Match needs
|T4+| ≤ 0.08 and |tier| ≤ 0.25 — T4+ misses by 0.009. Coverage is
adequate for this mix call (join 1.0, 57 boards, 113 unique cards,
96.7% pool-name). Card-frequency overlap is the weak axis (3 examples
per archetype; Jaccard 0.05) — do not route on identity.

Do **not** revise the 2Q recruit-value objective for “too much high
tier.” Next hour: isolate combat/scaling allocation by tavern tier
(2V leftover **B = 42%**). Do not change `_hero_damage`. Do not burn
confirm.

## Firestone reference (boardCount-weighted)

Joined every example minion to `bg_cards.json` (golden `_G` → base id)
and the active pool. Printed/base stats from the KB; Firestone combat
atk/hp are late-game scaled and are **not** the 2Q comparison.

| | Unweighted | Weighted (`boardCount / 3`) |
|---|---:|---:|
| Example boards | 57 | 57 (Σ weight = 4365) |
| Join rate | 1.0 | 1.0 |
| T4+ / T5+ / T6 | 0.852 / 0.630 / 0.321 | **0.881 / 0.697 / 0.343** |
| Mean printed tier | — | **4.89** |
| Mean printed raw | 13 median | **15.08** (golden ×2) |
| Golden share | 0.485 | **0.527** |
| Board size | — | 6.80 |
| T7+ (out of sim) | — | 0.012 |

Weighting reconciles: Σ example weight = Σ `boardCount` (4365). 0
unresolved card-IDs.

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#40](https://github.com/aferna6-cell/Replay/pull/40).

| Last-alive late (T12+) | Control | Treatment | Firestone | t−c | t−FS |
|---|---:|---:|---:|---:|---:|
| n boards / minions | 1701 / 11907 | 1330 / 9310 | 57 / 392 | — | — |
| T4+ share | 0.229 | **0.969** | 0.881 | +0.741 | +0.089 |
| T5+ share | 0.016 | **0.695** | 0.697 | +0.679 | **−0.002** |
| T6 share | 0.000 | **0.377** | 0.343 | +0.377 | +0.034 |
| Mean printed tier | 2.53 | **5.04** | 4.89 | +2.51 | +0.15 |
| Mean printed raw | 7.59 | 13.80 | **15.08** | +6.21 | −1.27 |
| Golden share | 0.001 | 0.022 | **0.527** | +0.021 | −0.505 |
| Board size | 7.0 | 7.0 | 6.80 | 0 | +0.20 |

Treatment histogram (share): T1 0.001 / T2 0.002 / T3 0.027 / T4 0.274 /
T5 0.318 / T6 0.377. Control is still a T1–T4 shop (T5+ = 0.016; T6 ≈ 0).
Firestone still fields ~12% T1–T3; treatment deleted that chaff.

### T12–T14 sensitivity (alive snapshots)

| Turn | Ctrl T4+ | Treat T4+ | Treat mean tier | Treat−FS tier | Treat−FS raw |
|---|---:|---:|---:|---:|---:|
| T12 | 0.229 | 0.924 | 4.75 | **−0.14** | −2.64 |
| T13 | 0.229 | 0.979 | 5.05 | +0.17 | −1.49 |
| T14 | 0.227 | 0.990 | 5.22 | +0.33 | −0.50 |

T12 is still *below* Firestone on mean printed tier. T14 is the only
window that clears the +0.25 tier floor — and printed raw is still
below Firestone. Primary last-alive pool (anyone who reached T12+)
does not overshoot.

### Tribe mix / card-frequency

Tribe shares are in the same ballpark (tribeless ~0.22 both treatment
and Firestone; no single-tribe takeover). Top-20 name Jaccard is **0.00**
control / **0.05** treatment (shared: Crimson Vindicator, Gatekeeper
Amalgam). That is expected with 3 Firestone examples per archetype —
identity is not a valid over-select call. Smallest stronger reference
for *identity* only: Firestone/HSReplay finalComp dump with ≫3 boards
per archetype, still joined to the active pool. Not required for the
tier/raw call.

## 2Q full-board replacements

| | Control | Treatment |
|---|---:|---:|
| Full-board replace rate | 0.013 | **0.283** |
| Accepted sell→buy pairs | 1466 | **25536** |
| Mean Δ printed tier (cand − inc) | +1.72 | +1.64 |
| Mean Δ printed raw | +6.91 | +3.71 |
| Share increase tier / raw | 0.85 / 1.00 | 0.79 / 1.00 |
| Mean candidate printed tier | 4.57 | 4.55 |
| Mean candidate printed raw | 13.51 | 11.40 |
| Cand tier > Firestone mean | 0.62 | 0.54 |
| Cand raw > Firestone mean | 0.34 | 0.14 |
| Disproportionate vs Firestone | no | **no** |

Accepted candidates upgrade the *incumbent* (that is the 2Q rule). They
do **not** land above Firestone-like printed tier/raw. Treatment
replaces ~20× more often; each replace is a milder printed-raw step
because 2Q values recruit stats, so it will sell a scaled T3 for a
printed T5.

## Reconciliation

| Check | |
|---|---|
| Firestone join + unresolved = n minions | 392 + 0 = 392 |
| Σ example weight = Σ boardCount | 4365 = 4365 |
| Last-board KB join mismatches | 0 / 0 |
| 2Q / 2S defaults | OFF outside arm context |
| `_hero_damage` | unchanged |

## Decision

```text
control last-alive T12+ is a T1–T4 shop (T4+ 0.23; T6 ≈ 0)
        ↓
2Q recruit-value unblocks full-board replaces (1.3% → 28%)
        ↓
treatment last-alive T5+ 0.695 = Firestone 0.697;
T6 0.377 ≈ 0.343; mean tier +0.15; printed raw −1.27
        ↓
T4+ +0.089 is deleted T1–T3, not extra T6 raw
        ↓
NOT a 2Q high-tier/raw overshoot vs Firestone
        ↓
next: isolate combat/scaling allocation by tavern tier
      (2V within-tier survival B = 0.42)
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–2V / **2W DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2w.py tests/test_phase_2v.py tests/test_sim.py -q
python -m ml.fidelity_phase_2w          # reused 14200–14699
```

Working tree was clean at contract time (`790692d`, 35.53s). Artifacts
landed at `3c68b5a`. Tracer is observational.
