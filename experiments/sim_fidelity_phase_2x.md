# Simulator Fidelity Phase 2X — synthetic allocation vs within-tier survival

Date: 2026-09-04 · Status: **`2x_v1` HOLD — `position_combat_order_dominates`** ·
Artifacts: [`results/sim_fidelity_phase_2x/`](../results/sim_fidelity_phase_2x/)

Stacked on PR #41 (`cursor/phase-2w-firestone-composition-d7c5`, head `1b18bf1`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 HOLD**. Do **not**
merge. Confirm **11500–11699** untouched. No α / residual / `_hero_damage` /
gate / behavior / default / recruit / scaling / 2Q changes. Reused consumed
2S–2W DEV **14200–14699** (no new seeds).

2W left treatment late-board mix Firestone-like (T5+ 0.695 vs 0.697). This
hour isolates the leftover 2V within-tier survival term **B = +1.678 / hit
(41.9%)** by tavern tier and synthetic abstract-pool share.

## Verdict

**HOLD.** Route: **`position_combat_order_dominates`**.

2V B is reproduced exactly. Holding printed tier + recruit/raw fixed, extra
synthetic allocation explains only **15.7%** of B. Residual survival after
also holding synthetic-share decile is **81.7%** — above the ~70% bar.
Within-tier recruit mix is noise (2.5%). Nested terms add to B (residual
~0).

Do **not** preregister alternative board-level pool rules yet. Next hour:
diagnose positioning / combat-order fidelity (board slot, who attacks
before death, teammate spillover). Do not rewrite 2Q. Do not change
`_hero_damage`. Do not burn confirm.

| Component of 2V B (+1.678) | Δ tier-sum / hit | Share of B |
|---|---:|---:|
| Extra synthetic allocation | +0.264 | 0.157 |
| Residual position / combat-order | **+1.372** | **0.817** |
| Within-tier recruit/raw mix | +0.043 | 0.025 |
| Nested residual | ~0 | — |

Of the standardized (tier + recruit) leftover, synthetic is only 16%.

## Classification (observational)

For every decisive T7–T14 hit, each winner starting minion is tagged by
printed tavern tier, recruit/base raw, synthetic share (`combat − recruit`)
at combat start, combat raw, board slot, golden, survived/died, and whether
it attacked in the combat loop before death.

```text
applied        = _hero_damage          unchanged
2V B           = Σ_t t · n̄_t · ΔP(survive|t)
ΔP(survive|t)  = recruit-mix + synth-mix + residual
```

Exclusive T6 stays in 2V A (composition), not B. Tracer does not consume
RNG. Per-minion synthetic shares sum to the winner's player pool
(2S ON: `abstract_pool`; 2S OFF: on-body implicit pool).

## Paired DEV 14200–14699 (500 lobbies)

Control: 2Q OFF, 2S OFF. Treatment: 2Q recruit-value + 2S pool. Same
seeds as [#41](https://github.com/aferna6-cell/Replay/pull/41). Game
length **15.692 → 13.510** reproduced. Hits 7747 / 7162.

2S does move synthetic off chaff and onto T4–T6. That is real — it is
just not most of B.

| Tier | Ctrl synth | Treat synth | Ctrl % combat | Treat % combat | Ctrl P | Treat P | ΔP |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 717 | 36 | 0.933 | 0.841 | 0.380 | 0.221 | −0.159 |
| T2 | 608 | 67 | 0.893 | 0.864 | 0.348 | 0.318 | −0.030 |
| T3 | 329 | 163 | 0.850 | 0.892 | 0.279 | 0.371 | +0.091 |
| T4 | 211 | **413** | 0.781 | 0.923 | 0.279 | 0.422 | +0.143 |
| T5 | 195 | **925** | 0.812 | 0.973 | 0.246 | 0.442 | +0.196 |
| T6 | — | 1762 | — | 0.990 | — | 0.452 | exclusive → A |

T4+T5 are **+1.456** of B. After standardization those two tiers are still
mostly residual (T4 residual 0.812 of 1.014; T5 0.298 of 0.442).

### Slot / attacked (same starting bodies)

| Tier | Ctrl slot | Treat slot | Ctrl P(attacked) | Treat P(attacked) |
|---|---:|---:|---:|---:|
| T3 | 3.47 | 2.92 | 0.588 | 0.730 |
| T4 | 4.76 | **3.46** | 0.420 | **0.696** |
| T5 | 5.89 | **3.84** | 0.277 | **0.678** |

Treatment high-tier bodies sit earlier and swing more often. Control T5s
are last-in-line (slot 5.89) and usually die without attacking.

Survivor-tier-sum / damage contribution is the same T4–T6 wave as 2V
(+1.91 / +1.78 / +1.12). Tokens remain ~0.

## Reweighting

Nested Kitagawa on pooled within-tier recruit-raw deciles, then
synthetic-share deciles. `n̄_t` matches 2V so B is the same +1.678.

```text
hold tier  (already in 2V B)
    ↓
hold P(recruit-raw decile | tier)     →  2.5% of B
    ↓
hold P(synthetic share | tier, recruit) → 15.7% of B
    ↓
leftover P(survive | tier, recruit, synth) → 81.7% of B
```

Synthetic-as-%-combat also rises at T4–T5 (+0.14 / +0.16), but matching
on the share itself leaves most of the survival gap. Next diagnostic
should split that residual into board-slot / attack-order vs teammate
protection (stronger rest-of-board). Do not retune total scaling to
chase the 16% synth term.

## Reconciliation

| Check | Control | Treatment |
|---|---|---|
| Σ minion synthetic shares = player pool | 0 mismatches | 0 mismatches |
| Tier-bucket sums = survivor tier sum | 0 | 0 |
| Survivors ⊆ starting ∪ created | 0 | 0 |
| 2V B reproduced | 1.6782901818400895 | 1.6782901818400895 |
| Nested mix+synth+resid − B | ~0 | ~0 |

Hooked vs unhooked placements / HP / RNG match. `_hero_damage` unchanged.
Largest-remainder vs painted combat is **not** a conservation check —
residual/ratio still paints on-body; the pool identity is Σ shares.

## Decision

```text
2S reallocates synthetic off T1–T2 (717→36 / 608→67)
        onto T4–T6 (211→413 / 195→925 / T6 1762)
        ↓
that allocation is only 15.7% of leftover 2V B
        ↓
after holding tier + recruit/raw + synth share,
same-tier bodies still survive more on treatment
        ↓
T4/T5 sit earlier (slot 4.76→3.46 / 5.89→3.84)
and attack more (0.42→0.70 / 0.28→0.68)
        ↓
position / combat-order (plus possible teammate spillover)
        ↓
next: diagnose positioning / combat fidelity
      do not rewrite 2Q; do not retune total scaling
```

**Not** a 2Q objective rewrite. **Not** a `_hero_damage` change.
**Not** a pool-allocation prereg. **Not** confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–2W / **2X DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2x.py tests/test_phase_2w.py tests/test_phase_2v.py tests/test_sim.py -q
python -m ml.fidelity_phase_2x          # reused 14200–14699
```

Working tree was clean at contract time (`870fe3f`, 46.6s). Tracer is
observational.
