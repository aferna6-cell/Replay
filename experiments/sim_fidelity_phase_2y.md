# Simulator Fidelity Phase 2Y — slot / attack-order vs teammate protection

Date: 2026-09-04 · Status: **`2y_v1` (DEV pending)** ·
Artifacts: [`results/sim_fidelity_phase_2y/`](../results/sim_fidelity_phase_2y/)

Stacked on PR #42 (`cursor/phase-2x-synthetic-allocation-e49b`, head `7d88748`).
Keep **#29 / #33 / #34 / #35 / #36 / #37 / #38 / #39 / #40 / #41 / #42 HOLD**.
Do **not** merge. Confirm **11500–11699** untouched. No α / residual /
`_hero_damage` / gate / behavior / default / recruit / scaling / 2Q /
damage changes. Reused consumed 2S–2X DEV **14200–14699** (no new seeds).

2X left **+1.372 / hit (81.7% of 2V B)** after holding tavern tier +
recruit/raw + synthetic share. This hour splits that residual into
board-slot / attack-opportunity vs teammate-protection vs leftover
combat mechanics.

## Protocol

For every decisive T7–T14 hit, each winner starting body is tagged with
printed tier, recruit/raw, synth share, board slot, first-attack index,
n_attacks, death-before-first-attack, taunt / defender-target counts
(if traced), teammate combat-raw excluding self, board size, and
survival.

```text
applied        = _hero_damage          unchanged
2X residual R  = +1.372  (hold tier + recruit + synth)
ΔP(survive|t,r,s) = slot-mix + teammate-mix + leftover
```

Nested Kitagawa: hold recruit-raw decile, then synth-share decile, then
slot bin (0 / 1 / 2 / 3 / 4+), then teammate-raw quintile. Exclusive T6
stays in 2V A. Tracer does not consume RNG.

## Decision rule

| Share of 2X residual R | Route |
|---|---|
| (A) slot / attack opportunity > ~70% | audit recruit/play positioning vs real BG evidence before implementation |
| (B) teammate protection > ~70% | diagnose board-level combat composition / effect fidelity |
| neither | isolate taunt / targeting / deathrattle / attack-cursor before any behavior change |

Do **not** rewrite 2Q. Do **not** change `_hero_damage`. Do **not** burn
confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–2X / **2Y DEV** | **14200–14699** | **reused; no new seeds** |

## Commands

```bash
python -m pytest tests/test_phase_2y.py tests/test_phase_2x.py tests/test_phase_2w.py tests/test_sim.py -q
python -m ml.fidelity_phase_2y          # reused 14200–14699
```

Tracer is observational. Play still appends (`board.append`); no
positioning rewrite.
