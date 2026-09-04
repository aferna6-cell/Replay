# Simulator Fidelity Phase 2Z — targeting / cursor / represented deathrattle

Date: 2026-09-04 · Status: **`2z_v1` measurement (DEV pending this commit)** ·
Artifacts: [`results/sim_fidelity_phase_2z/`](../results/sim_fidelity_phase_2z/)

Stacked on PR #43 (`cursor/phase-2y-slot-attack-order-7573`, head `c836390`).
Keep **#29 / #33–#43 HOLD**. Do **not** merge. Confirm **11500–11699**
untouched. No α / residual / `_hero_damage` / gate / behavior / default /
recruit / scaling / 2Q / damage changes. Reused consumed 2S–2Y DEV
**14200–14699** (no new seeds).

2Y left **+0.946 / hit (68.9% of 2X R)** after holding tavern tier +
recruit/raw + synth share + slot bin + teammate-raw. This hour splits that
leftover into targeting/taunt, attack-cursor/initiative, represented
generated-body/deathrattle, marked unsupported-effect coverage, and still
unexplained.

Unsupported registry stubs (Kaboom Bot, Spawn of N'Zoth) and the marked
approximate Rat Pack deathrattle are **tagged, not fitted**.

## Classification (observational)

For every decisive T7–T14 hit, each winner starting minion is tagged with
the 2Y covariates plus:

* attacker/defender identity and last attacker
* taunt-forced vs open targeting (`n_targeted_forced` / `n_targeted_open`)
* pre/post HP and death cause (`attack`, `counterattack`, `cleave`,
  `poison`, `start_of_combat`, `death_burst`)
* represented generated/deathrattle bodies vs placeholder/approximate
* attack-cursor advance and wrap/reset (`side_first`,
  `cursor_wrapped_before_first`)

```text
applied        = _hero_damage          unchanged
2Y leftover C  = Σ_t t · n̄_t · ΔP(survive|t,r,s,slot,teammates)
ΔP             = targeting + cursor + represented-DR + unsupported + leftover
```

Exclusive T6 stays in 2V A. Tracer does not consume RNG. Play still
appends. No positioning rewrite.

Target bins: never / open-only / taunt-or-forced.
Cursor bins: never reached / side-first no wrap / wrap or second-side.
Gen bins: no faithfully represented DR/generated exposure / has it.
Unsupported bins: clean / placeholder or approximate (marked).

## Decision rule

```text
if targeting/taunt > ~70% of C     → preregister only that correction
if attack-cursor > ~70% of C       → preregister only that correction
if represented gen/DR > ~70% of C  → preregister only that correction
if unsupported coverage > ~70% of C → audit that missing effect class
else rank residual and name the smallest extra observable
     (start-of-combat / DS / poison / cleave lethal cause)
```

Do **not** rewrite 2Q. Do **not** change `_hero_damage`. Do not burn confirm.

## Seeds

| Role | Range | Status |
|---|---|---|
| Confirmation | **11500–11699** | **reserved, untouched** |
| 2S–2Y / **2Z DEV** | **14200–14699** | **reused; no new seeds** |

## Protocol

```bash
python -m pytest tests/test_phase_2z.py tests/test_phase_2y.py tests/test_phase_2x.py tests/test_sim.py -q
python -m ml.fidelity_phase_2z          # reused 14200–14699
```

Tracer is observational. Event counts (attacks, forced vs open targets,
created, deaths) must reconcile on hooked fights. Hooked vs unhooked
RNG / HP / placement / outcome must match.
