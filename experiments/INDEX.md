# Replay experiment index

Canonical agent-strength evaluation: **Replay Benchmark v1** (`ml/benchmark.py`,
seeds 10,250,000–10,299,999). Do not reuse TEST for a changed simulator.

## PPO research arc — **CLOSED** (2026-09-01)

| # | Report | Outcome |
|---|---|---|
| Baseline | [`replay_baseline_v1.md`](replay_baseline_v1.md) | Greedy / BC / PPO on Benchmark v1 |
| 1 | [`ppo_degradation_diagnosis_v1.md`](ppo_degradation_diagnosis_v1.md) | PPO instability diagnosed |
| 2 | [`ppo_budget_study_v1.md`](ppo_budget_study_v1.md) | Budget curve |
| 3 | [`ppo_multiseed_replication_v1.md`](ppo_multiseed_replication_v1.md) | Multi-seed replication |
| 4 | [`ppo_policy_anchoring_v1.md`](ppo_policy_anchoring_v1.md) | KL anchoring machinery |
| 4b | [`ppo_matched_ab_v1.md`](ppo_matched_ab_v1.md) | **Anchoring causally stabilizes PPO** |
| 5 | [`ppo_dose_v1.md`](ppo_dose_v1.md) | **β=0.03 best fixed tradeoff; no reliable BC beat** |
| 6 | [`ppo_schedule_v1.md`](ppo_schedule_v1.md) | **STOP — schedule fails all criteria** |

**Conclusion:** Fixed β=0.03 is the best known PPO recipe in Simulator v1, but
PPO does not reliably improve on the BC warm start. No further PPO coefficient
tuning on the current simulator.

## Simulator fidelity — **IN PROGRESS**

| Phase | Report | Status |
|---|---|---|
| 2A | [`sim_fidelity_benchmark_v1.md`](sim_fidelity_benchmark_v1.md) | Measurement baseline (merged) |
| 2B | [`sim_fidelity_phase_2b.md`](sim_fidelity_phase_2b.md) | **Residual scaling — Simulator v1.1 merged** |
| 2C | [`sim_fidelity_phase_2c.md`](sim_fidelity_phase_2c.md) | Composition diagnostic **merged** |
| 2D | [`sim_fidelity_phase_2d.md`](sim_fidelity_phase_2d.md) | Build-aware path/5 A/B **merged (negative)** |
| 2E | [`sim_fidelity_phase_2e.md`](sim_fidelity_phase_2e.md) | Seeded-core conversion stress test |
| 2F | [`sim_fidelity_phase_2f.md`](sim_fidelity_phase_2f.md) | Post-purchase core lifecycle diagnosis **merged** |
| 2G | [`sim_fidelity_phase_2g.md`](sim_fidelity_phase_2g.md) | Seeded-core deployment board-slot stress test **merged** |
| 2H | [`sim_fidelity_phase_2h.md`](sim_fidelity_phase_2h.md) | Tempo board policy **merged (negative, 2h_v3)** |
| 2I | [`sim_fidelity_phase_2i.md`](sim_fidelity_phase_2i.md) | Seeded opportunity decision-margin diagnostic (**merged 2i_v2**) |
| 2J | [`sim_fidelity_phase_2j.md`](sim_fidelity_phase_2j.md) | Board-relative opportunity-cost policy (**merged ACCEPT α=0.5**) |
| 2K | [`sim_fidelity_phase_2k.md`](sim_fidelity_phase_2k.md) | Post-assembly gap (**merged**; 92% never-available) |
| 2L | [`sim_fidelity_phase_2l.md`](sim_fidelity_phase_2l.md) | Availability decomp (`2l_v2`: **63% A3** → approve; Phase 2M pool audit) |
| 2M | [`sim_fidelity_phase_2m.md`](sim_fidelity_phase_2m.md) | Shop/pool audit (`2m_v2`: multi-mismatch; mild draw undershoot) |
| 2N | [`sim_fidelity_phase_2n.md`](sim_fidelity_phase_2n.md) | Active Tavern-pool OK; **HOLD v1.x** (`2n_v3` macro gates fail; confirm 11500 reserved) |
| 2O | [`sim_fidelity_phase_2o.md`](sim_fidelity_phase_2o.md) | Midgame scaling budget (`2o_v1`): **pre-scale hole, post-scale ~OK at T10; recruit Δ≈0**; #29 still HOLD |
| 2P | [`sim_fidelity_phase_2p.md`](sim_fidelity_phase_2p.md) | Replacement-value contamination (`2p_v2`): **confirmed** (golden fix; weakest-golden share 0); #29 HOLD |
| 2Q | [`sim_fidelity_phase_2q.md`](sim_fidelity_phase_2q.md) | Recruit vs combat split (`2q_v1`): **HOLD** — replaces unblocked; post-scale macro collapses; #29 HOLD |
| 2R | [`sim_fidelity_phase_2r.md`](sim_fidelity_phase_2r.md) | Replacement churn (`2r_v1`): **HOLD** — churn/loss explains collapse (frac 0.994); preserve combat on replace; #29/#33 HOLD |
| 2R QA | [`sim_fidelity_phase_2r_qa.md`](sim_fidelity_phase_2r_qa.md) | Independent recompute **0.9938 survives**; sell→buy→play / residual / carry clean |
| 2S | [`sim_fidelity_phase_2s.md`](sim_fidelity_phase_2s.md) | Board-level abstract scaling (`2s_v1`): **HOLD `inconclusive`** — replace 0.298 + T10 1.007 recover; net-board loss **−3.46**; game length Δ **−2.18**; #29/#33/#34/#35/#36 HOLD |
| 2T | [`sim_fidelity_phase_2t.md`](sim_fidelity_phase_2t.md) | Game-length / damage attribution (`2t_v1`): **HOLD `damage_model_fidelity`** — amp share **0.79** of extra dpt; combat outcome 0.18; 2S post-scale still ≥ control; next is damage-model fidelity, not recruit/scaling; #29/#33/#34/#35/#36/#37 HOLD |
| 2U | [`sim_fidelity_phase_2u.md`](sim_fidelity_phase_2u.md) | Survivor-tier damage fidelity (`2u_v1`): **HOLD `isolate_survivor_composition`** — actual-survivor CF **widens** amp Δ to **+3.63** (proxy +2.78); treatment survivors 3.56 vs board 3.33; do not change `_hero_damage` yet; #29/#33–#38 HOLD |
| 2V | [`sim_fidelity_phase_2v.md`](sim_fidelity_phase_2v.md) | Survivor-composition attribution (`2v_v1`): **HOLD `fielded_composition_dominates`** — +4.01 survivor-tier-sum is **(A) 0.58** fielded / **(B) 0.42** within-tier survival / **(C) ~0** tokens; T6 exclusive to treatment; next is 2Q vs Firestone composition; #29/#33–#39 HOLD |
| 2W | [`sim_fidelity_phase_2w.md`](sim_fidelity_phase_2w.md) | Firestone final-board vs 2Q (`2w_v1`): **HOLD `mixed_or_undershoots_firestone`** — treatment last-alive T5+ **0.695 vs 0.697**; T6 0.377 vs 0.343; mean tier +0.15; printed raw −1.27; do **not** rewrite 2Q; leftover is 2V B=0.42; #29/#33–#40 HOLD |
| 2X | [`sim_fidelity_phase_2x.md`](sim_fidelity_phase_2x.md) | Synthetic allocation vs within-tier survival (`2x_v1`): **HOLD `position_combat_order_dominates`** — 2V B +1.678 is **synth 0.16 / residual 0.82 / recruit-mix 0.03**; T4–T5 sit earlier and attack more; next is positioning / combat fidelity; do not retune total scaling; #29/#33–#41 HOLD |
| 2Y | [`sim_fidelity_phase_2y.md`](sim_fidelity_phase_2y.md) | Slot/attack-order vs teammate protection (`2y_v1`): **HOLD `unexplained_combat_mechanics`** — 2X residual +1.372 is **slot −0.24 / teammate 0.55 / leftover 0.69**; neither A nor B ≥70%; next isolates taunt/targeting/deathrattle/attack-cursor; do not rewrite 2Q; #29/#33–#42 HOLD |
| 2Z | [`sim_fidelity_phase_2z.md`](sim_fidelity_phase_2z.md) | Targeting / cursor / represented DR (`2z_v1`): **HOLD `ranked_residual_needs_next_observable`** — 2Y leftover +0.946 is **target 0.05 / cursor 0.03 / gen-DR 0.07 / unsupported 0 / leftover 0.85**; none ≥70%; next records DS/poison/cleave/SOC lethal cause; do not rewrite 2Q; #29/#33–#43 HOLD |
| 3A | [`sim_fidelity_phase_3a.md`](sim_fidelity_phase_3a.md) | Lethal-cause / keyword attribution (`3a_v1`): **HOLD `ranked_residual_needs_next_observable`** — 2Z leftover +0.799 is **DS −0.02 / poison −0.01 / cleave 0 / SOC 0 / ordinary 0 / leftover 1.04**; represented jointly −3.5%; next records per-hit remaining HP / overkill / hit count; do not rewrite 2Q; #29/#33–#44 HOLD |
| 3B | [`sim_fidelity_phase_3b.md`](sim_fidelity_phase_3b.md) | HP depletion / overkill / hit-count (`3b_v1`): **HOLD `damage_per_hit_dominates`** — 3A leftover +0.828 is **hits −0.13 / dmg-per-hit 1.13 / overkill −0.01 / leftover 0.01**; next audits incoming punch vs start HP; do not rewrite 2Q; #29/#33–#45 HOLD |
| 3C | [`sim_fidelity_phase_3c.md`](sim_fidelity_phase_3c.md) | Attacker-punch attribution (`3c_v1`): **HOLD `jointly_explained_rank_largest`** — 3B +0.939 is **atk-strength 0.55 / synth 0.21 / pairing 0.11 / leftover 0.13**; next isolates board-strength / allocation of attacker-attack mix; do not rewrite 2Q; #29/#33–#46 HOLD |
| 3D | [`sim_fidelity_phase_3d.md`](sim_fidelity_phase_3d.md) | Attacker-punch source (`3d_v1`): **HOLD `board_pool_magnitude_dominates`** — 3C A +0.512 is **pool mag 0.82 / concentration 0.51 / combat Δ 0 / leftover 0.16**; next audits why opposing board-level synthetic strength differs across arms; do not rewrite 2Q; #29/#33–#47 HOLD |
| 3E | [`sim_fidelity_phase_3e.md`](sim_fidelity_phase_3e.md) | Board-pool lifecycle (`3e_v1`): **HOLD `carry_history_dominates`** — 3D A1 +0.422 is **carry 0.72 / scale-add 0.56 / replace 0.45 / leftover 0.06**; next traces when the pool divergence first appears; do not rewrite 2Q; #29/#33–#47/#50 HOLD |
| 3F | [`sim_fidelity_phase_3f.md`](sim_fidelity_phase_3f.md) | Carry divergence timing (`3f_v1`): **HOLD `selection_outcome_conditioning_dominates`** — unpaired punch Δcarry −196 is **paired uncond 9.1% / selection 90.9%**; T1 opposing carry 798→65; next isolates punch-sample selection; do not rewrite 2Q; #29/#33–#47/#50/#51 HOLD |
| 3G | [`sim_fidelity_phase_3g.md`](sim_fidelity_phase_3g.md) | Punch-sample selection decomp (`3g_v1`): **HOLD `mixture_role_selection_dominates`** — −196 is **mixture 1.001 / within-cell ~0 / role −0.19 / leftover 0**; T1–T3 rows are early/low-carry opponents, not a within-cell pool deficit; next traces winner-tier×turn matchups; do not rewrite 2Q; #29/#33–#47/#50/#51/#52 HOLD |
| 3H | [`sim_fidelity_phase_3h.md`](sim_fidelity_phase_3h.md) | Low-tier board-retention (`3h_v1`): **HOLD `mixed_route`** — late T1–T3 17924→4273 leftover **7155 (52.4%)**; next decomposes leftover pairing / who-wins; do not rewrite 2Q; #29/#33–#47/#50–#54 HOLD |
| 3I | [`sim_fidelity_phase_3i.md`](sim_fidelity_phase_3i.md) | Pairing / who-wins (`3i_v1`): **HOLD `opponent_schedule_dominates`** — leftover 7155 is pairing-schedule **5952 (83.2%)**; next audits matchmaking; do not rewrite 2Q; #29/#33–#47/#50–#55 HOLD |
| 3J | [`sim_fidelity_phase_3j.md`](sim_fidelity_phase_3j.md) | Matchmaking divergence (`3j_v1`): **HOLD `eligibility_dominates`** — 5952 is eligibility **5648 (94.9%)** / RNG **304 (5.1%)**; next traces elimination timing; do not rewrite 2Q; #29/#33–#47/#50–#56 HOLD |
| 3K | [`sim_fidelity_phase_3k.md`](sim_fidelity_phase_3k.md) | Elimination-timing (`3k_v1`): **HOLD `mixed_route_to_larger`** — 5648 is third-party **65.5%** / treat-earlier **19.6%** / ctrl-opp **14.9%**; (1)+(2) prior-HP **93.4%**; next traces the third-party elim chain one hop upstream; do not rewrite 2Q; #29/#33–#47/#50–#56 HOLD |
| 3L | [`sim_fidelity_phase_3l.md`](sim_fidelity_phase_3l.md) | Third-party elimination-chain (`3l_v1`): **HOLD `mixed_route_to_larger`** — 3701 is same-seat earlier **56.3%** / damage thresh **19.6%** / cascade **15.2%** / flip **9.0%**; of (1) prior-HP **89.2%**; next traces earliest same-seat HP split; do not rewrite 2Q; #29/#33–#47/#50–#57 HOLD |
| 3M | [`sim_fidelity_phase_3m.md`](sim_fidelity_phase_3m.md) | Earliest same-seat HP divergence (`3m_v1`): **HOLD `mixed_route_to_larger`** — 2082 is same-outcome damage **50.9%** / pairing **28.7%** / flip **20.5%** / inherited **0**; first splits T5 **59.5%**; next is matched-state damage fidelity; do not change `_hero_damage`; #29/#33–#47/#50–#58 HOLD |
| 3N | [`sim_fidelity_phase_3n.md`](sim_fidelity_phase_3n.md) | First-split matched-state damage (`3n_v1`): **HOLD `within_fight_survival_dominates`** — 1059 class-(3) already board-matched; CF +0.688 is **100% within-tier survival**; applied +0.047 after proxy cancel; next isolates the combat mechanic on T5/T6 survivors; do not change `_hero_damage`; #29/#33–#47/#50–#59 HOLD |

**Replay Benchmark v2** — define only after Simulator v1.x is frozen; retrain all
agents from scratch. Keep Benchmark v1 TEST untouched.
