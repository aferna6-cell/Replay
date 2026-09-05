"""Phase 3O matched-board survivor-mechanic attribution — observational locks."""

import random

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
)
from hsbg_coach.sim import Combatant, simulate_once
from ml.phase_2s_prereg import (
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
)
from ml.phase_3a_prereg import slot_bin
from ml.phase_3e_prereg import PHASE_3D_BOARD_POOL_MAGNITUDE as A1_3E
from ml.phase_3o_prereg import (
    BODY_EVENT_RECONCILE_IDENTITY,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    MECHANIC_COMPONENTS,
    METHODOLOGY_VERSION,
    NESTED_SURVIVAL_IDENTITY,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3N_CLASS3,
    PHASE_3N_CLASS3_T5,
    PHASE_3N_CLASS3_T6,
    PHASE_3N_WITHIN_TIER_B,
    PHASE_3O_LOBBIES,
    PHASE_3O_SEED,
    PRIMARY_TURNS,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_3o,
    keyword_bin,
)
from ml.survivor_mechanic_diagnostic import (
    SurvivorMechanicTracer,
    classify_mechanic_minion,
    reweight_survivor_mechanics,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3o_v1_default_off():
    assert METHODOLOGY_VERSION == "3o_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert PRIMARY_TURNS == (5, 6)
    assert MECHANIC_COMPONENTS == (
        "start_stats",
        "attack_opportunity",
        "target_exposure",
        "represented_keywords",
        "teammate_protection",
        "residual",
    )
    assert "winner starting body" in BODY_EVENT_RECONCILE_IDENTITY
    assert "start_stats +" in NESTED_SURVIVAL_IDENTITY


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3O_SEED == 14200
    assert PHASE_3O_LOBBIES == 500
    assert REUSED_SEED_LO == 14200
    assert REUSED_SEED_HI == 14699
    assert INSTRUMENT_TURNS == tuple(range(7, 15))
    assert (11500, 11699) in FORBIDDEN_RANGES
    assert (13700, 14199) in FORBIDDEN_RANGES
    assert_seed_range_allowed(14200, 500)
    assert_seed_range_allowed(14200, 8)
    try:
        assert_seed_range_allowed(11500, 10)
        raise AssertionError("expected confirm overlap to fail")
    except ValueError:
        pass
    try:
        assert_seed_range_allowed(14700, 1)
        raise AssertionError("expected new seeds to fail")
    except ValueError:
        pass


def test_hold_stack_includes_3n_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - A1_3E) < 1e-12
    assert PHASE_3N_CLASS3 == 1059
    assert PHASE_3N_CLASS3_T5 == 884
    assert PHASE_3N_CLASS3_T6 == 149
    assert abs(PHASE_3N_WITHIN_TIER_B - 0.6883852691218131) < 1e-12
    d = diagnose_phase_3o()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["keep_hold_prs"][-1] == 60
    assert d["history_filters_applied"] is False
    assert d["primary_turns"] == [5, 6]


def test_2s_gates_unchanged():
    assert GATE_REPLACE_RATE_MIN == 0.10
    assert GATE_T10_POST_SCALE_MIN == 0.85
    assert GATE_T10_POST_SCALE_DELTA_FLOOR == -0.10
    assert GATE_GAME_LENGTH_DELTA_FLOOR == -0.50
    assert GATE_MEAN_COMBAT_LOSS_MAX == 20.0


def test_hero_damage_formula_unchanged():
    board = [
        EnvMinion("id-a", "a", 5, 10, 10, [], []),
        EnvMinion("id-b", "b", 3, 8, 8, [], []),
        EnvMinion("id-c", "c", 4, 6, 6, [], []),
    ]
    raw = 4 + 3
    survivors = max(1, abs(raw) - 4)
    avg = (5 + 3 + 4) / 3
    assert BGEnv._hero_damage(raw, 4, board) == 4 + max(1, round(survivors * avg))


def test_trace_records_required_body_fields():
    r1 = random.Random(42)
    r2 = random.Random(42)
    a = [Combatant(5, 5, name="a", tier=3, divine_shield=True, taunt=True)]
    b = [Combatant(4, 4, name="b", tier=2, poisonous=True)]
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()
    start = list(trace.get("starting_a") or []) + list(trace.get("starting_b") or [])
    assert start
    for src in start:
        row = classify_mechanic_minion(src, int(src.get("board_slot") or 0), True)
        assert "tier" in row
        assert "recruit_attack" in row
        assert "recruit_health" in row
        assert "synthetic_share" in row
        assert "combat_raw" in row
        assert "board_slot" in row
        assert "n_attacks" in row
        assert "death_before_first_attack" in row
        assert "n_targeted" in row
        assert "taunt_forced_target" in row
        assert "open_target" in row
        assert "start_divine_shield" in row
        assert "poisonous" in row or "n_hits_poison" in row
        assert "cleave" in row or "n_cleave_primary" in row
        assert "n_soc_hits" in row
        assert "generated" in row
        assert "killer_attack" in row
        assert "killer_tier" in row
        assert "survived" in row
    counts = trace.get("event_counts") or {}
    assert counts.get("attacks_reconcile") is True
    assert counts.get("hits_reconcile") is True
    assert counts.get("deaths_reconcile") is True


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = SurvivorMechanicTracer(0, seed, "obs")
        tracer.attach_to_env(env)
    recs = env.play_scripted(
        [greedy_policy] * env.n_players,
        recruit_tracer=tracer,
    )
    return {
        "env": env,
        "length": max((r["turn"] for r in recs), default=0),
        "placements": [p.placement for p in env.players],
        "hp": [p.hp for p in env.players],
        "n_fights": len(tracer.fights) if tracer else 0,
        "tracer": tracer,
        "rng_state": env.rng.getstate(),
    }


def test_combat_pairing_hooks_are_observational_same_seed():
    """Hooked vs unhooked: placements, HP, length, outcome, and RNG state match."""
    plain = _play(14200, False)
    hooked = _play(14200, True)
    assert hooked["length"] == plain["length"]
    assert hooked["placements"] == plain["placements"]
    assert hooked["hp"] == plain["hp"]
    assert hooked["rng_state"] == plain["rng_state"]
    assert hooked["n_fights"] > 0
    tracer = hooked["tracer"]
    assert tracer.turn_rows
    assert tracer.pairing_decisions
    assert tracer.hp_rows
    assert tracer.eliminations
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    for f in hits:
        rows = f.get("start_minions") or []
        if not rows:
            continue
        assert f.get("shares_sum_to_pool") is True
        for r in rows:
            assert r.get("slot_bin") == slot_bin(r.get("board_slot"))
            assert "start_health" in r
            assert "n_targeted" in r
            assert "ds_bin" in r
            assert "keyword_bin" in r
            assert "killer_attack" in r
            assert keyword_bin(r) in (0, 1)
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )
        counts = f.get("event_counts") or {}
        if counts:
            assert counts.get("attacks_reconcile") is True
            assert f.get("event_counts_ok") is True


def test_unhooked_simulate_once_unchanged():
    r1 = random.Random(42)
    r2 = random.Random(42)
    a = [Combatant(8, 5, name="a", tier=3, recruit_attack=3)]
    b = [Combatant(4, 8, name="b", tier=2, recruit_attack=4)]
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()


def test_diagnose_routes():
    def _bag(start, attack, target, keywords, teammate=0.05, residual=None):
        if residual is None:
            residual = 1.0 - (start + attack + target + keywords + teammate)
        return {
            "primary": {
                "share_of_B_start_stats": start,
                "share_of_B_attack_opportunity": attack,
                "share_of_B_target_exposure": target,
                "share_of_B_represented_keywords": keywords,
                "share_of_B_teammate_protection": teammate,
                "share_of_B_residual": residual,
            },
        }

    stats = diagnose_phase_3o(_bag(0.80, 0.05, 0.05, 0.05))
    assert stats["primary_finding"] == "start_stats_synth_dominates"
    assert "allocate stats" in stats["recommended_next_step"]
    atk = diagnose_phase_3o(_bag(0.05, 0.80, 0.05, 0.05))
    assert atk["primary_finding"] == "attack_opportunity_dominates"
    assert "positioning" in atk["recommended_next_step"]
    tgt = diagnose_phase_3o(_bag(0.05, 0.05, 0.80, 0.05))
    assert tgt["primary_finding"] == "target_exposure_dominates"
    assert "targeting" in tgt["recommended_next_step"]
    key = diagnose_phase_3o(_bag(0.05, 0.05, 0.05, 0.80))
    assert key["primary_finding"] == "represented_keywords_dominates"
    assert "isolate that mechanic" in key["recommended_next_step"]
    leftover = diagnose_phase_3o(_bag(0.10, 0.10, 0.10, 0.10, teammate=0.10))
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    smoke = diagnose_phase_3o(stats, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def _row(
    tier, recruit, synth, survived, *,
    slot=2, teammate=40, start_health=None,
    n_targeted=0, n_forced=0, n_open=0,
    start_ds=False, n_pops=0, n_poison=0,
    n_cleave_p=0, n_soc=0, generated=False,
):
    combat = recruit + synth
    hp = start_health if start_health is not None else max(1, recruit // 2)
    return {
        "tier": tier,
        "recruit_raw": recruit,
        "synthetic_share": synth,
        "combat_raw": combat,
        "start_health": hp,
        "recruit_health": hp,
        "survived": survived,
        "died": not survived,
        "board_slot": slot,
        "slot_bin": slot_bin(slot),
        "n_attacks": 0 if not survived else 1,
        "death_before_first_attack": (not survived),
        "teammate_combat_raw": teammate,
        "board_size": 4,
        "n_targeted": n_targeted,
        "n_targeted_forced": n_forced,
        "n_targeted_open": n_open,
        "taunt_forced_target": n_forced > 0,
        "open_target": n_open > 0 and n_forced == 0,
        "start_divine_shield": start_ds,
        "n_shield_pops": n_pops,
        "n_hits_poison": n_poison,
        "n_cleave_primary": n_cleave_p,
        "n_cleave_secondary": 0,
        "n_soc_hits": n_soc,
        "has_represented_generated_effect": generated,
        "spawned_represented": 1 if generated else 0,
        "n_board_generated_represented": 1 if generated else 0,
        "has_unsupported_effect": False,
        "effect_status": "unregistered",
        "golden": False,
    }


def test_reweight_assigns_start_stats_when_only_synth_shifts():
    control = [_row(3, 10, 0, i == 0) for i in range(4)]
    treatment = [_row(3, 10, 40, i < 3) for i in range(4)]
    rw = reweight_survivor_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
    )
    assert abs(rw["residual_vs_direct_B"]) < 1e-9
    assert rw["share_of_B_start_stats"] > 0.70
    assert (rw["share_of_B_attack_opportunity"] or 0.0) < 0.20


def test_reweight_assigns_slot_when_only_slot_shifts():
    control = [_row(3, 10, 10, i == 0, slot=5) for i in range(4)]
    treatment = [_row(3, 10, 10, i < 3, slot=0) for i in range(4)]
    rw = reweight_survivor_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
    )
    assert abs(rw["residual_vs_direct_B"]) < 1e-9
    assert rw["share_of_B_attack_opportunity"] > 0.70
    assert (rw["share_of_B_start_stats"] or 0.0) < 0.20


def test_reweight_assigns_target_when_only_taunt_shifts():
    control = [_row(3, 10, 10, i == 0, n_open=1) for i in range(4)]
    treatment = [_row(3, 10, 10, i < 3, n_forced=1, n_targeted=1) for i in range(4)]
    rw = reweight_survivor_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
    )
    assert abs(rw["residual_vs_direct_B"]) < 1e-9
    assert rw["share_of_B_target_exposure"] > 0.70


def test_reweight_assigns_keywords_when_only_ds_shifts():
    control = [_row(3, 10, 10, i == 0, start_ds=False) for i in range(4)]
    treatment = [_row(3, 10, 10, i < 3, start_ds=True, n_pops=1) for i in range(4)]
    rw = reweight_survivor_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
    )
    assert abs(rw["residual_vs_direct_B"]) < 1e-9
    assert rw["share_of_B_represented_keywords"] > 0.70
    assert rw["share_of_B_divine_shield"] > 0.70


def test_reweight_parts_sum_to_within_tier_b():
    control = (
        [_row(2, 8, 4, False, slot=3, teammate=20) for _ in range(3)]
        + [_row(4, 16, 8, True, slot=1, teammate=50, start_ds=True)]
    )
    treatment = (
        [_row(2, 8, 12, True, slot=1, teammate=60) for _ in range(3)]
        + [_row(4, 16, 20, True, slot=0, teammate=80, n_forced=1)]
    )
    rw = reweight_survivor_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
    )
    explained = (
        rw["start_stats"] + rw["attack_opportunity"] + rw["target_exposure"]
        + rw["represented_keywords"] + rw["teammate_protection"] + rw["residual"]
    )
    assert abs(explained - rw["within_tier_B"]) < 1e-9
    assert abs(rw["residual_vs_direct_B"]) < 1e-9
