"""Phase 3A lethal-cause / keyword attribution — observational locks."""

import random

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
)
from hsbg_coach.effects import StartOfCombat
from hsbg_coach.sim import Combatant, simulate_once
from ml.phase_2s_prereg import (
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
)
from ml.phase_3a_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    PHASE_2Y_UNEXPLAINED,
    PHASE_2Z_UNEXPLAINED,
    PHASE_3A_LOBBIES,
    PHASE_3A_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    cleave_bin,
    diagnose_phase_3a,
    ds_bin,
    ordinary_bin,
    poison_bin,
    slot_bin,
    soc_bin,
)
from ml.lethal_cause_diagnostic import (
    LethalCauseTracer,
    reweight_lethal_cause,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3a_v1_default_off():
    assert METHODOLOGY_VERSION == "3a_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert slot_bin(4) == 4
    assert ds_bin({}) == 0
    assert ds_bin({"start_divine_shield": True}) == 1
    assert ds_bin({"n_shield_pops": 1}) == 2
    assert poison_bin({"n_hits_poison": 1}) == 1
    assert poison_bin({}) == 0
    assert cleave_bin({"n_cleave_primary": 1}) == 1
    assert cleave_bin({"n_cleave_secondary": 1}) == 2
    assert soc_bin({"n_soc_hits": 1}) == 1
    assert ordinary_bin({"n_ordinary_attack_hits": 1}) == 1
    assert ordinary_bin({"n_ordinary_counter_hits": 1}) == 2


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3A_SEED == 14200
    assert PHASE_3A_LOBBIES == 500
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


def test_hold_stack_includes_2z_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44)
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_2X_RESIDUAL_POSITION - 1.3719447683362298) < 1e-9
    assert abs(PHASE_2Y_UNEXPLAINED - 0.9456715648873479) < 1e-9
    assert abs(PHASE_2Z_UNEXPLAINED - 0.7993514476549548) < 1e-9
    d = diagnose_phase_3a()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["venomous_equals_poisonous_in_sim"] is True
    assert d["keep_hold_prs"][-1] == 44


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


def test_trace_records_ds_poison_cleave_soc_ordinary():
    r1 = random.Random(42)
    r2 = random.Random(42)
    a = [Combatant(5, 5, name="a", tier=3, divine_shield=True)]
    b = [Combatant(4, 4, name="b", tier=2, poisonous=True)]
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()
    start = list(trace.get("starting_a") or []) + list(trace.get("starting_b") or [])
    assert start
    for row in start:
        assert "n_shield_pops" in row
        assert "shield_pop_cause" in row
        assert "start_divine_shield" in row
        assert "end_divine_shield" in row
        assert "n_hits_poison" in row
        assert "poison_lethal" in row
        assert "n_cleave_primary" in row
        assert "n_cleave_secondary" in row
        assert "n_soc_hits" in row
        assert "n_ordinary_attack_hits" in row
        assert "n_ordinary_counter_hits" in row
        assert "death_cause" in row
    counts = trace.get("event_counts") or {}
    assert counts.get("hits_reconcile") is True
    assert counts.get("shield_pops_reconcile") is True
    assert counts.get("poison_hits_reconcile") is True
    assert counts.get("death_causes_reconcile") is True
    assert counts.get("attacks_reconcile") is True

    shielded = next(r for r in (trace.get("starting_a") or []) if r.get("name") == "a")
    assert shielded.get("start_divine_shield") is True
    assert int(shielded.get("n_shield_pops") or 0) >= 1
    assert shielded.get("ds_before_last_hit") is True
    assert shielded.get("ds_after_last_hit") is False
    assert shielded.get("shield_pop_cause") in ("attack", "counterattack", "poison")


def test_poison_lethal_flag_on_overkill_body():
    trace = {}
    simulate_once(
        [Combatant(1, 1, name="spore", poisonous=True, tier=1)],
        [Combatant(10, 10, name="tank", tier=5)],
        random.Random(0),
        trace=trace,
    )
    tank = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "tank")
    assert tank.get("poison_lethal") is True
    assert tank.get("death_cause") == "poison"
    assert int(tank.get("n_hits_poison") or 0) >= 1


def test_cleave_primary_and_secondary_attribution():
    trace = {}
    simulate_once(
        [Combatant(3, 10, name="cleaver", cleave=True, tier=4)],
        [
            Combatant(1, 2, name="left", tier=1),
            Combatant(1, 2, name="mid", tier=1),
            Combatant(1, 2, name="right", tier=1),
        ],
        random.Random(1),
        trace=trace,
    )
    defs = list(trace.get("starting_b") or [])
    assert sum(int(r.get("n_cleave_primary") or 0) for r in defs) >= 1
    assert sum(int(r.get("n_cleave_secondary") or 0) for r in defs) >= 1
    counts = trace.get("event_counts") or {}
    assert counts.get("cleave_primary_reconcile") is True
    assert counts.get("cleave_secondary_reconcile") is True
    assert int(counts.get("n_cleave_secondary_events") or 0) >= 1


def test_soc_hit_recorded_when_represented():
    soc = StartOfCombat(damage=2, targets=1)
    trace = {}
    simulate_once(
        [Combatant(2, 2, name="whelp", start_of_combat=soc, tier=1)],
        [Combatant(2, 2, name="victim", tier=1)],
        random.Random(5),
        trace=trace,
    )
    victim = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "victim")
    assert int(victim.get("n_soc_hits") or 0) >= 1
    assert victim.get("soc_lethal") is True
    assert victim.get("death_cause") == "start_of_combat"
    counts = trace.get("event_counts") or {}
    assert counts.get("soc_hits_reconcile") is True


def test_ordinary_attack_and_counterattack_lethal():
    trace = {}
    simulate_once(
        [Combatant(5, 1, name="atk", tier=1)],
        [Combatant(1, 1, name="def", tier=1)],
        random.Random(3),
        trace=trace,
    )
    defs = list(trace.get("starting_b") or [])
    atks = list(trace.get("starting_a") or [])
    assert any(r.get("ordinary_lethal") or r.get("death_cause") == "attack" for r in defs)
    assert any(
        int(r.get("n_ordinary_attack_hits") or 0) > 0
        or int(r.get("n_ordinary_counter_hits") or 0) > 0
        for r in atks + defs
    )


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = LethalCauseTracer(0, seed, "obs")
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


def test_combat_hook_is_observational_same_seed():
    """Hooked vs unhooked: placements, HP, length, outcome, and RNG state match."""
    plain = _play(14200, False)
    hooked = _play(14200, True)
    assert hooked["length"] == plain["length"]
    assert hooked["placements"] == plain["placements"]
    assert hooked["hp"] == plain["hp"]
    assert hooked["rng_state"] == plain["rng_state"]
    assert hooked["n_fights"] > 0
    hits = [
        f for f in hooked["tracer"].fights
        if int(f.get("applied_hp_loss") or 0) > 0
    ]
    assert hits
    for f in hits:
        assert f.get("shares_sum_to_pool") is True
        rows = f.get("start_minions") or []
        assert sum(int(r["synthetic_share"]) for r in rows) == int(
            f.get("winner_player_pool") or 0
        )
        for r in rows:
            assert "ds_bin" in r
            assert "poison_bin" in r
            assert "cleave_bin" in r
            assert "soc_bin" in r
            assert "ordinary_bin" in r
            assert "n_shield_pops" in r
            assert "death_cause" in r
        assert f["actual_survivor_count"] == f["survivor_count"]
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )
        counts = f.get("event_counts") or {}
        if counts:
            assert counts.get("attacks_reconcile") is True
            assert counts.get("hits_reconcile") is True
            assert counts.get("deaths_reconcile") is True


def _row(
    tier, recruit, synth, survived, *,
    slot=2, teammate=40,
    start_ds=False, n_pops=0,
    n_poison=0, poison_lethal=False,
    n_cleave_p=0, n_cleave_s=0,
    n_soc=0,
    n_ord_atk=0, n_ord_ctr=0,
):
    combat = recruit + synth
    return {
        "tier": tier,
        "recruit_raw": recruit,
        "synthetic_share": synth,
        "combat_raw": combat,
        "survived": survived,
        "died": not survived,
        "board_slot": slot,
        "slot_bin": slot_bin(slot),
        "golden": False,
        "attacked": False,
        "n_attacks": 0,
        "first_attack_index": None,
        "death_before_first_attack": (not survived),
        "teammate_combat_raw": teammate,
        "board_size": 4,
        "taunt": False,
        "n_targeted": 0,
        "n_targeted_forced": 0,
        "n_targeted_open": 0,
        "was_targeted": False,
        "side_first": False,
        "cursor_wrapped_before_first": False,
        "has_represented_generated_effect": False,
        "has_unsupported_effect": False,
        "n_board_generated_represented": 0,
        "spawned_represented": 0,
        "effect_status": "unregistered",
        "start_divine_shield": start_ds,
        "n_shield_pops": n_pops,
        "n_hits_poison": n_poison,
        "poison_lethal": poison_lethal,
        "n_cleave_primary": n_cleave_p,
        "n_cleave_secondary": n_cleave_s,
        "n_soc_hits": n_soc,
        "n_ordinary_attack_hits": n_ord_atk,
        "n_ordinary_counter_hits": n_ord_ctr,
    }


def _parts_sum(rw):
    return (
        rw["divine_shield"] + rw["poison_venomous"] + rw["cleave"]
        + rw["start_of_combat"] + rw["ordinary_combat"] + rw["still_unexplained"]
    )


def test_reweight_assigns_ds_when_only_shield_mix_shifts():
    control = [_row(4, 10, 10, i == 0, start_ds=False) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, start_ds=True, n_pops=1) for i in range(4)]
    rw = reweight_lethal_cause(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_2z=4.0,
    )
    assert rw["phase_2z_unexplained_hat"] > 0
    assert rw["share_of_leftover_divine_shield"] > 0.70
    assert (rw["share_of_leftover_poison_venomous"] or 0.0) < 0.20
    assert abs(_parts_sum(rw) - rw["phase_2z_unexplained_hat"]) < 1e-9


def test_reweight_assigns_poison_when_only_poison_shifts():
    control = [_row(4, 10, 10, i == 0, n_poison=0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, n_poison=1) for i in range(4)]
    rw = reweight_lethal_cause(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_2z=4.0,
    )
    assert rw["share_of_leftover_poison_venomous"] > 0.70
    assert (rw["share_of_leftover_divine_shield"] or 0.0) < 0.20


def test_reweight_assigns_cleave_when_only_cleave_shifts():
    control = [_row(4, 10, 10, i == 0, n_cleave_s=0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, n_cleave_s=1) for i in range(4)]
    rw = reweight_lethal_cause(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_2z=4.0,
    )
    assert rw["share_of_leftover_cleave"] > 0.70


def test_reweight_assigns_soc_when_only_soc_shifts():
    control = [_row(4, 10, 10, i == 0, n_soc=0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, n_soc=1) for i in range(4)]
    rw = reweight_lethal_cause(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_2z=4.0,
    )
    assert rw["share_of_leftover_start_of_combat"] > 0.70


def test_reweight_assigns_ordinary_when_only_ordinary_shifts():
    control = [_row(4, 10, 10, i == 0, n_ord_atk=0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, n_ord_atk=1) for i in range(4)]
    rw = reweight_lethal_cause(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_2z=4.0,
    )
    assert rw["share_of_leftover_ordinary_combat"] > 0.70


def test_reweight_assigns_unexplained_when_lethal_mix_matches():
    control = [_row(4, 10, 10, i == 0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3) for i in range(4)]
    rw = reweight_lethal_cause(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_2z=4.0,
    )
    assert rw["share_of_leftover_still_unexplained"] > 0.70
    assert (rw["share_of_leftover_divine_shield"] or 0.0) < 0.20
    assert (rw["share_of_leftover_poison_venomous"] or 0.0) < 0.20
    assert (rw["share_of_leftover_cleave"] or 0.0) < 0.20


def test_diagnose_routes_one_joint_and_leftover():
    ds = diagnose_phase_3a({
        "reweighting": {
            "share_of_leftover_divine_shield": 0.80,
            "share_of_leftover_poison_venomous": 0.05,
            "share_of_leftover_cleave": 0.05,
            "share_of_leftover_start_of_combat": 0.05,
            "share_of_leftover_ordinary_combat": 0.04,
            "share_of_leftover_still_unexplained": 0.01,
            "phase_2z_unexplained_hat": 0.799,
        }
    })
    assert ds["primary_finding"] == "divine_shield_dominates"
    assert "divine-shield correction/audit" in ds["recommended_next_step"]

    poi = diagnose_phase_3a({
        "reweighting": {
            "share_of_leftover_divine_shield": 0.05,
            "share_of_leftover_poison_venomous": 0.80,
            "share_of_leftover_cleave": 0.05,
            "share_of_leftover_start_of_combat": 0.04,
            "share_of_leftover_ordinary_combat": 0.04,
            "share_of_leftover_still_unexplained": 0.02,
            "phase_2z_unexplained_hat": 0.799,
        }
    })
    assert poi["primary_finding"] == "poison_venomous_dominates"

    joint = diagnose_phase_3a({
        "reweighting": {
            "share_of_leftover_divine_shield": 0.30,
            "share_of_leftover_poison_venomous": 0.25,
            "share_of_leftover_cleave": 0.15,
            "share_of_leftover_start_of_combat": 0.05,
            "share_of_leftover_ordinary_combat": 0.05,
            "share_of_leftover_still_unexplained": 0.20,
            "phase_2z_unexplained_hat": 0.799,
        }
    })
    assert joint["primary_finding"] == "jointly_explained_rank_largest"
    assert joint["ranked_represented"][0]["component"] == "divine_shield"
    assert "isolate the largest first" in joint["recommended_next_step"]

    leftover = diagnose_phase_3a({
        "reweighting": {
            "share_of_leftover_divine_shield": 0.10,
            "share_of_leftover_poison_venomous": 0.10,
            "share_of_leftover_cleave": 0.10,
            "share_of_leftover_start_of_combat": 0.05,
            "share_of_leftover_ordinary_combat": 0.10,
            "share_of_leftover_still_unexplained": 0.55,
            "phase_2z_unexplained_hat": 0.799,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    assert leftover["ranked_residual"][0]["component"] == "still_unexplained"
    assert "remaining HP" in leftover["smallest_additional_observable"]

    smoke = diagnose_phase_3a(ds, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
