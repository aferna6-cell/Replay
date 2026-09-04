"""Phase 3B HP depletion / overkill / hit-count attribution — observational locks."""

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
from ml.phase_3b_prereg import (
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
    PHASE_3A_UNEXPLAINED,
    PHASE_3B_LOBBIES,
    PHASE_3B_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_3b,
    hit_count_bin,
    hp_margin_value,
    overkill_bin,
    slot_bin,
)
from ml.hp_depletion_diagnostic import (
    HpDepletionTracer,
    reweight_hp_depletion,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3b_v1_default_off():
    assert METHODOLOGY_VERSION == "3b_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert slot_bin(4) == 4
    assert hit_count_bin({}) == 0
    assert hit_count_bin({"n_damaging_hits": 1}) == 1
    assert hit_count_bin({"n_damaging_hits": 3}) == 2
    assert overkill_bin({}) == 0
    assert overkill_bin({"overkill_on_death": 3}) == 1
    assert overkill_bin({"overkill_on_death": 8}) == 2
    assert hp_margin_value({"start_health": 10, "mean_incoming_dmg": 2}) == 5.0


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3B_SEED == 14200
    assert PHASE_3B_LOBBIES == 500
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


def test_hold_stack_includes_3a_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_2X_RESIDUAL_POSITION - 1.3719447683362298) < 1e-9
    assert abs(PHASE_2Y_UNEXPLAINED - 0.9456715648873479) < 1e-9
    assert abs(PHASE_2Z_UNEXPLAINED - 0.7993514476549548) < 1e-9
    assert abs(PHASE_3A_UNEXPLAINED - 0.8275878344476644) < 1e-9
    d = diagnose_phase_3b()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["venomous_equals_poisonous_in_sim"] is True
    assert d["keep_hold_prs"][-1] == 45


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


def test_trace_records_hp_flow_and_reconciles():
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
        assert "start_health" in row
        assert "end_health" in row
        assert "cumulative_incoming" in row
        assert "n_damaging_hits" in row
        assert "overkill_on_death" in row
        assert "last_hp_before" in row
        assert "last_hp_after" in row
        assert "incoming_ordinary" in row
        assert "incoming_poison" in row
        assert "hp_flow_ok" is not None
        assert row.get("hp_flow_ok") is True
        events = row.get("hit_events") or []
        if events:
            first = events[0]
            last = events[-1]
            assert first["hp_before"] == row["start_health"] or row.get("start_divine_shield")
            assert last["hp_after"] == row["end_health"]
            assert last["hp_before"] == row["last_hp_before"]
            assert last["incoming"] == row["last_incoming"]
    counts = trace.get("event_counts") or {}
    assert counts.get("hp_flow_reconcile") is True
    assert counts.get("incoming_reconcile") is True
    assert counts.get("applied_reconcile") is True
    assert counts.get("damaging_hits_reconcile") is True
    assert counts.get("overkill_reconcile") is True
    assert counts.get("hits_reconcile") is True


def test_ds_hit_is_incoming_not_damaging():
    trace = {}
    simulate_once(
        [Combatant(3, 5, name="shield", divine_shield=True, tier=2)],
        [Combatant(2, 2, name="chip", tier=1)],
        random.Random(7),
        trace=trace,
    )
    shield = next(r for r in (trace.get("starting_a") or []) if r.get("name") == "shield")
    assert int(shield.get("n_hits") or 0) >= 1
    assert int(shield.get("n_shield_pops") or 0) >= 1
    assert int(shield.get("cumulative_incoming") or 0) >= 1
    pops = [h for h in (shield.get("hit_events") or []) if h.get("hp_before") == h.get("hp_after")]
    assert pops
    assert all(int(h.get("applied") or 0) == 0 for h in pops)
    assert shield.get("hp_flow_ok") is True


def test_poison_overkill_on_high_hp_body():
    trace = {}
    simulate_once(
        [Combatant(1, 1, name="spore", poisonous=True, tier=1)],
        [Combatant(10, 10, name="tank", tier=5)],
        random.Random(0),
        trace=trace,
    )
    tank = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "tank")
    assert tank.get("poison_lethal") is True
    assert int(tank.get("overkill_on_death") or 0) >= 9
    assert int(tank.get("incoming_poison") or 0) >= 1
    assert tank.get("end_health") <= 0
    assert tank.get("hp_flow_ok") is True
    assert int(tank.get("n_damaging_hits") or 0) >= 1


def test_regular_overkill_records_excess_damage():
    trace = {}
    simulate_once(
        [Combatant(8, 2, name="smasher", tier=3)],
        [Combatant(1, 3, name="small", tier=1)],
        random.Random(2),
        trace=trace,
    )
    small = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "small")
    assert small.get("died") or small.get("end_health") <= 0 or small.get("death_cause")
    assert int(small.get("overkill_on_death") or 0) >= 5
    assert int(small.get("last_incoming") or 0) >= 8
    assert small.get("last_hp_before") == 3
    assert small.get("hp_flow_ok") is True


def test_soc_incoming_amount_recorded():
    soc = StartOfCombat(damage=2, targets=1)
    trace = {}
    simulate_once(
        [Combatant(2, 2, name="whelp", start_of_combat=soc, tier=1)],
        [Combatant(2, 2, name="victim", tier=1)],
        random.Random(5),
        trace=trace,
    )
    victim = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "victim")
    assert int(victim.get("incoming_soc") or 0) >= 2
    assert int(victim.get("n_soc_hits") or 0) >= 1
    events = victim.get("hit_events") or []
    soc_hits = [h for h in events if h.get("cause") == "start_of_combat"]
    assert soc_hits
    assert soc_hits[0]["incoming"] == 2
    assert soc_hits[0]["hp_before"] == 2


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = HpDepletionTracer(0, seed, "obs")
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
            assert "hit_count_bin" in r
            assert "overkill_bin" in r
            assert "n_damaging_hits" in r
            assert "overkill_on_death" in r
            assert "start_health" in r
            assert "cumulative_incoming" in r
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
            assert counts.get("hp_flow_reconcile") is True
            assert counts.get("incoming_reconcile") is True
            assert counts.get("damaging_hits_reconcile") is True


def _row(
    tier, recruit, synth, survived, *,
    slot=2, teammate=40,
    start_health=10, n_hits=1, n_damaging=1,
    incoming=6, mean_in=None, overkill=0,
    start_ds=False, n_pops=0,
    n_poison=0, n_cleave_p=0, n_cleave_s=0,
    n_soc=0, n_ord_atk=0, n_ord_ctr=0,
):
    combat = recruit + synth
    if mean_in is None:
        mean_in = float(incoming) / float(n_hits) if n_hits else 0.0
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
        "poison_lethal": False,
        "n_cleave_primary": n_cleave_p,
        "n_cleave_secondary": n_cleave_s,
        "n_soc_hits": n_soc,
        "n_ordinary_attack_hits": n_ord_atk,
        "n_ordinary_counter_hits": n_ord_ctr,
        "start_health": start_health,
        "end_health": start_health if survived else 0,
        "n_hits": n_hits,
        "n_damaging_hits": n_damaging,
        "cumulative_incoming": incoming,
        "mean_incoming_dmg": mean_in,
        "hp_depletion_margin": float(start_health) / max(float(mean_in), 1.0),
        "overkill_on_death": overkill,
    }


def _parts_sum(rw):
    return (
        rw["damaging_hits"] + rw["damage_per_hit"]
        + rw["overkill_threshold"] + rw["still_unexplained"]
    )


def test_reweight_assigns_hits_when_only_exposure_shifts():
    control = [_row(4, 10, 10, i == 0, n_damaging=2, n_hits=2, incoming=8) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, n_damaging=0, n_hits=0, incoming=0) for i in range(4)]
    rw = reweight_hp_depletion(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0,
    )
    assert rw["phase_3a_unexplained_hat"] > 0
    assert rw["share_of_leftover_damaging_hits"] > 0.70
    assert (rw["share_of_leftover_damage_per_hit"] or 0.0) < 0.20
    assert abs(_parts_sum(rw) - rw["phase_3a_unexplained_hat"]) < 1e-9


def test_reweight_assigns_margin_when_only_damage_per_hit_shifts():
    control = [
        _row(4, 10, 10, i == 0, n_damaging=1, n_hits=1, incoming=12, start_health=6)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, n_damaging=1, n_hits=1, incoming=2, start_health=6)
        for i in range(4)
    ]
    rw = reweight_hp_depletion(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0,
    )
    assert rw["share_of_leftover_damage_per_hit"] > 0.70
    assert (rw["share_of_leftover_damaging_hits"] or 0.0) < 0.20


def test_reweight_assigns_overkill_when_only_threshold_shifts():
    control = [
        _row(4, 10, 10, i == 0, n_damaging=1, n_hits=1, incoming=6, overkill=8)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, n_damaging=1, n_hits=1, incoming=6, overkill=0)
        for i in range(4)
    ]
    rw = reweight_hp_depletion(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0,
    )
    assert rw["share_of_leftover_overkill_threshold"] > 0.70


def test_reweight_assigns_unexplained_when_hp_mix_matches():
    control = [_row(4, 10, 10, i == 0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3) for i in range(4)]
    rw = reweight_hp_depletion(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0,
    )
    assert rw["share_of_leftover_still_unexplained"] > 0.70
    assert (rw["share_of_leftover_damaging_hits"] or 0.0) < 0.20
    assert (rw["share_of_leftover_damage_per_hit"] or 0.0) < 0.20
    assert (rw["share_of_leftover_overkill_threshold"] or 0.0) < 0.20


def test_diagnose_routes_one_joint_and_leftover():
    hits = diagnose_phase_3b({
        "reweighting": {
            "share_of_leftover_damaging_hits": 0.80,
            "share_of_leftover_damage_per_hit": 0.10,
            "share_of_leftover_overkill_threshold": 0.05,
            "share_of_leftover_still_unexplained": 0.05,
            "phase_3a_unexplained_hat": 0.828,
        }
    })
    assert hits["primary_finding"] == "damaging_hits_dominates"
    assert "how often winner-start bodies take damaging hits" in hits["recommended_next_step"]

    dmg = diagnose_phase_3b({
        "reweighting": {
            "share_of_leftover_damaging_hits": 0.05,
            "share_of_leftover_damage_per_hit": 0.80,
            "share_of_leftover_overkill_threshold": 0.10,
            "share_of_leftover_still_unexplained": 0.05,
            "phase_3a_unexplained_hat": 0.828,
        }
    })
    assert dmg["primary_finding"] == "damage_per_hit_dominates"

    joint = diagnose_phase_3b({
        "reweighting": {
            "share_of_leftover_damaging_hits": 0.35,
            "share_of_leftover_damage_per_hit": 0.25,
            "share_of_leftover_overkill_threshold": 0.15,
            "share_of_leftover_still_unexplained": 0.25,
            "phase_3a_unexplained_hat": 0.828,
        }
    })
    assert joint["primary_finding"] == "jointly_explained_rank_largest"
    assert joint["ranked_represented"][0]["component"] == "damaging_hits"
    assert "isolate the largest upstream cause first" in joint["recommended_next_step"]

    leftover = diagnose_phase_3b({
        "reweighting": {
            "share_of_leftover_damaging_hits": 0.10,
            "share_of_leftover_damage_per_hit": 0.10,
            "share_of_leftover_overkill_threshold": 0.10,
            "share_of_leftover_still_unexplained": 0.70,
            "phase_3a_unexplained_hat": 0.828,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    assert leftover["ranked_residual"][0]["component"] == "still_unexplained"
    assert "windfury" in leftover["smallest_additional_observable"]

    smoke = diagnose_phase_3b(hits, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
