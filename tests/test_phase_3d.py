"""Phase 3D attacker-punch source attribution — observational locks."""

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
from ml.phase_3d_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    PHASE_2Y_UNEXPLAINED,
    PHASE_2Z_UNEXPLAINED,
    PHASE_3A_UNEXPLAINED,
    PHASE_3B_DAMAGE_PER_HIT,
    PHASE_3C_ATTACKER_ATTACK_STRENGTH,
    PHASE_3D_LOBBIES,
    PHASE_3D_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    allocation_concentration_value,
    assert_seed_range_allowed,
    board_pool_value,
    combat_delta_value,
    diagnose_phase_3d,
)
from ml.attack_source_diagnostic import (
    AttackSourceTracer,
    reweight_attack_source,
)
from ml.attacker_punch_diagnostic import reweight_attacker_punch
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3d_v1_default_off():
    assert METHODOLOGY_VERSION == "3d_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert IMPACT_ATTACK_IDENTITY == (
        "impact_attack = start_recruit + start_pool_share + combat_delta"
    )
    assert board_pool_value({}) == 0.0
    assert board_pool_value({"opp_board_pool_attack": 40}) == 40.0
    assert allocation_concentration_value({
        "opp_pool_on_attackers_share": 0.8,
    }) == 0.8
    assert combat_delta_value({"mean_attacker_combat_delta": 2}) == 2.0


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3D_SEED == 14200
    assert PHASE_3D_LOBBIES == 500
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


def test_hold_stack_includes_3c_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_2X_RESIDUAL_POSITION - 1.3719447683362298) < 1e-9
    assert abs(PHASE_2Y_UNEXPLAINED - 0.9456715648873479) < 1e-9
    assert abs(PHASE_2Z_UNEXPLAINED - 0.7993514476549548) < 1e-9
    assert abs(PHASE_3A_UNEXPLAINED - 0.8275878344476644) < 1e-9
    assert abs(PHASE_3B_DAMAGE_PER_HIT - 0.9385531501941458) < 1e-9
    assert abs(PHASE_3C_ATTACKER_ATTACK_STRENGTH - 0.5120447786800975) < 1e-9
    d = diagnose_phase_3d()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["keep_hold_prs"][-1] == 47
    assert d["impact_attack_identity"] == IMPACT_ATTACK_IDENTITY


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


def test_impact_attack_equals_start_recruit_plus_pool_plus_delta():
    r1 = random.Random(42)
    r2 = random.Random(42)
    a = [Combatant(8, 5, name="a", tier=3, recruit_attack=3)]
    b = [Combatant(4, 8, name="b", tier=2, recruit_attack=4)]
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()
    start = list(trace.get("starting_a") or []) + list(trace.get("starting_b") or [])
    assert start
    n_repr = 0
    for row in start:
        assert row.get("attack_identity_ok") is True
        for ev in row.get("hit_events") or []:
            if not ev.get("damaging"):
                continue
            impact = int(ev["attacker_attack"])
            recruit = int(ev["attacker_start_recruit_attack"])
            pool = int(ev["attacker_start_pool_attack"])
            delta = int(ev["attacker_combat_delta"])
            assert impact == recruit + pool + delta
            if ev.get("attacker_start_represented"):
                n_repr += 1
                assert ev.get("attacker_attack_identity_ok") is True
    assert n_repr >= 1
    counts = trace.get("event_counts") or {}
    assert counts.get("attack_identity_reconcile") is True
    assert counts.get("ordinary_hp_loss_reconcile") is True
    smasher = next(r for r in (trace.get("starting_a") or []) if r.get("name") == "a")
    assert smasher.get("start_attack") == 8
    assert smasher.get("start_pool_attack") == 5
    loser = trace.get("starting_loser") or []
    assert loser


def test_opposing_board_pool_and_rank_are_recorded():
    trace = {}
    simulate_once(
        [
            Combatant(12, 4, name="big", tier=4, recruit_attack=2, board_slot=0),
            Combatant(3, 3, name="small", tier=1, recruit_attack=3, board_slot=1),
        ],
        [Combatant(1, 20, name="tank", tier=2, recruit_attack=1)],
        random.Random(3),
        tier_a=4, tier_b=2,
        trace=trace,
    )
    winner_side = trace.get("winner_side")
    assert winner_side in ("a", "b")
    tank = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "tank")
    damaging = [h for h in (tank.get("hit_events") or []) if h.get("damaging")]
    assert damaging
    hit = damaging[0]
    assert hit["attacker_attack"] == (
        int(hit["attacker_start_recruit_attack"])
        + int(hit["attacker_start_pool_attack"])
        + int(hit["attacker_combat_delta"])
    )
    assert hit.get("attacker_board_pool_attack") == (12 - 2) + (3 - 3)
    assert hit.get("attacker_board_recruit_attack") == 2 + 3
    assert hit.get("attacker_board_size") == 2
    assert hit.get("attacker_pool_rank") in (1, 2)
    start_w = list(trace.get("starting_winner") or [])
    if start_w and int(trace.get("survivor_count") or 0) > 0:
        # Winner-start rows carry opposing board totals when A won.
        pass
    counts = trace.get("event_counts") or {}
    assert counts.get("attack_identity_reconcile") is True


def test_generated_token_is_not_start_represented():
    from hsbg_coach.effects import Summon
    trace = {}
    simulate_once(
        [Combatant(1, 1, name="host", tier=1, deathrattle=Summon(1, 5, 5),
                   recruit_attack=1)],
        [Combatant(1, 1, name="victim", tier=1, recruit_attack=1)],
        random.Random(11),
        trace=trace,
    )
    created = list(trace.get("created") or [])
    if not created:
        return
    for row in created:
        for ev in row.get("hit_events") or []:
            if ev.get("attacker_origin") in ("token", "reborn"):
                assert ev.get("attacker_start_represented") is False
                impact = int(ev["attacker_attack"])
                recruit = int(ev["attacker_start_recruit_attack"])
                pool = int(ev["attacker_start_pool_attack"])
                delta = int(ev["attacker_combat_delta"])
                assert impact == recruit + pool + delta


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = AttackSourceTracer(0, seed, "obs")
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
            assert "mean_attacker_attack" in r
            assert "mean_attacker_start_recruit" in r
            assert "mean_attacker_start_pool" in r
            assert "mean_attacker_combat_delta" in r
            assert "opp_board_pool_attack" in r
            assert "opp_pool_on_attackers_share" in r
            assert "attack_identity_ok" in r
            assert "ordinary_hp_loss_ok" in r
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
            assert counts.get("ordinary_hp_loss_reconcile") is True
            assert counts.get("attack_identity_reconcile") is True


def _row(
    tier, recruit, synth, survived, *,
    slot=2, teammate=40,
    start_health=10, n_hits=1, n_damaging=1,
    incoming=6, mean_in=None, overkill=0,
    attacker_attack=6, attacker_synth_share=0.2,
    pairing=1.0,
    opp_pool=30, conc=0.5, combat_delta=0.0,
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
        "slot_bin": slot,
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
        "start_divine_shield": False,
        "n_shield_pops": 0,
        "n_hits_poison": 0,
        "poison_lethal": False,
        "n_cleave_primary": 0,
        "n_cleave_secondary": 0,
        "n_soc_hits": 0,
        "n_ordinary_attack_hits": 0,
        "n_ordinary_counter_hits": 0,
        "start_health": start_health,
        "end_health": start_health if survived else 0,
        "n_hits": n_hits,
        "n_damaging_hits": n_damaging,
        "n_punch_hits": n_damaging,
        "cumulative_incoming": incoming,
        "mean_incoming_dmg": mean_in,
        "hp_depletion_margin": float(start_health) / max(float(mean_in), 1.0),
        "overkill_on_death": overkill,
        "mean_attacker_attack": attacker_attack,
        "mean_attacker_recruit_attack": attacker_attack * (1.0 - attacker_synth_share),
        "mean_attacker_synth_share": attacker_synth_share,
        "mean_attacker_start_recruit": attacker_attack * (1.0 - attacker_synth_share),
        "mean_attacker_start_pool": attacker_attack * attacker_synth_share,
        "mean_attacker_combat_delta": combat_delta,
        "mean_relative_slot": 0.0,
        "mean_attacker_first_attack_index": pairing,
        "pairing_order_value": pairing,
        "ordinary_hp_loss_ok": True,
        "attack_identity_ok": True,
        "opp_board_pool_attack": opp_pool,
        "opp_pool_on_attackers_share": conc,
        "opp_board_recruit_attack": 12,
        "opp_board_size": 4,
        "opp_board_mean_tier": 3.0,
    }


def _parts_sum(rw):
    return (
        rw["board_pool_magnitude"] + rw["allocation_concentration"]
        + rw["combat_mutation"] + rw["attacker_attack_strength"]
    )


def test_reweight_assigns_pool_when_only_board_pool_shifts():
    control = [
        _row(4, 10, 10, i == 0, opp_pool=80, conc=0.5, attacker_attack=20)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, opp_pool=5, conc=0.5, attacker_attack=20)
        for i in range(4)
    ]
    rw = reweight_attack_source(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
        observed_attack_strength=4.0,
    )
    assert rw["share_of_a_board_pool_magnitude"] > 0.70
    assert (rw["share_of_a_allocation_concentration"] or 0.0) < 0.20
    assert (rw["share_of_a_combat_mutation"] or 0.0) < 0.20


def test_reweight_assigns_concentration_when_only_share_shifts():
    control = [
        _row(4, 10, 10, i == 0, opp_pool=40, conc=0.95, attacker_attack=10)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, opp_pool=40, conc=0.05, attacker_attack=10)
        for i in range(4)
    ]
    rw = reweight_attack_source(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
        observed_attack_strength=4.0,
    )
    assert rw["share_of_a_allocation_concentration"] > 0.70
    assert (rw["share_of_a_board_pool_magnitude"] or 0.0) < 0.20


def test_reweight_assigns_combat_delta_when_only_mutation_shifts():
    control = [
        _row(4, 10, 10, i == 0, combat_delta=9.0, attacker_attack=10)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, combat_delta=0.0, attacker_attack=10)
        for i in range(4)
    ]
    rw = reweight_attack_source(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
        observed_attack_strength=4.0,
    )
    assert rw["share_of_a_combat_mutation"] > 0.70


def test_reweight_assigns_unexplained_when_source_mix_matches():
    control = [_row(4, 10, 10, i == 0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3) for i in range(4)]
    rw = reweight_attack_source(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
        observed_attack_strength=4.0,
    )
    assert abs(rw["phase_3c_attacker_attack_strength_hat"] or 0.0) < 0.20
    assert (rw["share_of_a_board_pool_magnitude"] or 0.0) < 0.20
    assert (rw["share_of_a_allocation_concentration"] or 0.0) < 0.20
    assert (rw["share_of_a_combat_mutation"] or 0.0) < 0.20
    assert rw["still_unexplained"] > 0.70 * rw["phase_3b_damage_per_hit_hat"]


def test_3c_reweight_still_assigns_attack_strength_on_same_rows():
    control = [
        _row(4, 10, 10, i == 0, attacker_attack=20)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, attacker_attack=2)
        for i in range(4)
    ]
    rw = reweight_attacker_punch(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
    )
    assert rw["share_of_b_attacker_attack_strength"] > 0.70


def test_diagnose_routes_pool_conc_combat_joint_and_leftover():
    pool = diagnose_phase_3d({
        "reweighting": {
            "share_of_a_board_pool_magnitude": 0.80,
            "share_of_a_allocation_concentration": 0.10,
            "share_of_a_combat_mutation": 0.05,
            "share_of_a_still_unexplained": 0.05,
            "phase_3c_attacker_attack_strength_hat": 0.512,
        }
    })
    assert pool["primary_finding"] == "board_pool_magnitude_dominates"
    assert "synthetic strength" in pool["recommended_next_step"]

    conc = diagnose_phase_3d({
        "reweighting": {
            "share_of_a_board_pool_magnitude": 0.05,
            "share_of_a_allocation_concentration": 0.80,
            "share_of_a_combat_mutation": 0.10,
            "share_of_a_still_unexplained": 0.05,
            "phase_3c_attacker_attack_strength_hat": 0.512,
        }
    })
    assert conc["primary_finding"] == "allocation_concentration_dominates"
    assert "allocation fidelity" in conc["recommended_next_step"]

    mut = diagnose_phase_3d({
        "reweighting": {
            "share_of_a_board_pool_magnitude": 0.05,
            "share_of_a_allocation_concentration": 0.10,
            "share_of_a_combat_mutation": 0.80,
            "share_of_a_still_unexplained": 0.05,
            "phase_3c_attacker_attack_strength_hat": 0.512,
        }
    })
    assert mut["primary_finding"] == "combat_mutation_dominates"
    assert "effect fidelity" in mut["recommended_next_step"]

    joint = diagnose_phase_3d({
        "reweighting": {
            "share_of_a_board_pool_magnitude": 0.35,
            "share_of_a_allocation_concentration": 0.25,
            "share_of_a_combat_mutation": 0.15,
            "share_of_a_still_unexplained": 0.25,
            "phase_3c_attacker_attack_strength_hat": 0.512,
        }
    })
    assert joint["primary_finding"] == "jointly_explained_rank_largest"
    assert joint["ranked_represented"][0]["component"] == "board_pool_magnitude"

    leftover = diagnose_phase_3d({
        "reweighting": {
            "share_of_a_board_pool_magnitude": 0.10,
            "share_of_a_allocation_concentration": 0.10,
            "share_of_a_combat_mutation": 0.10,
            "share_of_a_still_unexplained": 0.70,
            "phase_3c_attacker_attack_strength_hat": 0.512,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    assert leftover["ranked_residual"][0]["component"] == "still_unexplained"

    smoke = diagnose_phase_3d(pool, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
