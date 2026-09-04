"""Phase 3C attacker-punch attribution — observational locks."""

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
from ml.phase_3c_prereg import (
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
    PHASE_3B_DAMAGE_PER_HIT,
    PHASE_3C_LOBBIES,
    PHASE_3C_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    attacker_attack_value,
    attacker_synth_share_value,
    diagnose_phase_3c,
    pairing_order_value,
    slot_bin,
)
from ml.attacker_punch_diagnostic import (
    AttackPunchTracer,
    reweight_attacker_punch,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3c_v1_default_off():
    assert METHODOLOGY_VERSION == "3c_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert slot_bin(4) == 4
    assert attacker_attack_value({}) == 0.0
    assert attacker_attack_value({"mean_attacker_attack": 9}) == 9.0
    assert attacker_synth_share_value({"mean_attacker_synth_share": 0.4}) == 0.4
    assert pairing_order_value({}) == -1.0
    assert pairing_order_value({
        "n_damaging_hits": 1, "pairing_order_value": 2.5,
    }) == 2.5


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3C_SEED == 14200
    assert PHASE_3C_LOBBIES == 500
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


def test_hold_stack_includes_3b_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46)
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_2X_RESIDUAL_POSITION - 1.3719447683362298) < 1e-9
    assert abs(PHASE_2Y_UNEXPLAINED - 0.9456715648873479) < 1e-9
    assert abs(PHASE_2Z_UNEXPLAINED - 0.7993514476549548) < 1e-9
    assert abs(PHASE_3A_UNEXPLAINED - 0.8275878344476644) < 1e-9
    assert abs(PHASE_3B_DAMAGE_PER_HIT - 0.9385531501941458) < 1e-9
    d = diagnose_phase_3c()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["venomous_equals_poisonous_in_sim"] is True
    assert d["keep_hold_prs"][-1] == 46
    assert d["ordinary_hp_loss_identity"] == (
        "min(pre_hit_hp, effective_incoming_attack)"
    )


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


def test_ordinary_hp_loss_equals_min_pre_hit_incoming():
    r1 = random.Random(42)
    r2 = random.Random(42)
    a = [Combatant(5, 5, name="a", tier=3, recruit_attack=4)]
    b = [Combatant(4, 8, name="b", tier=2, recruit_attack=3)]
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()
    start = list(trace.get("starting_a") or []) + list(trace.get("starting_b") or [])
    assert start
    n_ordinary = 0
    for row in start:
        assert row.get("ordinary_hp_loss_ok") is True
        for ev in row.get("hit_events") or []:
            kind = ev.get("hit_kind")
            if kind == "ordinary":
                n_ordinary += 1
                assert ev.get("ordinary_ok") is True
                assert ev["applied"] == min(int(ev["hp_before"]), int(ev["incoming"]))
                assert ev.get("attacker_attack") is not None
                assert ev.get("attacker_recruit_attack") is not None
                assert "attacker_survived_swing" in ev
            elif kind == "shield":
                assert int(ev.get("applied") or 0) == 0
            elif kind == "poison":
                assert kind == "poison"
    assert n_ordinary >= 1
    counts = trace.get("event_counts") or {}
    assert counts.get("ordinary_hp_loss_reconcile") is True
    assert counts.get("ordinary_kind_reconcile") is True
    assert counts.get("hp_flow_reconcile") is True


def test_shield_and_poison_are_not_ordinary():
    trace = {}
    simulate_once(
        [Combatant(3, 5, name="shield", divine_shield=True, tier=2)],
        [Combatant(2, 2, name="chip", poisonous=True, tier=1)],
        random.Random(7),
        trace=trace,
    )
    shield = next(r for r in (trace.get("starting_a") or []) if r.get("name") == "shield")
    kinds = [h.get("hit_kind") for h in (shield.get("hit_events") or [])]
    assert "ordinary" not in kinds or all(
        h.get("hit_kind") != "ordinary" or h.get("ordinary_ok")
        for h in (shield.get("hit_events") or [])
    )
    assert any(k in ("shield", "poison") for k in kinds)
    for h in shield.get("hit_events") or []:
        if h.get("hit_kind") == "shield":
            assert int(h.get("applied") or 0) == 0
            assert h.get("ordinary_ok") is None
        if h.get("hit_kind") == "poison":
            assert h.get("ordinary_ok") is None


def test_soc_records_dealer_not_as_ordinary():
    soc = StartOfCombat(damage=2, targets=1)
    trace = {}
    simulate_once(
        [Combatant(2, 2, name="whelp", start_of_combat=soc, tier=1)],
        [Combatant(2, 2, name="victim", tier=1)],
        random.Random(5),
        trace=trace,
    )
    victim = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "victim")
    soc_hits = [
        h for h in (victim.get("hit_events") or [])
        if h.get("cause") == "start_of_combat"
    ]
    assert soc_hits
    assert soc_hits[0]["hit_kind"] == "start_of_combat"
    assert soc_hits[0]["ordinary_ok"] is None
    assert soc_hits[0]["attacker_name"] == "whelp"
    assert soc_hits[0]["attacker_attack"] == 2


def test_overkill_ordinary_still_min_identity():
    trace = {}
    simulate_once(
        [Combatant(8, 2, name="smasher", tier=3, recruit_attack=5)],
        [Combatant(1, 3, name="small", tier=1, recruit_attack=1)],
        random.Random(2),
        trace=trace,
    )
    small = next(r for r in (trace.get("starting_b") or []) if r.get("name") == "small")
    damaging = [h for h in (small.get("hit_events") or []) if h.get("damaging")]
    assert damaging
    hit = damaging[0]
    if hit.get("hit_kind") == "ordinary":
        assert hit["applied"] == min(int(hit["hp_before"]), int(hit["incoming"]))
        assert hit["ordinary_ok"] is True
    assert small.get("mean_attacker_attack") == 8
    assert small.get("mean_attacker_recruit_attack") == 5
    assert abs(small.get("mean_attacker_synth_share") - 3 / 8) < 1e-9


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = AttackPunchTracer(0, seed, "obs")
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
            assert "mean_attacker_recruit_attack" in r
            assert "mean_attacker_synth_share" in r
            assert "pairing_order_value" in r
            assert "ordinary_hp_loss_ok" in r
            assert "p_attacker_survived_swing" in r
            assert "hit_count_bin" in r
            assert "n_damaging_hits" in r
            assert "start_health" in r
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


def _row(
    tier, recruit, synth, survived, *,
    slot=2, teammate=40,
    start_health=10, n_hits=1, n_damaging=1,
    incoming=6, mean_in=None, overkill=0,
    attacker_attack=6, attacker_synth_share=0.2,
    pairing=1.0,
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
        "mean_relative_slot": 0.0,
        "mean_attacker_first_attack_index": pairing,
        "pairing_order_value": pairing,
        "ordinary_hp_loss_ok": True,
    }


def _parts_sum(rw):
    return (
        rw["attacker_attack_strength"] + rw["attacker_synth_composition"]
        + rw["pairing_order"] + rw["still_unexplained"]
    )


def test_reweight_assigns_attack_strength_when_only_punch_shifts():
    control = [
        _row(4, 10, 10, i == 0, attacker_attack=20, pairing=1.0)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, attacker_attack=2, pairing=1.0)
        for i in range(4)
    ]
    rw = reweight_attacker_punch(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
    )
    assert rw["share_of_b_attacker_attack_strength"] > 0.70
    assert (rw["share_of_b_pairing_order"] or 0.0) < 0.20
    assert abs(_parts_sum(rw) - rw["phase_3b_damage_per_hit_hat"]) < 1e-9


def test_reweight_assigns_synth_when_only_composition_shifts():
    control = [
        _row(4, 10, 10, i == 0, attacker_attack=10, attacker_synth_share=0.9)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, attacker_attack=10, attacker_synth_share=0.0)
        for i in range(4)
    ]
    rw = reweight_attacker_punch(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
    )
    assert rw["share_of_b_attacker_synth_composition"] > 0.70
    assert (rw["share_of_b_attacker_attack_strength"] or 0.0) < 0.20


def test_reweight_assigns_pairing_when_only_order_shifts():
    control = [
        _row(4, 10, 10, i == 0, attacker_attack=10, pairing=0.0)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, attacker_attack=10, pairing=8.0)
        for i in range(4)
    ]
    rw = reweight_attacker_punch(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
    )
    assert rw["share_of_b_pairing_order"] > 0.70


def test_reweight_assigns_unexplained_when_punch_mix_matches():
    control = [_row(4, 10, 10, i == 0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3) for i in range(4)]
    rw = reweight_attacker_punch(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
    )
    assert rw["share_of_b_still_unexplained"] > 0.70
    assert (rw["share_of_b_attacker_attack_strength"] or 0.0) < 0.20
    assert (rw["share_of_b_attacker_synth_composition"] or 0.0) < 0.20
    assert (rw["share_of_b_pairing_order"] or 0.0) < 0.20


def test_diagnose_routes_strength_pairing_joint_and_leftover():
    atk = diagnose_phase_3c({
        "reweighting": {
            "share_of_b_attacker_attack_strength": 0.80,
            "share_of_b_attacker_synth_composition": 0.10,
            "share_of_b_pairing_order": 0.05,
            "share_of_b_still_unexplained": 0.05,
            "phase_3b_damage_per_hit_hat": 0.939,
        }
    })
    assert atk["primary_finding"] == "attacker_attack_strength_dominates"
    assert "board-strength / allocation" in atk["recommended_next_step"]

    pair = diagnose_phase_3c({
        "reweighting": {
            "share_of_b_attacker_attack_strength": 0.05,
            "share_of_b_attacker_synth_composition": 0.10,
            "share_of_b_pairing_order": 0.80,
            "share_of_b_still_unexplained": 0.05,
            "phase_3b_damage_per_hit_hat": 0.939,
        }
    })
    assert pair["primary_finding"] == "pairing_order_dominates"
    assert "targeting / initiative" in pair["recommended_next_step"]

    joint = diagnose_phase_3c({
        "reweighting": {
            "share_of_b_attacker_attack_strength": 0.35,
            "share_of_b_attacker_synth_composition": 0.25,
            "share_of_b_pairing_order": 0.15,
            "share_of_b_still_unexplained": 0.25,
            "phase_3b_damage_per_hit_hat": 0.939,
        }
    })
    assert joint["primary_finding"] == "jointly_explained_rank_largest"
    assert joint["ranked_represented"][0]["component"] == "attacker_attack_strength"

    leftover = diagnose_phase_3c({
        "reweighting": {
            "share_of_b_attacker_attack_strength": 0.10,
            "share_of_b_attacker_synth_composition": 0.10,
            "share_of_b_pairing_order": 0.10,
            "share_of_b_still_unexplained": 0.70,
            "phase_3b_damage_per_hit_hat": 0.939,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    assert leftover["ranked_residual"][0]["component"] == "still_unexplained"
    assert "windfury" in leftover["smallest_additional_observable"]

    smoke = diagnose_phase_3c(atk, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
