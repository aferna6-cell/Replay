"""Phase 3E board-pool lifecycle attribution — observational locks."""

import random

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
    A_SELL0,
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
from ml.phase_3d_prereg import PHASE_3C_ATTACKER_ATTACK_STRENGTH as A3C
from ml.phase_3e_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    IMPACT_ATTACK_IDENTITY,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_3C_ATTACKER_ATTACK_STRENGTH,
    PHASE_3D_BOARD_POOL_MAGNITUDE,
    PHASE_3D_SHARE_BOARD_POOL,
    PHASE_3E_LOBBIES,
    PHASE_3E_SEED,
    POOL_FLOW_IDENTITY,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    board_pool_value,
    carry_pool_value,
    diagnose_phase_3e,
    flow_identity_residual,
    replacement_loss_value,
    scaling_add_value,
)
from ml.attack_source_diagnostic import reweight_attack_source
from ml.pool_lifecycle_diagnostic import (
    PoolLifecycleTracer,
    board_attack_pool,
    reweight_pool_lifecycle,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_3e_v1_default_off():
    assert METHODOLOGY_VERSION == "3e_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert POOL_FLOW_IDENTITY == (
        "post = pre + add - represented_loss_or_transfer"
    )
    assert IMPACT_ATTACK_IDENTITY == (
        "impact_attack = start_recruit + start_pool_share + combat_delta"
    )
    assert carry_pool_value({"opp_carry_attack_pool": 40}) == 40.0
    assert scaling_add_value({"opp_scale_add_attack": 9}) == 9.0
    assert replacement_loss_value({"opp_replace_loss_attack": 3}) == 3.0
    assert board_pool_value({"opp_board_pool_attack": 50}) == 50.0
    row = {
        "opp_attack_pool_post_scale": 20,
        "opp_carry_attack_pool": 10,
        "opp_scale_add_attack": 15,
        "opp_replace_loss_attack": 5,
    }
    assert abs(flow_identity_residual(row)) < 1e-9


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_3E_SEED == 14200
    assert PHASE_3E_LOBBIES == 500
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


def test_hold_stack_includes_3d_and_frozen_alpha():
    assert HOLD_PRS == (
        29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50,
    )
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_3C_ATTACKER_ATTACK_STRENGTH - 0.5120447786800975) < 1e-9
    assert abs(A3C - PHASE_3C_ATTACKER_ATTACK_STRENGTH) < 1e-12
    assert abs(PHASE_3D_BOARD_POOL_MAGNITUDE - 0.4216721428553852) < 1e-9
    assert abs(PHASE_3D_SHARE_BOARD_POOL - 0.8235063814972068) < 1e-9
    d = diagnose_phase_3e()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["no_scaling_constant_change"] is True
    assert d["keep_hold_prs"][-1] == 50
    assert d["pool_flow_identity"] == POOL_FLOW_IDENTITY


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


def test_board_attack_pool_matches_combat_minus_recruit():
    a = EnvMinion("a", "A", 3, 4, 5, [], [])
    a.attack, a.health = 20, 15
    b = EnvMinion("b", "B", 2, 3, 3, [], [])
    assert board_attack_pool([a, b]) == (20 - 4) + (3 - 3)


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = PoolLifecycleTracer(0, seed, "obs")
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


def test_combat_and_scaling_hooks_are_observational_same_seed():
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
    window = [
        r for r in tracer.turn_rows
        if int(r["turn"]) in INSTRUMENT_TURNS
    ]
    assert window
    for r in window:
        post = float(r.get("attack_pool_post_scale") or 0.0)
        carry = float(r.get("attack_pool_recruit_start") or 0.0)
        add = float(r.get("scale_add_attack") or 0.0)
        loss = float(r.get("replace_loss_attack") or 0.0)
        assert abs(post - (carry + add - loss)) <= 1.0
        assert r.get("flow_ok") is True
        assert "firestone_target" in r or r.get("residual_add") is None or True
        assert "n_alive" in r
        assert "board_size_post_scale" in r
    hits = [
        f for f in tracer.fights
        if int(f.get("applied_hp_loss") or 0) > 0
        and int(f.get("turn") or 0) in INSTRUMENT_TURNS
    ]
    assert hits
    for f in hits:
        rows = f.get("start_minions") or []
        for r in rows:
            assert "opp_board_pool_attack" in r
            assert "opp_carry_attack_pool" in r
            assert "opp_scale_add_attack" in r
            assert "opp_replace_loss_attack" in r
            assert "attack_identity_ok" in r
        counts = f.get("event_counts") or {}
        if counts:
            assert counts.get("attack_identity_reconcile") is True
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )


def test_sell_buy_play_records_replacement_and_conserves_2s_pool():
    from hsbg_coach.bg_env import board_level_abstract_scaling_enabled
    env = BGEnv(seed=1)
    env.reset(seed=1)
    env.turn = 8
    p = env.players[0]
    keep = EnvMinion("k", "KeepMe", 3, 5, 5, [], [])
    keep.attack, keep.health = 25, 25
    sold = EnvMinion("s", "Sold", 2, 3, 3, [], [])
    sold.attack, sold.health = 13, 13
    newbie = EnvMinion("n", "New", 3, 6, 6, [], [])
    p.board = [keep, sold]
    p.shop = [newbie]
    p.gold = 10
    tracer = PoolLifecycleTracer(0, 1, "obs")
    with board_level_abstract_scaling_enabled(True):
        from hsbg_coach.bg_env import ensure_abstract_pool_current
        ensure_abstract_pool_current(p)
        tracer.begin_seat_recruit(0, 8, p)
        obs = {
            "board": [keep.view(), sold.view()],
            "shop": [newbie.view()],
            "hand": [],
        }
        tracer.before_action(0, 8, 0, obs, [])
        env._apply(0, A_SELL0 + 1)
        tracer.after_action(0, 8, 0, A_SELL0 + 1, False, p)
        obs2 = {
            "board": [m.view() for m in p.board],
            "shop": [m.view() for m in p.shop],
            "hand": [m.view() for m in p.hand],
        }
        tracer.before_action(0, 8, 0, obs2, [])
        env._apply(0, A_BUY0)
        tracer.after_action(0, 8, 0, A_BUY0, False, p)
        obs3 = {
            "board": [m.view() for m in p.board],
            "shop": [m.view() for m in p.shop],
            "hand": [m.view() for m in p.hand],
        }
        tracer.before_action(0, 8, 0, obs3, [])
        env._apply(0, A_PLAY0)
        tracer.after_action(0, 8, 0, A_PLAY0, False, p)
        tracer.end_seat_recruit(0, 8, p)
    assert tracer.replacement_events
    ev = tracer.replacement_events[0]
    assert ev["two_s_on"] is True
    assert abs(float(ev["stats_pool_delta"])) <= 1.0
    acc = tracer._acc(0, 8)
    assert acc["n_replacements"] == 1
    assert acc["n_sells"] >= 1


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


def _row(
    tier, recruit, synth, survived, *,
    slot=2, teammate=40,
    start_health=10, n_hits=1, n_damaging=1,
    incoming=6, mean_in=None, overkill=0,
    attacker_attack=6, attacker_synth_share=0.2,
    pairing=1.0,
    opp_pool=30, conc=0.5, combat_delta=0.0,
    carry=20, scale_add=10, replace_loss=0, board_size=4,
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
        "opp_board_size": board_size,
        "opp_board_mean_tier": 3.0,
        "opp_carry_attack_pool": carry,
        "opp_scale_add_attack": scale_add,
        "opp_replace_loss_attack": replace_loss,
        "opp_attack_pool_pre_scale": carry - replace_loss,
        "opp_attack_pool_post_scale": carry + scale_add - replace_loss,
        "opp_attack_pool_combat_start": opp_pool,
        "opp_select_board_size": board_size,
        "opp_flow_ok": True,
        "opp_flow_residual": 0.0,
    }


def test_reweight_assigns_carry_when_only_carry_shifts():
    control = [
        _row(4, 10, 10, i == 0, carry=80, scale_add=5, replace_loss=0, opp_pool=85)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, carry=5, scale_add=5, replace_loss=0, opp_pool=10)
        for i in range(4)
    ]
    rw = reweight_pool_lifecycle(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
        observed_attack_strength=4.0, observed_board_pool=4.0,
    )
    assert rw["share_of_a1_inherited_carry_pool"] > 0.70
    assert abs(rw["share_of_a1_current_turn_scaling_add"] or 0.0) < 0.25
    assert abs(rw["share_of_a1_replacement_churn"] or 0.0) < 0.25


def test_reweight_assigns_scale_when_only_scale_add_shifts():
    control = [
        _row(4, 10, 10, i == 0, carry=10, scale_add=70, replace_loss=0, opp_pool=80)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, carry=10, scale_add=2, replace_loss=0, opp_pool=12)
        for i in range(4)
    ]
    rw = reweight_pool_lifecycle(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
        observed_attack_strength=4.0, observed_board_pool=4.0,
    )
    assert rw["share_of_a1_current_turn_scaling_add"] > 0.70


def test_reweight_assigns_replace_when_only_loss_shifts():
    control = [
        _row(4, 10, 10, i == 0, carry=80, scale_add=5, replace_loss=0, opp_pool=85)
        for i in range(4)
    ]
    treatment = [
        _row(4, 10, 10, i < 3, carry=80, scale_add=5, replace_loss=70, opp_pool=15)
        for i in range(4)
    ]
    rw = reweight_pool_lifecycle(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover_3a=4.0, observed_damage_per_hit=4.0,
        observed_attack_strength=4.0, observed_board_pool=4.0,
    )
    assert rw["share_of_a1_replacement_churn"] > 0.55


def test_3d_reweight_still_assigns_pool_on_same_rows():
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


def test_diagnose_routes_carry_scale_replace_selection_joint_and_leftover():
    carry = diagnose_phase_3e({
        "reweighting": {
            "share_of_a1_inherited_carry_pool": 0.80,
            "share_of_a1_current_turn_scaling_add": 0.10,
            "share_of_a1_replacement_churn": 0.05,
            "share_of_a1_lifecycle_selection": 0.03,
            "share_of_a1_still_unexplained": 0.02,
            "share_of_a1_lifecycle_selection_plus_leftover": 0.05,
            "phase_3d_board_pool_magnitude_hat": 0.422,
            "reproduced_3d_board_pool_magnitude": 0.422,
        }
    })
    assert carry["primary_finding"] == "carry_history_dominates"
    assert "first appears" in carry["recommended_next_step"]

    scale = diagnose_phase_3e({
        "reweighting": {
            "share_of_a1_inherited_carry_pool": 0.05,
            "share_of_a1_current_turn_scaling_add": 0.80,
            "share_of_a1_replacement_churn": 0.05,
            "share_of_a1_lifecycle_selection": 0.05,
            "share_of_a1_still_unexplained": 0.05,
            "share_of_a1_lifecycle_selection_plus_leftover": 0.10,
            "phase_3d_board_pool_magnitude_hat": 0.422,
        }
    })
    assert scale["primary_finding"] == "current_turn_scaling_add_dominates"
    assert "inputs" in scale["recommended_next_step"]

    repl = diagnose_phase_3e({
        "reweighting": {
            "share_of_a1_inherited_carry_pool": 0.05,
            "share_of_a1_current_turn_scaling_add": 0.05,
            "share_of_a1_replacement_churn": 0.80,
            "share_of_a1_lifecycle_selection": 0.05,
            "share_of_a1_still_unexplained": 0.05,
            "share_of_a1_lifecycle_selection_plus_leftover": 0.10,
            "phase_3d_board_pool_magnitude_hat": 0.422,
        }
    })
    assert repl["primary_finding"] == "replacement_retention_dominates"
    assert "2S pool lifecycle" in repl["recommended_next_step"]

    sel = diagnose_phase_3e({
        "reweighting": {
            "share_of_a1_inherited_carry_pool": 0.05,
            "share_of_a1_current_turn_scaling_add": 0.05,
            "share_of_a1_replacement_churn": 0.05,
            "share_of_a1_lifecycle_selection": 0.40,
            "share_of_a1_still_unexplained": 0.40,
            "share_of_a1_lifecycle_selection_plus_leftover": 0.80,
            "phase_3d_board_pool_magnitude_hat": 0.422,
        }
    })
    assert sel["primary_finding"] == "lifecycle_selection_dominates"
    assert "matched alive" in sel["recommended_next_step"]

    joint = diagnose_phase_3e({
        "reweighting": {
            "share_of_a1_inherited_carry_pool": 0.35,
            "share_of_a1_current_turn_scaling_add": 0.25,
            "share_of_a1_replacement_churn": 0.15,
            "share_of_a1_lifecycle_selection": 0.10,
            "share_of_a1_still_unexplained": 0.15,
            "share_of_a1_lifecycle_selection_plus_leftover": 0.25,
            "phase_3d_board_pool_magnitude_hat": 0.422,
        }
    })
    assert joint["primary_finding"] == "jointly_explained_rank_largest"
    assert joint["ranked_represented"][0]["component"] == "inherited_carry_pool"

    leftover = diagnose_phase_3e({
        "reweighting": {
            "share_of_a1_inherited_carry_pool": 0.10,
            "share_of_a1_current_turn_scaling_add": 0.10,
            "share_of_a1_replacement_churn": 0.10,
            "share_of_a1_lifecycle_selection": 0.10,
            "share_of_a1_still_unexplained": 0.20,
            "share_of_a1_lifecycle_selection_plus_leftover": 0.30,
            "phase_3d_board_pool_magnitude_hat": 0.422,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"

    smoke = diagnose_phase_3e(carry, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
