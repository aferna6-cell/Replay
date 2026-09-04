"""Phase 2Z targeting / cursor / represented DR — observational locks."""

import random

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
)
from hsbg_coach.effects import (
    APPROXIMATE_DEATHRATTLE_NAMES,
    PLACEHOLDER_DEATHRATTLE_NAMES,
    Summon,
)
from hsbg_coach.sim import Combatant, classify_effect_status, simulate_once
from ml.phase_2s_prereg import (
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
)
from ml.phase_2z_prereg import (
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
    PHASE_2Z_LOBBIES,
    PHASE_2Z_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    cursor_bin,
    diagnose_phase_2z,
    gen_bin,
    slot_bin,
    target_bin,
    unsupported_bin,
)
from ml.combat_mechanics_diagnostic import (
    CombatMechanicsTracer,
    reweight_combat_mechanics,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_2z_v1_default_off():
    assert METHODOLOGY_VERSION == "2z_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert slot_bin(4) == 4
    assert target_bin({"taunt": True}) == 2
    assert target_bin({"n_targeted_open": 1}) == 1
    assert target_bin({}) == 0
    assert cursor_bin({}) == 0
    assert cursor_bin({"attacked": True, "side_first": True}) == 1
    assert cursor_bin({"attacked": True, "side_first": False}) == 2
    assert gen_bin({"has_represented_generated_effect": True}) == 1
    assert unsupported_bin({"has_unsupported_effect": True}) == 1
    assert "Kaboom Bot" in PLACEHOLDER_DEATHRATTLE_NAMES
    assert "Rat Pack" in APPROXIMATE_DEATHRATTLE_NAMES


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_2Z_SEED == 14200
    assert PHASE_2Z_LOBBIES == 500
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


def test_hold_stack_includes_2y_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_2X_RESIDUAL_POSITION - 1.3719447683362298) < 1e-9
    assert abs(PHASE_2Y_UNEXPLAINED - 0.9456715648873479) < 1e-9
    d = diagnose_phase_2z()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["unsupported_marked_not_approximated"] is True
    assert d["keep_hold_prs"][-1] == 43


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


def test_placeholder_and_approximate_are_marked_not_fired():
    boom = Combatant(2, 2, name="Kaboom Bot")
    boom.deathrattle = Summon(0, 0, 0)
    assert classify_effect_status(boom) == "unsupported_placeholder"
    spawn = Combatant(2, 2, name="Spawn of N'Zoth")
    spawn.deathrattle = Summon(0, 0, 0)
    assert classify_effect_status(spawn) == "unsupported_placeholder"
    rats = Combatant(3, 2, name="Rat Pack")
    rats.deathrattle = Summon(3, 1, 1, name="Rat")
    assert classify_effect_status(rats) == "represented_approximate"
    golem = Combatant(2, 3, name="Harvest Golem")
    golem.deathrattle = Summon(1, 2, 1, name="Damaged Golem")
    assert classify_effect_status(golem) == "represented"


def test_trace_records_forced_vs_open_and_death_cause():
    a = [
        Combatant(5, 5, name="a", tier=3, taunt=True),
        Combatant(3, 4, name="b", tier=2),
    ]
    b = [Combatant(4, 4, name="c", tier=4)]
    r1 = random.Random(42)
    r2 = random.Random(42)
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    trace = {}
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace=trace)
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()
    start = list(trace.get("starting_a") or []) + list(trace.get("starting_b") or [])
    assert start
    for row in start:
        assert "n_targeted_forced" in row
        assert "n_targeted_open" in row
        assert "death_cause" in row
        assert "start_health" in row
        assert "end_health" in row
        assert "effect_status" in row
        assert "side_first" in row
        assert "cursor_wrapped_before_first" in row
    counts = trace.get("event_counts") or {}
    assert counts.get("attacks_reconcile") is True
    assert counts.get("targets_reconcile") is True
    assert counts.get("forced_open_reconcile") is True
    assert counts.get("created_reconcile") is True
    assert int(counts.get("n_cursor_advance") or 0) >= 1

    # Larger attacking board so A swings first and must hit B's taunt.
    forced_trace = {}
    simulate_once(
        [
            Combatant(2, 2, name="atk", tier=1),
            Combatant(2, 2, name="atk2", tier=1),
            Combatant(2, 2, name="atk3", tier=1),
        ],
        [Combatant(10, 10, name="tank", taunt=True, tier=1),
         Combatant(1, 1, name="back", tier=1)],
        random.Random(0),
        trace=forced_trace,
    )
    defs = list(forced_trace.get("starting_b") or [])
    tank = next(r for r in defs if r.get("name") == "tank")
    back = next(r for r in defs if r.get("name") == "back")
    assert int(tank.get("n_targeted_forced") or 0) >= 1
    assert int(back.get("n_targeted_forced") or 0) == 0
    assert int(back.get("n_targeted_open") or 0) == 0


def test_open_targeting_when_no_taunts():
    a = [Combatant(3, 3, name="a", tier=1)]
    b = [Combatant(2, 8, name="b", tier=1), Combatant(2, 8, name="c", tier=1)]
    trace = {}
    simulate_once(a, b, random.Random(7), tier_a=1, tier_b=1, trace=trace)
    defs = list(trace.get("starting_b") or [])
    assert sum(int(r.get("n_targeted_open") or 0) for r in defs) >= 1
    assert sum(int(r.get("n_targeted_forced") or 0) for r in defs) == 0


def test_represented_golem_creates_token_placeholder_does_not():
    golem = Combatant(2, 1, name="Harvest Golem", tier=3)
    golem.deathrattle = Summon(1, 2, 1, name="Damaged Golem")
    boom = Combatant(2, 1, name="Kaboom Bot", tier=3)
    boom.deathrattle = Summon(0, 0, 0)
    killer = [Combatant(5, 5, name="x", tier=4)]
    g_trace: dict = {}
    simulate_once([golem], killer, random.Random(1), tier_a=3, tier_b=4, trace=g_trace)
    assert any(c.get("represented_generated") for c in (g_trace.get("created") or []))
    b_trace: dict = {}
    simulate_once([boom], killer, random.Random(1), tier_a=3, tier_b=4, trace=b_trace)
    assert not (b_trace.get("created") or [])
    start = (b_trace.get("starting_a") or [None])[0]
    assert start["effect_status"] == "unsupported_placeholder"


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = CombatMechanicsTracer(0, seed, "obs")
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
            assert "target_bin" in r
            assert "cursor_bin" in r
            assert "gen_bin" in r
            assert "unsupported_bin" in r
            assert "death_cause" in r
            assert "n_targeted_forced" in r
        assert f["actual_survivor_count"] == f["survivor_count"]
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )
        counts = f.get("event_counts") or {}
        if counts:
            assert counts.get("attacks_reconcile") is True
            assert counts.get("targets_reconcile") is True
            assert counts.get("forced_open_reconcile") is True


def _row(
    tier, recruit, synth, survived, *,
    slot=2, teammate=40,
    taunt=False, n_forced=0, n_open=0,
    attacked=False, first_idx=None, side_first=False, wrapped=False,
    represented=False, unsupported=False,
):
    combat = recruit + synth
    n_att = 1 if attacked else 0
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
        "attacked": attacked,
        "n_attacks": n_att,
        "first_attack_index": first_idx if attacked else None,
        "death_before_first_attack": (not survived) and n_att == 0,
        "teammate_combat_raw": teammate,
        "board_size": 4,
        "taunt": taunt,
        "n_targeted": n_forced + n_open,
        "n_targeted_forced": n_forced,
        "n_targeted_open": n_open,
        "was_targeted": (n_forced + n_open) > 0,
        "side_first": side_first,
        "cursor_wrapped_before_first": wrapped,
        "has_represented_generated_effect": represented,
        "has_unsupported_effect": unsupported,
        "n_board_generated_represented": 1 if represented else 0,
        "spawned_represented": 1 if represented else 0,
        "effect_status": (
            "unsupported_placeholder" if unsupported
            else ("represented" if represented else "unregistered")
        ),
    }


def test_reweight_assigns_targeting_when_only_taunt_shifts():
    control = [
        _row(4, 10, 10, True, n_open=1),
        _row(4, 10, 10, False, n_open=1),
        _row(4, 10, 10, False, n_open=1),
        _row(4, 10, 10, False, n_open=1),
    ]
    treatment = [
        _row(4, 10, 10, True, taunt=True, n_forced=1),
        _row(4, 10, 10, True, taunt=True, n_forced=1),
        _row(4, 10, 10, True, taunt=True, n_forced=1),
        _row(4, 10, 10, False, taunt=True, n_forced=1),
    ]
    rw = reweight_combat_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover=4.0,
    )
    assert rw["phase_2y_unexplained_hat"] > 0
    assert rw["share_of_leftover_targeting_taunt"] > 0.70
    assert (rw["share_of_leftover_attack_cursor"] or 0.0) < 0.20
    assert abs(
        rw["targeting_taunt"] + rw["attack_cursor"]
        + rw["represented_generated"] + rw["unsupported_coverage"]
        + rw["still_unexplained"] - rw["phase_2y_unexplained_hat"]
    ) < 1e-9


def test_reweight_assigns_cursor_when_only_initiative_shifts():
    control = [
        _row(4, 10, 10, True, attacked=False, side_first=False),
        _row(4, 10, 10, False, attacked=False, side_first=False),
        _row(4, 10, 10, False, attacked=False, side_first=False),
        _row(4, 10, 10, False, attacked=False, side_first=False),
    ]
    treatment = [
        _row(4, 10, 10, True, attacked=True, first_idx=0, side_first=True),
        _row(4, 10, 10, True, attacked=True, first_idx=0, side_first=True),
        _row(4, 10, 10, True, attacked=True, first_idx=1, side_first=True),
        _row(4, 10, 10, False, attacked=True, first_idx=1, side_first=True),
    ]
    rw = reweight_combat_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover=4.0,
    )
    assert rw["share_of_leftover_attack_cursor"] > 0.70
    assert (rw["share_of_leftover_targeting_taunt"] or 0.0) < 0.20


def test_reweight_assigns_generated_when_only_represented_dr_shifts():
    control = [_row(4, 10, 10, i == 0, represented=False) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, represented=True) for i in range(4)]
    rw = reweight_combat_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover=4.0,
    )
    assert rw["share_of_leftover_represented_generated"] > 0.70
    assert (rw["share_of_leftover_unsupported_coverage"] or 0.0) < 0.20


def test_reweight_assigns_unsupported_when_only_placeholder_shifts():
    control = [_row(4, 10, 10, i == 0, unsupported=False) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3, unsupported=True) for i in range(4)]
    rw = reweight_combat_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover=4.0,
    )
    assert rw["share_of_leftover_unsupported_coverage"] > 0.70
    assert (rw["share_of_leftover_represented_generated"] or 0.0) < 0.20


def test_reweight_assigns_unexplained_when_mechanics_match():
    control = [_row(4, 10, 10, i == 0) for i in range(4)]
    treatment = [_row(4, 10, 10, i < 3) for i in range(4)]
    rw = reweight_combat_mechanics(
        control, treatment, n_hits_c=1, n_hits_t=1,
        observed_leftover=4.0,
    )
    assert rw["share_of_leftover_still_unexplained"] > 0.70
    assert (rw["share_of_leftover_targeting_taunt"] or 0.0) < 0.20
    assert (rw["share_of_leftover_attack_cursor"] or 0.0) < 0.20


def test_diagnose_routes_five_ways():
    tgt = diagnose_phase_2z({
        "reweighting": {
            "share_of_leftover_targeting_taunt": 0.80,
            "share_of_leftover_attack_cursor": 0.05,
            "share_of_leftover_represented_generated": 0.05,
            "share_of_leftover_unsupported_coverage": 0.05,
            "share_of_leftover_still_unexplained": 0.05,
            "phase_2y_unexplained_hat": 0.946,
        }
    })
    assert tgt["primary_finding"] == "targeting_taunt_dominates"
    assert "taunt/targeting correction" in tgt["recommended_next_step"]

    cur = diagnose_phase_2z({
        "reweighting": {
            "share_of_leftover_targeting_taunt": 0.05,
            "share_of_leftover_attack_cursor": 0.80,
            "share_of_leftover_represented_generated": 0.05,
            "share_of_leftover_unsupported_coverage": 0.05,
            "share_of_leftover_still_unexplained": 0.05,
            "phase_2y_unexplained_hat": 0.946,
        }
    })
    assert cur["primary_finding"] == "attack_cursor_initiative_dominates"

    gen = diagnose_phase_2z({
        "reweighting": {
            "share_of_leftover_targeting_taunt": 0.05,
            "share_of_leftover_attack_cursor": 0.05,
            "share_of_leftover_represented_generated": 0.80,
            "share_of_leftover_unsupported_coverage": 0.05,
            "share_of_leftover_still_unexplained": 0.05,
            "phase_2y_unexplained_hat": 0.946,
        }
    })
    assert gen["primary_finding"] == "represented_generated_deathrattle_dominates"

    unsup = diagnose_phase_2z({
        "reweighting": {
            "share_of_leftover_targeting_taunt": 0.05,
            "share_of_leftover_attack_cursor": 0.05,
            "share_of_leftover_represented_generated": 0.05,
            "share_of_leftover_unsupported_coverage": 0.80,
            "share_of_leftover_still_unexplained": 0.05,
            "phase_2y_unexplained_hat": 0.946,
        }
    })
    assert unsup["primary_finding"] == "unsupported_effect_coverage_dominates"
    assert "audit that missing effect class" in unsup["recommended_next_step"]

    leftover = diagnose_phase_2z({
        "reweighting": {
            "share_of_leftover_targeting_taunt": 0.20,
            "share_of_leftover_attack_cursor": 0.20,
            "share_of_leftover_represented_generated": 0.15,
            "share_of_leftover_unsupported_coverage": 0.10,
            "share_of_leftover_still_unexplained": 0.35,
            "phase_2y_unexplained_hat": 0.946,
        }
    })
    assert leftover["primary_finding"] == "ranked_residual_needs_next_observable"
    assert leftover["ranked_residual"][0]["component"] == "still_unexplained"
    assert "start-of-combat" in leftover["smallest_additional_observable"]

    smoke = diagnose_phase_2z(tgt, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
