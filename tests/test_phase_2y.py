"""Phase 2Y slot/attack-order vs teammate protection — observational locks."""

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
from ml.phase_2y_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    PHASE_2X_SHARE_RESIDUAL,
    PHASE_2Y_LOBBIES,
    PHASE_2Y_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    SLOT_BIN_CAP,
    assert_seed_range_allowed,
    diagnose_phase_2y,
    slot_bin,
)
from ml.position_order_diagnostic import (
    PositionOrderTracer,
    reweight_position_protection,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_2y_v1_default_off():
    assert METHODOLOGY_VERSION == "2y_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert SHARE_DOMINANT == 0.70
    assert SLOT_BIN_CAP == 4
    assert slot_bin(0) == 0
    assert slot_bin(4) == 4
    assert slot_bin(6) == 4


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_2Y_SEED == 14200
    assert PHASE_2Y_LOBBIES == 500
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
    try:
        assert_seed_range_allowed(14199, 2)
        raise AssertionError("expected pre-2S band to fail")
    except ValueError:
        pass


def test_hold_stack_includes_2x_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2V_WITHIN_TIER_B - 1.6782901818400895) < 1e-9
    assert abs(PHASE_2X_RESIDUAL_POSITION - 1.3719447683362298) < 1e-9
    assert abs(PHASE_2X_SHARE_RESIDUAL - 0.8174657655638667) < 1e-9
    d = diagnose_phase_2y()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["no_2q_rewrite"] is True
    assert d["keep_hold_prs"] == [29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]


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


def test_trace_records_slot_attack_order_and_targeting():
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
        assert "n_attacks" in row
        assert "first_attack_index" in row
        assert "n_targeted" in row
        assert "taunt" in row
        assert "board_slot" in row
    assert any(row.get("attacked") for row in start)
    assert any(row.get("taunt") for row in start)
    assert any(row.get("was_targeted") for row in start)


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = PositionOrderTracer(0, seed, "obs")
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
    """Hooked vs unhooked: placements, HP, length, and RNG state match."""
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
        board_size = len(rows)
        total_raw = sum(int(r["combat_raw"]) for r in rows)
        for r in rows:
            assert r["board_size"] == board_size
            assert r["teammate_combat_raw"] == total_raw - int(r["combat_raw"])
            assert r["slot_bin"] == slot_bin(r.get("board_slot"))
            assert "death_before_first_attack" in r
            assert "n_targeted" in r
        assert f["actual_survivor_count"] == f["survivor_count"]
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )


def _row(
    tier, recruit, synth, survived, *,
    slot=0, attacked=False, n_attacks=None,
    teammate=20, board_size=4, first_attack_index=None,
):
    combat = recruit + synth
    n_att = n_attacks if n_attacks is not None else (1 if attacked else 0)
    return {
        "tier": tier,
        "recruit_raw": recruit,
        "synthetic_share": synth,
        "combat_raw": combat,
        "synthetic_share_of_combat": (synth / combat if combat else None),
        "survived": survived,
        "died": not survived,
        "board_slot": slot,
        "slot_bin": slot_bin(slot),
        "golden": False,
        "attacked": attacked,
        "n_attacks": n_att,
        "first_attack_index": first_attack_index if attacked else None,
        "death_before_first_attack": (not survived) and n_att == 0,
        "teammate_combat_raw": teammate,
        "board_size": board_size,
        "taunt": False,
        "n_targeted": 0,
        "was_targeted": False,
    }


def test_reweight_assigns_slot_when_only_slot_shifts():
    """Same tier/recruit/synth/teammate; treatment sits earlier and survives."""
    control = [
        _row(4, 10, 10, True, slot=4, attacked=False, teammate=40),
        _row(4, 10, 10, False, slot=4, attacked=False, teammate=40),
        _row(4, 10, 10, False, slot=5, attacked=False, teammate=40),
        _row(4, 10, 10, False, slot=5, attacked=False, teammate=40),
    ]
    treatment = [
        _row(4, 10, 10, True, slot=0, attacked=True, teammate=40),
        _row(4, 10, 10, True, slot=0, attacked=True, teammate=40),
        _row(4, 10, 10, True, slot=1, attacked=True, teammate=40),
        _row(4, 10, 10, False, slot=1, attacked=True, teammate=40),
    ]
    rw = reweight_position_protection(
        control, treatment, n_hits_c=1, n_hits_t=1, observed_residual=4.0
    )
    assert rw["residual_position_hat"] > 0
    assert rw["share_of_residual_slot_opportunity"] is not None
    assert rw["share_of_residual_slot_opportunity"] > 0.70
    assert (rw["share_of_residual_teammate_protection"] or 0.0) < 0.20
    assert abs(
        rw["slot_opportunity"] + rw["teammate_protection"] + rw["unexplained"]
        - rw["residual_position_hat"]
    ) < 1e-9


def test_reweight_assigns_teammate_when_only_protection_shifts():
    """Same tier/recruit/synth/slot; treatment has stronger teammates."""
    control = [
        _row(4, 10, 10, True, slot=1, teammate=10),
        _row(4, 10, 10, False, slot=1, teammate=10),
        _row(4, 10, 10, False, slot=1, teammate=12),
        _row(4, 10, 10, False, slot=1, teammate=12),
    ]
    treatment = [
        _row(4, 10, 10, True, slot=1, teammate=80),
        _row(4, 10, 10, True, slot=1, teammate=80),
        _row(4, 10, 10, True, slot=1, teammate=90),
        _row(4, 10, 10, False, slot=1, teammate=90),
    ]
    rw = reweight_position_protection(
        control, treatment, n_hits_c=1, n_hits_t=1, observed_residual=4.0
    )
    assert rw["residual_position_hat"] > 0
    assert rw["share_of_residual_teammate_protection"] is not None
    assert rw["share_of_residual_teammate_protection"] > 0.70
    assert (rw["share_of_residual_slot_opportunity"] or 0.0) < 0.20
    assert abs(
        rw["slot_opportunity"] + rw["teammate_protection"] + rw["unexplained"]
        - rw["residual_position_hat"]
    ) < 1e-9


def test_reweight_assigns_unexplained_when_slot_and_team_match():
    """Same covariates; only survival differs → leftover combat mechanics."""
    control = [
        _row(4, 10, 10, True, slot=2, teammate=40),
        _row(4, 10, 10, False, slot=2, teammate=40),
        _row(4, 10, 10, False, slot=2, teammate=40),
        _row(4, 10, 10, False, slot=2, teammate=40),
    ]
    treatment = [
        _row(4, 10, 10, True, slot=2, teammate=40),
        _row(4, 10, 10, True, slot=2, teammate=40),
        _row(4, 10, 10, True, slot=2, teammate=40),
        _row(4, 10, 10, False, slot=2, teammate=40),
    ]
    rw = reweight_position_protection(
        control, treatment, n_hits_c=1, n_hits_t=1, observed_residual=4.0
    )
    assert rw["residual_position_hat"] > 0
    assert rw["share_of_residual_unexplained"] is not None
    assert rw["share_of_residual_unexplained"] > 0.70
    assert (rw["share_of_residual_slot_opportunity"] or 0.0) < 0.20
    assert (rw["share_of_residual_teammate_protection"] or 0.0) < 0.20
    assert abs(
        rw["slot_opportunity"] + rw["teammate_protection"] + rw["unexplained"]
        - rw["residual_position_hat"]
    ) < 1e-9


def test_diagnose_routes_three_ways():
    slot = diagnose_phase_2y({
        "reweighting": {
            "share_of_residual_slot_opportunity": 0.80,
            "share_of_residual_teammate_protection": 0.10,
            "share_of_residual_unexplained": 0.10,
            "phase_2x_residual_position_hat": 1.372,
        }
    })
    assert slot["primary_finding"] == "slot_attack_opportunity_dominates"
    assert "positioning policy" in slot["recommended_next_step"]

    team = diagnose_phase_2y({
        "reweighting": {
            "share_of_residual_slot_opportunity": 0.15,
            "share_of_residual_teammate_protection": 0.75,
            "share_of_residual_unexplained": 0.10,
            "phase_2x_residual_position_hat": 1.372,
        }
    })
    assert team["primary_finding"] == "teammate_protection_dominates"
    assert "composition" in team["recommended_next_step"]

    leftover = diagnose_phase_2y({
        "reweighting": {
            "share_of_residual_slot_opportunity": 0.35,
            "share_of_residual_teammate_protection": 0.30,
            "share_of_residual_unexplained": 0.35,
            "phase_2x_residual_position_hat": 1.372,
        }
    })
    assert leftover["primary_finding"] == "unexplained_combat_mechanics"
    assert "taunt" in leftover["recommended_next_step"]

    smoke = diagnose_phase_2y(slot, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
