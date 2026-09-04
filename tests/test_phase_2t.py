"""Phase 2T game-length / damage attribution — observational tracer locks."""

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
)
from hsbg_coach.sim import Combatant, simulate_once
from ml.game_length_damage_diagnostic import (
    PHASE_2S_POST_SCALE,
    GameLengthDamageTracer,
    attribute_shortening,
    decompose_hero_damage,
)
from ml.phase_2s_prereg import (
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
)
from ml.phase_2t_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2T_LOBBIES,
    PHASE_2T_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_2t,
)


def test_methodology_is_2t_v1_default_off():
    assert METHODOLOGY_VERSION == "2t_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_2T_SEED == 14200
    assert PHASE_2T_LOBBIES == 500
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


def test_hold_stack_includes_2s_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37)
    assert FROZEN_ALPHA == 0.5
    d = diagnose_phase_2t()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_gate_change"] is True
    assert d["keep_hold_prs"] == [29, 33, 34, 35, 36, 37]


def test_2s_gates_unchanged():
    assert GATE_REPLACE_RATE_MIN == 0.10
    assert GATE_T10_POST_SCALE_MIN == 0.85
    assert GATE_T10_POST_SCALE_DELTA_FLOOR == -0.10
    assert GATE_GAME_LENGTH_DELTA_FLOOR == -0.50
    assert GATE_MEAN_COMBAT_LOSS_MAX == 20.0


def _minion(name, tier, attack, health, keywords=None):
    return EnvMinion(
        "id-" + name, name, tier, attack, health, [], list(keywords or []),
    )


def test_damage_decomposition_reconciles_to_hero_damage():
    board = [_minion("a", 5, 10, 10), _minion("b", 3, 8, 8), _minion("c", 4, 6, 6)]
    raw = 4 + 3  # tavern 4 + 3 survivors (sim.py count-only)
    decomp = decompose_hero_damage(raw, 4, board)
    assert decomp["survivor_count"] == 3
    assert decomp["count_only_damage"] == 7
    assert decomp["applied_damage"] == BGEnv._hero_damage(raw, 4, board)
    assert decomp["applied_damage"] == (
        decomp["count_only_damage"] + decomp["amplification"]
    )
    avg = (5 + 3 + 4) / 3
    assert decomp["applied_damage"] == 4 + max(1, round(3 * avg))
    assert decomp["amplification"] == decomp["applied_damage"] - 7


def test_empty_board_decomposition_still_reconciles():
    decomp = decompose_hero_damage(5, 2, [])
    assert decomp["applied_damage"] == BGEnv._hero_damage(5, 2, [])
    assert decomp["applied_damage"] == (
        decomp["count_only_damage"] + decomp["amplification"]
    )


def test_combat_hook_is_observational_same_seed():
    """Hook must not change placements, HP, or game length vs an unhooked run."""
    def _play(with_hook):
        env = BGEnv(seed=14200)
        tracer = None
        if with_hook:
            tracer = GameLengthDamageTracer(0, 14200, "obs")
            tracer.attach_to_env(env)
        recs = env.play_scripted(
            [greedy_policy] * env.n_players,
            recruit_tracer=tracer,
        )
        return {
            "length": max((r["turn"] for r in recs), default=0),
            "placements": [p.placement for p in env.players],
            "hp": [p.hp for p in env.players],
            "n_fights": len(tracer.fights) if tracer else 0,
        }

    plain = _play(False)
    hooked = _play(True)
    assert hooked["length"] == plain["length"]
    assert hooked["placements"] == plain["placements"]
    assert hooked["hp"] == plain["hp"]
    assert hooked["n_fights"] > 0


def test_applied_hp_loss_matches_formula_and_hp_delta():
    env = BGEnv(seed=14201)
    tracer = GameLengthDamageTracer(0, 14201, "obs")
    tracer.attach_to_env(env)
    env.play_scripted([greedy_policy] * env.n_players, recruit_tracer=tracer)
    hits = [f for f in tracer.fights if int(f.get("applied_hp_loss") or 0) > 0]
    assert hits
    for f in tracer.fights:
        assert f["applied_hp_loss"] == f["hp_delta"]
    for f in hits:
        assert f["applied_hp_loss"] == f["applied_damage"]
        assert f["applied_damage"] == f["count_only_damage"] + f["amplification"]


def test_direct_combat_hp_delta_matches_hero_damage():
    """Single live fight: HP deducted equals `_hero_damage` on the winner board."""
    env = BGEnv(seed=7)
    env.reset(seed=7)
    a, b = env.players[0], env.players[1]
    a.alive = True
    b.alive = True
    a.hp = 30
    b.hp = 30
    a.tier = 4
    b.tier = 3
    a.board = [_minion("x", 5, 80, 80), _minion("y", 4, 70, 70)]
    b.board = [_minion("z", 1, 1, 1)]
    for p in env.players[2:]:
        p.alive = False
        p.last_board = []
    env.turn = 10
    seen = []
    env.combat_audit_hook = lambda _e, fight: seen.append(fight)
    env._run_combat()
    assert seen
    fight = next(f for f in seen if f.get("kind") == "live")
    raw = int(fight["raw"])
    assert raw != 0
    expect = BGEnv._hero_damage(
        raw, int(fight["winner_tier"]), fight["winner_board"]
    )
    assert fight["applied"] == expect
    if fight["loser_seat"] == fight["seat_a"]:
        assert fight["pre_hp_a"] - fight["post_hp_a"] == expect
    else:
        assert fight["pre_hp_b"] - fight["post_hp_b"] == expect


def test_simulate_once_count_only_is_tier_plus_survivors():
    """Document sim.py raw: |raw| = winner_tier + survivor_count."""
    rng_a = __import__("random").Random(0)
    winners = [Combatant(50, 50, name="w")]
    losers = [Combatant(1, 1, name="l")]
    raw = simulate_once(winners, losers, rng_a, tier_a=4, tier_b=2)
    assert raw > 0
    assert raw == 4 + 1


def test_diagnose_routes_three_ways():
    amp = diagnose_phase_2t({
        "attribution": {
            "actual_shortening_turns": 2.18,
            "share_of_extra_hp_from_amplification": 0.72,
            "share_of_extra_hp_from_combat_outcome": 0.20,
            "share_of_shortening_unexplained_lifecycle": 0.08,
            "combat_strength_fidelity_healthy": True,
        }
    })
    assert amp["primary_finding"] == "damage_model_fidelity"
    assert "damage-model" in amp["recommended_next_step"]

    combat = diagnose_phase_2t({
        "attribution": {
            "actual_shortening_turns": 2.18,
            "share_of_extra_hp_from_amplification": 0.10,
            "share_of_extra_hp_from_combat_outcome": 0.80,
            "share_of_shortening_unexplained_lifecycle": 0.10,
            "combat_strength_fidelity_healthy": True,
        }
    })
    assert combat["primary_finding"] == "combat_outcome_dominance"

    life = diagnose_phase_2t({
        "attribution": {
            "actual_shortening_turns": 2.18,
            "share_of_extra_hp_from_amplification": 0.30,
            "share_of_extra_hp_from_combat_outcome": 0.30,
            "share_of_shortening_unexplained_lifecycle": 0.40,
            "combat_strength_fidelity_healthy": True,
        }
    })
    assert life["primary_finding"] == "lifecycle_or_unexplained"
    assert SHARE_DOMINANT == 0.55

    smoke = diagnose_phase_2t(combat, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False


def test_2s_post_scale_prior_is_healthy():
    assert all(
        float(v["treatment"]) >= float(v["control"])
        for v in PHASE_2S_POST_SCALE.values()
    )


def test_attribute_shortening_amp_share_routes_damage_model():
    control = {
        "mean_game_length": 15.692,
        "mean_applied_per_alive_seat_turn": 4.38,
        "mean_count_only_when_hit": 7.46,
        "mean_amplification_when_hit": 3.41,
        "hit_rate_per_alive_seat_turn": 0.40,
        "mean_applied_when_hit": 10.87,
        "mean_hp_at_t7": 18.6,
        "mean_winner_strength": 3600,
        "hp_flow_identity_ok": True,
    }
    treatment = {
        "mean_game_length": 13.510,
        "mean_applied_per_alive_seat_turn": 5.79,
        "mean_count_only_when_hit": 7.76,
        "mean_amplification_when_hit": 6.19,
        "hit_rate_per_alive_seat_turn": 0.41,
        "mean_applied_when_hit": 13.95,
        "mean_hp_at_t7": 18.3,
        "mean_winner_strength": 3000,
        "hp_flow_identity_ok": True,
    }
    attr = attribute_shortening(control, treatment)
    assert attr["combat_strength_fidelity_healthy"] is True
    assert attr["share_of_extra_hp_from_amplification"] >= SHARE_DOMINANT
    routed = diagnose_phase_2t({"attribution": attr})
    assert routed["primary_finding"] == "damage_model_fidelity"
