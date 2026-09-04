"""Phase 2U survivor-tier damage fidelity — observational tracer locks."""

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
from ml.phase_2u_prereg import (
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2U_LOBBIES,
    PHASE_2U_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_REMOVED_MOST,
    assert_seed_range_allowed,
    diagnose_phase_2u,
)
from ml.survivor_tier_damage_diagnostic import (
    SurvivorTierTracer,
    rules_faithful_hero_damage,
)


def test_methodology_is_2u_v1_default_off():
    assert METHODOLOGY_VERSION == "2u_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_2U_SEED == 14200
    assert PHASE_2U_LOBBIES == 500
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


def test_hold_stack_includes_2t_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38)
    assert FROZEN_ALPHA == 0.5
    d = diagnose_phase_2u()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["keep_hold_prs"] == [29, 33, 34, 35, 36, 37, 38]


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


def test_rules_faithful_is_tavern_plus_survivor_tier_sum():
    assert rules_faithful_hero_damage(5, [6, 4, 1]) == 16
    assert rules_faithful_hero_damage(3, []) == 3


def test_survivor_trace_matches_living_combatants():
    """Trace identities/tiers equal the living winner board after combat."""
    winners = [
        Combatant(80, 80, name="big", card_id="B", tier=6),
        Combatant(70, 70, name="mid", card_id="M", tier=4),
    ]
    losers = [Combatant(1, 1, name="chump", card_id="C", tier=1)]
    rng = random.Random(0)
    trace = {}
    raw = simulate_once(winners, losers, rng, tier_a=5, tier_b=2, trace=trace)
    assert raw == 5 + 2
    assert trace["survivor_count"] == 2
    assert {s["name"] for s in trace["survivors"]} == {"big", "mid"}
    assert {s["tier"] for s in trace["survivors"]} == {6, 4}
    assert trace["survivor_tier_sum"] == 10
    assert trace["rules_faithful_damage"] == 5 + 10
    assert raw == 5 + trace["survivor_count"]


def test_survivor_trace_excludes_dead_winner_minion():
    """A weak winner body dies; trace must drop it and keep the living tier."""
    # Winner attacks first (2 vs 1). Index-0 1/1 T6 trades into 30/1 and dies;
    # 40/40 T5 never takes a hit.
    winners = [
        Combatant(1, 1, name="chaff", card_id="X", tier=6),
        Combatant(40, 40, name="tank", card_id="T", tier=5),
    ]
    losers = [Combatant(30, 1, name="spike", card_id="S", tier=2)]
    rng = random.Random(1)
    trace = {}
    raw = simulate_once(winners, losers, rng, tier_a=4, tier_b=3, trace=trace)
    assert raw > 0
    names = {s["name"] for s in trace["survivors"]}
    assert "chaff" not in names
    assert "tank" in names
    assert trace["survivor_count"] == 1
    assert trace["survivor_tier_sum"] == 5
    assert raw == 4 + 1
    assert trace["rules_faithful_damage"] == 4 + 5


def test_trace_does_not_change_simulate_once_return_or_rng():
    a = [Combatant(5, 5, name="a", tier=3), Combatant(3, 4, name="b", tier=2)]
    b = [Combatant(4, 4, name="c", tier=4)]
    r1 = random.Random(42)
    r2 = random.Random(42)
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace={})
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()


def test_from_minion_carries_env_tier():
    m = EnvMinion("id-z", "Zebra", 5, 8, 8, [], [])
    c = Combatant.from_minion(m.view())
    assert c.tier == 5
    assert c.name == "Zebra"


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = SurvivorTierTracer(0, seed, "obs")
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
        assert f["actual_survivor_count"] == f["survivor_count"]
        assert f["counterfactual_damage"] == rules_faithful_hero_damage(
            f["winner_tavern_tier"],
            [s["tier"] for s in f["actual_survivors"]],
        )


def test_direct_combat_survivors_match_trace_and_formula():
    env = BGEnv(seed=7)
    env.reset(seed=7)
    a, b = env.players[0], env.players[1]
    a.alive = True
    b.alive = True
    a.hp = 30
    b.hp = 30
    a.tier = 4
    b.tier = 3
    # T1 chaff dies into the 50/1; T6 tank survives. Board-mean proxy
    # therefore disagrees with sum(actual survivor tiers).
    a.board = [
        EnvMinion("id-x", "x", 1, 1, 1, [], []),
        EnvMinion("id-y", "y", 6, 80, 80, [], []),
    ]
    b.board = [EnvMinion("id-z", "z", 1, 50, 1, [], [])]
    for p in env.players[2:]:
        p.alive = False
        p.last_board = []
    env.turn = 10
    seen = []
    env.combat_audit_hook = lambda _e, fight: seen.append(fight)
    env._run_combat()
    fight = next(f for f in seen if f.get("kind") == "live")
    raw = int(fight["raw"])
    assert raw != 0
    expect_proxy = BGEnv._hero_damage(
        raw, int(fight["winner_tier"]), fight["winner_board"]
    )
    assert fight["applied"] == expect_proxy
    assert fight["survivor_count_actual"] == abs(raw) - int(fight["winner_tier"])
    assert {s["name"] for s in fight["survivors"]} == {"y"}
    assert fight["survivor_tier_sum"] == 6
    cf = rules_faithful_hero_damage(
        int(fight["winner_tier"]),
        [int(s["tier"]) for s in fight["survivors"]],
    )
    assert cf == 4 + 6
    assert expect_proxy != cf  # board-mean proxy ≠ actual survivor-tier sum


def test_diagnose_routes_two_ways():
    gone = diagnose_phase_2u({
        "fidelity": {
            "share_of_amp_delta_removed": 0.80,
            "share_of_amp_delta_remaining": 0.20,
            "proxy_amplification_delta_when_hit": 2.78,
            "counterfactual_amplification_delta_when_hit": 0.56,
        }
    })
    assert gone["primary_finding"] == "preregister_default_off_damage_formula"
    assert "default-OFF" in gone["recommended_next_step"]

    remain = diagnose_phase_2u({
        "fidelity": {
            "share_of_amp_delta_removed": 0.30,
            "share_of_amp_delta_remaining": 0.70,
            "proxy_amplification_delta_when_hit": 2.78,
            "counterfactual_amplification_delta_when_hit": 1.95,
        }
    })
    assert remain["primary_finding"] == "isolate_survivor_composition"
    assert SHARE_REMOVED_MOST == 0.55

    smoke = diagnose_phase_2u(remain, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
