"""Phase 2V survivor-composition attribution — observational tracer locks."""

import random

from hsbg_coach.bg_env import (
    PHASE_2Q_RECRUIT_VALUE_STATS,
    PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING,
    BGEnv,
    EnvMinion,
    greedy_policy,
)
from hsbg_coach.effects import Summon
from hsbg_coach.sim import Combatant, simulate_once
from ml.phase_2s_prereg import (
    GATE_GAME_LENGTH_DELTA_FLOOR,
    GATE_MEAN_COMBAT_LOSS_MAX,
    GATE_REPLACE_RATE_MIN,
    GATE_T10_POST_SCALE_DELTA_FLOOR,
    GATE_T10_POST_SCALE_MIN,
)
from ml.phase_2v_prereg import (
    CHAFF_TIER_MAX,
    FEATURE_TOGGLE,
    FEATURE_TOGGLE_DEFAULT,
    FORBIDDEN_RANGES,
    FROZEN_ALPHA,
    HIGH_TIER_MIN,
    HOLD_PRS,
    INSTRUMENT_TURNS,
    METHODOLOGY_VERSION,
    PHASE_2U_SURVIVOR_TIER_SUM_DELTA,
    PHASE_2V_LOBBIES,
    PHASE_2V_SEED,
    REUSED_SEED_HI,
    REUSED_SEED_LO,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_2v,
)
from ml.survivor_composition_diagnostic import (
    SurvivorCompositionTracer,
    classify_env_minion,
    decompose_gap,
    fight_tier_buckets_reconcile,
    survivors_subset_of_traced,
    tier_sum,
)
from ml.survivor_tier_damage_diagnostic import rules_faithful_hero_damage


def test_methodology_is_2v_v1_default_off():
    assert METHODOLOGY_VERSION == "2v_v1"
    assert FEATURE_TOGGLE == "PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING"
    assert FEATURE_TOGGLE_DEFAULT is False
    assert PHASE_2Q_RECRUIT_VALUE_STATS is False
    assert PHASE_2S_BOARD_LEVEL_ABSTRACT_SCALING is False
    assert HIGH_TIER_MIN == 4
    assert CHAFF_TIER_MAX == 2


def test_reuses_2s_dev_and_forbids_confirm():
    assert PHASE_2V_SEED == 14200
    assert PHASE_2V_LOBBIES == 500
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


def test_hold_stack_includes_2u_and_frozen_alpha():
    assert HOLD_PRS == (29, 33, 34, 35, 36, 37, 38, 39)
    assert FROZEN_ALPHA == 0.5
    assert abs(PHASE_2U_SURVIVOR_TIER_SUM_DELTA - 4.009722998772222) < 1e-9
    d = diagnose_phase_2v()
    assert d["no_merge"] is True
    assert d["no_hero_damage_retune"] is True
    assert d["no_behavior_change"] is True
    assert d["keep_hold_prs"] == [29, 33, 34, 35, 36, 37, 38, 39]


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


def test_from_minion_carries_composition_fields():
    m = EnvMinion(
        "id-z", "Zebra", 5, 8, 12, ["Beast"], ["TAUNT"], True, 4, 6,
    )
    c = Combatant.from_minion(m.view())
    assert c.tier == 5
    assert c.golden is True
    assert c.tribes == ("Beast",)
    assert c.recruit_attack == 4
    assert c.recruit_health == 6
    assert c.attack == 8
    assert c.health == 12
    row = classify_env_minion(m, 2)
    assert row["board_slot"] == 2
    assert row["archetype"] == "Beast"
    assert row["recruit_raw"] == 10
    assert row["combat_raw"] == 20
    assert row["golden"] is True


def test_tier_bucket_sum_equals_survivor_tier_sum():
    winners = [
        Combatant(80, 80, name="big", card_id="B", tier=6),
        Combatant(70, 70, name="mid", card_id="M", tier=4),
    ]
    losers = [Combatant(1, 1, name="chump", card_id="C", tier=1)]
    rng = random.Random(0)
    trace = {}
    raw = simulate_once(winners, losers, rng, tier_a=5, tier_b=2, trace=trace)
    assert raw == 5 + 2
    assert tier_sum(trace["survivors"]) == trace["survivor_tier_sum"] == 10
    fight = {
        "actual_survivors": trace["survivors"],
        "actual_survivor_tier_sum": trace["survivor_tier_sum"],
    }
    assert fight_tier_buckets_reconcile(fight)


def test_survivors_are_subset_of_starting_or_created():
    """Living starting tank stays; deathrattle token is a created body."""
    golem = Combatant(2, 2, name="Harvest Golem", card_id="G", tier=2)
    golem.deathrattle = Summon(1, 2, 1, name="Damaged Golem")
    tank = Combatant(40, 40, name="tank", card_id="T", tier=5)
    winners = [golem, tank]
    losers = [Combatant(2, 1, name="spike", card_id="S", tier=1)]
    rng = random.Random(1)
    trace = {}
    raw = simulate_once(winners, losers, rng, tier_a=4, tier_b=2, trace=trace)
    assert raw > 0
    assert survivors_subset_of_traced(
        trace["survivors"], trace["starting_winner"], trace["created_winner"]
    )
    start_ids = {s["body_id"] for s in trace["starting_winner"]}
    created_ids = {s["body_id"] for s in trace["created_winner"]}
    surv_ids = {s["body_id"] for s in trace["survivors"]}
    assert surv_ids <= (start_ids | created_ids)
    assert any(s.get("token") for s in trace["created"]) or any(
        s.get("origin") == "token" for s in trace["created"]
    )
    assert "tank" in {s["name"] for s in trace["survivors"]}


def test_trace_does_not_change_simulate_once_return_or_rng():
    a = [Combatant(5, 5, name="a", tier=3), Combatant(3, 4, name="b", tier=2)]
    b = [Combatant(4, 4, name="c", tier=4)]
    r1 = random.Random(42)
    r2 = random.Random(42)
    raw1 = simulate_once(a, b, r1, tier_a=3, tier_b=2)
    raw2 = simulate_once(a, b, r2, tier_a=3, tier_b=2, trace={})
    assert raw1 == raw2
    assert r1.getstate() == r2.getstate()


def _play(seed, with_hook):
    env = BGEnv(seed=seed)
    tracer = None
    if with_hook:
        tracer = SurvivorCompositionTracer(0, seed, "obs")
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
        assert f.get("tier_buckets_reconcile") is True
        assert f.get("survivors_subset_of_traced") is True
        assert fight_tier_buckets_reconcile(f)
        assert survivors_subset_of_traced(
            f["actual_survivors"],
            f.get("start_combat_bodies") or [],
            f.get("created_winner") or [],
        )


def test_kitagawa_adds_to_starting_origin_gap():
    control = {
        "start_combat_tier_histogram": {"1": 3.0, "2": 2.0, "3": 1.0,
                                        "4": 0.5, "5": 0.2, "6": 0.0},
        "survival_prob_by_tier": {"1": 0.2, "2": 0.3, "3": 0.4,
                                  "4": 0.5, "5": 0.6, "6": None},
        "mean_surv_generated_tier_sum": 0.10,
        "mean_survivor_tier_sum": 5.336130114883181,
    }
    treatment = {
        "start_combat_tier_histogram": {"1": 1.5, "2": 1.5, "3": 1.5,
                                        "4": 1.5, "5": 0.8, "6": 0.2},
        "survival_prob_by_tier": {"1": 0.15, "2": 0.35, "3": 0.45,
                                  "4": 0.65, "5": 0.75, "6": 0.8},
        "mean_surv_generated_tier_sum": 0.20,
        "mean_survivor_tier_sum": 9.345853113655403,
    }
    d = decompose_gap(control, treatment)
    start = d["starting_origin_gap"]
    assert abs(d["fielded_composition_A"] + d["within_tier_survival_B"] - start) < 1e-9
    assert abs(
        d["explained_A_plus_B_plus_C"]
        - (d["fielded_composition_A"] + d["within_tier_survival_B"]
           + d["token_generated_C"])
    ) < 1e-9
    # Control never fields T6 — exclusive support is composition, not survival.
    t6 = d["per_tier"]["6"]
    assert t6["exclusive_support"] is True
    assert t6["kitagawa_survival"] == 0.0
    assert abs(t6["kitagawa_fielded"] - t6["starting_origin_tier_sum_delta"]) < 1e-9


def test_diagnose_routes_three_ways():
    fielded = diagnose_phase_2v({
        "decomposition": {
            "share_fielded_composition": 0.70,
            "share_within_tier_survival": 0.20,
            "share_token_generated": 0.10,
        }
    })
    assert fielded["primary_finding"] == "fielded_composition_dominates"
    assert "2Q" in fielded["recommended_next_step"]

    surv = diagnose_phase_2v({
        "decomposition": {
            "share_fielded_composition": 0.20,
            "share_within_tier_survival": 0.65,
            "share_token_generated": 0.15,
        }
    })
    assert surv["primary_finding"] == "within_tier_survival_dominates"

    tok = diagnose_phase_2v({
        "decomposition": {
            "share_fielded_composition": 0.10,
            "share_within_tier_survival": 0.20,
            "share_token_generated": 0.70,
        }
    })
    assert tok["primary_finding"] == "token_generated_dominates"

    mixed = diagnose_phase_2v({
        "decomposition": {
            "share_fielded_composition": 0.40,
            "share_within_tier_survival": 0.35,
            "share_token_generated": 0.25,
        }
    })
    assert mixed["primary_finding"] == "mixed_survivor_composition"
    assert SHARE_DOMINANT == 0.55

    smoke = diagnose_phase_2v(fielded, non_evaluative=True)
    assert smoke["primary_finding"] == "measurement_smoke_non_evaluative"
    assert smoke["evaluative"] is False
