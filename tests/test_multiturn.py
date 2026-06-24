"""Multi-turn planner tests — assert robust *qualitative* strategy behavior
(stdlib; uses the committed pace curves)."""

from hsbg_coach.multiturn import plan_multiturn, best_plan, STRATEGIES
from hsbg_coach.pace import load_pace

PACE = load_pace()


def _snap(turn, tier, hp, stats):
    return {"turn": turn, "tavern_tier": tier, "hero_health": hp,
            "board": [{"name": "x", "attack": stats // 2, "health": stats - stats // 2}]}


def _rank(plans):
    return {p.name: i for i, p in enumerate(plans)}


def test_under_tiered_prefers_leveling_over_pure_tempo():
    plans = plan_multiturn(_snap(6, 2, 25, 50), PACE, horizon=3)
    r = _rank(plans)
    assert r["Level now"] < r["Tempo"]            # behind on tier -> catch up


def test_low_hp_below_curve_never_greeds():
    plans = plan_multiturn(_snap(6, 3, 6, 35), PACE, horizon=3)
    assert plans[0].name == "Tempo"               # stabilize, don't die
    assert plans[0].died is False
    # every leveling line is projected to kill you here
    assert all(p.died for p in plans if "Level" in p.name)


def test_longer_horizon_rewards_leveling_more():
    """The further ahead you plan, the more a tier investment pays off."""
    short = _rank(plan_multiturn(_snap(6, 3, 35, 80), PACE, horizon=3))
    long = _rank(plan_multiturn(_snap(6, 3, 35, 80), PACE, horizon=5))
    assert long["Level now"] <= short["Level now"]   # leveling climbs the ranking


def test_projection_shape_and_hp_decay():
    plan = best_plan(_snap(5, 2, 30, 20), PACE, horizon=4)
    assert len(plan.projection) == 4
    assert [p.turn for p in plan.projection] == [5, 6, 7, 8]
    assert plan.projection[-1].hp <= 30                # HP only goes down


def test_all_strategies_scored():
    plans = plan_multiturn(_snap(7, 3, 25, 100), PACE, horizon=3)
    assert {p.name for p in plans} == set(STRATEGIES)
    assert plans == sorted(plans, key=lambda p: p.value, reverse=True)
