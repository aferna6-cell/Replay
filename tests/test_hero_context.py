"""Hero/comp-aware advice plumbing tests."""

from hsbg_coach.economy import advise, HeroContext
from hsbg_coach.bg import ActionType


def snap(**over):
    base = {
        "turn": 5, "tavern_tier": 2, "gold": 6, "hero_health": 30,
        "board": [{"name": "filler", "attack": 1, "health": 1}],
        "shop": [
            {"name": "Big Stats", "attack": 6, "health": 6, "tribe": "Beast"},
            {"name": "Murloc Knight", "attack": 2, "health": 2, "tribe": "Murloc"},
        ],
    }
    base.update(over)
    return base


def _buy(suggestions):
    return next((s for s in suggestions if s.action == ActionType.BUY.value), None)


def test_no_context_buys_highest_stats():
    buy = _buy(advise(snap()))
    assert buy is not None
    assert "Big Stats" in buy.rationale          # raw-stat pick without a plan


def test_target_tribe_prefers_on_comp_minion():
    ctx = HeroContext(hero="Old Murk-Eye", target_tribe="Murloc")
    buy = _buy(advise(snap(), hero_ctx=ctx))
    assert "Murloc Knight" in buy.rationale       # on-comp beats raw stats
    assert buy.detail["on_comp"] is True


def test_recommended_minion_is_prioritized():
    ctx = HeroContext(recommended_minions=["Murloc Knight"])
    buy = _buy(advise(snap(), hero_ctx=ctx))
    assert "Murloc Knight" in buy.rationale


def test_level_aggression_raises_tier_up_priority():
    base = advise(snap())
    eager = advise(snap(), hero_ctx=HeroContext(hero="Galakrond", level_aggression=0.25))
    lvl_base = next(s for s in base if s.action == ActionType.TIER_UP.value)
    lvl_eager = next(s for s in eager if s.action == ActionType.TIER_UP.value)
    assert lvl_eager.priority > lvl_base.priority


def test_context_is_optional_backward_compatible():
    # Calling without context still works (existing behavior).
    assert advise(snap()) and isinstance(advise(snap()), list)
