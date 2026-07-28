"""Heuristic economy advisor tests — assert the fundamentals fire correctly."""

from hsbg_coach.economy import advise, top_advice, EconomyConfig
from hsbg_coach.bg import ActionType


def snap(**over):
    base = {
        "turn": 2, "tavern_tier": 1, "gold": 4, "hero_health": 30,
        "board": [], "shop": [{"name": "Alleycat", "attack": 1, "health": 1}],
    }
    base.update(over)
    return base


def _actions(suggestions):
    return [s.action for s in suggestions]


def test_buys_to_fill_board_early():
    s = advise(snap(turn=1, gold=3, board=[]))
    assert ActionType.BUY.value in _actions(s)
    assert s[0].action == ActionType.BUY.value           # buying is top priority early


def test_levels_when_healthy_with_gold():
    s = advise(snap(turn=3, gold=6, hero_health=35, tavern_tier=1,
                    board=[{"name": "x", "attack": 1, "health": 1}]))
    assert ActionType.TIER_UP.value in _actions(s)


def test_low_health_deprioritizes_leveling():
    healthy = advise(snap(turn=4, gold=6, hero_health=35, tavern_tier=2,
                          board=[{"attack": 2, "health": 2}]))
    hurt = advise(snap(turn=4, gold=6, hero_health=8, tavern_tier=2,
                       board=[{"attack": 2, "health": 2}]))
    lvl_h = next(x for x in healthy if x.action == ActionType.TIER_UP.value)
    lvl_u = next(x for x in hurt if x.action == ActionType.TIER_UP.value)
    assert lvl_u.priority < lvl_h.priority                # greed costs more when hurt


def test_full_board_suggests_sell_for_upgrade():
    board = [{"name": f"m{i}", "attack": 1, "health": 1} for i in range(7)]
    s = advise(snap(gold=3, board=board,
                    shop=[{"name": "Big", "attack": 5, "health": 5}]))
    assert ActionType.SELL.value in _actions(s)


def test_freeze_when_multiple_wanted_unaffordable():
    s = advise(snap(gold=3, board=[{"attack": 1, "health": 1}],
                    shop=[{"name": "a", "attack": 3, "health": 3},
                          {"name": "b", "attack": 3, "health": 3},
                          {"name": "c", "attack": 3, "health": 3}]))
    assert ActionType.FREEZE.value in _actions(s)         # 3 wanted, can afford 1


def test_roll_when_board_full_and_nothing_better():
    board = [{"attack": 9, "health": 9} for _ in range(7)]
    s = advise(snap(gold=2, tavern_tier=6, board=board,
                    shop=[{"attack": 1, "health": 1}]))
    assert ActionType.ROLL.value in _actions(s)


def test_unknown_gold_is_surfaced_not_guessed():
    s = advise(snap(gold=None))
    assert len(s) == 1 and "Gold unknown" in s[0].rationale


def test_top_advice_is_a_string():
    assert isinstance(top_advice(snap(turn=1, gold=3)), str)


def test_thresholds_are_configurable():
    cfg = EconomyConfig(level_min_gold=10)
    # With a high level threshold, 6 gold no longer triggers leveling.
    s = advise(snap(turn=3, gold=6, hero_health=35, tavern_tier=1,
                    board=[{"attack": 1, "health": 1}]), cfg)
    assert ActionType.TIER_UP.value not in _actions(s)
