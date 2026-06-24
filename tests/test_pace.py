"""Pace benchmark tests: lookups, advice, recommend integration, real snapshot."""

from hsbg_coach.pace import load_pace, pace_advice, board_stats, _at
from hsbg_coach.recommend import recommend
from hsbg_coach.bg import ActionType

# Synthetic top-10%-shaped pace.
PACE = {
    "leveling": {1: 1.0, 2: 1.3, 3: 1.65, 4: 2.0, 5: 2.5, 6: 3.0, 7: 3.3},
    "scaling": {1: 3.0, 2: 6.0, 3: 14.0, 4: 25.0, 5: 42.0, 6: 80.0, 7: 141.0},
}


def snap(**over):
    base = {"turn": 6, "tavern_tier": 3, "gold": 6, "hero_health": 30,
            "board": [{"attack": 10, "health": 10}]}
    base.update(over)
    return base


def test_at_lookup_uses_nearest_below():
    assert _at(PACE["leveling"], 6) == 3.0
    assert _at(PACE["leveling"], 9) == 3.3        # beyond max -> last known
    assert _at(PACE["leveling"], 1) == 1.0


def test_board_stats_sums_attack_and_health():
    assert board_stats(snap(board=[{"attack": 3, "health": 4},
                                   {"attack": 1, "health": 1}])) == 9


def test_behind_leveling_flagged():
    v = pace_advice(snap(turn=6, tavern_tier=2), PACE)   # tier 2 on turn 6, bench 3.0
    assert v.behind_leveling is True
    assert any("Behind top-10% leveling" in n for n in v.notes)


def test_on_pace_not_flagged():
    v = pace_advice(snap(turn=6, tavern_tier=3), PACE)   # exactly on pace
    assert v.behind_leveling is False and v.ahead_leveling is False


def test_ahead_leveling_flagged():
    v = pace_advice(snap(turn=4, tavern_tier=3), PACE)   # tier 3 on turn 4, bench 2.0
    assert v.ahead_leveling is True


def test_behind_scaling_flagged():
    v = pace_advice(snap(turn=6, tavern_tier=3, board=[{"attack": 5, "health": 5}]), PACE)
    assert v.behind_scaling is True               # 10 stats vs ~80 expected


def test_recommend_emits_pace_tier_up_when_behind():
    recs = recommend(snap(turn=6, tavern_tier=2, gold=5), pace=PACE)
    pace_recs = [r for r in recs if r.source == "pace"]
    assert pace_recs and pace_recs[0].action == ActionType.TIER_UP.value


def test_recommend_no_pace_without_data():
    recs = recommend(snap())
    assert all(r.source != "pace" for r in recs)


def test_real_committed_pace_loads_and_is_monotonic():
    pace = load_pace()
    assert pace and pace["leveling"] and pace["scaling"]
    lv = pace["leveling"]
    # leveling tier should be non-decreasing across the early turns
    turns = sorted(t for t in lv if t <= 8)
    vals = [lv[t] for t in turns]
    assert vals == sorted(vals)
