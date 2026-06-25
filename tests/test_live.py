"""Live-coaching tests: the advice provider + overlay rendering are pure and run
headless (no Tk display, no torch — heuristic scorer)."""

from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.live import advice_lines, _key
from hsbg_coach.overlay import format_overlay_text

EMB = {"b1": [1.0, 0.0], "b2": [1.0, 0.0], "b3": [1.0, 0.0],
       "good": [1.0, 0.0], "bad": [0.0, 1.0]}


def _recruit():
    return {"phase": "recruit", "turn": 5, "tavern_tier": 2, "gold": 7,
            "hero_health": 30,
            "board": [{"name": f"b{i+1}", "attack": 3, "health": 3} for i in range(3)],
            "shop": [{"name": "good", "attack": 5, "health": 5},
                     {"name": "bad", "attack": 2, "health": 1}]}


def test_advice_lines_during_recruit_recommends_the_strong_buy():
    # The strong minion is recommended and preferred over the weak one. (We don't
    # assert it's #1: with the learned whole-game value, leveling can legitimately
    # outrank a single buy when you're behind on tier.)
    lines = advice_lines(_recruit(), kb=None, scorer=HeuristicScorer(EMB))
    assert lines and any("Buy good" in l for l in lines)
    if any("Buy bad" in l for l in lines):
        good_i = next(i for i, l in enumerate(lines) if "Buy good" in l)
        bad_i = next(i for i, l in enumerate(lines) if "Buy bad" in l)
        assert good_i < bad_i


def test_no_advice_without_a_shop():
    snap = {"phase": "combat", "board": [{"name": "b1", "attack": 3, "health": 3}],
            "shop": []}
    assert advice_lines(snap, kb=None, scorer=HeuristicScorer(EMB)) == []


def test_cache_key_changes_with_shop_and_gold():
    a = _recruit()
    b = dict(a, gold=4)
    c = dict(a, shop=[{"name": "good"}])
    assert _key(a) != _key(b)
    assert _key(a) != _key(c)
    assert _key(a) == _key(dict(a))


def test_livecoach_waits_for_hearthstone_before_active():
    from hsbg_coach.live import LiveCoach
    coach = LiveCoach(power_log=None)          # no log yet, thread not started
    snap, odds, recos = coach.frame()
    assert snap["phase"] == "waiting" and recos == []
    assert any("Battlegrounds" in n for n in snap["notes"])


def test_overlay_renders_recommendations_prominently():
    text = format_overlay_text(_recruit(), recommendations=["Buy good (+5%)", "Roll the shop"])
    assert "▸ Recommended:" in text
    assert "1. Buy good (+5%)" in text
    # recommendations appear above the board section
    assert text.index("Recommended") < text.index("Your board")
