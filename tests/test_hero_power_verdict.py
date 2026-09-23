"""HSReplay's hero-power verdicts drive hero select and Hero Power in NEXT."""
from hsbg_coach.draft import hero_draft_plan, rank_heroes
from hsbg_coach.hero_pick import hsreplay_row
from hsbg_coach.hero_power_verdict import usable_hp_bullets, weak_quote
from hsbg_coach.hsreplay_guides import hero_guide_hp_adjust, hero_guide_lines
from hsbg_coach.stats import StatsDB

TAE = "Tae'thelan Bloodwatcher"


def _snap(hero, **kw):
    s = {"hero_name": hero, "gold": 5, "tavern_tier": 3, "turn": 5,
         "hero_power": {"usable": True, "cost": 1}, "shop": []}
    s.update(kw)
    return s


def test_weak_verdicts_come_from_hsreplay_text():
    assert "Don't expect much value" in weak_quote(hsreplay_row(TAE))
    assert "isn't that good" in weak_quote(hsreplay_row("Edwin VanCleef"))
    assert weak_quote(hsreplay_row("Lord Jaraxxus")) is None


def test_weak_hero_power_hero_is_not_picked():
    offer = ["Xyrella", "The Great Akazamzarak", "Guff Runetotem", TAE]
    # Tae'thelan's raw 1st rate beats Xyrella's; the weak-HP verdict sinks it.
    first = lambda n: hsreplay_row(n)["stats"]["final_placement_distribution"][0]
    assert first(TAE) > first("Xyrella")
    ranked = [c.name for c in rank_heroes(offer, StatsDB([], [], []))]
    assert ranked[-1] == TAE
    plan = hero_draft_plan(offer, StatsDB([], [], []))
    assert plan["lines"][0].startswith(f"Reroll: {TAE}")
    assert "weak hero power" in plan["lines"][0]


def test_weak_hero_power_is_never_next():
    d, why = hero_guide_hp_adjust(_snap(TAE), 1)
    assert d > 0 and "weak hero power" in why
    assert not any(l.startswith("Hero Power") for l in hero_guide_lines(_snap(TAE)))


def test_do_not_lines_are_not_hero_power_recommendations():
    row = hsreplay_row("Professor Putricide")
    assert not any("Do not" in b for b in usable_hp_bullets(row))
    # "Do not hero power on tavern 2" — no promotion at tavern 2, fine at 3.
    assert hero_guide_hp_adjust(_snap("Professor Putricide", tavern_tier=2), 1) == (0.0, None)
    assert hero_guide_hp_adjust(_snap("Professor Putricide", tavern_tier=3), 1)[0] < 0
