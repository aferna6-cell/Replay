"""Hero select ranks by HSReplay hero placement, with a small lobby-fit nudge."""
from hsbg_coach.draft import hero_draft_plan, rank_heroes
from hsbg_coach.hero_pick import hsreplay_row, lobby_fit
from hsbg_coach.stats import StatsDB, HeroStats

OFFER = ["Trade Prince Gallywix", "Deathwing", "Lord Jaraxxus", "N'Zoth"]


def _avg(name):
    return hsreplay_row(name)["stats"]["avg_final_placement"]


def test_heroes_ranked_by_hsreplay_placement():
    out = rank_heroes(OFFER, StatsDB([], [], []))
    names = [c.name for c in out]
    assert names == sorted(OFFER, key=_avg), names
    assert out[0].reason.startswith("HSReplay avg")
    assert "1st " in out[0].reason and "top-4 " in out[0].reason


def test_hsreplay_beats_firestone_fallback():
    """Firestone says Gallywix is best; HSReplay placement wins."""
    db = StatsDB([HeroStats(card_id="x", name="Trade Prince Gallywix",
                            average_position=2.0)], [], [])
    out = rank_heroes(OFFER, db)
    assert out[0].name != "Trade Prince Gallywix"
    assert out[-1].name == "Trade Prince Gallywix"


def test_lobby_fit_uses_hsreplay_favorable_tribes():
    no_mech = ["Beast", "Demon", "Murloc", "Pirate", "Undead"]
    pen, note = lobby_fit("Deathwing", no_mech)            # HSReplay: Mech hero
    assert pen > 0 and "not in this lobby" in note
    ok, note2 = lobby_fit("Deathwing", no_mech[:-1] + ["Mech"])
    assert ok <= 0 and "Mech" in note2
    assert lobby_fit("Deathwing", None) == (0.0, None)      # lobby unknown: no nudge


def test_reroll_line_shows_real_hsreplay_average():
    plan = hero_draft_plan(OFFER, StatsDB([], [], []),
                           available_tribes=["Beast", "Demon", "Murloc", "Pirate", "Undead"])
    worst = max(OFFER, key=_avg)
    assert plan["lines"][0].startswith(f"Reroll: {worst} — HSReplay avg {_avg(worst):.2f}")
