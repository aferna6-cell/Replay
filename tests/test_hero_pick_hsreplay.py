"""Hero select ranks by HSReplay 1st-place rate, with a small lobby-fit nudge."""
from hsbg_coach.draft import hero_draft_plan, rank_heroes
from hsbg_coach.hero_pick import estimate_first, hsreplay_row, lobby_fit
from hsbg_coach.stats import StatsDB, HeroStats

OFFER = ["Trade Prince Gallywix", "Deathwing", "Lord Jaraxxus", "N'Zoth"]


def _first(name):
    return hsreplay_row(name)["stats"]["final_placement_distribution"][0]


def _avg(name):
    return hsreplay_row(name)["stats"]["avg_final_placement"]


def test_heroes_ranked_by_hsreplay_first_place_rate():
    out = rank_heroes(OFFER, StatsDB([], [], []))
    names = [c.name for c in out]
    assert names == sorted(OFFER, key=_first, reverse=True), names
    assert out[0].reason.startswith("HSReplay 1st ")
    assert "top-4 " in out[0].reason and "avg " in out[0].reason


def test_more_firsts_beats_better_average():
    """Gallywix averages worse than Deathwing on HSReplay but wins more lobbies."""
    assert _avg("Trade Prince Gallywix") > _avg("Deathwing")
    assert _first("Trade Prince Gallywix") > _first("Deathwing")
    names = [c.name for c in rank_heroes(OFFER, StatsDB([], [], []))]
    assert names.index("Trade Prince Gallywix") < names.index("Deathwing")


def test_hsreplay_beats_firestone_fallback():
    """Firestone says Deathwing is best; HSReplay 1st-place rate wins."""
    db = StatsDB([HeroStats(card_id="x", name="Deathwing", average_position=2.0)], [], [])
    out = rank_heroes(OFFER, db)
    assert out[0].name != "Deathwing"
    assert out[-1].name == "Deathwing"


def test_firestone_fallback_estimates_first_rate():
    db = StatsDB([HeroStats(card_id="a", name="Alpha Test Hero", average_position=3.0),
                  HeroStats(card_id="b", name="Beta Test Hero", average_position=4.2)], [], [])
    out = rank_heroes(["Beta Test Hero", "Alpha Test Hero"], db)
    assert [c.name for c in out] == ["Alpha Test Hero", "Beta Test Hero"]
    assert f"1st ~{estimate_first(3.0):.0f}% (est.)" in out[0].reason


def test_lobby_fit_uses_hsreplay_favorable_tribes():
    no_mech = ["Beast", "Demon", "Murloc", "Pirate", "Undead"]
    pen, note = lobby_fit("Deathwing", no_mech)            # HSReplay: Mech hero
    assert pen > 0 and "not in this lobby" in note
    ok, note2 = lobby_fit("Deathwing", no_mech[:-1] + ["Mech"])
    assert ok <= 0 and "Mech" in note2
    assert lobby_fit("Deathwing", None) == (0.0, None)      # lobby unknown: no nudge


def test_reroll_line_shows_real_hsreplay_first_rate():
    plan = hero_draft_plan(OFFER, StatsDB([], [], []),
                           available_tribes=["Beast", "Demon", "Murloc", "Pirate", "Undead"])
    worst = min(OFFER, key=_first)
    assert plan["lines"][0].startswith(f"Reroll: {worst} — HSReplay 1st {_first(worst):.0f}%")


def test_trinkets_ranked_by_first_place_rate(monkeypatch):
    """A real HSReplay 1st-place rate beats a better average placement."""
    from hsbg_coach import hsreplay_guides
    from hsbg_coach.draft import rank_trinkets
    from hsbg_coach.stats import TrinketStats
    rows = {
        "Wins Lobbies": {"name": "Wins Lobbies", "stats": {
            "avg_final_placement": 4.1,
            "final_placement_distribution": [24.0, 12, 10, 9, 9, 10, 12, 14]}},
        "Safe Top4": {"name": "Safe Top4", "stats": {
            "avg_final_placement": 3.6,
            "final_placement_distribution": [15.0, 17, 16, 15, 12, 10, 8, 7]}},
    }
    monkeypatch.setattr(hsreplay_guides, "lookup_trinket", lambda n: rows.get(n))
    db = StatsDB([], [], [TrinketStats(name=n, card_id="", average_position=4.0)
                          for n in rows])
    out = rank_trinkets(["Safe Top4", "Wins Lobbies"], db)
    assert [c.name for c in out] == ["Wins Lobbies", "Safe Top4"]
    assert out[0].reason.startswith("1st 24%")
