"""Draft/choice recommender tests — all stdlib (heuristic scorer), no torch."""

import pytest

from hsbg_coach.draft import (
    rank_heroes, rank_trinkets, rank_discover, recommend_choice,
)
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.cards import CardKnowledge
from hsbg_coach.stats import StatsDB, HeroStats, TrinketStats


def _db():
    heroes = [
        HeroStats(card_id="H1", name="Strong Hero", average_position=2.8,
                  best_tribes=["Murloc"], playstyle="tempo"),
        HeroStats(card_id="H2", name="Weak Hero", average_position=4.6),
    ]
    trinkets = [
        TrinketStats(name="Great Trinket", card_id="T1", average_position=3.1, tier="A"),
        TrinketStats(name="Bad Trinket", card_id="T2", average_position=4.9, tier="D"),
    ]
    return StatsDB(heroes, [], trinkets)


def test_rank_heroes_best_placement_first():
    out = rank_heroes(["Weak Hero", "Strong Hero"], _db())
    assert out[0].name == "Strong Hero" and out[1].name == "Weak Hero"
    assert "favors Murloc" in out[0].reason


def test_rank_heroes_unknown_defaults_gracefully():
    out = rank_heroes(["Nonexistent Hero"], _db())
    assert out[0].name == "Nonexistent Hero" and "no stats" in out[0].reason


def test_rank_trinkets_best_first():
    out = rank_trinkets(["Bad Trinket", "Great Trinket"], _db())
    assert out[0].name == "Great Trinket"


EMB = {"m1": [1.0, 0.0], "m2": [1.0, 0.0], "good": [1.0, 0.0], "bad": [0.0, 1.0]}


def _kb():
    cks = [
        CardKnowledge(card_id="m1", name="m1", tier=2, attack=3, health=3,
                      tribes=["Murloc"], keywords=[], text=""),
        CardKnowledge(card_id="m2", name="m2", tier=2, attack=3, health=3,
                      tribes=["Murloc"], keywords=[], text=""),
        CardKnowledge(card_id="good", name="good", tier=3, attack=5, health=5,
                      tribes=["Murloc"], keywords=[], text=""),
        CardKnowledge(card_id="bad", name="bad", tier=3, attack=2, health=1,
                      tribes=["Beast"], keywords=[], text=""),
    ]
    return {c.card_id: c for c in cks}


def test_rank_discover_prefers_board_fit():
    board = [{"name": "m1", "attack": 3, "health": 3},
             {"name": "m2", "attack": 3, "health": 3}]
    out = rank_discover(["bad", "good"], board, _kb(), scorer=HeuristicScorer(EMB))
    assert out[0].name == "good"          # on-tribe, higher-stat minion wins
    assert "equity" in out[0].reason


def _mech_kb():
    cks = [CardKnowledge(card_id=f"mm{i}", name=f"mm{i}", tier=2, attack=3,
                         health=3, tribes=["Mech"], keywords=[], text="")
           for i in range(2)]
    return {c.card_id: c for c in cks}


def test_rank_trinkets_uses_board_fit():
    from hsbg_coach.economy import HeroContext
    db = StatsDB([], [], [
        TrinketStats(name="MechT", card_id="mt", average_position=3.6, tier="A",
                     text="Give your Mechs +2/+2."),
        TrinketStats(name="GenericT", card_id="gt", average_position=3.0, tier="S",
                     text="Gain 2 gold."),
        TrinketStats(name="MurlocT", card_id="lt", average_position=3.5, tier="A",
                     text="Your Murlocs gain Poisonous."),
    ])
    board = [{"name": "mm0"}, {"name": "mm1"}]
    out = rank_trinkets(["MechT", "GenericT", "MurlocT"], db, board=board,
                        kb=_mech_kb(), hero_ctx=HeroContext(target_tribe="Mech"))
    names = [c.name for c in out]
    # Mech trinket gets a fit boost (passes the off-tribe Murloc one); generic
    # S-tier still leads on raw strength.
    assert names.index("MechT") < names.index("MurlocT")
    assert any("matches your Mechs" in c.reason for c in out if c.name == "MechT")


def test_trinket_placement_only_without_board():
    db = StatsDB([], [], [
        TrinketStats(name="A", card_id="a", average_position=4.0, tier="B",
                     text="Give your Mechs +2/+2."),
    ])
    out = rank_trinkets(["A"], db)            # no board context -> no fit applied
    assert out[0].rank_value == 4.0


def test_recommend_choice_dispatch_and_unknown():
    assert recommend_choice("hero", ["Strong Hero"], db=_db())[0].name == "Strong Hero"
    with pytest.raises(ValueError):
        recommend_choice("bogus", ["x"])
