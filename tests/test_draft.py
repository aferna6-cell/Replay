"""Draft/choice recommender tests — all stdlib (heuristic scorer), no torch."""

import pytest

from hsbg_coach.draft import (
    F2P_HERO_CHOICES, hero_draft_plan, rank_heroes, rank_trinkets,
    rank_discover, recommend_choice,
)
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.cards import CardKnowledge
from hsbg_coach.stats import StatsDB, HeroStats, TrinketStats


def _db():
    heroes = [
        HeroStats(card_id="H1", name="Strong Hero", average_position=2.8,
                  best_tribes=["Murloc"], playstyle="tempo"),
        HeroStats(card_id="H2", name="Weak Hero", average_position=4.6),
        HeroStats(card_id="H3", name="Mid Hero", average_position=3.5,
                  best_tribes=["Mech"], playstyle="flexible"),
        HeroStats(card_id="H4", name="Okay Hero", average_position=3.9),
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
    # Ranked by (estimated) 1st-place rate: rank_value = -1st %, no fit term.
    from hsbg_coach.first_place import estimate_first
    assert out[0].rank_value == -estimate_first(4.0)
    assert "1st ~" in out[0].reason and "avg 4.00" in out[0].reason


def test_recommend_choice_dispatch_and_unknown():
    assert recommend_choice("hero", ["Strong Hero"], db=_db())[0].name == "Strong Hero"
    with pytest.raises(ValueError):
        recommend_choice("bogus", ["x"])


def test_default_hero_choices_is_four():
    assert F2P_HERO_CHOICES == 4


def test_rank_heroes_default_includes_all_four():
    offered = ["Weak Hero", "Okay Hero", "Mid Hero", "Strong Hero"]
    out = rank_heroes(offered, _db())
    assert len(out) == 4
    assert out[0].name == "Strong Hero"
    assert out[-1].name == "Weak Hero"


def test_rank_heroes_env_cap_override():
    offered = ["Weak Hero", "Okay Hero", "Mid Hero", "Strong Hero"]
    out = rank_heroes(offered, _db(), max_choices=2)
    assert len(out) == 2
    # Truncates by offer order (first two), then ranks those.
    assert {c.name for c in out} == {"Weak Hero", "Okay Hero"}


def test_hero_draft_plan_rerolls_worst_of_four():
    offered = ["Weak Hero", "Okay Hero", "Mid Hero", "Strong Hero"]
    plan = hero_draft_plan(offered, _db(), rerolls_available=1)
    assert plan["reroll"] is not None
    assert plan["reroll"].name == "Weak Hero"
    pick_names = [c.name for c in plan["picks"]]
    assert pick_names == ["Strong Hero", "Mid Hero", "Okay Hero"]
    assert "Weak Hero" not in pick_names
    assert plan["lines"][0].startswith("Reroll: Weak Hero")
    assert "weakest of 4" in plan["lines"][0]
    assert plan["lines"][1] == "Then pick (best first):"
    assert "1. Strong Hero" in plan["lines"][2]


def test_hero_draft_plan_close_worst_still_rerolled():
    """Even when worst is within ~0.15 of 3rd-best, still name a reroll target."""
    db = StatsDB([
        HeroStats(card_id="a", name="A", average_position=3.00),
        HeroStats(card_id="b", name="B", average_position=3.10),
        HeroStats(card_id="c", name="C", average_position=3.20),
        HeroStats(card_id="d", name="D", average_position=3.30),  # within 0.15 of C
    ], [], [])
    plan = hero_draft_plan(["A", "B", "C", "D"], db)
    assert plan["reroll"].name == "D"
    assert [c.name for c in plan["picks"]] == ["A", "B", "C"]


def test_hero_draft_plan_no_reroll_when_all_unknown():
    plan = hero_draft_plan(
        ["Mystery1", "Mystery2", "Mystery3", "Mystery4"], _db(),
    )
    assert plan["reroll"] is None
    assert len(plan["picks"]) == 4
    assert plan["lines"][0].startswith("Hero ranking")


def test_hero_draft_plan_skips_reroll_with_two_heroes():
    plan = hero_draft_plan(["Strong Hero", "Weak Hero"], _db())
    assert plan["reroll"] is None
    assert [c.name for c in plan["picks"]] == ["Strong Hero", "Weak Hero"]


def test_hero_draft_plan_respects_rerolls_available_zero():
    plan = hero_draft_plan(
        ["Weak Hero", "Okay Hero", "Mid Hero", "Strong Hero"], _db(),
        rerolls_available=0,
    )
    assert plan["reroll"] is None
    assert len(plan["picks"]) == 4


def test_hsbg_hero_choices_env_override(monkeypatch):
    """HSBG_HERO_CHOICES env still controls the module default after reload."""
    import importlib
    import hsbg_coach.draft as draft
    monkeypatch.setenv("HSBG_HERO_CHOICES", "2")
    importlib.reload(draft)
    try:
        assert draft.F2P_HERO_CHOICES == 2
        offered = ["Weak Hero", "Okay Hero", "Mid Hero", "Strong Hero"]
        out = draft.rank_heroes(offered, _db())
        assert len(out) == 2
    finally:
        monkeypatch.delenv("HSBG_HERO_CHOICES", raising=False)
        importlib.reload(draft)
        assert draft.F2P_HERO_CHOICES == 4


def test_offer_advice_lines_hero_shows_reroll():
    from hsbg_coach.choices import ChoiceOffer, offer_advice_lines
    offer = ChoiceOffer(
        kind="hero",
        card_ids=["H1", "H2", "H3", "H4"],
        names=["Strong Hero", "Weak Hero", "Mid Hero", "Okay Hero"],
    )
    lines = offer_advice_lines(offer, db=_db())
    assert lines[0].startswith("Reroll: Weak Hero")
    assert "Then pick (best first):" in lines
    joined = "\n".join(lines)
    assert "Strong Hero" in joined and "Weak Hero" in lines[0]
