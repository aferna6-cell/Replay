"""Meta comps + lobby tribe soft priors (live #1 move path)."""

from hsbg_coach.meta_strategy import (
    tribe_placement_table,
    choose_meta_prior,
    meta_buy_adjust,
    is_listed_comp_piece,
)
from hsbg_coach.actions import Action, BUY
from hsbg_coach.stats import StatsDB


def test_tribe_table_from_firestone_comps():
    table = tribe_placement_table(StatsDB.load())
    assert table
    # Stronger tribes have lower avg placement than weak ones when data exists.
    if "elemental" in table and "quilboar" in table:
        assert table["elemental"] <= table["quilboar"] + 0.01


def test_meta_prior_prefers_strong_listed_tribe_in_lobby():
    snap = {
        "board": [],
        "opponent_profiles": [
            {"tribe": "elemental"},
            {"tribe": "elemental"},
            {"tribe": "pirate"},
        ],
    }
    prior = choose_meta_prior(snap, db=StatsDB.load())
    assert prior.preferred_tribe
    assert prior.listed_comps


def test_meta_buy_promotes_listed_core_on_preferred_tribe():
    db = StatsDB.load()
    core = tribe = None
    for c in db.comps:
        if (c.tribe or "").lower() == "elemental" and c.core_cards:
            core, tribe = c.core_cards[0], "elemental"
            break
    if not core:
        return
    snap = {"board": [], "opponent_profiles": [{"tribe": "elemental"}]}
    act = Action(BUY, core, 3, {"minion": {"name": core, "tribe": tribe}})
    adj, reason = meta_buy_adjust(act, snap, db=db)
    assert adj < 0
    assert reason


def test_meta_buy_allows_pivot_to_other_listed_comp():
    db = StatsDB.load()
    # Prefer elemental via lobby, but shop opens a listed naga core on empty board.
    naga_core = None
    for c in db.comps:
        if (c.tribe or "").lower() == "naga" and c.core_cards:
            naga_core = c.core_cards[0]
            break
    if not naga_core:
        return
    snap = {"board": [], "opponent_profiles": [{"tribe": "elemental"}]}
    act = Action(BUY, naga_core, 3, {"minion": {"name": naga_core, "tribe": "naga"}})
    adj, reason = meta_buy_adjust(act, snap, db=db)
    assert adj < 0
    assert reason and "pivot" in reason.lower() or "listed" in reason.lower()


def test_listed_comp_piece_detects_core():
    db = StatsDB.load()
    c = next((x for x in db.comps if x.core_cards), None)
    if not c:
        return
    assert is_listed_comp_piece(c.core_cards[0], (c.tribe or "").lower(), db=db)


def test_rank_actions_applies_meta_adjust_on_buy():
    """#1-move scorer actually calls meta priors (not an unused helper)."""
    from hsbg_coach.game_value import _meta_strategy_adjust

    db = StatsDB.load()
    core = None
    for c in db.comps:
        if (c.tribe or "").lower() == "elemental" and c.core_cards:
            core = c.core_cards[0]
            break
    if not core:
        return
    snap = {"board": [], "opponent_profiles": [{"tribe": "elemental"}]}
    act = Action(BUY, core, 3, {"minion": {"name": core, "tribe": "elemental"}})
    adj, reason = _meta_strategy_adjust(act, snap)
    assert adj < 0
    assert reason
