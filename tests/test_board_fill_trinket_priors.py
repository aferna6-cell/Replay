"""Soft priors: fill board before hard-roll, cut off-direction, play into trinkets.

Aidan playtest (soft rules, never absolute):
  * Sparse board + buyable shop → NEXT is buy, not roll.
  * Full board + trash shop → roll OK.
  * Filled-enough + solid on-direction in shop → don't endless-roll.
  * Direction committed → soft-prefer cutting useless off-direction.
  * Trinket pick must align with board/plan (no random Battlecry premium).
  * Equipped trinket soft-boosts matching buys / demotes selling enablers.
"""

from __future__ import annotations

from hsbg_coach.actions import Action, BUY, ROLL, SELL
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.comp_signals import (
    minion_trinket_buy_adjust, off_trinket_buy_penalty, sell_trinket_penalty,
)
from hsbg_coach.draft import _trinket_fit
from hsbg_coach.game_value import rank_actions
from hsbg_coach.jeef_priors import (
    anti_stuck_roll_adjust, board_fill_roll_adjust, direction_cut_sell_adjust,
    midgame_solid_buy_adjust,
)
from hsbg_coach.pace import load_pace

PACE = load_pace()
SC = HeuristicScorer({})


def _rank(snap):
    recs, _ = rank_actions(snap, kb=None, scorer=SC, pace=PACE)
    return recs


def test_sparse_board_mediocre_shop_prefers_buy_not_roll():
    snap = {
        "turn": 5, "tavern_tier": 3, "gold": 7, "hero_health": 30,
        "board": [
            {"name": "a", "attack": 3, "health": 3, "tribes": ["Beast"], "tier": 2},
            {"name": "b", "attack": 2, "health": 4, "tribes": ["Beast"], "tier": 2},
        ],
        "shop": [
            {"name": "filler", "attack": 4, "health": 4, "tribes": ["Beast"], "tier": 3},
            {"name": "ok", "attack": 3, "health": 5, "tribes": ["Mech"], "tier": 3},
        ],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    recs = _rank(snap)
    assert recs[0].action.kind == BUY, (
        f"NEXT should be buy on sparse board, got {recs[0].action.kind}: {recs[0].reason}"
    )
    roll = next(r for r in recs if r.action.kind == ROLL)
    buy = next(r for r in recs if r.action.kind == BUY and r.action.target == "filler")
    assert buy.placement < roll.placement


def test_full_board_trash_shop_roll_ok():
    snap = {
        "turn": 11, "tavern_tier": 5, "gold": 4, "hero_health": 20,
        "board": [
            {"name": f"m{i}", "attack": 10, "health": 10, "tribes": ["Dragon"], "tier": 5}
            for i in range(7)
        ],
        "shop": [
            {"name": "t1", "attack": 1, "health": 1, "tribes": ["Murloc"], "tier": 1},
            {"name": "t2", "attack": 2, "health": 1, "tribes": ["Pirate"], "tier": 2},
        ],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    # Direct prior: roll should NOT be demoted (trash shop / full board).
    assert board_fill_roll_adjust(snap) == (0.0, None)
    assert anti_stuck_roll_adjust(snap)[0] == 0.0
    recs = _rank(snap)
    assert recs[0].action.kind == ROLL


def test_direction_only_mediocre_buy_still_beats_roll_when_sparse():
    """Works with existing direction-only good buys — on-dir weak body > roll."""
    snap = {
        "turn": 5, "tavern_tier": 3, "gold": 6, "hero_health": 30,
        "board": [
            {"name": "d1", "attack": 3, "health": 3, "tribes": ["Dragon"], "tier": 2},
            {"name": "d2", "attack": 2, "health": 4, "tribes": ["Dragon"], "tier": 2},
        ],
        "shop": [
            {"name": "ondir_weak", "attack": 2, "health": 2, "tribes": ["Dragon"], "tier": 3},
            {"name": "off_strong", "attack": 8, "health": 8, "tribes": ["Pirate"], "tier": 3},
        ],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "ondir_weak"


def test_anti_stuck_roll_when_stable_board_and_solid_shop():
    snap = {
        "turn": 8, "tavern_tier": 4, "gold": 6, "hero_health": 25,
        "board": [
            {"name": f"d{i}", "attack": 5 + i, "health": 5 + i, "tribes": ["Dragon"], "tier": 3}
            for i in range(5)
        ],
        "shop": [
            {"name": "solid", "attack": 6, "health": 6, "tribes": ["Dragon"], "tier": 4},
            {"name": "trash", "attack": 1, "health": 1, "tribes": ["Pirate"], "tier": 1},
        ],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    adj, reason = anti_stuck_roll_adjust(snap)
    assert adj > 0 and reason and "solid" in reason.lower()
    act = Action(BUY, target="solid", cost=3,
                 detail={"minion": snap["shop"][0]})
    badj, _ = midgame_solid_buy_adjust(act, snap)
    assert badj < 0
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "solid"


def test_direction_cut_prefers_selling_off_direction_chaff():
    board = [
        {"name": f"d{i}", "attack": 5, "health": 5, "tribes": ["Dragon"], "tier": 3}
        for i in range(4)
    ] + [{"name": "junk", "attack": 2, "health": 2, "tribes": ["Pirate"], "tier": 2}]
    snap = {
        "turn": 7, "tavern_tier": 4, "gold": 3, "hero_health": 25,
        "board": board, "shop": [],
        "available_tribes": ["Dragon", "Beast", "Mech", "Pirate", "Murloc"],
    }
    cut = Action(SELL, target="junk", cost=0, detail={"minion": board[-1]})
    keep = Action(SELL, target="d0", cost=0, detail={"minion": board[0]})
    cadj, creason = direction_cut_sell_adjust(cut, snap)
    kadj, _ = direction_cut_sell_adjust(keep, snap)
    assert cadj < 0 and creason and "cut" in creason.lower()
    assert kadj == 0.0


def test_trinket_pick_demotes_battlecry_without_plan():
    bc = "The first two Battlecry minions you buy each turn are free."
    ds = "Whenever a friendly Dragon attacks, give it Divine Shield."
    board_tribes = {"dragon": 4}
    board_kw = {"divine shield", "taunt"}
    bc_fit, bc_bits = _trinket_fit(bc, board_tribes, "dragon", board_kw)
    ds_fit, ds_bits = _trinket_fit(ds, board_tribes, "dragon", board_kw)
    assert bc_fit > 0  # demoted — no battlecry plan
    assert any("battlecry" in b.lower() for b in bc_bits)
    assert ds_fit < bc_fit  # dragon/DS plan ranks better
    # With battlecry density, BC trinket becomes good.
    bc_ok, _ = _trinket_fit(bc, board_tribes, "dragon", {"battlecry"})
    assert bc_ok < bc_fit


def test_equipped_trinket_boosts_matching_buy_and_protects_sell():
    snap = {
        "trinkets": [{
            "name": "Warcry Totem",
            "text": "The first two Battlecry minions you buy each turn are free.",
        }],
        "board": [
            {"name": f"b{i}", "keywords": ["BATTLECRY"], "tribes": ["Neutral"]}
            for i in range(4)
        ],
    }
    bc = {"name": "Cryer", "keywords": ["BATTLECRY"], "tribes": ["Neutral"],
          "attack": 3, "health": 3}
    plain = {"name": "Vanilla", "keywords": [], "tribes": ["Beast"],
             "attack": 5, "health": 5}
    adj, reason = minion_trinket_buy_adjust(bc, snap)
    assert adj < 0 and reason and "battlecry" in reason.lower()
    pen, preason = off_trinket_buy_penalty(plain, snap)
    assert pen > 0 and preason
    assert off_trinket_buy_penalty(bc, snap)[0] == 0.0
    assert sell_trinket_penalty(bc, snap) > 0
    assert sell_trinket_penalty(plain, snap) == 0.0


def test_naga_not_in_trinket_tribe_list():
    from hsbg_coach.draft import _TRIBES
    assert "naga" not in _TRIBES
    assert "aberration" in _TRIBES
