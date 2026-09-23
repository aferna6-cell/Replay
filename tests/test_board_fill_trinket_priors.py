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
from hsbg_coach.advisor import advise_actions
from hsbg_coach.jeef_priors import (
    anti_stuck_roll_adjust, board_fill_roll_adjust, direction_cut_sell_adjust,
    midgame_solid_buy_adjust, roll_must_not_be_next,
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
    radj, _ = board_fill_roll_adjust(snap)
    assert radj > 0
    assert roll_must_not_be_next(snap)

    recs = _rank(snap)
    assert recs[0].action.kind == BUY, (
        f"NEXT should be buy on sparse board, got {recs[0].action.kind}: {recs[0].reason}"
    )
    assert recs[0].action.kind != ROLL
    roll = next(r for r in recs if r.action.kind == ROLL)
    buy = next(r for r in recs if r.action.kind == BUY and r.action.target == "filler")
    assert buy.placement < roll.placement
    # Full-strength demotion: roll at least +0.6 vs unadjusted base gap to buy.
    assert roll.placement >= buy.placement + 0.15

    plan = advise_actions(snap, scorer=SC, include_reposition=False)
    assert plan.best.action.kind == BUY
    adv_roll = next(a for a in plan.ranked if a.action.kind == ROLL)
    assert adv_roll.priority <= 0.05, (
        f"advisor roll prio must be <=0.05 after full demotion, got {adv_roll.priority}"
    )


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
    assert not roll_must_not_be_next(snap)
    recs = _rank(snap)
    assert recs[0].action.kind == ROLL


def test_hard_rule_board_lt5_fill_shop_top_not_roll():
    """board size < 5 + gold>=3 + acceptable fill → top kind != ROLL."""
    snap = {
        "turn": 6, "tavern_tier": 3, "gold": 5, "hero_health": 28,
        "board": [
            {"name": "a", "attack": 4, "health": 4, "tribes": ["Dragon"], "tier": 2},
            {"name": "b", "attack": 3, "health": 5, "tribes": ["Dragon"], "tier": 2},
            {"name": "c", "attack": 2, "health": 2, "tribes": ["Dragon"], "tier": 1},
            {"name": "d", "attack": 3, "health": 3, "tribes": ["Dragon"], "tier": 2},
        ],
        "shop": [
            {"name": "fill", "attack": 3, "health": 4, "tribes": ["Dragon"], "tier": 3},
            {"name": "trash", "attack": 1, "health": 1, "tribes": ["Pirate"], "tier": 1},
        ],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    assert len(snap["board"]) < 5
    assert roll_must_not_be_next(snap)
    recs = _rank(snap)
    assert recs[0].action.kind != ROLL
    assert recs[0].action.kind == BUY
    plan = advise_actions(snap, scorer=SC, include_reposition=False)
    assert plan.best.action.kind != ROLL


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


def test_board_profile_resolves_by_card_id():
    """Live MinionView dicts: card_id must populate tribes/keywords even if name misses."""
    from hsbg_coach.cards import load_kb
    from hsbg_coach.draft import _board_profile
    kb = load_kb()
    assert kb, "kb required"
    # Electric Synthesizer BG26_963 — Battlecry Dragon in KB
    board = [
        {"name": "WRONG_LIVE_NAME", "card_id": "BG26_963"},
        {"name": "also_wrong", "card_id": "BG26_963"},
        {"name": "also_wrong", "card_id": "BG26_963"},
    ]
    tribes, kws = _board_profile(board, kb)
    assert tribes.get("dragon", 0) >= 3
    assert "battlecry" in kws


def test_trinket_strategy_beats_meta_not_position():
    """Deathrattle/Battlecry plan must beat raw meta avg; not offer-order biased.

    Meta-best Battlecry premium on the RIGHT, and again on the LEFT — both times
    pick the plan-aligned trinket when the board is Battlecry Dragons.
    """
    from hsbg_coach.cards import load_kb
    from hsbg_coach.draft import rank_trinkets
    from hsbg_coach.stats import TrinketStats
    from hsbg_coach.choices import ChoiceOffer, offer_advice_lines
    from hsbg_coach.overlay import format_next, format_overlay_text

    class FakeDB:
        def __init__(self, items):
            self.trinkets = items

    kb = load_kb()
    board = [
        {"name": "WRONG", "card_id": "BG26_963"},  # Electric Synthesizer — BC Dragon
        {"name": "WRONG", "card_id": "BG26_963"},
        {"name": "WRONG", "card_id": "BG26_963"},
        {"name": "WRONG", "card_id": "BG34_633"},  # Draconic Warden — BC Dragon
    ]
    tribes, kws = __import__("hsbg_coach.draft", fromlist=["_board_profile"])._board_profile(board, kb)
    assert "battlecry" in kws and tribes.get("dragon", 0) >= 3

    bc = TrinketStats(
        name="Warcry Totem", card_id="T_BC", average_position=4.30, pick_rate=0.2,
        tier="B",
        text="The first two Battlecry minions you buy each turn are free.",
    )
    meta_best = TrinketStats(
        name="Premium Charm", card_id="T_META", average_position=3.05, pick_rate=0.6,
        tier="S",
        text="At the end of your turn, give a random minion +2/+2.",
    )
    mid = TrinketStats(
        name="Random Bauble", card_id="T_MID", average_position=3.70, pick_rate=0.3,
        tier="A",
        text="Gain 1 Gold.",
    )
    db = FakeDB([bc, meta_best, mid])

    # Case A: meta-best LEFT, battlecry RIGHTMOST
    offered_a = ["Premium Charm", "Random Bauble", "Warcry Totem"]
    ranked_a = rank_trinkets(offered_a, db, board=board, kb=kb)
    assert ranked_a[0].name == "Warcry Totem", (
        f"expected Battlecry plan pick, got {ranked_a[0].name}: {ranked_a[0].reason}"
    )

    # Case B: battlecry LEFT, meta-best RIGHTMOST — still Battlecry (not position)
    offered_b = ["Warcry Totem", "Random Bauble", "Premium Charm"]
    ranked_b = rank_trinkets(offered_b, db, board=board, kb=kb)
    assert ranked_b[0].name == "Warcry Totem", (
        f"position-biased? got {ranked_b[0].name}: {ranked_b[0].reason}"
    )

    # Overlay: one clear NEXT → PICK with effect reason; alts not labeled PICK
    offer = ChoiceOffer("trinket",
                        ["T_META", "T_MID", "T_BC"],
                        offered_a)
    lines = offer_advice_lines(offer, board=board, kb=kb, db=db)
    assert lines[0].startswith("PICK Warcry Totem — ")
    assert "battlecry" in lines[0].lower() or "plays into" in lines[0].lower()
    assert all(not l.startswith("PICK ") for l in lines[1:])
    text = format_next({"phase": "choose trinket"}, None, lines)
    assert "NEXT → PICK Warcry Totem" in text
    assert "—" in text.split("\n")[0] or "—" in text
    rich = format_overlay_text({"phase": "choose trinket", "board": board,
                                "turn": 6, "tavern_tier": 4, "gold": 5,
                                "hero_health": 30}, None, lines)
    assert "NEXT → PICK Warcry Totem" in rich

def test_hero_power_beats_roll_when_shop_mediocre():
    """Usable HP + spare gold + mediocre shop → HP ranks above Roll."""
    from hsbg_coach.actions import HERO_POWER, ROLL, BUY
    from hsbg_coach.jeef_priors import hero_power_adjust, hero_power_buy_adjust
    from hsbg_coach.economy import HeroContext
    snap = {
        "turn": 7, "tavern_tier": 4, "gold": 5, "hero_health": 28, "phase": "recruit",
        "board": [
            {"name": f"m{i}", "attack": 5, "health": 5, "tribes": ["Dragon"], "tier": 3}
            for i in range(5)
        ],
        "shop": [
            {"name": "trash", "attack": 1, "health": 1, "tribes": ["Pirate"], "tier": 1},
            {"name": "meh", "attack": 2, "health": 2, "tribes": ["Beast"], "tier": 2},
        ],
        "hero_power": {
            "name": "Brew", "cost": 1, "usable": True,
            "text": "Give a Dragon +2/+2.",
        },
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    adj, reason = hero_power_adjust(snap, 1)
    assert adj < -0.5 and reason
    recs = _rank(snap)
    assert recs[0].action.kind == HERO_POWER, (
        f"expected HP NEXT on mediocre shop, got {recs[0].action.kind}: {recs[0].reason}"
    )
    roll = next(r for r in recs if r.action.kind == ROLL)
    hp = next(r for r in recs if r.action.kind == HERO_POWER)
    assert hp.placement < roll.placement


def test_on_plan_buy_can_still_beat_hero_power():
    """Clear on-plan solid buy may outrank HP (don't spam HP over real pieces)."""
    from hsbg_coach.actions import HERO_POWER, BUY
    from hsbg_coach.jeef_priors import hero_power_buy_adjust
    from hsbg_coach.economy import HeroContext
    from hsbg_coach.actions import Action
    from hsbg_coach.cards import load_kb
    snap = {
        "turn": 7, "tavern_tier": 4, "gold": 6, "hero_health": 28, "phase": "recruit",
        "board": [
            {"name": f"m{i}", "attack": 5, "health": 5, "tribes": ["Dragon"], "tier": 3}
            for i in range(5)
        ],
        "shop": [
            {"name": "solid", "attack": 7, "health": 7, "tribes": ["Dragon"], "tier": 4},
            {"name": "trash", "attack": 1, "health": 1, "tribes": ["Pirate"], "tier": 1},
        ],
        "hero_power": {
            "name": "Brew", "cost": 1, "usable": True,
            "text": "Give a Dragon +2/+2.",
        },
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    recs = _rank(snap)
    assert recs[0].action.kind == BUY
    assert recs[0].action.target == "solid"
    ctx = HeroContext(hero="Test", target_tribe="dragon",
                      recommended_minions=["solid"])
    act = Action(BUY, target="solid", cost=3, detail={"minion": snap["shop"][0]})
    badj, breason = hero_power_buy_adjust(act, snap, load_kb(), hero_ctx=ctx)
    assert badj < 0 and breason and "hero" in breason.lower()
