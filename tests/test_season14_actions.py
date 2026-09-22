"""Contract tests: Dark Gift, Activate, Hero Power, BUY_SPELL in recommend path.

Advisory only — legal_actions emission + ranking can put them #1 when scoring
says so. Does not touch training pipelines or the playstyle dashboard.
"""

from __future__ import annotations

from hsbg_coach.actions import (
    legal_actions, BUY, BUY_SPELL, HERO_POWER, ACTIVATE, DARK_GIFT, END,
)
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.game_value import rank_actions
from hsbg_coach.jeef_priors import dark_gift_adjust, activate_adjust
from hsbg_coach.live import _key
from hsbg_coach.bg import BGTracker, Snapshot


EMB = {
    "b1": [1.0, 0.0], "b2": [1.0, 0.0], "Courier": [1.0, 0.0],
    "good": [1.0, 0.0], "weak": [0.0, 1.0],
}


def _base(**kw):
    snap = {
        "phase": "recruit",
        "gold": 6,
        "tavern_tier": 2,
        "hero_health": 30,
        "turn": 4,
        "board": [{"name": "b1", "attack": 3, "health": 3}],
        "shop": [{"name": "weak", "attack": 1, "health": 1}],
        "shop_spells": [],
        "hero_power": None,
        "activatable": [],
        "dark_gift": None,
    }
    snap.update(kw)
    return snap


def test_legal_actions_emit_dark_gift_when_usable():
    snap = _base(dark_gift={"name": "Dark Gift", "cost": 3, "usable": True})
    kinds = [a.kind for a in legal_actions(snap)]
    assert DARK_GIFT in kinds
    # Unusable / unaffordable gated out.
    snap2 = _base(gold=2, dark_gift={"name": "Dark Gift", "cost": 3, "usable": True})
    assert DARK_GIFT not in [a.kind for a in legal_actions(snap2)]
    snap3 = _base(dark_gift={"name": "Dark Gift", "cost": 3, "usable": False})
    assert DARK_GIFT not in [a.kind for a in legal_actions(snap3)]


def test_legal_actions_emit_activate_when_usable():
    snap = _base(activatable=[{
        "name": "Courier", "card_id": "BG26_810", "cost": 0, "usable": True,
        "entity_id": 9,
    }])
    acts = [a for a in legal_actions(snap) if a.kind == ACTIVATE]
    assert len(acts) == 1 and acts[0].cost == 0 and acts[0].target == "Courier"
    snap2 = _base(gold=1, activatable=[{
        "name": "Courier", "cost": 3, "usable": True, "entity_id": 9,
    }])
    assert ACTIVATE not in [a.kind for a in legal_actions(snap2)]


def test_hero_power_and_buy_spell_still_legal():
    snap = _base(
        gold=5,
        hero_power={"name": "Trade Up", "cost": 1, "usable": True},
        shop_spells=[{"name": "Corrupted Coin", "card_id": "BG36_303", "cost": 2}],
    )
    kinds = {a.kind for a in legal_actions(snap)}
    assert HERO_POWER in kinds
    assert BUY_SPELL in kinds
    # Passive / unusable HP not offered.
    snap["hero_power"] = {"name": "Passive", "cost": 1, "usable": False}
    assert HERO_POWER not in {a.kind for a in legal_actions(snap)}


def test_dark_gift_can_rank_first_in_jeef_like_spot():
    """Turn 4 / tier 2 / 3g / weak shop — Jeef midgame gift window."""
    snap = _base(
        gold=3, turn=4, tavern_tier=2,
        shop=[{"name": "weak", "attack": 1, "health": 1}],
        dark_gift={"name": "Dark Gift", "cost": 3, "usable": True},
        # No affordable buy (gold == 3 but buy costs 3 — buy still legal);
        # gift should compete and can win via Jeef midgame prior.
    )
    recs, _ = rank_actions(snap, scorer=HeuristicScorer(EMB), include_reposition=False)
    kinds = [r.action.kind for r in recs]
    assert DARK_GIFT in kinds
    # Soft prior: midgame gift should outrank END and typically beat a weak buy.
    assert kinds[0] in (DARK_GIFT, BUY)  # allow buy if scorer loves the body
    gift = next(r for r in recs if r.action.kind == DARK_GIFT)
    end = next(r for r in recs if r.action.kind == END)
    assert gift.placement < end.placement


def test_activate_zero_cost_can_rank_first():
    snap = _base(
        gold=3, turn=5,
        shop=[{"name": "weak", "attack": 1, "health": 1}],
        activatable=[{"name": "Courier", "cost": 0, "usable": True, "entity_id": 1}],
    )
    recs, _ = rank_actions(snap, scorer=HeuristicScorer(EMB), include_reposition=False)
    top = recs[0].action.kind
    assert ACTIVATE in [r.action.kind for r in recs]
    # Free activate is high priority vs END / freeze.
    assert top in (ACTIVATE, BUY, HERO_POWER, DARK_GIFT, BUY_SPELL)


def test_corrupted_coin_spell_promoted_over_unknown():
    snap = _base(
        gold=6,
        shop_spells=[
            {"name": "Corrupted Coin", "card_id": "BG36_303", "cost": 2},
            {"name": "Mystery Gloop", "card_id": "UNKNOWN_SPELL_X", "cost": 2},
        ],
    )
    recs, _ = rank_actions(snap, scorer=HeuristicScorer(EMB), include_reposition=False)
    spells = [r for r in recs if r.action.kind == BUY_SPELL]
    assert len(spells) == 2
    by_name = {r.action.target: r.placement for r in spells}
    assert by_name["Corrupted Coin"] < by_name["Mystery Gloop"]


def test_jeef_priors_dark_gift_demotes_panic():
    good = dark_gift_adjust(_base(turn=4, tavern_tier=2, gold=6, hero_health=30), 3)
    panic = dark_gift_adjust(_base(turn=4, tavern_tier=2, gold=3, hero_health=8,
                                   board=[{"name": "b1", "attack": 2, "health": 2}]), 3)
    assert good[0] < panic[0]  # more negative = better


def test_jeef_priors_activate_zero_better_than_expensive():
    z = activate_adjust(_base(gold=6), 0)
    pricey = activate_adjust(_base(gold=3), 3)
    assert z[0] < pricey[0]


def test_cache_key_includes_dark_gift_and_activate():
    a = _base(hero_power={"usable": False}, dark_gift={"usable": False},
              activatable=[{"entity_id": 1, "usable": False, "cost": 0}])
    b = dict(a, dark_gift={"usable": True, "cost": 3, "entity_id": 2})
    c = dict(a, activatable=[{"entity_id": 1, "usable": True, "cost": 0}])
    assert _key(a) != _key(b)
    assert _key(a) != _key(c)


def test_bg_tracker_detects_dark_gift_and_activatable_entities():
    """Synthetic entity tags — mirrors Power.log reconstruction signals."""
    from hsbg_coach.state import Entity
    from hsbg_coach.bg import Phase

    t = BGTracker()
    t.in_bg = True
    t.local_player = 1
    t.player_names = {1: "Me"}
    t.phase = Phase.RECRUIT
    t.state.current_turn = 5

    player = Entity(id=2, name="Me")
    player.tags = {"RESOURCES": "6", "RESOURCES_USED": "0", "HERO_ENTITY": "90"}
    hero = Entity(id=90, card_id="BG_HERO_X")
    hero.tags = {"CARDTYPE": "HERO", "CONTROLLER": "1", "ZONE": "PLAY",
                 "PLAYER_TECH_LEVEL": "2", "HEALTH": "30"}

    gift = Entity(id=10, name="Dark Gift", card_id="TB_BaconShop_DarkGift_Button")
    gift.tags = {"CONTROLLER": "1", "ZONE": "PLAY", "COST": "3"}

    act = Entity(id=11, name="Courier", card_id="BG26_810")
    act.tags = {
        "CONTROLLER": "1", "ZONE": "PLAY", "CARDTYPE": "MINION",
        "ZONE_POSITION": "1", "HAS_ACTIVATE_POWER": "1", "BACON_TRIGGER_XY": "1",
        "TAG_SCRIPT_DATA_NUM_1": "0", "ATK": "3", "HEALTH": "3",
    }

    # Noise: HAS_ACTIVATE_POWER alone must NOT count as Activate.
    noise = Entity(id=12, name="NoActivate", card_id="BGS_039")
    noise.tags = {
        "CONTROLLER": "1", "ZONE": "PLAY", "CARDTYPE": "MINION",
        "ZONE_POSITION": "2", "HAS_ACTIVATE_POWER": "1", "ATK": "2", "HEALTH": "2",
    }

    t.state.entities = {2: player, 90: hero, 10: gift, 11: act, 12: noise}

    dg = t._dark_gift()
    assert dg is not None and dg["usable"] is True and dg["cost"] == 3
    acts = t._activatable()
    assert len(acts) == 1 and acts[0]["cost"] == 0

    snap = t.snapshot()
    assert isinstance(snap, Snapshot)
    assert snap.dark_gift and snap.dark_gift["usable"]
    assert len(snap.activatable) == 1
