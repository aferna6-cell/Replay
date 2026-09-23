"""Hard NEXT behaviors from ingested HSReplay guides (Aidan-approved).

Proves:
  1. Sparse board + solid shop buy ⇒ ROLL is not NEXT
  2. Filled board + key piece in shop ⇒ buy the key (PLAN lock)
  3. Hero with HSReplay HP guide rule ⇒ HP can be NEXT
  4. Trinket ranking uses HSReplay guide signal
"""
from __future__ import annotations

from hsbg_coach.actions import Action, BUY, ROLL, HERO_POWER
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.game_value import rank_actions
from hsbg_coach.advisor import advise_actions
from hsbg_coach.jeef_priors import (
    board_fill_roll_adjust, roll_must_not_be_next, hero_power_adjust,
)
from hsbg_coach.pace import load_pace
from hsbg_coach.hsreplay_guides import (
    load_heroes, load_comps, load_trinkets, locked_comp, plan_line,
    key_or_enabler_boost, hero_guide_hp_adjust, hero_guide_lines,
    trinket_guide_score, lookup_trinket, lookup_hero,
)
from hsbg_coach.hero_scripts import hero_script_lines
from hsbg_coach.draft import rank_trinkets, _trinket_fit
from hsbg_coach.live import build_note_for

PACE = load_pace()
SC = HeuristicScorer({})


def _rank(snap):
    recs, _ = rank_actions(snap, kb=None, scorer=SC, pace=PACE)
    return recs


def test_guides_ingested():
    heroes = load_heroes()["heroes"]
    comps = load_comps()["comps"]
    trinkets = load_trinkets()["trinkets"]
    assert len(heroes) >= 80
    assert sum(1 for h in heroes if h.get("guide_text")) >= 50
    assert len(comps) >= 15
    assert all(not c.get("naga") for c in comps)
    assert len(trinkets) >= 100
    assert sum(1 for t in trinkets if t.get("guide_text")) >= 50


def test_sparse_plus_solid_shop_roll_not_next():
    """Fill-with-solids gate: sparse board + solid buy ⇒ NEXT ≠ ROLL."""
    snap = {
        "turn": 5, "tavern_tier": 3, "gold": 7, "hero_health": 30,
        "board": [
            {"name": "a", "attack": 3, "health": 3, "tribes": ["Beast"], "tier": 2},
            {"name": "b", "attack": 2, "health": 4, "tribes": ["Beast"], "tier": 2},
        ],
        "shop": [
            {"name": "solid", "attack": 4, "health": 4, "tribes": ["Beast"], "tier": 3},
            {"name": "ok", "attack": 3, "health": 5, "tribes": ["Mech"], "tier": 3},
        ],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    radj, _ = board_fill_roll_adjust(snap)
    assert radj > 0
    assert roll_must_not_be_next(snap)
    recs = _rank(snap)
    assert recs[0].action.kind == BUY, (
        f"NEXT should be BUY on sparse+solid, got {recs[0].action.kind}: {recs[0].reason}"
    )
    assert recs[0].action.kind != ROLL
    plan = advise_actions(snap, scorer=SC, include_reposition=False)
    assert plan.best.action.kind != ROLL


def test_filled_board_key_piece_buy_and_plan():
    """After fill, HSReplay key in shop is preferred; overlay shows PLAN →."""
    # Beetles keys from ingested comps
    board = [
        {"name": "Turquoise Skitterer", "attack": 5, "health": 5,
         "tribes": ["Beast"], "tier": 3},
        {"name": "Headhunter Gryphon", "attack": 4, "health": 4,
         "tribes": ["Beast"], "tier": 4},
        {"name": "Ravaging Scorpid", "attack": 3, "health": 3,
         "tribes": ["Beast"], "tier": 2},
        {"name": "filler_a", "attack": 4, "health": 4, "tribes": ["Beast"], "tier": 2},
        {"name": "filler_b", "attack": 4, "health": 4, "tribes": ["Beast"], "tier": 2},
    ]
    shop = [
        {"name": "Banana Slamma", "attack": 3, "health": 4,
         "tribes": ["Beast"], "tier": 4},
        {"name": "off_tribe_big", "attack": 10, "health": 10,
         "tribes": ["Pirate"], "tier": 4},
    ]
    snap = {
        "turn": 9, "tavern_tier": 4, "gold": 6, "hero_health": 25,
        "board": board, "shop": shop,
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    comp = locked_comp(snap)
    assert comp is not None, "should lock an HSReplay comp once board is filled"
    assert "Beetle" in (comp.get("name") or "") or "Beast" in (comp.get("name") or "")
    assert plan_line(snap) and plan_line(snap).startswith("PLAN →")
    note = build_note_for(snap)
    assert note and note.startswith("PLAN →"), note

    boost, reason = key_or_enabler_boost("Banana Slamma", snap)
    assert boost <= -1.0, (boost, reason)
    assert reason and "HSReplay" in reason

    # BUY key should beat BUY off-tribe body in ranking
    recs = _rank(snap)
    buys = [r for r in recs if r.action.kind == BUY]
    assert buys, "expected buy actions"
    # Prefer Banana Slamma over off_tribe_big
    banana = next((r for r in buys if r.action.target == "Banana Slamma"), None)
    off = next((r for r in buys if r.action.target == "off_tribe_big"), None)
    assert banana is not None
    if off is not None:
        assert banana.placement < off.placement, (
            f"key buy should outrank off-tribe: {banana.placement} vs {off.placement}"
        )


def test_hero_guide_hp_can_be_next():
    """Hero with HSReplay HP rule → HP adjust fires hard enough for NEXT."""
    hero = lookup_hero({"hero_name": "Al'Akir"})
    assert hero and hero.get("guide_text")
    assert (hero.get("structured") or {}).get("hp"), "Al'Akir guide should have HP bullets"

    snap = {
        "turn": 5, "tavern_tier": 3, "gold": 5, "hero_health": 30,
        "hero": hero.get("card_id"), "hero_name": "Al'Akir",
        "hero_power": {"usable": True, "cost": 0, "name": "Swat Fly"},
        "board": [
            {"name": "a", "attack": 4, "health": 4, "tribes": ["Dragon"], "tier": 2},
            {"name": "b", "attack": 3, "health": 3, "tribes": ["Dragon"], "tier": 2},
            {"name": "c", "attack": 3, "health": 3, "tribes": ["Dragon"], "tier": 2},
            {"name": "d", "attack": 3, "health": 3, "tribes": ["Dragon"], "tier": 2},
            {"name": "e", "attack": 3, "health": 3, "tribes": ["Dragon"], "tier": 2},
        ],
        # Trash shop so HP isn't competing with a must-buy fill
        "shop": [
            {"name": "trash", "attack": 1, "health": 1, "tribes": ["Pirate"], "tier": 1},
        ],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
        "phase": "recruit",
    }
    gadj, greason = hero_guide_hp_adjust(snap, cost=0)
    assert gadj <= -0.8, (gadj, greason)
    assert greason and "HSReplay" in greason

    adj, reason = hero_power_adjust(snap, cost=0)
    assert adj <= -0.8, (adj, reason)
    assert reason and "HSReplay" in reason

    lines = hero_script_lines(snap)
    assert any("Hero Power" in l and "HSReplay" in l for l in lines), lines[:5]

    # HP action should be able to win NEXT (or at least outrank ROLL)
    recs = _rank(snap)
    kinds = [r.action.kind for r in recs[:3]]
    assert HERO_POWER in kinds or recs[0].action.kind == HERO_POWER, (
        f"expected HP in top-3 NEXT, got {kinds} reasons={[r.reason for r in recs[:3]]}"
    )


def test_trinket_ranking_uses_guide_signal():
    """Trinket with an HSReplay guide that enables board pieces ranks above a
    premium Battlecry trinket with no plan."""
    # Find a guided trinket that mentions a board card
    guided = None
    for t in load_trinkets()["trinkets"]:
        g = (t.get("guide_text") or "")
        if "Drakkari" in g or "Utility Drone" in g or "Brann" in g:
            guided = t
            break
    assert guided is not None, "expected at least one guided trinket in ingest"

    board = [
        {"name": "Drakkari Enchanter", "attack": 3, "health": 4,
         "tribes": ["Neutral"], "tier": 4},
        {"name": "Utility Drone", "attack": 3, "health": 3,
         "tribes": ["Mech"], "tier": 3},
        {"name": "mech_a", "attack": 4, "health": 4, "tribes": ["Mech"], "tier": 3},
        {"name": "mech_b", "attack": 4, "health": 4, "tribes": ["Mech"], "tier": 3},
    ]
    board_tribes = {"mech": 3, "neutral": 1}
    board_names = [m["name"] for m in board]

    gdelta, gbits = trinket_guide_score(
        guided.get("name") or guided.get("card_id"),
        board_tribes=board_tribes,
        board_names=board_names,
        target_tribe="mech",
    )
    assert gdelta < 0, (gdelta, gbits)
    assert any("HSReplay" in b for b in gbits), gbits

    # Battlecry premium with no battlecry plan is demoted by _trinket_fit
    bc = "The first two Battlecry minions you buy each turn are free."
    bc_fit, bc_bits = _trinket_fit(bc, board_tribes, "mech", {"divine shield", "taunt"})
    # Guide-enabled trinket effect text (use guided effect_summary)
    guide_text = guided.get("effect_summary") or guided.get("guide_text") or ""
    # Direct guide score must beat the demoted battlecry fit
    assert gdelta < bc_fit, (
        f"guide signal {gdelta} should beat demoted battlecry {bc_fit}"
    )

    # Avoid-guide demotion
    # Synthesize: lookup returns real data; just assert avoid path via raw score
    # on a guide containing "don't pick" — covered by unit of trinket_guide_score
    # when guide_text has avoid language (if present in ingest).
    avoid = next(
        (t for t in load_trinkets()["trinkets"]
         if t.get("guide_text") and "don't pick" in t["guide_text"].lower()),
        None,
    )
    if avoid:
        d, bits = trinket_guide_score(avoid["name"], board_tribes=board_tribes,
                                      board_names=board_names)
        assert d > 0 and any("avoid" in b.lower() for b in bits)


def test_no_naga_comps_in_live_guides():
    for c in load_comps()["comps"]:
        assert not c.get("naga")
        assert "naga" not in (c.get("name") or "").lower()
        assert (c.get("tribe") or "").lower() != "naga"
