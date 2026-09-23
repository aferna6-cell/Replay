"""Hero/hand contextual scripts: Lens Case play, Gallywix cycle, HP lean."""

from __future__ import annotations

from hsbg_coach.hero_scripts import (
    hand_item_play_lines, gallywix_cycle_lines, is_hand_playable_item,
    hero_script_lines,
)
from hsbg_coach.live import advice_lines, _is_hand_minion
from hsbg_coach.overlay import format_next, _short_move
from hsbg_coach.board_value import HeuristicScorer
from hsbg_coach.cards import load_kb


def test_lens_case_detected_as_hand_playable():
    m = {
        "name": "Lens Case",
        "card_id": "BG35_MagicItem_817",
        "tags": {"CARDTYPE": "BATTLEGROUND_TRINKET"},
    }
    assert is_hand_playable_item(m)
    assert _is_hand_minion(m), "MagicItem in hand must not be dropped by CARDTYPE filter"


def test_lens_case_in_hand_is_next_play():
    """Had Lens Case — NEVER silent; NEXT should Play Lens Case."""
    kb = load_kb()
    snap = {
        "phase": "recruit", "turn": 6, "tavern_tier": 3, "gold": 5,
        "hero_health": 30, "board": [
            {"name": "a", "card_id": "a", "attack": 3, "health": 3},
        ],
        "shop": [
            {"name": "filler", "card_id": "f", "attack": 4, "health": 4,
             "tribes": ["Beast"], "tier": 3},
        ],
        "hand": [{
            "name": "Lens Case",
            "card_id": "BG35_MagicItem_817",
            "tags": {"CARDTYPE": "BATTLEGROUND_TRINKET"},
        }],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    scripts = hand_item_play_lines(snap)
    assert scripts and "Lens Case" in scripts[0]
    assert "Play" in scripts[0]
    lines = advice_lines(snap, kb, HeuristicScorer({}))
    assert lines, "advice must not be empty with Lens Case in hand"
    assert "Lens Case" in lines[0], lines[:3]
    text = format_next(snap, None, lines)
    assert "Lens Case" in text
    assert "Play" in text


def test_gallywix_cycle_bias_emits_sell_buy_or_buy():
    kb = load_kb()
    board = [
        {"name": f"Big{i}", "card_id": f"b{i}", "attack": 8, "health": 8}
        for i in range(6)
    ] + [{"name": "Runt", "card_id": "r", "attack": 1, "health": 1}]
    snap = {
        "phase": "recruit", "turn": 8, "tavern_tier": 4, "gold": 5,
        "hero_health": 28,
        "hero": "TB_BaconShop_HERO_10",
        "hero_name": "Trade Prince Gallywix",
        "board": board,
        "shop": [
            {"name": "CycleBody", "card_id": "cb", "attack": 5, "health": 5,
             "tribes": ["Pirate"], "tier": 3},
        ],
        "hand": [],
        "available_tribes": ["Beast", "Mech", "Pirate", "Dragon", "Murloc"],
    }
    lines = gallywix_cycle_lines(snap, kb)
    assert lines, "Gallywix with gold+shop must emit a cycle line"
    assert any("Gallywix" in l for l in lines)
    # Full board + gold>=4 → sell then buy
    assert any("Sell" in l and "buy" in l.lower() for l in lines), lines
    adv = advice_lines(snap, kb, HeuristicScorer({}))
    assert adv and ("Gallywix" in adv[0] or "cycle" in adv[0].lower()
                    or "Lens" in adv[0]), adv[:3]
    short = _short_move(adv[0])
    # Compound sell+buy must survive overlay short form
    if "Sell" in adv[0] and "buy" in adv[0].lower():
        assert "Sell" in short and "buy" in short.lower(), short
