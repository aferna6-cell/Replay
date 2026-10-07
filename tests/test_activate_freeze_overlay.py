"""Regression: Activate false positives, freeze-over-eager, minimal overlay."""

from __future__ import annotations

from hsbg_coach.actions import legal_actions, ACTIVATE, FREEZE, END
from hsbg_coach.activate_cards import is_activate_minion, ACTIVATE_CARD_IDS
from hsbg_coach.advisor import advise_actions, _FREEZE_STRONG
from hsbg_coach.bg import BGTracker, Snapshot
from hsbg_coach.board_value import get_scorer
from hsbg_coach.game_value import rank_actions
from hsbg_coach.live import advice_lines
from hsbg_coach.overlay import format_next, format_overlay_text
from hsbg_coach.state import Entity
from hsbg_coach import cards


def test_activate_allowlist_covers_prisonguard_not_joyous():
    assert is_activate_minion("BG36_345")
    assert is_activate_minion("BG36_345_G")
    assert is_activate_minion("BG36_099")  # 0-cost Activate
    assert not is_activate_minion("BG36_110")  # Joyous — no Activate
    assert not is_activate_minion("BG26_135")  # Southsea Busker
    assert not is_activate_minion("TB_BaconShop_DragSell")
    assert "BG36_345" in ACTIVATE_CARD_IDS


def test_board_minion_without_activate_keyword_does_not_emit():
    """HAS_ACTIVATE_POWER=1 on a normal board minion must NOT yield Activate."""
    t = BGTracker()
    t.in_bg = True
    t.local_player = 1
    t.player_names = {1: "Me"}
    from hsbg_coach.bg import Phase
    t.phase = Phase.RECRUIT
    t.state.current_turn = 2
    player = Entity(id=2, name="Me")
    player.tags = {"RESOURCES": "0", "RESOURCES_USED": "0", "HERO_ENTITY": "90"}
    hero = Entity(id=90, card_id="BG_HERO_X")
    hero.tags = {"CARDTYPE": "HERO", "CONTROLLER": "1", "ZONE": "PLAY",
                 "PLAYER_TECH_LEVEL": "2", "HEALTH": "30"}
    joyous = Entity(id=11, name="Joyous", card_id="BG36_110")
    joyous.tags = {
        "CONTROLLER": "1", "ZONE": "PLAY", "CARDTYPE": "MINION",
        "ZONE_POSITION": "1", "HAS_ACTIVATE_POWER": "1",
        "ATK": "2", "HEALTH": "3",
    }
    t.state.entities = {2: player, 90: hero, 11: joyous}
    assert t._activatable() == []
    snap = t.snapshot()
    assert isinstance(snap, Snapshot)
    assert snap.activatable == []
    assert ACTIVATE not in [a.kind for a in legal_actions(snap.to_dict())]


def test_board_minion_with_activate_tag_and_allowlist_emits():
    t = BGTracker()
    t.in_bg = True
    t.local_player = 1
    from hsbg_coach.bg import Phase
    t.phase = Phase.RECRUIT
    t.state.current_turn = 4
    player = Entity(id=2, name="Me")
    player.tags = {"RESOURCES": "3", "RESOURCES_USED": "0", "HERO_ENTITY": "90"}
    hero = Entity(id=90, card_id="BG_HERO_X")
    hero.tags = {"CARDTYPE": "HERO", "CONTROLLER": "1", "ZONE": "PLAY",
                 "PLAYER_TECH_LEVEL": "2", "HEALTH": "30"}
    act = Entity(id=11, name="Suspicious Prisonguard", card_id="BG36_345")
    act.tags = {
        "CONTROLLER": "1", "ZONE": "PLAY", "CARDTYPE": "MINION",
        "ZONE_POSITION": "1", "HAS_ACTIVATE_POWER": "1",
        "TAG_SCRIPT_DATA_NUM_1": "1", "ATK": "2", "HEALTH": "2",
    }
    t.state.entities = {2: player, 90: hero, 11: act}
    acts = t._activatable()
    assert len(acts) == 1 and acts[0]["name"] == "Suspicious Prisonguard"
    assert acts[0]["cost"] == 1


def _millhouse_turn2_snap():
    """Aidan's live spot: gold 0, Joyous board, generic T2 shop."""
    return {
        "phase": "recruit",
        "gold": 0,
        "tavern_tier": 2,
        "hero_health": 30,
        "turn": 2,
        "board": [{"name": "Joyous", "card_id": "BG36_110", "attack": 2, "health": 3}],
        "shop": [
            {"name": "Suspicious Prisonguard", "card_id": "BG36_345",
             "attack": 2, "health": 2, "tags": {"TECH_LEVEL": "1"}},
            {"name": "Flighty Scout", "card_id": "BG32_330",
             "attack": 3, "health": 3},
            {"name": "Lullabot", "card_id": "BG26_146",
             "attack": 1, "health": 2},
        ],
        "shop_spells": [],
        "hero_power": None,
        "activatable": [],
        "dark_gift": None,
    }


def test_turn2_mediocre_shop_does_not_rank_freeze_first():
    """Freeze must not be NEXT on Joyous + generic T2 shop at 0 gold."""
    snap = _millhouse_turn2_snap()
    kb, scorer = cards.load_kb(), get_scorer()
    plan = advise_actions(snap, kb=kb, scorer=scorer, include_reposition=False)
    frz = next(s for s in plan.ranked if s.action.kind == FREEZE)
    assert frz.priority < 0.4
    assert "worth keeping" in frz.reason

    recs, _ = rank_actions(snap, kb=kb, scorer=scorer, include_reposition=False)
    # After burial, Freeze must not beat End turn.
    end = next(r for r in recs if r.action.kind == END)
    freeze = next(r for r in recs if r.action.kind == FREEZE)
    assert freeze.placement > end.placement

    lines = advice_lines(snap, kb=kb, scorer=scorer)
    assert lines, "should still advise something"
    top = lines[0]
    assert not top.startswith("Freeze"), f"Freeze must not be NEXT, got: {top}"


def test_format_next_is_minimal_no_board_shop_odds_status():
    snap = _millhouse_turn2_snap()
    recs = [
        "End turn (finish 6.4) — pass the turn",
        "Sell Joyous (finish 7.8) — frees a slot",
        "Freeze the shop (finish 8.0) — buried",
    ]
    text = format_next(snap, odds="win 62% / tie 8% / loss 30%", recommendations=recs)
    assert text.startswith("→ End turn")
    assert "then:" in text
    assert "Sell Joyous" in text
    # Clutter banned from default view:
    assert "Your board" not in text
    assert "Shop (" not in text
    assert "Combat:" not in text
    assert "synced" not in text
    assert "finish" not in text
    assert "—" not in text.split("\n")[0]  # primary has no long reason


def test_format_overlay_text_still_has_rich_dump():
    snap = _millhouse_turn2_snap()
    text = format_overlay_text(snap, odds="win 50% / tie 0% / loss 50%",
                               recommendations=["Buy X", "Roll"])
    assert "NEXT → Buy X" in text
    assert "Your board" in text
    assert "Shop (" in text
    assert "Combat:" in text
