"""HSReplay XML adapter + states.v1 builder, on a hand-written synthetic replay.

Optional real-replay test: set HSBG_REPLAY_XML=<path to a Firestone .xml.gz>
(and optionally HSBG_REPLAY_MANIFEST=<manifest json> to also require every
fidelity check to pass).
"""

import json
import os
from collections import Counter

import pytest

from hsbg_coach.bg import BGTracker, Phase, Snapshot
from hsbg_coach.hsreplay_xml import iter_events
from hsbg_coach.parser import Event
from hsbg_coach.replay_states import (SCHEMA_VERSION, _Fail, _check_picks, _check_quests,
                                       _check_row, _final_checks, build_game)
from hsbg_coach.state import Entity

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_bg_replay.xml")
META = {
    "reviewId": "synthetic", "buildNumber": 1, "mmr": 5000, "placement": 1,
    "tribes": {"available": [{"id": 14, "name": "MURLOC"}]},
    "finalComp": {"board": [{"cardId": "BG_TEST_GIFTED", "golden": False,
                             "atk": 5, "health": 5}]},
}


def _snapshots():
    """Tracker snapshot at every Options block of the fixture."""
    tracker, snaps = BGTracker(), []
    for ev in iter_events(FIXTURE):
        tracker.feed(ev)
        if ev.kind == "OPTIONS":
            snaps.append(tracker.snapshot())
    return tracker, snaps


def test_events_use_powerlog_names_and_mark_local_player():
    events = list(iter_events(FIXTURE))
    players = [e for e in events if e.kind == "PLAYER"]
    assert [(p.fields["player_id"], p.fields["hi"]) for p in players] == [("1", "1"), ("9", "0")]
    tags = {(e.tag, e.value) for e in events if e.kind == "TAG"}
    assert ("ZONE", "PLAY") in tags and ("CARDTYPE", "MINION") in tags
    assert ("CARDRACE", "MURLOC") in tags and ("HAS_DARK_GIFT", "1") in tags
    assert sum(e.kind == "RESET_ENTITIES" for e in events) == 1
    opts = [e for e in events if e.kind == "OPTIONS"]
    assert opts[0].items[2] == {"index": 2, "type": "POWER", "entity": 12, "error": -1,
                                "targets": [20], "sub_options": []}
    assert opts[-1].items[1]["sub_options"] == [
        {"index": 0, "entity": 61, "error": -1, "targets": [50]},
        {"index": 1, "entity": 62, "error": 12, "targets": []}]
    assert any(e.kind == "RAW" and "BlockType=ATTACK" in e.text for e in events)


def test_tracker_state_from_xml():
    tracker, snaps = _snapshots()
    s = snaps[0]
    assert tracker.local_player == 1
    assert s.phase == "recruit"
    assert s.gold == 3 and s.tavern_tier == 1
    assert s.hero == "BG36_HERO_002" and s.hero_armor == 5 and s.hero_health == 30 + 5 - 2
    assert [m.card_id for m in s.board] == ["BG_TEST_MURLOC", "BG_TEST_GIFTED"]
    # Dark Gift via DARK_GIFT_ENTITY while the offer tag is still present.
    assert s.board[1].dark_gift["card_id"] == "BG36_MidGameEffect_000t51"
    assert s.board[0].dark_gift is None
    # ChangeEntity renamed the shop minion; HideEntity removed the other one.
    assert [m.card_id for m in s.shop] == ["BG_TEST_SHOP_CHANGED"]
    assert s.shop_frozen is True


def test_redump_drops_stale_entities_and_gift_follows_enchantment():
    _, snaps = _snapshots()
    s = snaps[-1]
    # Minion 20 died before the re-dump and is not re-emitted: it must be gone.
    assert [(m.entity_id, m.card_id) for m in s.board] == [(50, "BG_TEST_GIFTED")]
    # Re-created minion: no DARK_GIFT_ENTITY, gift found via attached ..._000t51e.
    assert s.board[0].dark_gift["card_id"] == "BG36_MidGameEffect_000t51"
    assert s.tavern_tier == 2 and s.hero_health == 30 + 5 - 4
    # Player tags appended to the re-dumped GameEntity land on the player
    # (gold 5 - 1), and the game's TURN is not overwritten by the players'.
    assert s.gold == 4 and s.turn == 3
    assert [m.card_id for m in s.hand] == ["BG_TEST_SPELL"]


def test_snapshot_round_trips_through_json():
    _, snaps = _snapshots()
    for snap in snaps:
        again = Snapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
        assert again == snap


def test_build_game_rows_and_checks():
    res = build_game(FIXTURE, META, card_names={"BG36_MidGameEffect_000t51": "Steady Growth"})
    assert dict(res["failures"].count) == {}
    assert res["game_warnings"] == []
    assert res["stats"]["options_blocks"] == 3 and res["stats"]["options_resent"] == 1
    assert [(r["kind"], r["dp_index"], r.get("options_index")) for r in res["rows"]] == [
        ("choice", 0, None), ("choice", 1, None), ("options", 2, 0), ("options", 3, 1)]
    rows = [r for r in res["rows"] if r["kind"] == "options"]
    assert [(r["raw_turn"], r["turn"]) for r in rows] == [(1, 1), (3, 2)]
    first, last = rows
    assert first["schema_version"] == SCHEMA_VERSION and first["lobby_tribes"] == ["MURLOC"]
    assert [o["entity_id"] for o in first["options"]] == [0, 30, 12]   # legal only
    assert first["options"][2] == {"index": 2, "type": "POWER", "entity_id": 12,
                                   "card_id": "BG36_HERO_002p", "zone": "PLAY",
                                   "targets": [20], "sub_options": []}
    assert first["hero_power"]["activatable"] is True
    assert first["dark_discovery"] == {"available": False}
    assert first["snapshot"]["board"][1]["dark_gift"] == {
        "card_id": "BG36_MidGameEffect_000t51", "name": "Steady Growth"}
    # Re-sent Options id 2: the later block (two options) is the one kept.
    assert [o["entity_id"] for o in last["options"]] == [0, 60]
    # Choose One: only the legal SubOption (61) is kept; card id via the tracker.
    assert last["options"][1]["sub_options"] == [
        {"index": 0, "entity_id": 61, "card_id": "BG_TEST_SPELL_a", "targets": [50]}]
    assert res["stats"]["options_with_sub_options"] == 1 and res["stats"]["sub_options"] == 1
    assert res["stats"]["dark_discovery_picks"] == 1
    assert res["stats"]["dark_discovery_picks_verified"] == 1
    assert res["warnings"] == []
    json.dumps(rows)                      # rows are JSON-serializable


def test_build_game_flags_final_board_and_placement_mismatch():
    meta = dict(META, placement=4, finalComp={"board": [{"cardId": "OTHER", "golden": False}]})
    res = build_game(FIXTURE, meta)
    assert set(res["failures"].count) == {"final_board_mismatch", "placement_mismatch"}


def _sire_tracker():
    """Entities as in replay d062a6b7 (Sire Denathrius, controller 7): the
    Whodunit? enchantment offers two quest+reward pairs; Unlikely Duo picked."""
    t = BGTracker()
    t.local_player = 7

    def put(eid, card_id, **tags):
        t.state.entities[eid] = Entity(id=eid, card_id=card_id, tags=dict(
            CONTROLLER="7", **{k: str(v) for k, v in tags.items()}))

    put(395, "BG24_QuestsPlayerEnch_t", CARDTYPE="ENCHANTMENT", ZONE="REMOVEDFROMGAME")
    put(397, "BG24_Quest_151", CARDTYPE="SPELL", ZONE="SECRET", QUEST=1, CREATOR=395,
        QUEST_PROGRESS=2, QUEST_PROGRESS_TOTAL=3, TAG_SCRIPT_DATA_ENT_1=398)
    put(398, "BG27_Reward_504", CARDTYPE="BATTLEGROUND_QUEST_REWARD", ZONE="SETASIDE",
        CREATOR=395)
    put(399, "BG24_Quest_313", CARDTYPE="SPELL", ZONE="REMOVEDFROMGAME", QUEST=1,
        CREATOR=395, QUEST_PROGRESS_TOTAL=6)                     # the quest not picked
    return t


def test_quests_active_then_completed():
    t = _sire_tracker()
    assert t._quests() == [{
        "entity_id": 397, "card_id": "BG24_Quest_151", "name": "BG24_Quest_151",
        "progress": 2, "goal": 3, "reward_card_id": "BG27_Reward_504",
        "source_card_id": "BG24_QuestsPlayerEnch_t", "completed": False}]
    # Completion: the quest is removed with its tags reset; the reward is in PLAY.
    q = t.state.entities[397]
    q.tags.update(ZONE="REMOVEDFROMGAME", QUEST_PROGRESS="0", QUEST_PROGRESS_TOTAL="4",
                  TAG_SCRIPT_DATA_ENT_1="0")
    t.state.entities[1635] = Entity(id=1635, card_id="BG27_Reward_504", tags={
        "CONTROLLER": "7", "CARDTYPE": "BATTLEGROUND_QUEST_REWARD", "ZONE": "PLAY",
        "CREATOR": "397", "BACON_IS_HEROPOWER_QUESTREWARD": "1"})
    assert t._quests() == [{
        "entity_id": 397, "card_id": "BG24_Quest_151", "name": "BG24_Quest_151",
        "progress": None, "goal": None, "reward_card_id": "BG27_Reward_504",
        "source_card_id": "BG24_QuestsPlayerEnch_t", "completed": True}]
    t.local_player = 3                     # somebody else's quests are not ours
    assert t._quests() == []


def _quest_row(dp, hero, *quests):
    return {"dp_index": dp, "snapshot": {"hero": hero, "quests": [
        {"entity_id": e, "card_id": "BG24_Quest_151", "progress": p, "goal": g,
         "completed": c} for e, p, g, c in quests]}}


def test_quest_checks():
    def run(*rows):
        fail, seen, stats = _Fail(), {}, Counter()
        for r in rows:
            _check_quests(r, seen, fail, stats)
        return dict(fail.count)

    ok = [_quest_row(0, "BG24_HERO_100", (397, 0, 3, False)),
          _quest_row(1, "BG24_HERO_100_SKIN_A", (397, 2, 3, False)),
          _quest_row(2, "BG24_HERO_100_SKIN_E", (397, None, None, True))]
    assert run(*ok) == {}
    assert run(_quest_row(0, "BG24_HERO_100", (397, 2, 3, False)),
               _quest_row(1, "BG24_HERO_100", (397, 1, 3, False))) == {
        "quest_progress_decreased": 1}
    assert run(_quest_row(0, "BG24_HERO_100", (397, 4, 3, False))) == {
        "quest_progress_over_goal": 1}
    assert run(_quest_row(0, "BG24_HERO_100", (397, 0, None, False))) == {
        "quest_goal_missing": 1}
    assert run(_quest_row(0, "BG24_HERO_100", (397, None, None, True)),
               _quest_row(1, "BG24_HERO_100", (397, 1, 3, False))) == {"quest_uncompleted": 1}
    assert run(_quest_row(0, "BG36_HERO_002", (397, 0, 3, False))) == {
        "quests_on_non_sire_hero": 1}
    assert run(_quest_row(0, "BG24_HERO_100p", (397, 0, 3, False))) == {
        "quests_on_non_sire_hero": 1}      # the hero power is not the hero
    # Any hero can sell Shady Aristocrat (Sire's buddy) and Discover a Quest
    # (5ec28cd6: The Rat King got it from a Wisdomball refresh).
    buddy = _quest_row(0, "TB_BaconShop_HERO_12", (15859, 1, 3, False))
    buddy["snapshot"]["quests"][0]["source_card_id"] = "BG24_HERO_100_Buddy"
    assert run(buddy) == {}


def test_build_game_non_sire_has_no_quests():
    res = build_game(FIXTURE, META)
    assert all(r["snapshot"]["quests"] == [] for r in res["rows"])
    assert res["stats"]["quests_seen"] == 0


def _shop_tracker():
    """Recruit phase, us = controller 7, Bob's shop = controller 10. Tags as in
    the Firestone replays (1c96a514: a shop showing only spells)."""
    t = BGTracker()
    t.local_player, t.phase = 7, Phase.RECRUIT

    def put(eid, card_id, **tags):
        t.state.entities[eid] = Entity(id=eid, card_id=card_id,
                                       tags={k: str(v) for k, v in tags.items()})

    put(50, "TB_BaconShopBob", CARDTYPE="HERO", ZONE="PLAY", CONTROLLER=10)
    put(16155, "BG36_301t", CARDTYPE="BATTLEGROUND_SPELL", ZONE="PLAY", CONTROLLER=10,
        ZONE_POSITION=2, COST=1)
    put(16156, "BG35_951", CARDTYPE="BATTLEGROUND_SPELL", ZONE="PLAY", CONTROLLER=10,
        ZONE_POSITION=1, COST=2)
    put(16157, "SPELL_X", CARDTYPE="SPELL", ZONE="PLAY", CONTROLLER=10,
        ZONE_POSITION=3, COST=3)
    put(900, "BG28_897", CARDTYPE="BATTLEGROUND_SPELL", ZONE="SETASIDE", CONTROLLER=10)
    # Our buy handles (tag 2442 = the shop entity) and refresh button.
    put(16160, "TB_BaconShop_DragBuy_Spell", CARDTYPE="MOVE_MINION_HOVER_TARGET",
        ZONE="PLAY", CONTROLLER=7, COST=1, **{"2442": 16155})
    put(16161, "TB_BaconShop_DragBuy_Spell", CARDTYPE="MOVE_MINION_HOVER_TARGET",
        ZONE="PLAY", CONTROLLER=7, COST=2, **{"2442": 16156})
    put(1679, "TB_BaconShop_8p_Reroll_Button", CARDTYPE="GAME_MODE_BUTTON",
        ZONE="PLAY", CONTROLLER=7)                    # no COST tag = free refresh
    return t


def test_shop_spells_without_minions_and_buy_costs():
    t = _shop_tracker()
    assert [(s["entity_id"], s["card_id"], s["cost"], s["buy_cost"])
            for s in t._shop_spells()] == [(16156, "BG35_951", 2, 2),
                                           (16155, "BG36_301t", 1, 1),
                                           (16157, "SPELL_X", 3, None)]
    assert t._reroll_cost() == 0
    t.state.entities[1679].tags["COST"] = "1"
    assert t._reroll_cost() == 1
    # A shop minion gets its handle's COST as buy_cost.
    t.state.entities[1776] = Entity(id=1776, card_id="BG25_011", tags={
        "CARDTYPE": "MINION", "ZONE": "PLAY", "CONTROLLER": "10", "ZONE_POSITION": "1"})
    t.state.entities[1777] = Entity(id=1777, card_id="TB_BaconShop_DragBuy", tags={
        "CARDTYPE": "MOVE_MINION_HOVER_TARGET", "ZONE": "PLAY", "CONTROLLER": "7",
        "COST": "3", "2442": "1776"})
    snap = t.snapshot()
    assert [(m.entity_id, m.buy_cost) for m in snap.shop] == [(1776, 3)]
    assert snap.reroll_cost == 1 and len(snap.shop_spells) == 3
    assert Snapshot.from_dict(json.loads(json.dumps(snap.to_dict()))) == snap
    # No Bob and no shop minion: no anchor, no spells.
    del t.state.entities[50], t.state.entities[1776]
    assert t._shop_spells() == []


def test_hand_spells_accept_spell_cardtype_and_zero_cost():
    t = BGTracker()
    t.local_player = 7
    t.state.entities = {
        1: Entity(id=1, card_id="BG28_500", tags={"CARDTYPE": "SPELL", "ZONE": "HAND",
                                                 "CONTROLLER": "7"}),
        2: Entity(id=2, card_id="BG28_168", tags={"CARDTYPE": "SPELL", "ZONE": "HAND",
                                                 "CONTROLLER": "7", "COST": "2"}),
        3: Entity(id=3, card_id="BG28_169", tags={"CARDTYPE": "SPELL", "ZONE": "SETASIDE",
                                                 "CONTROLLER": "7", "COST": "1"}),
        4: Entity(id=4, card_id="BG_M", tags={"CARDTYPE": "MINION", "ZONE": "HAND",
                                             "CONTROLLER": "7"})}
    assert sorted((s["card_id"], s["cost"]) for s in t._hand_spells()) == [
        ("BG28_168", 2), ("BG28_500", 0)]


def test_hero_power_zero_cost_and_only_in_play():
    t = BGTracker()
    t.local_player = 2
    old = Entity(id=185, card_id="BG36_HERO_002p", tags={
        "CARDTYPE": "HERO_POWER", "CONTROLLER": "2", "ZONE": "SETASIDE", "COST": "1"})
    cur = Entity(id=2619, card_id="BG26_HERO_102p", tags={
        "CARDTYPE": "HERO_POWER", "CONTROLLER": "2", "ZONE": "PLAY"})   # COST omitted = 0
    t.state.entities = {185: old, 2619: cur}
    hp = t._hero_power()
    assert (hp["card_id"], hp["cost"], hp["entity_id"]) == ("BG26_HERO_102p", 0, 2619)
    cur.tags["HIDE_COST"] = "1"                 # passive power in PLAY
    assert t._hero_power() is None              # the stale SETASIDE power is not used


def test_choice_rows_hero_pick_and_dark_discovery():
    res = build_game(FIXTURE, META, card_names={"BG36_MidGameEffect_000t51": "Steady Growth"})
    hero, dd = [r for r in res["rows"] if r["kind"] == "choice"]
    assert res["stats"]["choice_blocks"] == 2 and res["stats"]["choice_rows"] == 2
    assert res["stats"]["choice_blocks_other_player"] == 1      # opponent's offer ignored
    assert res["stats"]["choice_kind_hero"] == 1
    assert res["stats"]["choice_kind_dark_discovery"] == 1
    ch = hero["choice"]
    assert (ch["choice_type"], ch["choice_kind"], ch["source_entity_id"], ch["min"],
            ch["max"]) == ("MULLIGAN", "hero", 1, 1, 1)
    assert [(c["entity_id"], c["card_id"], c["cardtype"]) for c in ch["cards"]] == [
        (70, "BG36_HERO_002", "HERO"), (71, "BG_TEST_HERO_B", "HERO")]
    assert "chosen" not in ch and "picked" not in json.dumps(ch)   # pick not recorded
    ch = dd["choice"]
    assert (ch["choice_type"], ch["choice_kind"], ch["source_card_id"]) == (
        "GENERAL", "dark_discovery", "BG36_MidGameEffect_010")
    # 21 carries HAS_DARK_GIFT; 22 only DARK_GIFT_ENTITY: both resolve to the gift.
    assert [(c["entity_id"], c["dark_gift"]) for c in ch["cards"]] == [
        (21, {"card_id": "BG36_MidGameEffect_000t51", "name": "Steady Growth"}),
        (22, {"card_id": "BG36_MidGameEffect_000t51", "name": "Steady Growth"})]
    assert dd["snapshot"]["hero"] == "BG36_HERO_002"
    for r in (hero, dd):
        assert r["schema_version"] == SCHEMA_VERSION
        assert Snapshot.from_dict(json.loads(json.dumps(r["snapshot"]))).to_dict() == r["snapshot"]


def test_choice_kind_and_checks():
    from hsbg_coach.replay_states import _check_choice, _choice_kind

    def card(ct, **tags):
        return {"cardtype": ct, "tags": tags}

    assert _choice_kind("MULLIGAN", None, [card("HERO")]) == "hero"
    # Friendly Wager copies the next combat pair's heroes into a GENERAL choice.
    assert _choice_kind("GENERAL", "TB_BaconShop_HP_081",
                        [card("HERO"), card("HERO")]) == "other"
    assert _choice_kind("GENERAL", "BG36_MidGameEffect_010", [card("MINION")]) == "dark_discovery"
    assert _choice_kind("GENERAL", "BG24_QuestsPlayerEnch_t",
                        [card("SPELL", QUEST="1")] * 2) == "quest"
    assert _choice_kind("GENERAL", "BG30_Trinket_1st", [card("BATTLEGROUND_TRINKET")]) == "trinket"
    assert _choice_kind("GENERAL", "TB_BaconShop_Triples_01",
                        [card("MINION"), card("BATTLEGROUND_SPELL")]) == "discover"
    assert _choice_kind("GENERAL", "EBG_Spell_037", [card("HERO_POWER")]) == "other"
    fail, stats = _Fail(), Counter()
    _check_choice({"dp_index": 0, "choice": {"source_card_id": "X", "cards": []}}, fail, stats)
    _check_choice({"dp_index": 1, "choice": {"source_card_id": "X", "cards": [
        {"entity_id": 5, "card_id": None, "dark_gift": None}]}}, fail, stats)
    assert dict(fail.count) == {"choice_no_cards": 1, "choice_card_missing": 1}


def test_hero_pick_missing_is_a_warning(tmp_path):
    xml = open(FIXTURE, encoding="utf-8").read()
    start = xml.index("<!-- Hero pick")
    end = xml.index("</ChosenEntities>", start) + len("</ChosenEntities>")
    path = tmp_path / "no_hero_pick.xml"
    path.write_text(xml[:start] + xml[end:], encoding="utf-8")
    res = build_game(str(path), META, card_names={"BG36_MidGameEffect_000t51": "Steady Growth"})
    assert res["game_warnings"] == ["hero_pick_missing"]
    assert dict(res["failures"].count) == {}                     # a warning, not a failure


def test_encoder_handles_options_rows():
    np = pytest.importorskip("numpy")
    encode = pytest.importorskip("hsbg_coach.encode")
    res = build_game(FIXTURE, META)
    for r in res["rows"]:
        snap = Snapshot.from_dict(json.loads(json.dumps(r["snapshot"])))
        v = encode.encode_state(snap)
        assert np.isfinite(v).all()
        for o in encode.legal_options(snap):
            assert np.isfinite(encode.encode_option(snap, o)).all()


@pytest.mark.skipif(not os.environ.get("HSBG_REPLAY_XML"),
                    reason="set HSBG_REPLAY_XML to a local Firestone replay (.xml.gz)")
def test_real_replay():
    path = os.environ["HSBG_REPLAY_XML"]
    gid = os.path.basename(path).split(".")[0]
    meta = {"reviewId": gid}
    if os.environ.get("HSBG_REPLAY_MANIFEST"):
        games = json.load(open(os.environ["HSBG_REPLAY_MANIFEST"], encoding="utf-8"))
        meta = {g["reviewId"]: g for g in games.get("games", games)}[gid]
    res = build_game(path, meta, track_opponents=False)
    assert res["rows"]
    for row in res["rows"]:
        snap = row["snapshot"]
        assert len(snap["board"]) <= 7 and len(snap["hand"]) <= 10
        assert Snapshot.from_dict(json.loads(json.dumps(snap))).to_dict() == snap
    if os.environ.get("HSBG_REPLAY_MANIFEST"):
        assert dict(res["failures"].count) == {}


def test_redump_keeps_card_ids_of_dropped_entities():
    """fc4633cb: the gift spell is gone after a re-dump; the gifted minion's
    enchantment (EDR_100t13e) still names it as CREATOR."""
    t = BGTracker()
    t.state.entities[50] = Entity(id=50, card_id="BG36_MidGameEffect_000t13",
                                  tags={"ZONE": "GRAVEYARD"})
    t.state.apply(Event(kind="RESET_ENTITIES"))
    assert 50 not in t.state.entities and t.state.card_id_of(50) == "BG36_MidGameEffect_000t13"
    minion = Entity(id=60, card_id="BG36_097", tags={"HAS_DARK_GIFT": "1"})
    t.state.entities[60] = minion
    t.state.entities[61] = Entity(id=61, card_id="EDR_100t13e", tags={
        "ATTACHED": "60", "ZONE": "PLAY", "CARDTYPE": "ENCHANTMENT", "CREATOR": "50"})
    assert t._minion_dark_gift(minion)["card_id"] == "BG36_MidGameEffect_000t13"
    # CREATOR never seen in the replay: EDR_100t13e itself names the gift
    t.state.entities[61].tags["CREATOR"] = "99"
    assert t._minion_dark_gift(minion)["card_id"] == "BG36_MidGameEffect_000t13"
    t.state.apply(Event(kind="CREATE_GAME"))
    assert t.state.card_id_of(50) is None


def _gold_row(dp, gold):
    return {"dp_index": dp, "options": [],
            "snapshot": {"gold": gold, "board": [], "hand": [], "shop": []}}


def test_gold_unknown_until_first_set():
    fail, stats = _Fail(), Counter()
    for dp, gold in enumerate([None, None, 3, 0]):
        _check_row(BGTracker(), _gold_row(dp, gold), fail, stats)
    assert not fail.count and stats["gold_unknown_rows"] == 2
    _check_row(BGTracker(), _gold_row(4, None), fail, stats)
    _check_row(BGTracker(), _gold_row(5, -1), fail, stats)
    assert fail.count == {"invariant_gold": 2}


def test_dark_discovery_pick_checked_only_while_held():
    def run(held_gift):
        fail, stats = _Fail(), Counter()
        board = [] if held_gift is False else [
            {"entity_id": 7, "dark_gift": held_gift and {"card_id": held_gift}}]
        pending = [{"entity_id": 7, "card_id": "X", "gift": "G1", "persistent": True}]
        _check_picks({"dp_index": 0, "snapshot": {"board": board, "hand": []}},
                     pending, fail, stats)
        return dict(fail.count), stats
    assert run("G1")[1]["dark_discovery_picks_verified"] == 1
    assert run(False)[1]["dark_discovery_picks_gone"] == 1   # e.g. tripled
    # gone, but the golden it was tripled into carries the gift
    fail, stats = _Fail(), Counter()
    _check_picks({"dp_index": 0, "snapshot": {"hand": [], "board": [
        {"entity_id": 9, "dark_gift": {"card_id": "G1"}}]}},
        [{"entity_id": 7, "card_id": "X", "gift": "G1", "persistent": True}], fail, stats)
    assert not fail.count and stats["dark_discovery_picks_verified"] == 1
    # the offer named no gift (no DARK_GIFT_ENTITY): any gift on the held pick
    fail, stats = _Fail(), Counter()
    _check_picks({"dp_index": 0, "snapshot": {"hand": [], "board": [
        {"entity_id": 7, "dark_gift": {"card_id": "G3"}}]}},
        [{"entity_id": 7, "card_id": "X", "gift": None, "persistent": True}], fail, stats)
    assert not fail.count and stats["dark_discovery_picks_verified"] == 1
    assert run("G2")[0] == {"dark_discovery_pick_missing": 1}


def _board_row(turn, *cards):
    return {"turn": turn, "snapshot": {"board": [
        {"card_id": c, "tags": {}, "attack": 1, "health": 1} for c in cards]}}


def test_final_board_uses_manifest_turn_and_combat_board():
    rows = [_board_row(1, "A"), _board_row(2, "A", "B")]

    def run(fc_turn, cards, combat=None):
        fail, stats = _Fail(), Counter()
        meta = {"finalComp": {"turn": fc_turn, "board": [
            {"cardId": c, "golden": False, "atk": 1, "health": 1} for c in cards]}}
        _final_checks(BGTracker(), rows, meta, fail, combat, stats)
        return dict(fail.count), stats
    assert run(2, ["A", "B"]) == ({}, Counter())
    # finalComp from the turn before the last one (0272ec87)
    assert run(1, ["A"]) == ({}, Counter(final_comp_earlier_turn=1))
    # a minion played after the last decision point: the combat board matches
    assert run(2, ["A", "B", "C"], {2: [("A", False, 1, 1), ("B", False, 1, 1),
                                        ("C", False, 1, 1)]})[0] == {}
    assert run(2, ["B", "A"])[0] == {"final_board_order": 1}
    assert run(2, ["A"])[0] == {"manifest_final_comp_suspect": 1}   # our turn-1 board
    assert run(2, ["Z"])[0] == {"final_board_mismatch": 1}
