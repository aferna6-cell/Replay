"""HSReplay XML adapter + states.v1 builder, on a hand-written synthetic replay.

Optional real-replay test: set HSBG_REPLAY_XML=<path to a Firestone .xml.gz>
(and optionally HSBG_REPLAY_MANIFEST=<manifest json> to also require every
fidelity check to pass).
"""

import io
import json
import os
from collections import Counter

import pytest

from hsbg_coach.bg import BGTracker, Phase, Snapshot
from hsbg_coach.hsreplay_xml import iter_events
from hsbg_coach.parser import EntityRef, Event
from hsbg_coach.replay_states import (SCHEMA_VERSION, _Fail, _check_gold_options, _check_picks,
                                       _check_quests, _check_row, _final_checks, build_game)
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


def _play_xml(text):
    tracker, golds = BGTracker(), []
    for ev in iter_events(io.BytesIO(text.encode())):
        tracker.feed(ev)
        if ev.kind == "OPTIONS":
            golds.append(tracker._gold())
    return tracker, golds


def _redump_xml(resources, used_before, dump_resources, extra_before="", extra_after="",
                temp_before=0, played_before=0):
    """Two GameEntity dumps. The second player segment omits RESOURCES_USED,
    TEMP_RESOURCES and NUM_OPTIONS_PLAYED_THIS_TURN."""
    return f"""<?xml version="1.0" encoding="utf-8"?>
<HSReplay><Game>
<GameEntity id="1"><Tag tag="202" value="1"/><Tag tag="49" value="1"/><Tag tag="20" value="5"/></GameEntity>
<Player id="2" accountHi="0" accountLo="0" playerID="1" name="Hero A" isMainPlayer="true">
  <Tag tag="50" value="1"/><Tag tag="202" value="2"/><Tag tag="30" value="1"/>
  <Tag tag="27" value="10"/><Tag tag="26" value="{resources}"/>
  <Tag tag="25" value="0"/></Player>
<Player id="3" accountHi="0" accountLo="0" playerID="9" name="Bob" isMainPlayer="false">
  <Tag tag="50" value="9"/><Tag tag="202" value="2"/><Tag tag="30" value="9"/></Player>
<TagChange entity="2" tag="25" value="{used_before}"/>
<TagChange entity="2" tag="295" value="{temp_before}"/>
<TagChange entity="2" tag="358" value="{played_before}"/>
{extra_before}
<Options id="1"><Option index="0" type="2" entity="0" error="-1"/></Options>
<GameEntity id="1"><Tag tag="202" value="1"/><Tag tag="49" value="1"/><Tag tag="20" value="7"/>
  <Tag tag="53" value="1"/>
  <Tag tag="50" value="1"/><Tag tag="202" value="2"/><Tag tag="30" value="1"/>
  <Tag tag="20" value="6"/><Tag tag="26" value="{dump_resources}"/><Tag tag="27" value="10"/>
  <Tag tag="53" value="2"/>
  <Tag tag="50" value="9"/><Tag tag="202" value="2"/><Tag tag="30" value="9"/>
  <Tag tag="53" value="3"/></GameEntity>
{extra_after}
</Game></HSReplay>
"""


def test_redump_replaces_player_tags_absent_resources_used_is_zero():
    """RESOURCES_USED=5 (and temp / this-turn counters) before the second dump
    and absent from it. Spendable gold is RESOURCES afterwards."""
    text = _redump_xml(resources=10, used_before=5, dump_resources=10,
                       temp_before=2, played_before=8)
    tracker, golds = _play_xml(text)
    pe = tracker._player_entity()
    assert golds == [10 - 5 + 2]
    assert tracker._gold() == 10
    assert pe.id == 2 and pe.name == "Hero A#1" and pe.card_id is None
    assert pe.tags["RESOURCES"] == "10"
    assert pe.tags["CARDTYPE"] == "PLAYER" and pe.tags["CONTROLLER"] == "1"
    assert pe.tags["HERO_ENTITY"] == "10" and pe.tags["PLAYER_ID"] == "1"
    for stale in ("RESOURCES_USED", "TEMP_RESOURCES", "NUM_OPTIONS_PLAYED_THIS_TURN"):
        assert stale not in pe.tags
    # The player's own TURN in the dump does not overwrite the game TURN.
    assert tracker.snapshot().turn == 7
    # Both players are kept.
    assert 3 in tracker.state.entities
    assert tracker.state.entities[3].tags.get("CARDTYPE") == "PLAYER"


def test_redump_4bea8b1a_resources_without_resources_used_is_full_gold(tmp_path):
    """4bea8b1a: after the re-dump RESOURCES=6 and RESOURCES_USED is omitted.
    A merge would keep the stale 5 (gold 1). The next logged spend is
    RESOURCES_USED 5, which only changes gold if the tag was actually cleared."""
    hp = """<FullEntity id="12" cardID="BG_TEST_HP"><Tag tag="50" value="1"/>
  <Tag tag="202" value="10"/><Tag tag="49" value="1"/><Tag tag="48" value="2"/></FullEntity>
<Options id="2"><Option index="0" type="2" entity="0" error="-1"/>
  <Option index="1" type="3" entity="12" error="-1"/></Options>
<TagChange entity="2" tag="25" value="5"/>
"""
    text = _redump_xml(resources=6, used_before=5, dump_resources=6, extra_after=hp)
    tracker, golds = _play_xml(text)
    assert golds == [1, 6]
    assert tracker._gold() == 6 - 5
    assert "RESOURCES_USED" in tracker._player_entity().tags

    path = tmp_path / "redump_gold.xml"
    path.write_text(text, encoding="utf-8")
    res = build_game(str(path), {
        "reviewId": "4bea8b1a-style", "buildNumber": 1, "mmr": 1,
        "placement": None, "finalComp": {"board": []}})
    assert "gold_option_mismatch" not in res["failures"].count
    row = [r for r in res["rows"] if r["kind"] == "options"][-1]
    assert row["snapshot"]["gold"] == 6
    assert row["gold_options"] == [{
        "entity_id": 12, "card_id": "BG_TEST_HP", "kind": "hero_power",
        "cost": 2, "error": -1, "resource": "gold"}]
    assert [o["entity_id"] for o in row["options"]] == [0, 12]
    assert "error" not in row["options"][1]


def test_live_full_entity_merges_player_tags():
    """Power.log never emits RESET_ENTITIES. A later FULL_ENTITY keeps tags
    the new block does not repeat."""
    t = BGTracker()
    t.local_player = 1
    t.player_names[1] = "Hero A#1"
    t.state.entities[2] = Entity(
        id=2, name="Hero A#1", tags={
            "CARDTYPE": "PLAYER", "CONTROLLER": "1", "PLAYER_ID": "1",
            "RESOURCES": "6", "RESOURCES_USED": "5", "TEMP_RESOURCES": "1",
            "NUM_OPTIONS_PLAYED_THIS_TURN": "8", "HERO_ENTITY": "10"})
    t.feed(Event(kind="FULL_ENTITY", logger="GameState",
                 entity=EntityRef(id=2, name="Hero A#1")))
    t.feed(Event(kind="TAG", logger="GameState", tag="RESOURCES", value="8"))
    t.feed(Event(kind="TAG", logger="GameState", tag="CARDTYPE", value="PLAYER"))
    pe = t._player_entity()
    assert pe.tags["RESOURCES_USED"] == "5"
    assert pe.tags["TEMP_RESOURCES"] == "1"
    assert pe.tags["NUM_OPTIONS_PLAYED_THIS_TURN"] == "8"
    assert t._gold() == 8 - 5 + 1


def test_gold_option_mismatch_both_directions():
    t = BGTracker()

    def check(gold, opts, turn=5, health=None):
        snap = {"gold": gold, "board": [], "hand": [], "shop": []}
        if health is not None:
            snap["hero_health"] = health
        row = {
            "dp_index": 1, "turn": turn,
            "snapshot": snap,
            "hero_power": None, "hero_powers": [], "options": [],
            "gold_options": opts,
        }
        fail, stats = _Fail(), Counter()
        _check_row(t, row, fail, stats)
        return fail.count["gold_option_mismatch"]

    def opt(kind, cost, err):
        return {"entity_id": 12, "card_id": "X", "kind": kind, "cost": cost, "error": err}

    assert check(0, [opt("hero_power", 2, -1)]) == 1
    assert check(0, [opt("buy", 3, -1)]) == 1
    assert check(5, [opt("hero_power", 1, 14)]) == 1
    assert check(1, [opt("buy", 1, 14)]) == 1
    assert check(0, [opt("hero_power", 2, 14)]) == 0
    assert check(2, [opt("hero_power", 2, -1)]) == 0
    assert check(None, [opt("hero_power", 2, -1)]) == 0
    # ea39f046 options idx 3: turn-1 Queen of Dragons, cost 1, gold 0, legal
    # for one block ("Unlocks at Tier 4"). Gold-priced. Not special-cased.
    assert check(0, [opt("hero_power", 1, -1)], turn=1) == 1
    # Missing resource stays on the gold check. Null hero health skips a
    # Health-paid option rather than treating it as gold.
    assert check(0, [dict(opt("buy", 3, -1), resource="health")]) == 0
    assert check(0, [dict(opt("buy", 3, -1), resource="health")], health=2) == 1
    assert check(22, [dict(opt("buy", 3, 14), resource="health")], health=20) == 1


def test_gold_option_mismatch_from_built_row(tmp_path):
    text = """<?xml version="1.0" encoding="utf-8"?>
<HSReplay><Game>
<GameEntity id="1"><Tag tag="202" value="1"/><Tag tag="49" value="1"/><Tag tag="20" value="5"/></GameEntity>
<Player id="2" accountHi="0" accountLo="0" playerID="1" name="Hero A" isMainPlayer="true">
  <Tag tag="50" value="1"/><Tag tag="202" value="2"/><Tag tag="26" value="1"/>
  <Tag tag="25" value="0"/></Player>
<Player id="3" accountHi="0" accountLo="0" playerID="9" name="Bob" isMainPlayer="false">
  <Tag tag="50" value="9"/><Tag tag="202" value="2"/></Player>
<FullEntity id="12" cardID="BG_TEST_HP"><Tag tag="50" value="1"/>
  <Tag tag="202" value="10"/><Tag tag="49" value="1"/><Tag tag="48" value="2"/></FullEntity>
<FullEntity id="40" cardID="TB_BaconShop_DragBuy"><Tag tag="50" value="9"/>
  <Tag tag="202" value="4"/><Tag tag="48" value="1"/></FullEntity>
<Options id="1"><Option index="0" type="2" entity="0" error="-1"/>
  <Option index="1" type="3" entity="12" error="-1"/>
  <Option index="2" type="3" entity="40" error="14"/></Options>
</Game></HSReplay>
"""
    path = tmp_path / "gold_mismatch.xml"
    path.write_text(text, encoding="utf-8")
    res = build_game(str(path), {
        "reviewId": "gold-mismatch", "buildNumber": 1, "mmr": 1,
        "placement": None, "finalComp": {"board": []}})
    assert res["failures"].count["gold_option_mismatch"] == 2
    row = res["rows"][-1]
    assert [o["entity_id"] for o in row["options"]] == [0, 12]
    assert {(o["entity_id"], o["error"], o["cost"], o["resource"])
            for o in row["gold_options"]} == {
        (12, -1, 2, "gold"), (40, 14, 1, "gold")}


def test_shop_minion_option_is_not_a_buy(tmp_path):
    """The shop minion stays error=-1 when its DragBuy handle is error 14.
    Pricing the minion would flag every unaffordable slot."""
    text = """<?xml version="1.0" encoding="utf-8"?>
<HSReplay><Game>
<GameEntity id="1"><Tag tag="202" value="1"/><Tag tag="49" value="1"/><Tag tag="20" value="1"/></GameEntity>
<Player id="2" accountHi="0" accountLo="0" playerID="1" name="Hero A" isMainPlayer="true">
  <Tag tag="50" value="1"/><Tag tag="202" value="2"/><Tag tag="26" value="0"/>
  <Tag tag="25" value="0"/></Player>
<Player id="3" accountHi="0" accountLo="0" playerID="9" name="Bob" isMainPlayer="false">
  <Tag tag="50" value="9"/><Tag tag="202" value="2"/></Player>
<FullEntity id="30" cardID="BG_TEST_SHOP"><Tag tag="50" value="9"/>
  <Tag tag="202" value="4"/><Tag tag="49" value="1"/></FullEntity>
<FullEntity id="31" cardID="TB_BaconShop_DragBuy"><Tag tag="50" value="1"/>
  <Tag tag="202" value="22"/><Tag tag="49" value="1"/><Tag tag="48" value="3"/>
  <Tag tag="2442" value="30"/></FullEntity>
<Options id="1"><Option index="0" type="2" entity="0" error="-1"/>
  <Option index="1" type="3" entity="30" error="-1"/>
  <Option index="2" type="3" entity="31" error="14"/></Options>
</Game></HSReplay>
"""
    path = tmp_path / "shop_not_buy.xml"
    path.write_text(text, encoding="utf-8")
    res = build_game(str(path), {
        "reviewId": "shop-not-buy", "buildNumber": 1, "mmr": 1,
        "placement": None, "finalComp": {"board": []}})
    assert "gold_option_mismatch" not in res["failures"].count
    row = res["rows"][-1]
    assert row["snapshot"]["gold"] == 0
    assert [o["entity_id"] for o in row["options"]] == [0, 30]
    assert row["gold_options"] == [{
        "entity_id": 31, "card_id": "TB_BaconShop_DragBuy", "kind": "buy",
        "cost": 3, "error": 14, "resource": "gold"}]


def test_pre_fix_stale_gold_still_flags_4bea8b1a_hero_power():
    """4bea8b1a on merge-semantics gold: a legal cost-2 hero power at gold 1
    is a mismatch. The same cost paid with health at hero_health 30 is not."""
    def run(resource, health):
        row = {
            "dp_index": 16,
            "snapshot": {"gold": 1, "hero_health": health,
                         "board": [], "hand": [], "shop": []},
            "gold_options": [{
                "kind": "hero_power", "cost": 2, "error": -1,
                "resource": resource,
            }],
        }
        fail = _Fail()
        _check_gold_options(row, fail)
        return fail

    stale = run("gold", 30)
    assert stale.count["gold_option_mismatch"] == 1
    assert stale.examples["gold_option_mismatch"][0]["resource"] == "gold"
    assert run("health", 30).count["gold_option_mismatch"] == 0


def _health_game(tmp_path, name, resources, health, damage, middle):
    """One options row. Hero survivability is HEALTH + ARMOR - DAMAGE."""
    text = f"""<?xml version="1.0" encoding="utf-8"?>
<HSReplay><Game>
<GameEntity id="1"><Tag tag="202" value="1"/><Tag tag="49" value="1"/><Tag tag="20" value="5"/></GameEntity>
<Player id="2" accountHi="0" accountLo="0" playerID="1" name="Hero A" isMainPlayer="true">
  <Tag tag="50" value="1"/><Tag tag="202" value="2"/><Tag tag="27" value="10"/>
  <Tag tag="26" value="{resources}"/><Tag tag="25" value="0"/></Player>
<Player id="3" accountHi="0" accountLo="0" playerID="9" name="Bob" isMainPlayer="false">
  <Tag tag="50" value="9"/><Tag tag="202" value="2"/></Player>
<FullEntity id="10" cardID="BG_TEST_HERO"><Tag tag="50" value="1"/>
  <Tag tag="202" value="3"/><Tag tag="49" value="1"/>
  <Tag tag="45" value="{health}"/><Tag tag="44" value="{damage}"/></FullEntity>
{middle}
</Game></HSReplay>
"""
    path = tmp_path / f"{name}.xml"
    path.write_text(text, encoding="utf-8")
    return build_game(str(path), {
        "reviewId": name, "buildNumber": 1, "mmr": 1,
        "placement": None, "finalComp": {"board": []}})


def _handle(eid, card, cost, error, alternate=True):
    alt = '<Tag tag="2837" value="1"/>' if alternate else ""
    return (
        f'<FullEntity id="{eid}" cardID="{card}"><Tag tag="50" value="1"/>'
        f'<Tag tag="202" value="22"/><Tag tag="49" value="1"/>'
        f'<Tag tag="48" value="{cost}"/>{alt}</FullEntity>',
        f'<Option index="1" type="3" entity="{eid}" error="{error}"/>',
    )


def test_health_buy_legal_while_gold_is_zero(tmp_path):
    """Hasty Excavation (03cfc343 dp 213): DragBuy_Spell is legal at gold 0
    because CARD_ALTERNATE_COST=1 and hero health 13 is above cost 3."""
    entity, option = _handle(40, "TB_BaconShop_DragBuy_Spell", 3, -1)
    res = _health_game(tmp_path, "hasty-legal", 0, 30, 17,
                       entity + "\n<Options id=\"1\">"
                       "<Option index=\"0\" type=\"2\" entity=\"0\" error=\"-1\"/>"
                       + option + "</Options>")
    assert "gold_option_mismatch" not in res["failures"].count
    row = res["rows"][-1]
    assert row["snapshot"]["gold"] == 0
    assert row["snapshot"]["hero_health"] == 13
    assert row["gold_options"] == [{
        "entity_id": 40, "card_id": "TB_BaconShop_DragBuy_Spell", "kind": "buy",
        "cost": 3, "error": -1, "resource": "health"}]


def test_health_buy_refused_when_health_is_not_above_cost(tmp_path):
    """27147d0d dp 456: error 14 at health 1, gold 22. b55b2f23 dp 357:
    error 14 when health equals cost. Both are correct refusals."""
    low_ent, low_opt = _handle(40, "TB_BaconShop_DragBuy_Spell", 3, 14)
    low = _health_game(tmp_path, "hasty-low", 22, 1, 0,
                       low_ent + "\n<Options id=\"1\">"
                       "<Option index=\"0\" type=\"2\" entity=\"0\" error=\"-1\"/>"
                       + low_opt + "</Options>")
    assert "gold_option_mismatch" not in low["failures"].count
    assert low["rows"][-1]["snapshot"]["hero_health"] == 1
    assert low["rows"][-1]["gold_options"][0]["resource"] == "health"

    eq_ent, eq_opt = _handle(40, "TB_BaconShop_DragBuy_Spell", 3, 14)
    equal = _health_game(tmp_path, "hasty-equal", 19, 3, 0,
                         eq_ent + "\n<Options id=\"1\">"
                         "<Option index=\"0\" type=\"2\" entity=\"0\" error=\"-1\"/>"
                         + eq_opt + "</Options>")
    assert "gold_option_mismatch" not in equal["failures"].count
    assert equal["rows"][-1]["snapshot"]["gold"] == 19
    assert equal["rows"][-1]["snapshot"]["hero_health"] == 3


def test_health_buy_contradiction(tmp_path):
    """A legal Health buy at health 2 / cost 3, and an error-14 Health buy
    at health 20 / cost 3, both disagree with the snapshot."""
    legal_ent, legal_opt = _handle(40, "TB_BaconShop_DragBuy_Spell", 3, -1)
    legal = _health_game(tmp_path, "health-short", 0, 2, 0,
                         legal_ent + "\n<Options id=\"1\">"
                         "<Option index=\"0\" type=\"2\" entity=\"0\" error=\"-1\"/>"
                         + legal_opt + "</Options>")
    assert legal["failures"].count["gold_option_mismatch"] == 1
    assert legal["failures"].examples["gold_option_mismatch"][0]["resource"] == "health"
    assert legal["failures"].examples["gold_option_mismatch"][0]["hero_health"] == 2

    rich_ent, rich_opt = _handle(40, "TB_BaconShop_DragBuy_Spell", 3, 14)
    rich = _health_game(tmp_path, "health-rich", 0, 20, 0,
                        rich_ent + "\n<Options id=\"1\">"
                        "<Option index=\"0\" type=\"2\" entity=\"0\" error=\"-1\"/>"
                        + rich_opt + "</Options>")
    assert rich["failures"].count["gold_option_mismatch"] == 1
    example = rich["failures"].examples["gold_option_mismatch"][0]
    assert example["resource"] == "health" and example["hero_health"] == 20
    assert example["error"] == 14 and example["cost"] == 3


def test_health_refresh_blocked_at_one_health(tmp_path):
    """634db4af dp 837: Malchezaar's reroll is Health-paid, error 14 at
    health 1 (30 health, 29 damage) while gold is 10."""
    entity, option = _handle(40, "TB_BaconShop_8p_Reroll_Button", 1, 14)
    res = _health_game(tmp_path, "health-refresh", 10, 30, 29,
                       entity + "\n<Options id=\"1\">"
                       "<Option index=\"0\" type=\"2\" entity=\"0\" error=\"-1\"/>"
                       + option + "</Options>")
    assert "gold_option_mismatch" not in res["failures"].count
    row = res["rows"][-1]
    assert row["snapshot"]["gold"] == 10
    assert row["snapshot"]["hero_health"] == 1
    assert row["gold_options"] == [{
        "entity_id": 40, "card_id": "TB_BaconShop_8p_Reroll_Button", "kind": "roll",
        "cost": 1, "error": 14, "resource": "health"}]


def test_effect_health_minion_buy_legal_at_zero_gold(tmp_path):
    """3fa73f4e dp 143: a DragBuy handle with CARD_ALTERNATE_COST=1 is a
    legal Health buy at gold 0."""
    entity, option = _handle(40, "TB_BaconShop_DragBuy", 3, -1)
    res = _health_game(tmp_path, "effect-health-buy", 0, 30, 0,
                       entity + "\n<Options id=\"1\">"
                       "<Option index=\"0\" type=\"2\" entity=\"0\" error=\"-1\"/>"
                       + option + "</Options>")
    assert "gold_option_mismatch" not in res["failures"].count
    row = res["rows"][-1]
    assert row["snapshot"]["gold"] == 0
    assert row["snapshot"]["hero_health"] == 30
    assert row["gold_options"] == [{
        "entity_id": 40, "card_id": "TB_BaconShop_DragBuy", "kind": "buy",
        "cost": 3, "error": -1, "resource": "health"}]


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
    # One power: the snapshot stays free of hero_powers; the row lists it.
    assert "hero_powers" not in first["snapshot"]
    assert first["hero_powers"][0]["card_id"] == "BG36_HERO_002p"
    assert first["hero_powers"][0]["activatable"] is True
    assert first["hero_powers"][0]["passive"] is False
    assert first["dark_discovery"] == {"available": False}
    # Error stays off the legal-action list. The error-14 board minion is not
    # a buy, so only the priced hero power is checked against gold.
    assert "error" not in first["options"][2]
    assert first["gold_options"] == [{
        "entity_id": 12, "card_id": "BG36_HERO_002p", "kind": "hero_power",
        "cost": 1, "error": -1, "resource": "gold"}]
    assert last["gold_options"] == []
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


def _quest_row(dp, hero, *quests, card_id="BG24_Quest_151", board=None):
    snap = {"hero": hero, "quests": [
        {"entity_id": e, "card_id": card_id, "progress": p, "goal": g,
         "completed": c} for e, p, g, c in quests]}
    if board is not None:
        snap["board"] = board
    return {"dp_index": dp, "snapshot": snap}


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
    # Pressure the Authorities tracks live warband attack (a8aebfa4: 9 → 0
    # on an empty board, goal 20 not the card baseline 28). A drop that is
    # not the board's attack still fails, and a cumulative quest still fails.
    attack = dict(card_id="BG27_Quest_801")
    assert run(_quest_row(0, "BG24_HERO_100", (363, 9, 20, False)),
               _quest_row(1, "BG24_HERO_100", (363, 0, 20, False),
                          board=[{"attack": 0}], **attack)) == {}
    assert run(_quest_row(0, "BG24_HERO_100", (363, 9, 20, False)),
               _quest_row(1, "BG24_HERO_100", (363, 0, 20, False),
                          board=[{"attack": 9}], **attack)) == {
        "quest_progress_decreased": 1}


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
    # No reroll: cards stay the offer, and offer_initial echoes them.
    assert ch["offer_initial"] == ch["cards"]
    assert "rerolls" not in ch
    assert "chosen" not in ch and "picked" not in json.dumps(ch)   # pick not recorded
    ch = dd["choice"]
    assert "offer_initial" not in ch and "rerolls" not in ch
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


def _tag(eid, tag, value, kind="TAG_CHANGE", card_id=None):
    return Event(kind=kind, logger="GameState",
                 entity=EntityRef(id=eid, card_id=card_id), tag=tag, value=value)


def test_poet_permanent_enchantment_names_the_gift_spell():
    t = BGTracker()
    assert t._gift_spell_id("BG36_MidGameEffect_000t64") == "BG36_MidGameEffect_000t64"
    assert t._gift_spell_id("BG36_MidGameEffect_000t64e") == "BG36_MidGameEffect_000t64"
    assert t._gift_spell_id("BG36_MidGameEffect_000t64e2") == "BG36_MidGameEffect_000t64"
    assert t._gift_spell_id("BG36_MidGameEffect_000t64e3") == "BG36_MidGameEffect_000t64"
    assert t._gift_spell_id("BG36_MidGameEffect_000t2e3") == "BG36_MidGameEffect_000t2"
    assert t._gift_spell_id("BG36_MidGameEffect_000t64te") == "BG36_MidGameEffect_000t64"
    assert t._gift_spell_id("BG36_MidGameEffect_000t64te2") == "BG36_MidGameEffect_000t64"
    assert t._gift_spell_id("BG36_MidGameEffect_000t51e") == "BG36_MidGameEffect_000t51"
    # Extra trailing t on the enchantment, not on the spell.
    assert t._gift_spell_id("BG36_MidGameEffect_000t64t") == "BG36_MidGameEffect_000t64"
    assert t._gift_spell_id("EDR_100t13e") is None
    # Offensive Sacrifice is the spell …000t. Its own trailing t stays, and
    # its enchantment …000te must not collapse to the parent game-effect.
    assert t._gift_spell_id("BG36_MidGameEffect_000t") == "BG36_MidGameEffect_000t"
    assert t._gift_spell_id("BG36_MidGameEffect_000te") == "BG36_MidGameEffect_000t"
    assert t._gift_spell_id("BG36_MidGameEffect_000te2") == "BG36_MidGameEffect_000t"
    assert t._gift_spell_id("BG36_MidGameEffect_000tte") == "BG36_MidGameEffect_000t"
    assert t._gift_spell_id("BG36_MidGameEffect_000") is None


def _attach_gift(t, minion, ench, ench_card):
    t.local_player = 1
    t.feed(_tag(minion, None, None, "FULL_ENTITY", "BG_TEST"))
    for tag, value in (("CARDTYPE", "MINION"), ("ZONE", "PLAY"), ("CONTROLLER", "1"),
                       ("HAS_DARK_GIFT", "1")):
        t.feed(_tag(minion, tag, value))
    t.feed(_tag(ench, None, None, "SHOW_ENTITY", ench_card))
    for tag, value in (("CARDTYPE", "ENCHANTMENT"), ("ZONE", "PLAY"),
                       ("ATTACHED", str(minion))):
        t.feed(_tag(ench, tag, value))


def test_offensive_sacrifice_keeps_000t():
    """…000te is the enchantment of Offensive Sacrifice (…000t), not the
    parent game-effect entity …000."""
    t = BGTracker()
    t.state.card_names["BG36_MidGameEffect_000t"] = "Offensive Sacrifice"
    _attach_gift(t, 10, 11, "BG36_MidGameEffect_000te")
    gift = t._minion_dark_gift(t.state.entities[10])
    assert gift == {"card_id": "BG36_MidGameEffect_000t", "name": "Offensive Sacrifice"}
    # The spell entity itself, and a parent pointer, must not become …000.
    t.feed(_tag(12, None, None, "FULL_ENTITY", "BG36_MidGameEffect_000t"))
    t.feed(_tag(10, "DARK_GIFT_ENTITY", "12"))
    assert t._minion_dark_gift(t.state.entities[10])["card_id"] == "BG36_MidGameEffect_000t"
    t.feed(_tag(13, None, None, "FULL_ENTITY", "BG36_MidGameEffect_000"))
    t.feed(_tag(10, "DARK_GIFT_ENTITY", "13"))
    # Parent is ignored; the attached enchantment still names the spell.
    assert t._minion_dark_gift(t.state.entities[10])["card_id"] == "BG36_MidGameEffect_000t"


def test_trailing_t_enchantment_maps_to_gift_spell():
    t = BGTracker()
    t.state.card_names["BG36_MidGameEffect_000t64"] = "Dexterity"
    _attach_gift(t, 10, 11, "BG36_MidGameEffect_000t64t")
    assert t._minion_dark_gift(t.state.entities[10]) == {
        "card_id": "BG36_MidGameEffect_000t64", "name": "Dexterity"}


def test_dark_gift_unknown_card_quarantines():
    t = BGTracker()

    def row(gift):
        return {"dp_index": 1, "options": [], "snapshot": {
            "gold": 1,
            "board": [{"entity_id": 7, "card_id": "BG_M",
                       "tags": {"HAS_DARK_GIFT": "1"}, "dark_gift": gift}],
            "hand": [], "shop": []}}

    fail = _Fail()
    _check_row(t, row({"card_id": "BG36_MidGameEffect_000", "name": None}), fail, Counter())
    assert fail.count["dark_gift_unknown_card"] == 1
    assert fail.examples["dark_gift_unknown_card"][0]["card_id"] == "BG36_MidGameEffect_000"
    # An enchantment id that was not mapped back to a spell.
    fail = _Fail()
    _check_row(t, row({"card_id": "BG36_MidGameEffect_000te", "name": None}), fail, Counter())
    assert fail.count["dark_gift_unknown_card"] == 1
    # No card-name table: a structural spell id is enough.
    fail = _Fail()
    _check_row(t, row({"card_id": "BG36_MidGameEffect_000t",
                       "name": "BG36_MidGameEffect_000t"}), fail, Counter())
    assert "dark_gift_unknown_card" not in fail.count
    t.state.card_names["BG36_MidGameEffect_000t"] = "Offensive Sacrifice"
    fail = _Fail()
    _check_row(t, row({"card_id": "BG36_MidGameEffect_000t",
                       "name": "Offensive Sacrifice"}), fail, Counter())
    assert "dark_gift_unknown_card" not in fail.count
    # Table loaded, but this id never received a name.
    fail = _Fail()
    _check_row(t, row({"card_id": "BG36_MidGameEffect_000t64",
                       "name": "BG36_MidGameEffect_000t64"}), fail, Counter())
    assert fail.count["dark_gift_unknown_card"] == 1


def test_hand_copy_inherits_dark_gift_after_enchantment_is_removed():
    """b138f295: Radio Star copies Persistent Poet after the killer's gift
    enchantment (000t64te) was removed on death. The hand copy keeps
    HAS_DARK_GIFT; COPIED_FROM on the intermediate is later cleared."""
    t = BGTracker()
    t.local_player = 1
    t.feed(_tag(10, None, None, "FULL_ENTITY", "BG29_813"))
    for tag, value in (("CARDTYPE", "MINION"), ("ZONE", "PLAY"), ("CONTROLLER", "1"),
                       ("HAS_DARK_GIFT", "1")):
        t.feed(_tag(10, tag, value))
    t.feed(_tag(11, None, None, "SHOW_ENTITY", "BG36_MidGameEffect_000t64te"))
    for tag, value in (("CARDTYPE", "ENCHANTMENT"), ("ZONE", "PLAY"), ("ATTACHED", "10")):
        t.feed(_tag(11, tag, value))
    assert t._minion_dark_gift(t.state.entities[10])["card_id"] == "BG36_MidGameEffect_000t64"
    t.feed(_tag(11, "ZONE", "REMOVEDFROMGAME"))
    t.feed(_tag(12, None, None, "FULL_ENTITY", "BG29_813"))
    for tag, value in (("CARDTYPE", "MINION"), ("CONTROLLER", "1"), ("ZONE", "SETASIDE"),
                       ("HAS_DARK_GIFT", "1"), ("COPIED_FROM_ENTITY_ID", "10")):
        t.feed(_tag(12, tag, value))
    t.feed(_tag(13, None, None, "FULL_ENTITY", "BG29_813"))
    for tag, value in (("CARDTYPE", "MINION"), ("CONTROLLER", "1"), ("ZONE", "HAND"),
                       ("HAS_DARK_GIFT", "1"), ("COPIED_FROM_ENTITY_ID", "12")):
        t.feed(_tag(13, tag, value))
    t.feed(_tag(12, "COPIED_FROM_ENTITY_ID", "0"))
    t.feed(_tag(12, "HAS_DARK_GIFT", "0"))
    assert t._minion_dark_gift(t.state.entities[13])["card_id"] == "BG36_MidGameEffect_000t64"
    # A flagged minion with no enchantment and no copied gift stays unresolved.
    t.feed(_tag(14, None, None, "FULL_ENTITY", "BG29_813"))
    for tag, value in (("CARDTYPE", "MINION"), ("CONTROLLER", "1"), ("ZONE", "HAND"),
                       ("HAS_DARK_GIFT", "1")):
        t.feed(_tag(14, tag, value))
    assert t._minion_dark_gift(t.state.entities[14]) is None


# The two 36.6.3 first-place games the builder quarantined. Final-board and
# placement checks need Firestone's manifest, which is not in the fixture;
# these tests assert the accuracy reasons that quarantined them are gone.
_REPLAYS = os.path.join(os.path.dirname(__file__), "fixtures", "replays")
_MANIFEST_REASONS = {"manifest_final_comp_suspect", "final_board_mismatch",
                     "final_board_order", "placement_mismatch",
                     "manifest_placement_suspect"}


def _replay(name):
    path = os.path.join(_REPLAYS, name)
    if not os.path.isfile(path):
        pytest.skip("replay fixture missing")
    gid = name.split(".")[0]
    return build_game(path, {"reviewId": gid, "placement": 1}, track_opponents=False)


def test_radio_star_poet_copy_resolves_dark_gift():
    res = _replay("b138f295-2101-43ce-8fb5-4f7a934eb84d.xml.gz")
    assert "dark_gift_unresolved" not in res["failures"].count
    assert set(res["failures"].count) <= _MANIFEST_REASONS
    gifted = [m for r in res["rows"] for z in ("board", "hand", "shop")
              for m in r["snapshot"][z]
              if m["entity_id"] == 9052 and m["tags"].get("HAS_DARK_GIFT") == "1"]
    assert gifted and {m["dark_gift"]["card_id"] for m in gifted} == {
        "BG36_MidGameEffect_000t64"}


def test_pressure_the_authorities_attack_may_fall():
    res = _replay("a8aebfa4-8e71-4418-93ab-52873dec5647.xml.gz")
    assert "quest_progress_decreased" not in res["failures"].count
    assert set(res["failures"].count) <= _MANIFEST_REASONS
    quests = [(r["dp_index"], q) for r in res["rows"]
              for q in r["snapshot"]["quests"] if q["entity_id"] == 363]
    drop = next(q for dp, q in quests if dp == 21)
    assert (drop["progress"], drop["goal"], drop["card_id"]) == (0, 20, "BG27_Quest_801")
    assert sum(m["attack"] or 0 for m in next(
        r["snapshot"]["board"] for r in res["rows"] if r["dp_index"] == 21)) == 0
    assert any(q["completed"] for _, q in quests)
