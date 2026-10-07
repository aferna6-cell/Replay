"""Dual hero powers and hero-pick rerolls. Synthetic entities and XML only.

(a) passive first, active second
(b) two active powers, both legal: two hero_power actions
(c) a single power is unchanged (no hero_powers key)
(d) passive only
(e) a second power created by a trinket, with no card-id allowlist
(f) reroll of the picked hero slot
(g) reroll of an unpicked slot
(h) no reroll: cards unchanged, offer_initial echoes them
"""

import json
from collections import Counter

from hsbg_coach.actions import HERO_POWER, legal_actions
from hsbg_coach.bg import BGTracker, Snapshot
from hsbg_coach.replay_states import (
    _Fail, _check_row, _select_row_hero_power, build_game,
)
from hsbg_coach.state import Entity

# Reinvigorating Light: "Discover a second Hero Power. Gain 3 Gold."
TRINKET_HP = "BG36_MagicItem_411"


def _tracker(gold="10"):
    t = BGTracker()
    t.local_player = 1
    t.player_names = {1: "Me"}
    t.state.entities[2] = Entity(id=2, name="Me", tags={
        "CARDTYPE": "PLAYER", "HERO_ENTITY": "9", "RESOURCES": gold,
        "RESOURCES_USED": "0"})
    t.state.entities[9] = Entity(id=9, card_id="BG_HERO", tags={
        "CARDTYPE": "HERO", "CONTROLLER": "1", "ZONE": "PLAY", "HEALTH": "30"})
    return t


def _power(eid, card, name=None, **tags):
    base = {"CARDTYPE": "HERO_POWER", "CONTROLLER": "1", "ZONE": "PLAY"}
    base.update({k: str(v) for k, v in tags.items()})
    return Entity(id=eid, card_id=card, name=name, tags=base)


def _ids(powers):
    return [p["card_id"] for p in powers]


# --- snapshot / actions ----------------------------------------------------
def test_a_passive_first_does_not_hide_active_second():
    t = _tracker()
    # Lower id and lower zone position: the old scan returned None on the
    # first HIDE_COST=1 power and never reached the clickable one.
    t.state.entities[12] = _power(12, "BG34_HERO_004p", HIDE_COST=1, ZONE_POSITION=1)
    t.state.entities[13] = _power(13, "TB_BaconShop_HP_102", COST=1, ZONE_POSITION=2)
    hp = t._hero_power()
    assert hp["card_id"] == "TB_BaconShop_HP_102"
    assert hp["cost"] == 1 and hp["usable"] is True and hp["entity_id"] == 13
    snap = t.snapshot()
    assert snap.hero_power["card_id"] == "TB_BaconShop_HP_102"
    # Only one non-passive power: the optional list stays off the snapshot.
    assert snap.hero_powers is None
    assert "hero_powers" not in snap.to_dict()


def test_b_two_active_powers_emit_two_actions():
    t = _tracker()
    # Usable power with the worse zone position must still sort first.
    t.state.entities[20] = _power(20, "TB_BaconShop_HP_022", COST=1, ZONE_POSITION=2,
                                  EXHAUSTED=0)
    t.state.entities[21] = _power(21, "BG28_HERO_400p", COST=2, ZONE_POSITION=1,
                                  EXHAUSTED=1)
    powers = t._hero_powers()
    assert _ids(powers) == ["TB_BaconShop_HP_022", "BG28_HERO_400p"]
    assert powers[0]["usable"] is True and powers[1]["usable"] is False
    assert t._hero_power()["entity_id"] == 20

    # Both usable, same zone: lowest entity id. Then a lower zone wins.
    t.state.entities[21].tags["EXHAUSTED"] = "0"
    t.state.entities[20].tags["ZONE_POSITION"] = "1"
    t.state.entities[21].tags["ZONE_POSITION"] = "1"
    assert _ids(t._hero_powers()) == ["TB_BaconShop_HP_022", "BG28_HERO_400p"]
    t.state.entities[20].tags["ZONE_POSITION"] = "2"
    powers = t._hero_powers()
    assert _ids(powers) == ["BG28_HERO_400p", "TB_BaconShop_HP_022"]
    assert all(p["usable"] for p in powers)

    snap = t.snapshot()
    assert snap.hero_powers is not None and snap.hero_power == snap.hero_powers[0]
    again = Snapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
    assert again == snap
    acts = [a for a in legal_actions(snap) if a.kind == HERO_POWER]
    assert len(acts) == 2
    assert [a.target for a in acts] == ["Hero Power", "Hero Power"]
    assert [a.cost for a in acts] == [2, 1]
    assert [a.detail["hero_power"]["card_id"] for a in acts] == _ids(powers)

    # Present but one unusable: do not also fall back to hero_power.
    snap.hero_powers = [dict(powers[0], usable=False), dict(powers[1], usable=True)]
    acts = [a for a in legal_actions(snap) if a.kind == HERO_POWER]
    assert [a.detail["hero_power"]["card_id"] for a in acts] == ["TB_BaconShop_HP_022"]


def test_c_single_power_output_is_unchanged():
    t = _tracker()
    stale = _power(185, "BG36_HERO_002p", ZONE="SETASIDE", COST=1)
    cur = _power(2619, "BG26_HERO_102p")          # COST omitted = 0
    t.state.entities[185] = stale
    t.state.entities[2619] = cur
    hp = t._hero_power()
    assert hp == {
        "name": "Hero Power",
        "card_id": "BG26_HERO_102p",
        "cost": 0,
        "usable": True,
        "entity_id": 2619,
    }
    raw = json.dumps(t.snapshot().to_dict(), sort_keys=False)
    assert '"hero_powers"' not in raw
    # Absent hero_powers falls back to the one hero_power.
    acts = [a for a in legal_actions({"gold": 3, "hero_power": hp, "shop": [], "board": []})
            if a.kind == HERO_POWER]
    assert len(acts) == 1 and acts[0].cost == 0
    # A dict that simply has no hero_powers key is the historical input.
    assert len([a for a in legal_actions({"gold": 0, "hero_power": hp, "shop": [], "board": []})
                if a.kind == HERO_POWER]) == 1


def test_d_passive_only_is_not_offered():
    t = _tracker()
    t.state.entities[122] = _power(122, "BG34_HERO_004p", name="Warped Conflux", HIDE_COST=1)
    assert t._hero_power() is None
    snap = t.snapshot()
    assert snap.hero_power is None and snap.hero_powers is None
    assert "hero_powers" not in snap.to_dict()
    assert HERO_POWER not in [a.kind for a in legal_actions(snap)]


def test_e_trinket_granted_second_power_is_included():
    """No allowlist. The second power's card id is not a known hero power;
    CREATOR points at Reinvigorating Light and that is enough."""
    t = _tracker()
    t.state.entities[40] = Entity(id=40, card_id=TRINKET_HP, tags={
        "CARDTYPE": "BATTLEGROUND_TRINKET", "ZONE": "PLAY", "CONTROLLER": "1"})
    t.state.entities[12] = _power(12, "BG36_HERO_000p", COST=1, ZONE_POSITION=1)
    t.state.entities[13] = _power(13, "BG_TRINKET_GRANTED_HP", COST=0, ZONE_POSITION=2,
                                  CREATOR=40)
    powers = t._hero_powers()
    assert _ids(powers) == ["BG36_HERO_000p", "BG_TRINKET_GRANTED_HP"]
    snap = t.snapshot()
    assert [t["card_id"] for t in snap.trinkets] == [TRINKET_HP]
    assert _ids(snap.hero_powers) == ["BG36_HERO_000p", "BG_TRINKET_GRANTED_HP"]
    acts = [a for a in legal_actions(snap) if a.kind == HERO_POWER]
    assert len(acts) == 2
    assert {a.detail["hero_power"]["card_id"] for a in acts} == {
        "BG36_HERO_000p", "BG_TRINKET_GRANTED_HP"}


def test_hero_power_option_mismatch_quarantine():
    t = _tracker()
    t.state.entities[5] = _power(5, "HP_A", COST=1)
    row = {
        "dp_index": 3,
        "snapshot": {"gold": 10, "board": [], "hand": [], "shop": []},
        "hero_power": {"card_id": "HP_A", "name": "HP_A", "cost": 1,
                       "used": False, "activatable": False},
        "hero_powers": [{"card_id": "HP_A", "name": "HP_A", "cost": 1,
                         "used": False, "activatable": False, "passive": False}],
        "options": [{"index": 1, "type": "POWER", "entity_id": 5, "card_id": "HP_A",
                     "zone": "PLAY", "targets": [], "sub_options": []}],
    }
    fail, stats = _Fail(), Counter()
    _check_row(t, row, fail, stats)
    assert fail.count["hero_power_option_mismatch"] >= 1


# --- hero-pick rows (synthetic XML) ----------------------------------------
_HEAD = """<?xml version="1.0" encoding="utf-8"?>
<HSReplay><Game>
<GameEntity id="1"><Tag tag="202" value="1"/><Tag tag="49" value="1"/><Tag tag="20" value="1"/></GameEntity>
<Player id="2" accountHi="0" accountLo="0" playerID="1" name="Hero A" isMainPlayer="true">
  <Tag tag="50" value="1"/><Tag tag="202" value="2"/><Tag tag="27" value="10"/>
  <Tag tag="26" value="10"/><Tag tag="25" value="0"/></Player>
<Player id="3" accountHi="0" accountLo="0" playerID="9" name="Bob" isMainPlayer="false">
  <Tag tag="50" value="9"/><Tag tag="202" value="2"/></Player>
"""


def _build(tmp_path, name, body):
    path = tmp_path / f"{name}.xml"
    path.write_text(_HEAD + body + "</Game></HSReplay>", encoding="utf-8")
    meta = {"reviewId": name, "buildNumber": 1, "mmr": 1, "placement": None,
            "finalComp": {"board": []}}
    return build_game(str(path), meta)


def _entry(card, activatable, passive=False, used=False):
    return {"card_id": card, "name": card, "cost": 1, "used": used,
            "activatable": activatable, "passive": passive}


def test_row_hero_power_picks_the_legal_power():
    # 3,950 rows: a power is legal, but the zone-first power is the other one.
    chosen = _select_row_hero_power([
        _entry("PASSIVE", False, passive=True),
        _entry("ACTIVE", True),
    ])
    assert chosen == {"card_id": "ACTIVE", "name": "ACTIVE", "cost": 1,
                      "used": False, "activatable": True}
    both = _select_row_hero_power([_entry("A", True), _entry("B", True)])
    assert both["card_id"] == "A"
    neither = _select_row_hero_power([
        _entry("PASSIVE", False, passive=True), _entry("ACTIVE", False)])
    assert neither["card_id"] == "ACTIVE"
    only = _select_row_hero_power([_entry("PASSIVE", False, passive=True)])
    assert only["card_id"] == "PASSIVE" and "passive" not in only


def _hero_offer(extra="", offered="BG36_HERO_002", played="BG36_HERO_002", played_tags=""):
    return f"""
<FullEntity id="104" cardID="{offered}"><Tag tag="50" value="1"/><Tag tag="202" value="3"/>
  <Tag tag="49" value="6"/></FullEntity>
<FullEntity id="105" cardID="BG_TEST_OTHER"><Tag tag="50" value="1"/><Tag tag="202" value="3"/>
  <Tag tag="49" value="6"/></FullEntity>
<Choices id="0" playerID="2" source="1" type="1" min="1" max="1">
  <Choice entity="104" index="0"/><Choice entity="105" index="1"/></Choices>
{extra}
<ChosenEntities playerID="2"><Choice entity="104" index="0"/></ChosenEntities>
<FullEntity id="10" cardID="{played}"><Tag tag="50" value="1"/><Tag tag="202" value="3"/>
  <Tag tag="49" value="1"/><Tag tag="45" value="30"/>{played_tags}</FullEntity>
"""


def _hero_row(res):
    rows = [r for r in res["rows"] if r["kind"] == "choice" and r["choice"]["choice_kind"] == "hero"]
    assert len(rows) == 1
    return rows[0]["choice"]


def test_f_reroll_of_the_picked_slot(tmp_path):
    # 29bc2989: slot 104 was logged as Kith'ix and rerolled to The Lich King.
    body = _hero_offer(
        extra='<ChangeEntity entity="104" cardID="BG_TEST_LICH_KING">'
              '<Tag tag="202" value="3"/></ChangeEntity>',
        played="BG_TEST_LICH_KING")
    res = _build(tmp_path, "reroll-picked", body)
    ch = _hero_row(res)
    assert [c["card_id"] for c in ch["cards"]] == ["BG_TEST_LICH_KING", "BG_TEST_OTHER"]
    assert [c["card_id"] for c in ch["offer_initial"]] == ["BG36_HERO_002", "BG_TEST_OTHER"]
    assert ch["rerolls"] == [{
        "entity_id": 104, "from_card_id": "BG36_HERO_002", "to_card_id": "BG_TEST_LICH_KING"}]
    assert "hero_pick_mismatch" not in res["failures"].count


def test_f_two_rerolls_of_the_picked_slot_keep_the_last_card(tmp_path):
    extra = (
        '<ChangeEntity entity="104" cardID="BG_TEST_MID"><Tag tag="202" value="3"/></ChangeEntity>'
        '<ChangeEntity entity="104" cardID="BG_TEST_LICH_KING"><Tag tag="202" value="3"/></ChangeEntity>'
    )
    res = _build(tmp_path, "reroll-twice", _hero_offer(extra=extra, played="BG_TEST_LICH_KING"))
    ch = _hero_row(res)
    assert ch["cards"][0]["card_id"] == "BG_TEST_LICH_KING"
    assert ch["offer_initial"][0]["card_id"] == "BG36_HERO_002"
    assert [r["to_card_id"] for r in ch["rerolls"]] == ["BG_TEST_MID", "BG_TEST_LICH_KING"]
    assert "hero_pick_mismatch" not in res["failures"].count


def test_g_reroll_of_an_unpicked_slot(tmp_path):
    extra = ('<ChangeEntity entity="105" cardID="BG_TEST_REROLLED">'
             '<Tag tag="202" value="3"/></ChangeEntity>')
    res = _build(tmp_path, "reroll-other", _hero_offer(extra=extra, played="BG36_HERO_002"))
    ch = _hero_row(res)
    assert [c["card_id"] for c in ch["cards"]] == ["BG36_HERO_002", "BG_TEST_REROLLED"]
    assert [c["card_id"] for c in ch["offer_initial"]] == ["BG36_HERO_002", "BG_TEST_OTHER"]
    assert ch["rerolls"] == [{
        "entity_id": 105, "from_card_id": "BG_TEST_OTHER", "to_card_id": "BG_TEST_REROLLED"}]
    assert "hero_pick_mismatch" not in res["failures"].count


def test_h_no_reroll_offer_initial_equals_cards(tmp_path):
    res = _build(tmp_path, "no-reroll", _hero_offer())
    ch = _hero_row(res)
    assert ch["offer_initial"] == ch["cards"]
    assert [c["card_id"] for c in ch["cards"]] == ["BG36_HERO_002", "BG_TEST_OTHER"]
    assert "rerolls" not in ch
    assert "hero_pick_mismatch" not in res["failures"].count


def test_hero_pick_mismatch_and_known_swaps(tmp_path):
    bad = _build(tmp_path, "wrong-hero", _hero_offer(played="BG_TEST_OTHER"))
    assert bad["failures"].count["hero_pick_mismatch"] == 1
    ex = bad["failures"].examples["hero_pick_mismatch"][0]
    assert ex["picked"] == "BG36_HERO_002" and ex["hero"] == "BG_TEST_OTHER"

    skin = _build(tmp_path, "skin", _hero_offer(played="BG36_HERO_002_SKIN_A"))
    assert "hero_pick_mismatch" not in skin["failures"].count

    aranna = _build(tmp_path, "aranna", _hero_offer(
        offered="TB_BaconShop_HERO_59", played="TB_BaconShop_HERO_59t"))
    assert _hero_row(aranna)["cards"][0]["card_id"] == "TB_BaconShop_HERO_59"
    assert "hero_pick_mismatch" not in aranna["failures"].count

    legacy = _build(tmp_path, "legacy-skin", _hero_offer(
        played="TB_BaconShop_HERO_44_SKIN_A",
        played_tags='<Tag tag="2038" value="1"/>'))
    assert "hero_pick_mismatch" not in legacy["failures"].count


def test_a_options_row_points_at_the_active_power(tmp_path):
    body = """
<FullEntity id="10" cardID="BG34_HERO_004"><Tag tag="50" value="1"/><Tag tag="202" value="3"/>
  <Tag tag="49" value="1"/><Tag tag="45" value="30"/></FullEntity>
<FullEntity id="12" cardID="BG34_HERO_004p"><Tag tag="50" value="1"/><Tag tag="202" value="10"/>
  <Tag tag="49" value="1"/><Tag tag="263" value="1"/><Tag tag="684" value="1"/></FullEntity>
<FullEntity id="13" cardID="TB_BaconShop_HP_102"><Tag tag="50" value="1"/><Tag tag="202" value="10"/>
  <Tag tag="49" value="1"/><Tag tag="263" value="2"/><Tag tag="48" value="1"/></FullEntity>
<Options id="1">
  <Option index="0" type="2" entity="0" error="-1"/>
  <Option index="1" type="3" entity="13" error="-1"/>
</Options>
"""
    res = _build(tmp_path, "passive-then-active", body)
    assert "hero_power_option_mismatch" not in res["failures"].count
    row = next(r for r in res["rows"] if r["kind"] == "options")
    assert row["hero_power"]["card_id"] == "TB_BaconShop_HP_102"
    assert row["hero_power"]["activatable"] is True
    assert [(p["card_id"], p["passive"], p["activatable"]) for p in row["hero_powers"]] == [
        ("BG34_HERO_004p", True, False),
        ("TB_BaconShop_HP_102", False, True),
    ]
    assert "hero_powers" not in row["snapshot"]
    assert row["snapshot"]["hero_power"]["card_id"] == "TB_BaconShop_HP_102"
    acts = [a for a in legal_actions(row["snapshot"]) if a.kind == HERO_POWER]
    assert len(acts) == 1


def test_b_options_row_both_powers_legal(tmp_path):
    body = """
<FullEntity id="10" cardID="BG35_HERO_001"><Tag tag="50" value="1"/><Tag tag="202" value="3"/>
  <Tag tag="49" value="1"/><Tag tag="45" value="30"/></FullEntity>
<FullEntity id="12" cardID="TB_BaconShop_HP_022"><Tag tag="50" value="1"/><Tag tag="202" value="10"/>
  <Tag tag="49" value="1"/><Tag tag="263" value="1"/><Tag tag="48" value="1"/></FullEntity>
<FullEntity id="13" cardID="BG28_HERO_400p"><Tag tag="50" value="1"/><Tag tag="202" value="10"/>
  <Tag tag="49" value="1"/><Tag tag="263" value="2"/><Tag tag="48" value="2"/></FullEntity>
<Options id="1">
  <Option index="0" type="2" entity="0" error="-1"/>
  <Option index="1" type="3" entity="12" error="-1"/>
  <Option index="2" type="3" entity="13" error="-1"/>
</Options>
"""
    res = _build(tmp_path, "two-legal", body)
    assert "hero_power_option_mismatch" not in res["failures"].count
    row = next(r for r in res["rows"] if r["kind"] == "options")
    assert row["hero_power"]["card_id"] == "TB_BaconShop_HP_022"
    assert row["hero_power"]["activatable"] is True
    assert [p["card_id"] for p in row["hero_powers"]] == ["TB_BaconShop_HP_022", "BG28_HERO_400p"]
    assert all(p["activatable"] for p in row["hero_powers"])
    acts = [a for a in legal_actions(row["snapshot"]) if a.kind == HERO_POWER]
    assert len(acts) == 2
    assert [a.detail["hero_power"]["card_id"] for a in acts] == [
        "TB_BaconShop_HP_022", "BG28_HERO_400p"]


def test_d_options_row_passive_only(tmp_path):
    body = """
<FullEntity id="10" cardID="BG34_HERO_004"><Tag tag="50" value="1"/><Tag tag="202" value="3"/>
  <Tag tag="49" value="1"/><Tag tag="45" value="30"/></FullEntity>
<FullEntity id="12" cardID="BG34_HERO_004p"><Tag tag="50" value="1"/><Tag tag="202" value="10"/>
  <Tag tag="49" value="1"/><Tag tag="684" value="1"/></FullEntity>
<Options id="1">
  <Option index="0" type="2" entity="0" error="-1"/>
</Options>
"""
    res = _build(tmp_path, "passive-only", body)
    assert "hero_power_option_mismatch" not in res["failures"].count
    row = next(r for r in res["rows"] if r["kind"] == "options")
    assert row["hero_power"]["card_id"] == "BG34_HERO_004p"
    assert row["hero_power"]["activatable"] is False
    assert row["hero_powers"] == [{
        "card_id": "BG34_HERO_004p",
        "name": row["hero_powers"][0]["name"],
        "cost": 0,
        "used": False,
        "activatable": False,
        "passive": True,
    }]
    assert row["snapshot"]["hero_power"] is None
    assert "hero_powers" not in row["snapshot"]


def test_hero_power_cost_is_the_live_tag_and_gold_includes_temp():
    """Spendable gold is RESOURCES - RESOURCES_USED + TEMP_RESOURCES, and the
    cost legal_actions checks is the power's current COST tag. A discount
    that sets COST to 0 is free at 0 gold; a positive COST above gold is not
    offered."""
    t = _tracker(gold="1")
    t.state.entities[2].tags["RESOURCES_USED"] = "1"
    t.state.entities[2].tags["TEMP_RESOURCES"] = "2"
    t.state.entities[12] = _power(12, "TB_BaconShop_HP_022", COST=2)
    assert t._gold() == 2
    snap = t.snapshot()
    assert snap.gold == 2 and snap.hero_power["cost"] == 2 and snap.hero_power["usable"]
    assert any(a.kind == HERO_POWER for a in legal_actions(snap))
    # The log changes COST (a discount, or a free power whose 0 was omitted).
    t.state.entities[12].tags["COST"] = "0"
    t.state.entities[2].tags["TEMP_RESOURCES"] = "0"
    assert t._gold() == 0
    snap = t.snapshot()
    assert snap.hero_power["cost"] == 0 and snap.hero_power["usable"]
    acts = [a for a in legal_actions(snap) if a.kind == HERO_POWER]
    assert len(acts) == 1 and acts[0].cost == 0
    t.state.entities[12].tags["COST"] = "1"
    snap = t.snapshot()
    assert snap.gold == 0 and snap.hero_power["cost"] == 1
    assert snap.hero_power["usable"] is False
    assert not any(a.kind == HERO_POWER for a in legal_actions(snap))
