"""HSReplay XML adapter + states.v1 builder, on a hand-written synthetic replay.

Optional real-replay test: set HSBG_REPLAY_XML=<path to a Firestone .xml.gz>
(and optionally HSBG_REPLAY_MANIFEST=<manifest json> to also require every
fidelity check to pass).
"""

import json
import os

import pytest

from hsbg_coach.bg import BGTracker, Snapshot
from hsbg_coach.hsreplay_xml import iter_events
from hsbg_coach.replay_states import build_game

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
                                "targets": [20]}
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
    rows = res["rows"]
    assert dict(res["failures"].count) == {}
    assert res["stats"]["options_blocks"] == 3 and res["stats"]["options_resent"] == 1
    assert [r["dp_index"] for r in rows] == [0, 1]
    assert [(r["raw_turn"], r["turn"]) for r in rows] == [(1, 1), (3, 2)]
    first, last = rows
    assert first["schema_version"] == "states.v1" and first["lobby_tribes"] == ["MURLOC"]
    assert [o["entity_id"] for o in first["options"]] == [0, 30, 12]   # legal only
    assert first["options"][2] == {"index": 2, "type": "POWER", "entity_id": 12,
                                   "card_id": "BG36_HERO_002p", "zone": "PLAY",
                                   "targets": [20]}
    assert first["hero_power"]["activatable"] is True
    assert first["dark_discovery"] == {"available": False}
    assert first["snapshot"]["board"][1]["dark_gift"] == {
        "card_id": "BG36_MidGameEffect_000t51", "name": "Steady Growth"}
    # Re-sent Options id 2: the later block (two options) is the one kept.
    assert [o["entity_id"] for o in last["options"]] == [0, 60]
    assert res["stats"]["dark_discovery_picks"] == 1
    assert res["stats"]["dark_discovery_picks_verified"] == 1
    assert res["warnings"] == []
    json.dumps(rows)                      # rows are JSON-serializable


def test_build_game_flags_final_board_and_placement_mismatch():
    meta = dict(META, placement=4, finalComp={"board": [{"cardId": "OTHER", "golden": False}]})
    res = build_game(FIXTURE, meta)
    assert set(res["failures"].count) == {"final_board_mismatch", "placement_mismatch"}


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
