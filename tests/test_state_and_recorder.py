"""Integration-ish tests: events -> GameState, and recorder round-trip."""

import json
import os
import tempfile

from hsbg_coach.parser import parse_line
from hsbg_coach.state import GameState
from hsbg_coach.bg import BGTracker, Snapshot, ActionType
from hsbg_coach.recorder import TrajectoryRecorder


def _feed(state: GameState, lines):
    for line in lines:
        ev = parse_line(line)
        if ev:
            state.apply(ev)


def test_gamestate_tracks_entity_and_turn():
    state = GameState()
    _feed(state, [
        "D 09:25:43.1 GameState.DebugPrintPower() - CREATE_GAME",
        "D 09:25:43.1 GameState.DebugPrintPower() - FULL_ENTITY - Updating "
        "[entityName=Foo id=27 zone=PLAY zonePos=1 cardId=X player=1] CardID=X",
        "D 09:25:43.1 GameState.DebugPrintPower() - TAG_CHANGE "
        "Entity=[entityName=Foo id=27 zone=PLAY zonePos=1 cardId=X player=1] "
        "tag=ATK value=4",
        "D 09:25:43.1 GameState.DebugPrintPower() - TAG_CHANGE Entity=1 "
        "tag=TURN value=3",
    ])
    assert state.game_counter == 1
    assert state.current_turn == 3
    ent = state.entities[27]
    assert ent.tag_int("ATK") == 4
    assert ent.zone == "PLAY"


def test_bgtracker_detects_bacon_scene():
    t = BGTracker()
    for line in [
        "D 09:25:43.1 LoadingScreen.OnSceneLoaded() - prevMode=HUB currMode=BACON",
        "D 09:25:43.1 GameState.DebugPrintPower() - CREATE_GAME",
    ]:
        ev = parse_line(line)
        if ev:
            t.feed(ev)
    assert t.in_bg is True


def test_recorder_round_trip(tmp_path=None):
    d = tmp_path or tempfile.mkdtemp()
    rec = TrajectoryRecorder(str(d))
    rec.start_game("testgame")
    snap = Snapshot(game_counter=1, turn=4, phase="recruit", tavern_tier=2,
                    gold=8, hero_health=30)
    rec.record(snap, ActionType.BUY, {"card_id": "BGS_039"})
    rec.record(snap, ActionType.END_TURN)
    path = rec.finish_game(placement=1)

    assert path.endswith("game-testgame.jsonl")
    assert os.path.isfile(path)
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 2
    assert rows[0]["action_type"] == "buy"
    assert rows[0]["action_detail"]["card_id"] == "BGS_039"
    assert all(r["placement"] == 1 for r in rows)  # outcome backfilled to all decisions
