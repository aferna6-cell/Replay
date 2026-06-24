"""Unit tests for the raw log-line parser.

These run with no Hearthstone install and no real log — they pin the *stable*
core (text -> events). The BG semantic layer is calibrated separately against a
captured real log.
"""

from hsbg_coach.parser import parse_line


def test_non_log_line_returns_none():
    assert parse_line("this is not a log line") is None
    assert parse_line("") is None


def test_create_game():
    ev = parse_line("D 09:25:43.123 GameState.DebugPrintPower() - CREATE_GAME")
    assert ev is not None
    assert ev.kind == "CREATE_GAME"
    assert ev.logger == "GameState"
    assert ev.method == "DebugPrintPower"


def test_scene_load_detects_fields():
    ev = parse_line(
        "D 09:25:43.123 LoadingScreen.OnSceneLoaded() - prevMode=HUB currMode=BACON"
    )
    assert ev.kind == "SCENE"
    assert ev.fields["currMode"] == "BACON"
    assert ev.fields["prevMode"] == "HUB"


def test_full_entity_bracket_and_cardid():
    line = ("D 09:25:43.123 GameState.DebugPrintPower() - FULL_ENTITY - Updating "
            "[entityName=Murloc Tidecaller id=27 zone=PLAY zonePos=2 cardId=EX1_509 "
            "player=1] CardID=EX1_509")
    ev = parse_line(line)
    assert ev.kind == "FULL_ENTITY"
    assert ev.entity.id == 27
    assert ev.entity.name == "Murloc Tidecaller"  # space-containing name survives
    assert ev.entity.zone == "PLAY"
    assert ev.entity.zone_pos == 2
    assert ev.entity.player == 1
    assert ev.entity.card_id == "EX1_509"


def test_tag_change_with_bracket_entity():
    line = ("D 09:25:43.123 GameState.DebugPrintPower() - TAG_CHANGE "
            "Entity=[entityName=Foo id=27 zone=PLAY zonePos=2 cardId=X player=1] "
            "tag=ZONE value=GRAVEYARD")
    ev = parse_line(line)
    assert ev.kind == "TAG_CHANGE"
    assert ev.tag == "ZONE"
    assert ev.value == "GRAVEYARD"
    assert ev.entity.id == 27


def test_indented_tag_line():
    ev = parse_line("D 09:25:43.123 GameState.DebugPrintPower() -     tag=TURN value=5")
    assert ev.kind == "TAG"
    assert ev.tag == "TURN"
    assert ev.value == "5"
    assert ev.indent == 4


def test_tag_change_bare_entity_id():
    line = ("D 09:25:43.123 GameState.DebugPrintPower() - TAG_CHANGE Entity=27 "
            "tag=DAMAGE value=3")
    ev = parse_line(line)
    assert ev.kind == "TAG_CHANGE"
    assert ev.entity.id == 27
    assert ev.tag == "DAMAGE"
    assert ev.value == "3"
