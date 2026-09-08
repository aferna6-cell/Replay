import hashlib

from ml.phase_3u_powerlog_play_membership import audit_powerlog_play_membership


def _line(payload: str) -> bytes:
    return f"D 12:00:00 GameState.DebugPrintPower() - {payload}\n".encode()


def test_numeric_play_membership_is_direct_even_without_position():
    source = b"".join([
        _line("Player EntityID=1 PlayerID=1"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=PLAY zonePos=2 cardId=BG_TEST player=1] tag=ATK value=5"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
    ])
    result = audit_powerlog_play_membership(source)
    assert result["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert result["play_membership_intervals"] == 1
    assert result["tag_change_membership_intervals"] == 1
    assert result["identity_grounded"] == 1
    assert result["explicit_position_grounded"] == 1
    row = result["per_game"][0]["intervals"][0]
    assert row["closed_by_zone"] is True
    assert row["exit_zone"] == "GRAVEYARD"
    assert row["identity_grounded"] is True
    assert result["phase_3u_schema_ready"] is False
    assert result["ranking_ready"] is False


def test_identity_can_be_grounded_while_position_remains_unknown():
    source = b"".join([
        _line("Player EntityID=1 PlayerID=1"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=CONTROLLER value=1"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=HEALTH value=8"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND"),
    ])
    result = audit_powerlog_play_membership(source)
    assert result["identity_grounded"] == 1
    assert result["explicit_position_grounded"] == 0
    assert result["identity_grounded_without_position"] == 1
    row = result["per_game"][0]["intervals"][0]
    assert row["card_id_known_at_entry"] == "BG_TEST"
    assert row["player_known_at_entry"] == 1
    assert row["position_grounded"] is False


def test_full_entity_zone_play_is_distinct_direct_membership_source():
    source = b"".join([
        _line("Player EntityID=1 PlayerID=1"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=CONTROLLER value=1"),
        _line("    tag=ZONE value=PLAY"),
        _line("    tag=ZONE_POSITION value=4"),
    ])
    result = audit_powerlog_play_membership(source)
    assert result["play_membership_intervals"] == 1
    assert result["full_entity_tag_membership_intervals"] == 1
    row = result["per_game"][0]["intervals"][0]
    assert row["membership_source"] == "full_entity_tag"
    assert row["identity_grounded"] is True
    assert row["explicit_position"] == 4
    assert row["position_source"] == "full_entity_tag"


def test_next_zone_closes_membership_before_late_identity_or_position():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=GRAVEYARD zonePos=0 cardId=BG_TEST player=1] tag=ATK value=5"),
    ])
    result = audit_powerlog_play_membership(source)
    row = result["per_game"][0]["intervals"][0]
    assert row["closed_by_zone"] is True
    assert row["card_id_grounded"] is False
    assert row["position_grounded"] is False


def test_create_game_boundary_prevents_entity_identity_leakage():
    source = b"".join([
        _line("CREATE_GAME"),
        _line("Player EntityID=1 PlayerID=1"),
        _line("FULL_ENTITY - Creating ID=25 CardID=OLD"),
        _line("    tag=CONTROLLER value=1"),
        _line("CREATE_GAME"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
    ])
    result = audit_powerlog_play_membership(source)
    assert result["game_segments"] == 2
    row = result["per_game"][1]["intervals"][0]
    assert row["card_id_grounded"] is False
    assert row["player_grounded"] is False


def test_unvalidated_controller_does_not_ground_player_identity():
    source = b"".join([
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=CONTROLLER value=7"),
        _line("    tag=ZONE value=PLAY"),
    ])
    result = audit_powerlog_play_membership(source)
    row = result["per_game"][0]["intervals"][0]
    assert row["card_id_grounded"] is True
    assert row["player_grounded"] is False
    assert row["identity_grounded"] is False


def test_noncanonical_mirror_cannot_create_membership_or_identity():
    source = (
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND")
        + b"D 12:00:00 PowerTaskList.DebugPrintPower() - TAG_CHANGE Entity=[entityName=Test id=25 zone=PLAY zonePos=2 cardId=BG_TEST player=1] tag=ZONE value=PLAY\n"
    )
    result = audit_powerlog_play_membership(source)
    assert result["play_membership_intervals"] == 0
    assert result["identity_grounded"] == 0
