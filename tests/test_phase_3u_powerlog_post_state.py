import hashlib

from ml.phase_3u_powerlog_post_state import audit_powerlog_post_state


def _line(payload: str) -> bytes:
    return f"D 12:00:00 GameState.DebugPrintPower() - {payload}\n".encode()


def test_forward_zone_position_is_post_state_only_and_never_admits():
    source = b"".join(
        [
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
            _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=3"),
        ]
    )
    result = audit_powerlog_post_state(source)

    assert result["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert result["unresolved_numeric_zone_events"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 1
    assert result["post_zone_position_coverage"] == 1.0
    assert result["per_game"][0]["intervals"][0]["post_zone_position_distance"] == 1
    assert result["pre_state_repaired_from_future_events"] is False
    assert result["phase_3u_schema_ready"] is False
    assert result["ranking_ready"] is False


def test_next_zone_closes_interval_before_late_position():
    source = b"".join(
        [
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
            _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=0"),
        ]
    )
    result = audit_powerlog_post_state(source)

    assert result["unresolved_numeric_zone_events"] == 2
    first, second = result["per_game"][0]["intervals"]
    assert first["closed_by_next_zone"] is True
    assert first["post_zone_position_observed"] is False
    assert first["next_zone_distance"] == 1
    assert second["post_zone_position_observed"] is True
    assert result["closed_by_next_zone_without_post_position"] == 1


def test_previously_missing_card_id_can_be_observed_forward_without_repairing_pre_state():
    source = b"".join(
        [
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
            _line(
                "TAG_CHANGE Entity=[entityName=Test id=25 zone=PLAY zonePos=2 cardId=BG_TEST player=1] "
                "tag=ATK value=5"
            ),
        ]
    )
    result = audit_powerlog_post_state(source)

    assert result["missing_card_id_pre_events"] == 1
    assert result["missing_card_id_pre_with_forward_card_id"] == 1
    assert result["forward_card_id_coverage_for_missing_pre_card"] == 1.0
    event = result["per_game"][0]["intervals"][0]
    assert event["card_id_observed_after_distance"] == 1
    assert "card_id" in event["missing_pre_state_fields"]


def test_recoverable_numeric_pre_state_is_not_part_of_unresolved_forward_cohort():
    source = b"".join(
        [
            _line(
                "TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=1 cardId=BG_TEST player=1] "
                "tag=ATK value=5"
            ),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
            _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=2"),
        ]
    )
    result = audit_powerlog_post_state(source)

    assert result["unresolved_numeric_zone_events"] == 0
    assert result["post_zone_position_coverage"] is None
    assert result["per_game"][0]["intervals"] == []


def test_pending_intervals_do_not_cross_create_game_boundary():
    source = b"".join(
        [
            _line("CREATE_GAME"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
            _line("CREATE_GAME"),
            _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=4"),
        ]
    )
    result = audit_powerlog_post_state(source)

    assert result["game_segments"] == 2
    assert result["unresolved_numeric_zone_events"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 0
    assert result["per_game"][0]["intervals"][0]["post_zone_position_observed"] is False


def test_noncanonical_mirror_cannot_supply_forward_position():
    source = (
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY")
        + b"D 12:00:00 PowerTaskList.DebugPrintPower() - TAG_CHANGE Entity=25 tag=ZONE_POSITION value=3\n"
    )
    result = audit_powerlog_post_state(source)

    assert result["unresolved_numeric_zone_events"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 0
