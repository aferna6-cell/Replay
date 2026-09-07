import hashlib

from ml.phase_3u_powerlog_state_recovery import audit_powerlog_state_recovery


def _line(payload: str) -> bytes:
    return f"D 12:00:00 GameState.DebugPrintPower() - {payload}\n".encode()


def test_inline_complete_descriptor_is_recoverable_but_never_admitted():
    source = _line(
        "TAG_CHANGE Entity=[entityName=Test id=25 zone=PLAY zonePos=2 cardId=BG_TEST player=1] "
        "tag=ZONE value=HAND"
    )
    result = audit_powerlog_state_recovery(source)

    assert result["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert result["zone_changes"] == 1
    assert result["zone_changes_with_recoverable_pre_state"] == 1
    assert result["zone_pre_state_recovery_coverage"] == 1.0
    assert result["ranking_ready"] is False
    assert result["candidate_scoring_performed"] is False


def test_numeric_zone_change_can_recover_from_prior_observed_descriptor_and_position():
    source = b"".join(
        [
            _line(
                "TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=1 cardId=BG_TEST player=1] "
                "tag=ATK value=5"
            ),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["numeric_zone_changes"] == 1
    assert result["numeric_zone_changes_with_recoverable_pre_state"] == 1
    assert result["numeric_zone_pre_state_recovery_coverage"] == 1.0


def test_zone_change_invalidates_position_until_position_is_observed_again():
    source = b"".join(
        [
            _line(
                "TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=1 cardId=BG_TEST player=1] "
                "tag=ZONE value=PLAY"
            ),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
            _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=3"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["zone_changes"] == 3
    assert result["zone_changes_with_recoverable_pre_state"] == 2
    assert result["numeric_zone_changes_with_recoverable_pre_state"] == 1
    assert result["zone_changes_without_recoverable_pre_state"] == 1


def test_empty_card_id_is_not_claimed_as_body_pre_state():
    source = _line(
        "TAG_CHANGE Entity=[entityName=Player id=2 zone=PLAY zonePos=0 cardId= player=1] "
        "tag=ZONE value=SETASIDE"
    )
    result = audit_powerlog_state_recovery(source)

    assert result["zone_changes"] == 1
    assert result["zone_changes_with_recoverable_pre_state"] == 0
    assert result["zone_pre_state_recovery_coverage"] == 0.0


def test_noncanonical_mirror_does_not_contribute_state():
    source = (
        b"D 12:00:00 PowerTaskList.DebugPrintPower() - "
        b"TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=1 cardId=BG_TEST player=1] "
        b"tag=ATK value=5\n"
        + _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY")
    )
    result = audit_powerlog_state_recovery(source)

    assert result["zone_changes"] == 1
    assert result["zone_changes_with_recoverable_pre_state"] == 0
    assert result["numeric_zone_pre_state_recovery_coverage"] == 0.0
