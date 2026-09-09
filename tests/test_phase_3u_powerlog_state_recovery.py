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


def test_full_entity_block_can_ground_numeric_zone_pre_state():
    source = b"".join(
        [
            _line("Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=2]"),
            _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
            _line("    tag=ENTITY_ID value=25"),
            _line("    tag=CONTROLLER value=1"),
            _line("    tag=ZONE value=HAND"),
            _line("    tag=ZONE_POSITION value=3"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["probe_version"] == "3u_powerlog_state_recovery_v4"
    assert result["player_id_collection_passes_per_game"] == 2
    assert result["player_id_validation_order_invariant"] is True
    assert result["player_id_validation_game_local"] is True
    assert result["entity_state_reset_at_game_boundary"] is True
    assert result["observed_player_ids"] == [1]
    assert result["full_entity_records"] == 1
    assert result["full_entity_records_with_card_id"] == 1
    assert result["full_entity_entities_with_complete_state"] == 1
    assert result["numeric_zone_changes"] == 1
    assert result["numeric_zone_changes_recovered_from_prior_state"] == 1
    assert result["numeric_zone_pre_state_recovery_coverage"] == 1.0
    assert result["phase_3u_schema_ready"] is False
    assert result["ranking_ready"] is False


def test_full_entity_controller_can_reference_player_record_later_in_same_game():
    source = b"".join(
        [
            _line("CREATE_GAME"),
            _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
            _line("    tag=CONTROLLER value=1"),
            _line("    tag=ZONE value=HAND"),
            _line("    tag=ZONE_POSITION value=3"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
            _line("Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=2]"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["game_segments"] == 1
    assert result["observed_player_ids"] == [1]
    assert result["full_entity_entities_with_complete_state"] == 1
    assert result["numeric_zone_changes_with_recoverable_pre_state"] == 1


def test_player_id_from_prior_game_cannot_validate_reused_controller():
    source = b"".join(
        [
            _line("CREATE_GAME"),
            _line("Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=2]"),
            _line("CREATE_GAME"),
            _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
            _line("    tag=CONTROLLER value=1"),
            _line("    tag=ZONE value=HAND"),
            _line("    tag=ZONE_POSITION value=3"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["game_segments"] == 2
    assert result["per_game"][0]["observed_player_ids"] == [1]
    assert result["per_game"][1]["observed_player_ids"] == []
    assert result["numeric_zone_changes"] == 1
    assert result["numeric_zone_changes_with_recoverable_pre_state"] == 0
    assert result["numeric_unrecoverable_missing_player"] == 1
    assert result["numeric_unrecoverable_missing_field_sets"] == {"player": 1}


def test_entity_state_does_not_cross_create_game_boundary():
    source = b"".join(
        [
            _line("CREATE_GAME"),
            _line(
                "TAG_CHANGE Entity=[entityName=Old id=25 zone=HAND zonePos=1 cardId=BG_OLD player=1] "
                "tag=ATK value=5"
            ),
            _line("CREATE_GAME"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["game_segments"] == 2
    assert result["numeric_zone_changes"] == 1
    assert result["numeric_zone_changes_with_recoverable_pre_state"] == 0
    assert result["numeric_unrecoverable_missing_field_sets"] == {
        "zone+zone_pos+card_id+player": 1
    }


def test_full_entity_controller_requires_observed_player_id_in_same_game():
    source = b"".join(
        [
            _line("CREATE_GAME"),
            _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
            _line("    tag=CONTROLLER value=1"),
            _line("    tag=ZONE value=HAND"),
            _line("    tag=ZONE_POSITION value=3"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["observed_player_ids"] == []
    assert result["full_entity_entities_with_complete_state"] == 0
    assert result["numeric_zone_changes_with_recoverable_pre_state"] == 0
    assert result["numeric_unrecoverable_missing_player"] == 1
    assert result["numeric_unrecoverable_missing_field_sets"] == {"player": 1}


def test_missing_field_classification_reports_each_unobserved_requirement():
    source = _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY")
    result = audit_powerlog_state_recovery(source)

    assert result["numeric_zone_changes_without_recoverable_pre_state"] == 1
    assert result["numeric_unrecoverable_missing_zone"] == 1
    assert result["numeric_unrecoverable_missing_zone_pos"] == 1
    assert result["numeric_unrecoverable_missing_card_id"] == 1
    assert result["numeric_unrecoverable_missing_player"] == 1
    assert result["numeric_unrecoverable_missing_field_sets"] == {
        "zone+zone_pos+card_id+player": 1
    }


def test_full_entity_context_ends_at_first_non_tag_record():
    source = b"".join(
        [
            _line("Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=2]"),
            _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
            _line("    tag=CONTROLLER value=1"),
            _line("BLOCK_START BlockType=TRIGGER Entity=2 EffectCardId= EffectIndex=0 Target=0"),
            _line("    tag=ZONE value=HAND"),
            _line("    tag=ZONE_POSITION value=3"),
            _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        ]
    )
    result = audit_powerlog_state_recovery(source)

    assert result["full_entity_state_tags_applied"] == 1
    assert result["full_entity_entities_with_complete_state"] == 0
    assert result["numeric_zone_changes_with_recoverable_pre_state"] == 0


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
    assert result["numeric_unrecoverable_missing_zone_pos"] == 1


def test_empty_card_id_is_not_claimed_as_body_pre_state():
    source = _line(
        "TAG_CHANGE Entity=[entityName=Player id=2 zone=PLAY zonePos=0 cardId= player=1] "
        "tag=ZONE value=SETASIDE"
    )
    result = audit_powerlog_state_recovery(source)

    assert result["zone_changes"] == 1
    assert result["zone_changes_with_recoverable_pre_state"] == 0
    assert result["zone_pre_state_recovery_coverage"] == 0.0


def test_noncanonical_mirror_does_not_contribute_state_or_player_identity():
    source = (
        b"D 12:00:00 PowerTaskList.DebugPrintPower() - "
        b"Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=2]\n"
        + _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST")
        + _line("    tag=CONTROLLER value=1")
        + _line("    tag=ZONE value=HAND")
        + _line("    tag=ZONE_POSITION value=1")
        + _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY")
    )
    result = audit_powerlog_state_recovery(source)

    assert result["observed_player_ids"] == []
    assert result["zone_changes"] == 1
    assert result["zone_changes_with_recoverable_pre_state"] == 0
    assert result["numeric_zone_pre_state_recovery_coverage"] == 0.0
