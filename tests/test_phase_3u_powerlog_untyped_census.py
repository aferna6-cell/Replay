from ml.phase_3u_powerlog_untyped_census import audit_powerlog_untyped_census


def _log(*payloads: str) -> bytes:
    return "\n".join(f"D GameState.DebugPrintPower() - {p}" for p in payloads).encode()


def test_census_finds_show_change_records_without_promoting_type():
    result = audit_powerlog_untyped_census(_log(
        "CREATE_GAME",
        "Player EntityID=1 PlayerID=1",
        "TAG_CHANGE Entity=[entityName=x id=10 zone=HAND zonePos=1 cardId=X player=1] tag=ZONE value=PLAY",
        "SHOW_ENTITY - Updating Entity=[entityName=x id=10 zone=PLAY zonePos=1 cardId=X player=1] CardID=X",
        "CHANGE_ENTITY - Updating Entity=[entityName=x id=10 zone=PLAY zonePos=1 cardId=X player=1] CardID=Y",
        "TAG_CHANGE Entity=10 tag=ZONE value=GRAVEYARD",
    ))
    assert result["play_membership_intervals"] == 1
    assert result["causally_untyped_intervals"] == 1
    assert result["untyped_entities_with_show_entity"] == 1
    assert result["untyped_entities_with_change_entity"] == 1
    assert result["untyped_intervals_with_causal_show_entity_cardtype"] == 0
    assert result["cardtype_inference_performed"] is False


def test_show_entity_raw_cardtype_is_measured_as_literal_omitted_signal_only():
    result = audit_powerlog_untyped_census(_log(
        "CREATE_GAME",
        "Player EntityID=1 PlayerID=1",
        "SHOW_ENTITY - Updating Entity=10 CardID=X",
        "    tag=CARDTYPE value=MINION",
        "TAG_CHANGE Entity=[entityName=x id=10 zone=HAND zonePos=1 cardId=X player=1] tag=ZONE value=PLAY",
        "TAG_CHANGE Entity=10 tag=ZONE value=GRAVEYARD",
    ))
    assert result["causally_untyped_intervals"] == 1
    assert result["untyped_intervals_with_causal_show_entity_cardtype"] == 1
    assert result["untyped_intervals_with_causal_show_entity_explicit_minion"] == 1
    assert result["causal_show_entity_cardtype_value_counts"] == {"MINION": 1}
    assert result["show_entity_cardtype_is_source_literal_only"] is True
    assert result["cardtype_inference_performed"] is False


def test_literal_cardtype_inside_interval_removes_row_from_untyped_census():
    result = audit_powerlog_untyped_census(_log(
        "CREATE_GAME",
        "Player EntityID=1 PlayerID=1",
        "TAG_CHANGE Entity=[entityName=x id=10 zone=HAND zonePos=1 cardId=X player=1] tag=ZONE value=PLAY",
        "TAG_CHANGE Entity=10 tag=CARDTYPE value=MINION",
        "TAG_CHANGE Entity=10 tag=ZONE value=GRAVEYARD",
    ))
    assert result["play_membership_intervals"] == 1
    assert result["causally_untyped_intervals"] == 0
    assert result["intervals"] == []
