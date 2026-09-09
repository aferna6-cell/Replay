from ml.phase_3u_powerlog_position_census import audit_powerlog_position_census


def _line(payload: str) -> bytes:
    return f"D 12:00:00 GameState.DebugPrintPower() - {payload}\n".encode()


def test_literal_minion_with_explicit_position_is_counted_without_reconstruction():
    source = b"".join([
        _line("Player EntityID=1 PlayerID=1"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=CONTROLLER value=1"),
        _line("    tag=CARDTYPE value=MINION"),
        _line("    tag=ZONE value=PLAY"),
        _line("    tag=ZONE_POSITION value=3"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
    ])
    result = audit_powerlog_position_census(source)
    assert result["explicit_minion_grounded"] == 1
    assert result["explicit_minion_position_grounded"] == 1
    assert result["explicit_minion_position_forward"] == 1
    assert result["board_order_reconstructed"] is False
    assert result["ranking_ready"] is False


def test_descriptor_position_on_zone_mutation_is_not_promoted():
    source = b"".join([
        _line("Player EntityID=1 PlayerID=1"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=CONTROLLER value=1"),
        _line("    tag=CARDTYPE value=MINION"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=4 cardId=BG_TEST player=1] tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
    ])
    result = audit_powerlog_position_census(source)
    assert result["explicit_minion_grounded"] == 1
    assert result["explicit_minion_position_grounded"] == 0
    assert result["explicit_minion_without_position"] == 1
