import hashlib

from ml.phase_3u_powerlog_entity_type import audit_powerlog_entity_type


def _line(payload: str) -> bytes:
    return f"D 12:00:00 GameState.DebugPrintPower() - {payload}\n".encode()


def test_full_entity_cardtype_minion_before_play_is_grounded():
    source = b"".join([
        _line("Player EntityID=1 PlayerID=1"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=CARDTYPE value=MINION"),
        _line("    tag=CONTROLLER value=1"),
        _line("    tag=ZONE value=PLAY"),
    ])
    result = audit_powerlog_entity_type(source)
    assert result["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert result["play_membership_intervals"] == 1
    assert result["cardtype_grounded"] == 1
    assert result["explicit_minion_grounded"] == 1
    assert result["identity_and_explicit_minion_grounded"] == 1
    row = result["per_game"][0]["intervals"][0]
    assert row["cardtype_known_at_entry"] == "MINION"
    assert row["cardtype_source"] == "full_entity_tag"
    assert row["retrospective_cardtype_candidate"] is None
    assert result["board_set_reconstructed"] is False
    assert result["phase_3u_schema_ready"] is False


def test_forward_cardtype_before_next_zone_is_allowed_but_later_type_is_retrospective_only():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=CARDTYPE value=MINION"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
        _line("TAG_CHANGE Entity=26 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=26 tag=ZONE value=HAND"),
        _line("TAG_CHANGE Entity=26 tag=CARDTYPE value=MINION"),
    ])
    result = audit_powerlog_entity_type(source)
    first, second = result["per_game"][0]["intervals"]
    assert first["forward_cardtype"] == "MINION"
    assert first["explicit_minion_grounded"] is True
    assert second["cardtype_grounded"] is False
    assert second["post_exit_cardtype"] == "MINION"
    assert second["retrospective_cardtype_candidate"] == "MINION"
    assert second["retrospective_candidate_is_causal_grounding"] is False
    assert result["explicit_minion_grounded"] == 1
    assert result["retrospective_explicit_minion_candidates"] == 1


def test_retrospective_candidate_requires_same_game_type_consistency():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND"),
        _line("TAG_CHANGE Entity=25 tag=CARDTYPE value=MINION"),
        _line("TAG_CHANGE Entity=25 tag=CARDTYPE value=SPELL"),
    ])
    result = audit_powerlog_entity_type(source)
    row = result["per_game"][0]["intervals"][0]
    assert row["cardtype_grounded"] is False
    assert row["same_game_cardtype_distinct_values"] == ["MINION", "SPELL"]
    assert row["same_game_cardtype_consistent"] is False
    assert row["retrospective_cardtype_candidate"] is None
    assert result["same_game_type_conflict_intervals"] == 1
    assert result["retrospective_cardtype_candidates"] == 0


def test_numeric_cardtype_is_reported_but_not_interpreted_as_minion():
    source = b"".join([
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=CARDTYPE value=4"),
        _line("    tag=ZONE value=PLAY"),
    ])
    result = audit_powerlog_entity_type(source)
    row = result["per_game"][0]["intervals"][0]
    assert row["cardtype_grounded"] is True
    assert row["cardtype_value"] == "4"
    assert row["explicit_minion_grounded"] is False
    assert result["cardtype_value_counts"] == {"4": 1}


def test_create_game_boundary_prevents_type_leakage_and_retrospective_repair():
    source = b"".join([
        _line("CREATE_GAME"),
        _line("FULL_ENTITY - Creating ID=25 CardID=OLD"),
        _line("    tag=CARDTYPE value=MINION"),
        _line("CREATE_GAME"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("CREATE_GAME"),
        _line("TAG_CHANGE Entity=25 tag=CARDTYPE value=MINION"),
    ])
    result = audit_powerlog_entity_type(source)
    assert result["game_segments"] == 3
    row = result["per_game"][1]["intervals"][0]
    assert row["cardtype_grounded"] is False
    assert row["retrospective_cardtype_candidate"] is None


def test_noncanonical_mirror_cardtype_is_ignored():
    source = (
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY")
        + b"D 12:00:00 PowerTaskList.DebugPrintPower() - TAG_CHANGE Entity=25 tag=CARDTYPE value=MINION\n"
    )
    result = audit_powerlog_entity_type(source)
    assert result["cardtype_grounded"] == 0
    assert result["explicit_minion_grounded"] == 0
    assert result["retrospective_cardtype_candidates"] == 0
