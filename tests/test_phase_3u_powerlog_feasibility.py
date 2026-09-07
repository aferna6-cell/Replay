import hashlib
import json

import pytest

from ml.phase_3u_powerlog_feasibility import (
    audit_powerlog_file,
    audit_powerlog_observability,
    main,
)


SAMPLE = b"""D 16:42:13 GameState.DebugPrintPower() - CREATE_GAME
D 16:42:13 GameState.DebugPrintPower() -     Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=2]
D 16:42:13 GameState.DebugPrintPower() -     FULL_ENTITY - Creating ID=25 CardID=BG_TEST
D 16:42:13 GameState.DebugPrintPower() -         tag=ENTITY_ID value=25
D 16:42:13 GameState.DebugPrintPower() -         tag=ATK value=7
D 16:42:13 GameState.DebugPrintPower() -         tag=HEALTH value=8
D 16:42:13 GameState.DebugPrintPower() -         tag=ZONE value=PLAY
D 16:42:14 GameState.DebugPrintPower() -     TAG_CHANGE Entity=25 tag=ATK value=9
D 16:42:14 GameState.DebugPrintPower() -     TAG_CHANGE Entity=25 tag=HEALTH value=10
D 16:42:14 GameState.DebugPrintPower() -     TAG_CHANGE Entity=25 tag=ZONE value=HAND
"""


def test_concrete_powerlog_primitives_are_detected_but_never_admitted():
    result = audit_powerlog_observability(SAMPLE)

    assert result["source_sha256"] == hashlib.sha256(SAMPLE).hexdigest()
    assert result["source_bytes"] == len(SAMPLE)
    assert result["canonical_stream"] == "GameState.DebugPrintPower"
    assert result["stable_identity_primitives_observed"] is True
    assert result["per_body_stat_primitives_observed"] is True
    assert result["ordered_change_primitives_observed"] is True
    assert result["membership_change_primitives_observed"] is True
    assert result["raw_observability_candidate"] is True
    assert result["phase_3u_schema_ready"] is False
    assert result["ranking_ready"] is False
    assert result["candidate_scoring_performed"] is False
    assert "conserved_pool_reconstruction_not_defined" in result["blockers"]
    assert "digest_bound_parser_loader_not_implemented" in result["blockers"]


def test_duplicate_powertasklist_mirror_is_ignored():
    mirror = SAMPLE.replace(b"GameState.DebugPrintPower()", b"PowerTaskList.DebugPrintPower()")
    result = audit_powerlog_observability(SAMPLE + mirror)
    baseline = audit_powerlog_observability(SAMPLE)

    for key in (
        "create_game_count",
        "player_records",
        "full_entity_records",
        "entity_id_tags",
        "attack_tags",
        "health_tags",
        "zone_tags",
        "tag_changes",
        "attack_changes",
        "health_changes",
        "zone_changes",
        "changed_entity_count",
    ):
        assert result[key] == baseline[key]
    assert result["ignored_noncanonical_power_lines"] == len(SAMPLE.splitlines())


def test_bracketed_entity_descriptor_normalizes_to_numeric_id():
    source = SAMPLE.replace(
        b"TAG_CHANGE Entity=25 tag=ZONE value=HAND",
        b"TAG_CHANGE Entity=[entityName=Test Minion id=25 zone=PLAY zonePos=1 cardId=BG_TEST player=1] tag=ZONE value=HAND",
    )
    result = audit_powerlog_observability(source)

    assert result["zone_changes"] == 1
    assert result["changed_entity_count"] == 1
    assert result["membership_change_primitives_observed"] is True


def test_missing_membership_change_primitives_fail_observability():
    source = SAMPLE.replace(
        b"D 16:42:14 GameState.DebugPrintPower() -     TAG_CHANGE Entity=25 tag=ZONE value=HAND\n",
        b"",
    )
    result = audit_powerlog_observability(source)

    assert result["membership_change_primitives_observed"] is False
    assert result["raw_observability_candidate"] is False
    assert result["blockers"][0] == "required_powerlog_primitives_missing"
    assert result["ranking_ready"] is False


def test_noncanonical_stream_alone_is_not_observable():
    source = SAMPLE.replace(b"GameState.DebugPrintPower()", b"PowerTaskList.DebugPrintPower()")
    result = audit_powerlog_observability(source)

    assert result["canonical_power_lines"] == 0
    assert result["raw_observability_candidate"] is False
    assert result["ranking_ready"] is False


def test_source_digest_changes_with_exact_bytes():
    first = audit_powerlog_observability(SAMPLE)
    second = audit_powerlog_observability(SAMPLE + b"\n")
    assert first["source_sha256"] != second["source_sha256"]


@pytest.mark.parametrize("bad", [None, "text", bytearray(SAMPLE)])
def test_requires_exact_bytes(bad):
    with pytest.raises(TypeError, match="exact bytes"):
        audit_powerlog_observability(bad)


def test_rejects_empty_and_non_utf8_sources():
    with pytest.raises(ValueError, match="non-empty"):
        audit_powerlog_observability(b"")
    with pytest.raises(ValueError, match="UTF-8"):
        audit_powerlog_observability(b"\xff\xfe")


def test_file_probe_reads_exact_bytes_and_remains_fail_closed(tmp_path):
    source = tmp_path / "Power.log"
    source.write_bytes(SAMPLE)

    result = audit_powerlog_file(source)

    assert result["source_sha256"] == hashlib.sha256(SAMPLE).hexdigest()
    assert result["source_bytes"] == len(SAMPLE)
    assert result["raw_observability_candidate"] is True
    assert result["phase_3u_schema_ready"] is False
    assert result["ranking_ready"] is False


def test_cli_writes_machine_readable_measurement_artifact(tmp_path, monkeypatch, capsys):
    source = tmp_path / "Power.log"
    output = tmp_path / "evidence" / "powerlog_probe.json"
    source.write_bytes(SAMPLE)
    monkeypatch.setattr(
        "sys.argv",
        ["phase_3u_powerlog_feasibility", "--source", str(source), "--out", str(output)],
    )

    main()

    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["source_sha256"] == hashlib.sha256(SAMPLE).hexdigest()
    assert written["candidate_scoring_performed"] is False
    assert written["ranking_ready"] is False
