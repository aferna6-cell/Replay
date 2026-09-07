import hashlib

import pytest

from ml.phase_3u_powerlog_feasibility import audit_powerlog_observability


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
