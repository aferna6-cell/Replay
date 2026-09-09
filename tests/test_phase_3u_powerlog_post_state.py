import hashlib

from ml.phase_3u_powerlog_post_state import audit_powerlog_post_state


def _line(payload: str) -> bytes:
    return f"D 12:00:00 GameState.DebugPrintPower() - {payload}\n".encode()


def test_forward_zone_position_is_post_state_only_and_never_admits():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=3"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert result["unresolved_numeric_zone_events"] == 1
    assert result["tag_zone_position_observed_before_next_zone"] == 1
    assert result["post_zone_position_coverage"] == 1.0
    event = result["per_game"][0]["intervals"][0]
    assert event["position_observations"] == [
        {"ordinal": 1, "distance": 1, "source": "tag",
         "temporal_side": "post", "position": 3}
    ]
    assert result["pre_state_repaired_from_future_events"] is False
    assert result["phase_3u_schema_ready"] is False
    assert result["ranking_ready"] is False


def test_matching_descriptor_zonepos_counts_as_explicit_forward_channel():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=PLAY zonePos=2 cardId=BG_TEST player=1] tag=ATK value=5"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["descriptor_zone_position_observed_before_next_zone"] == 1
    assert result["descriptor_pre_zone_position_observed"] == 0
    assert result["post_zone_position_observed_before_next_zone"] == 1
    event = result["per_game"][0]["intervals"][0]
    assert event["post_zone_position"] == 2
    assert event["post_zone_position_source"] == "descriptor"
    assert event["card_id_observed_after_distance"] == 1
    assert event["position_observations"][0]["temporal_side"] == "post"


def test_descriptor_zone_must_match_pending_post_zone():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=2 cardId=BG_TEST player=1] tag=ATK value=5"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["descriptor_zone_position_observed_before_next_zone"] == 0
    assert result["descriptor_zone_mismatch_events"] == 1


def test_delayed_position_change_is_evolution_not_contradiction():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND"),
        _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=3"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=2 cardId=BG_TEST player=1] tag=ATK value=5"),
    ])
    result = audit_powerlog_post_state(source)
    event = result["per_game"][0]["intervals"][0]
    assert result["position_contradictions"] == 0
    assert result["same_ordinal_position_conflicts"] == 0
    assert result["position_evolution_events"] == 1
    assert event["position_evolution_events"] == 1
    assert [o["position"] for o in event["position_observations"]] == [3, 2]
    assert [o["temporal_side"] for o in event["position_observations"]] == ["post", "post"]


def test_same_record_descriptor_and_tag_agreement_is_pre_to_post_transition():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=2 cardId=BG_TEST player=1] tag=ZONE_POSITION value=2"),
    ])
    result = audit_powerlog_post_state(source)
    event = result["per_game"][0]["intervals"][0]
    assert result["same_ordinal_position_conflicts"] == 0
    assert result["descriptor_pre_zone_position_observed"] == 1
    assert result["descriptor_zone_position_observed_before_next_zone"] == 0
    assert result["tag_zone_position_observed_before_next_zone"] == 1
    assert [o["ordinal"] for o in event["position_observations"]] == [1, 1]
    assert [o["source"] for o in event["position_observations"]] == ["descriptor", "tag"]
    assert [o["temporal_side"] for o in event["position_observations"]] == ["pre", "post"]
    assert event["post_zone_position"] == 2
    assert event["post_zone_position_source"] == "tag"


def test_same_record_descriptor_tag_disagreement_is_transition_not_conflict():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=2 cardId=BG_TEST player=1] tag=ZONE_POSITION value=3"),
    ])
    result = audit_powerlog_post_state(source)
    event = result["per_game"][0]["intervals"][0]
    assert result["position_contradictions"] == 0
    assert result["same_ordinal_position_conflicts"] == 0
    assert result["position_evolution_events"] == 0
    assert result["descriptor_pre_zone_position_observed"] == 1
    assert result["descriptor_zone_position_observed_before_next_zone"] == 0
    assert result["post_zone_position_observed_before_next_zone"] == 1
    assert event["position_contradiction"] is False
    assert event["descriptor_pre_zone_position"] == 2
    assert event["post_zone_position"] == 3
    assert event["post_zone_position_source"] == "tag"
    assert [(o["temporal_side"], o["position"]) for o in event["position_observations"]] == [
        ("pre", 2), ("post", 3)
    ]


def test_descriptor_pre_alone_cannot_satisfy_post_coverage():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=2 cardId=BG_TEST player=1] tag=ZONE_POSITION value=not-a-number"),
    ])
    result = audit_powerlog_post_state(source)
    event = result["per_game"][0]["intervals"][0]
    assert result["descriptor_pre_zone_position_observed"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 0
    assert event["post_zone_position_observed"] is False


def test_unrelated_later_descriptor_snapshot_remains_forward_evidence():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=HAND"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=4 cardId=BG_TEST player=1] tag=HEALTH value=6"),
    ])
    result = audit_powerlog_post_state(source)
    event = result["per_game"][0]["intervals"][0]
    assert result["descriptor_pre_zone_position_observed"] == 0
    assert result["descriptor_zone_position_observed_before_next_zone"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 1
    assert event["post_zone_position"] == 4
    assert event["position_observations"][0]["temporal_side"] == "post"


def test_zone_summary_reports_coverage_and_closure():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=2"),
        _line("TAG_CHANGE Entity=26 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=26 tag=ZONE value=GRAVEYARD"),
    ])
    result = audit_powerlog_post_state(source)
    play = result["by_zone"]["PLAY"]
    assert play["unresolved"] == 2
    assert play["union"] == 1
    assert play["closed_without_position"] == 1
    assert play["union_coverage"] == 0.5


def test_next_zone_closes_interval_before_late_position():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
        _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=0"),
    ])
    result = audit_powerlog_post_state(source)
    first, second = result["per_game"][0]["intervals"]
    assert first["closed_by_next_zone"] is True
    assert first["post_zone_position_observed"] is False
    assert first["next_zone_distance"] == 1
    assert second["post_zone_position_observed"] is True
    assert result["closed_by_next_zone_without_post_position"] == 1


def test_previously_missing_card_id_can_be_observed_forward_without_repairing_pre_state():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=PLAY zonePos=2 cardId=BG_TEST player=1] tag=ATK value=5"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["missing_card_id_pre_events"] == 1
    assert result["missing_card_id_pre_with_forward_card_id"] == 1
    assert result["forward_card_id_coverage_for_missing_pre_card"] == 1.0


def test_recoverable_numeric_pre_state_is_not_part_of_unresolved_forward_cohort():
    source = b"".join([
        _line("TAG_CHANGE Entity=[entityName=Test id=25 zone=HAND zonePos=1 cardId=BG_TEST player=1] tag=ATK value=5"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=2"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["unresolved_numeric_zone_events"] == 0
    assert result["post_zone_position_coverage"] is None
    assert result["per_game"][0]["intervals"] == []


def test_pending_intervals_do_not_cross_create_game_boundary():
    source = b"".join([
        _line("CREATE_GAME"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("CREATE_GAME"),
        _line("TAG_CHANGE Entity=25 tag=ZONE_POSITION value=4"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["game_segments"] == 2
    assert result["unresolved_numeric_zone_events"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 0


def test_noncanonical_mirror_cannot_supply_forward_position():
    source = (_line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY")
              + b"D 12:00:00 PowerTaskList.DebugPrintPower() - TAG_CHANGE Entity=25 tag=ZONE_POSITION value=3\n")
    result = audit_powerlog_post_state(source)
    assert result["unresolved_numeric_zone_events"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 0


def test_full_entity_raw_zone_position_counts_as_distinct_forward_channel():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=ZONE value=PLAY"),
        _line("    tag=ZONE_POSITION value=4"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["probe_version"] == "3u_powerlog_post_state_v5"
    assert result["full_entity_tag_zone_position_observed_before_next_zone"] == 1
    assert result["full_entity_tag_zone_position_coverage"] == 1.0
    event = result["per_game"][0]["intervals"][0]
    assert event["post_zone_position_source"] == "full_entity_tag"
    assert event["position_observations"] == [
        {"ordinal": 3, "distance": 3, "source": "full_entity_tag",
         "temporal_side": "post", "position": 4}
    ]
    assert result["pre_state_repaired_from_future_events"] is False


def test_full_entity_position_rejected_when_block_zone_disagrees_with_pending_post_zone():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=ZONE value=HAND"),
        _line("    tag=ZONE_POSITION value=4"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["full_entity_tag_zone_position_observed_before_next_zone"] == 0
    assert result["full_entity_zone_mismatch_events"] == 1
    assert result["post_zone_position_observed_before_next_zone"] == 0


def test_full_entity_block_ends_before_unrelated_raw_position_line():
    source = b"".join([
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=99 tag=ATK value=2"),
        _line("    tag=ZONE_POSITION value=4"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["full_entity_tag_zone_position_observed_before_next_zone"] == 0
    assert result["post_zone_position_observed_before_next_zone"] == 0


def test_full_entity_forward_position_cannot_cross_next_zone_or_game_boundary():
    source = b"".join([
        _line("CREATE_GAME"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=PLAY"),
        _line("TAG_CHANGE Entity=25 tag=ZONE value=GRAVEYARD"),
        _line("CREATE_GAME"),
        _line("FULL_ENTITY - Creating ID=25 CardID=BG_TEST"),
        _line("    tag=ZONE value=PLAY"),
        _line("    tag=ZONE_POSITION value=4"),
    ])
    result = audit_powerlog_post_state(source)
    assert result["game_segments"] == 2
    first_game = result["per_game"][0]
    assert first_game["closed_by_next_zone_without_post_position"] == 1
    first = first_game["intervals"][0]
    assert first["post_zone_position_observed"] is False
