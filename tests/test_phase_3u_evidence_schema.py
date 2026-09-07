import math

import pytest

from ml.phase_3u_evidence_schema import validate_external_transition_evidence


def _body(entity_id, attack, health):
    return {"entity_id": entity_id, "attack": attack, "health": health}


def _row(index=1, *, pool=10, observation_id=None):
    row = {
        "observation_id": observation_id or f"obs-{index}",
        "game_id": "g1",
        "player_id": "p1",
        "event_index": index,
        "event_kind": "play",
        "pre_board": [_body("a", 3, 4), _body("b", 5, 6)],
        "post_board": [_body("a", 4, 5), _body("b", 6, 7), _body("c", 2, 3)],
    }
    if pool is not None:
        row["conserved_pool"] = pool
    return row


def _source(**overrides):
    source = {
        "generated_by": "external_log_parser",
        "independent": True,
        "source_reference": "external://phase3u/observations-v1",
        "source_sha256": "b" * 64,
    }
    source.update(overrides)
    return source


def test_valid_external_transition_is_schema_ready_but_not_ranking_ready():
    out = validate_external_transition_evidence([_row()], source=_source())
    assert out["valid"] is True
    assert out["schema_version"] == "3u_evidence_v3"
    assert out["schema_ready"] is True
    assert out["ranking_ready"] is False
    assert out["ranking_blocker"] == "admission_thresholds_not_yet_satisfied"
    assert out["candidate_scoring_performed"] is False
    assert out["persistent_entity_links"] == 2
    assert out["observation_ids"] == ["obs-1"]
    assert out["observation_ids_unique"] is True
    assert out["per_body_stats_finite"] is True
    assert out["source_reference"] == "external://phase3u/observations-v1"
    assert out["source_sha256"] == "b" * 64
    assert out["source_provenance_verified"] is True


def test_single_complete_row_cannot_prematurely_authorize_ranking():
    out = validate_external_transition_evidence([_row()], source=_source())
    assert out["conserved_pool_complete"] is True
    assert out["ranking_ready"] is False


def test_missing_conserved_pool_keeps_evidence_not_schema_or_ranking_ready():
    out = validate_external_transition_evidence([_row(pool=None)], source=_source())
    assert out["valid"] is True
    assert out["conserved_pool_complete"] is False
    assert out["schema_ready"] is False
    assert out["ranking_ready"] is False


def test_replay_or_candidate_generated_evidence_is_rejected():
    with pytest.raises(ValueError, match="independent"):
        validate_external_transition_evidence(
            [_row()], source=_source(generated_by="replay_simulator")
        )
    with pytest.raises(ValueError, match="independent"):
        validate_external_transition_evidence(
            [_row()], source=_source(independent=False)
        )


def test_external_source_requires_reference_and_digest():
    with pytest.raises(ValueError, match="source_reference"):
        validate_external_transition_evidence(
            [_row()], source=_source(source_reference="")
        )
    with pytest.raises(ValueError, match="source_sha256"):
        validate_external_transition_evidence(
            [_row()], source=_source(source_sha256="not-a-digest")
        )


def test_simulator_downstream_selection_labels_are_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        validate_external_transition_evidence(
            [_row()],
            source=_source(),
            selection_labels=("simulated_damage_under_candidate_rule",),
        )


def test_stable_entity_identity_is_required():
    row = _row()
    row["post_board"] = [_body("c", 2, 3), _body("d", 4, 5)]
    with pytest.raises(ValueError, match="stable entity_id"):
        validate_external_transition_evidence([row], source=_source())


def test_event_order_must_be_strictly_increasing_per_trajectory():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_external_transition_evidence(
            [_row(2, observation_id="obs-a"), _row(2, observation_id="obs-b")],
            source=_source(),
        )


def test_complete_body_stats_are_required():
    row = _row()
    del row["pre_board"][0]["health"]
    with pytest.raises(ValueError, match="body missing required fields"):
        validate_external_transition_evidence([row], source=_source())


def test_body_stats_must_be_observed_numeric_values():
    row = _row()
    row["post_board"][0]["attack"] = "unknown"
    with pytest.raises(ValueError, match="observed numeric"):
        validate_external_transition_evidence([row], source=_source())


def test_duplicate_entity_ids_are_rejected():
    row = _row()
    row["pre_board"].append(_body("a", 8, 9))
    with pytest.raises(ValueError, match="duplicate entity_id"):
        validate_external_transition_evidence([row], source=_source())


def test_unsupported_event_kind_is_rejected():
    row = _row()
    row["event_kind"] = "combat_result"
    with pytest.raises(ValueError, match="unsupported membership event_kind"):
        validate_external_transition_evidence([row], source=_source())


def test_observation_ids_are_required_nonempty_and_unique():
    missing = _row()
    del missing["observation_id"]
    with pytest.raises(ValueError, match="observation_id"):
        validate_external_transition_evidence([missing], source=_source())

    blank = _row()
    blank["observation_id"] = "  "
    with pytest.raises(ValueError, match="observation_id must be non-empty"):
        validate_external_transition_evidence([blank], source=_source())

    with pytest.raises(ValueError, match="duplicate observation_id"):
        validate_external_transition_evidence(
            [_row(1, observation_id="same"), _row(2, observation_id="same")],
            source=_source(),
        )


def test_trajectory_ids_must_be_nonempty():
    row = _row()
    row["game_id"] = ""
    with pytest.raises(ValueError, match="game_id must be non-empty"):
        validate_external_transition_evidence([row], source=_source())

    row = _row()
    row["player_id"] = "   "
    with pytest.raises(ValueError, match="player_id must be non-empty"):
        validate_external_transition_evidence([row], source=_source())


def test_event_index_must_be_exact_nonnegative_integer():
    for invalid in (1.5, "2", True, -1):
        row = _row()
        row["event_index"] = invalid
        with pytest.raises(ValueError, match="event_index"):
            validate_external_transition_evidence([row], source=_source())


def test_body_stats_and_conserved_pool_must_be_finite():
    for invalid in (math.nan, math.inf, -math.inf):
        row = _row()
        row["post_board"][0]["health"] = invalid
        with pytest.raises(ValueError, match="health must be finite"):
            validate_external_transition_evidence([row], source=_source())

        row = _row(pool=invalid)
        with pytest.raises(ValueError, match="conserved_pool must be finite"):
            validate_external_transition_evidence([row], source=_source())


def test_boards_and_rows_require_structured_mappings():
    row = _row()
    row["pre_board"] = "not-a-board"
    with pytest.raises(ValueError, match="board must be a sequence"):
        validate_external_transition_evidence([row], source=_source())

    with pytest.raises(ValueError, match="transition rows must be mappings"):
        validate_external_transition_evidence(["not-a-row"], source=_source())
