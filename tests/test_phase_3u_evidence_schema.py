import pytest

from ml.phase_3u_evidence_schema import validate_external_transition_evidence


def _body(entity_id, attack, health):
    return {"entity_id": entity_id, "attack": attack, "health": health}


def _row(index=1, *, pool=10):
    row = {
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
    source = {"generated_by": "external_log_parser", "independent": True}
    source.update(overrides)
    return source


def test_valid_external_transition_is_admitted_but_not_scored():
    out = validate_external_transition_evidence([_row()], source=_source())
    assert out["valid"] is True
    assert out["ranking_ready"] is True
    assert out["candidate_scoring_performed"] is False
    assert out["persistent_entity_links"] == 2


def test_missing_conserved_pool_keeps_evidence_not_ranking_ready():
    out = validate_external_transition_evidence([_row(pool=None)], source=_source())
    assert out["valid"] is True
    assert out["conserved_pool_complete"] is False
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
            [_row(2), _row(2)], source=_source()
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
