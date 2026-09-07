import pytest

from ml.phase_3u_observation_identity import (
    IDENTITY_CONTRACT_VERSION,
    canonical_observation_identity_payload,
    derive_observation_id,
    derive_observation_ids,
)


def _body(entity_id, attack=1, health=1):
    return {"entity_id": entity_id, "attack": attack, "health": health}


def _row(index=1):
    return {
        "game_id": "g1",
        "player_id": "p1",
        "event_index": index,
        "event_kind": "play",
        "pre_board": [_body("a", 3, 4), _body("b", 5, 6)],
        "post_board": [_body("a", 4, 5), _body("b", 6, 7), _body("c", 2, 3)],
        "conserved_pool": 10,
    }


def test_identity_is_deterministic_and_source_bound():
    row = _row()
    first = derive_observation_id(row, source_sha256="a" * 64)
    second = derive_observation_id(dict(row), source_sha256="a" * 64)
    other_source = derive_observation_id(row, source_sha256="b" * 64)
    assert first == second
    assert len(first) == 64
    assert first != other_source


def test_measurements_do_not_change_observation_identity():
    row = _row()
    baseline = derive_observation_id(row, source_sha256="a" * 64)
    changed = _row()
    changed["pre_board"][0]["attack"] = 999
    changed["post_board"][1]["health"] = 888
    changed["conserved_pool"] = 777
    assert derive_observation_id(changed, source_sha256="a" * 64) == baseline


def test_identity_changes_when_immutable_event_coordinates_change():
    baseline = derive_observation_id(_row(), source_sha256="a" * 64)
    for mutate in (
        lambda row: row.__setitem__("game_id", "g2"),
        lambda row: row.__setitem__("player_id", "p2"),
        lambda row: row.__setitem__("event_index", 2),
        lambda row: row.__setitem__("event_kind", "sell"),
        lambda row: row["post_board"].append(_body("d")),
        lambda row: row["post_board"].reverse(),
    ):
        row = _row()
        mutate(row)
        assert derive_observation_id(row, source_sha256="a" * 64) != baseline


def test_payload_is_explicitly_versioned_and_excludes_candidate_outcomes():
    row = _row()
    row["simulated_damage_under_candidate_rule"] = 123
    payload = canonical_observation_identity_payload(row, source_sha256="a" * 64)
    assert payload["identity_contract_version"] == IDENTITY_CONTRACT_VERSION
    assert "simulated_damage_under_candidate_rule" not in payload
    assert "conserved_pool" not in payload
    assert "attack" not in str(payload)
    assert "health" not in str(payload)


def test_invalid_or_ambiguous_identity_fields_fail_closed():
    with pytest.raises(ValueError, match="source_sha256"):
        derive_observation_id(_row(), source_sha256="not-a-digest")

    row = _row()
    row["event_index"] = True
    with pytest.raises(ValueError, match="event_index"):
        derive_observation_id(row, source_sha256="a" * 64)

    row = _row()
    row["pre_board"].append(_body("a"))
    with pytest.raises(ValueError, match="duplicate entity_id"):
        derive_observation_id(row, source_sha256="a" * 64)


def test_batch_derivation_rejects_duplicate_transition_identity():
    with pytest.raises(ValueError, match="collision or duplicate"):
        derive_observation_ids([_row(), _row()], source_sha256="a" * 64)
