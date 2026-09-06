import hashlib

import pytest

from ml.phase_3u_observation_identity import derive_observation_id
from ml.phase_3u_parser_reconciliation import (
    PARSER_RECONCILIATION_VERSION,
    reconcile_parser_output_to_manifest,
)


def _body(entity_id, attack=1, health=1):
    return {"entity_id": entity_id, "attack": attack, "health": health}


def _base_row(index=1):
    return {
        "game_id": "g1",
        "player_id": "p1",
        "event_index": index,
        "event_kind": "play",
        "pre_board": [_body("a", 3, 4), _body("b", 5, 6)],
        "post_board": [_body("a", 4, 5), _body("b", 6, 7), _body(f"c{index}", 2, 3)],
        "conserved_pool": 10,
    }


def _parser_for(rows):
    def parser(source_bytes):
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        parsed = []
        for original in rows:
            row = dict(original)
            row["pre_board"] = [dict(body) for body in original["pre_board"]]
            row["post_board"] = [dict(body) for body in original["post_board"]]
            row["observation_id"] = derive_observation_id(row, source_sha256=source_sha)
            parsed.append(row)
        return parsed

    return parser


def _ids_for(source_bytes, rows):
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    return [derive_observation_id(row, source_sha256=source_sha) for row in rows]


def test_exact_source_parser_identity_manifest_chain_reconciles():
    source = b"real-provider-artifact-placeholder-format-v1"
    rows = [_base_row(1), _base_row(2)]
    result = reconcile_parser_output_to_manifest(
        source,
        parser=_parser_for(rows),
        parser_version="provider-parser-v1",
        expected_manifest_observation_ids=_ids_for(source, rows),
    )
    assert result["reconciliation_version"] == PARSER_RECONCILIATION_VERSION
    assert result["parser_version"] == "provider-parser-v1"
    assert result["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert result["row_count"] == 2
    assert result["source_parser_identity_manifest_bound"] is True
    assert result["candidate_scoring_performed"] is False


def test_source_byte_change_breaks_frozen_manifest_binding():
    source = b"artifact-v1"
    rows = [_base_row()]
    frozen = _ids_for(source, rows)
    with pytest.raises(ValueError, match="frozen manifest"):
        reconcile_parser_output_to_manifest(
            b"artifact-v1-tampered",
            parser=_parser_for(rows),
            parser_version="provider-parser-v1",
            expected_manifest_observation_ids=frozen,
        )


def test_independently_supplied_observation_id_is_rejected():
    source = b"artifact-v1"
    row = _base_row()

    def parser(_source_bytes):
        parsed = dict(row)
        parsed["observation_id"] = "externally-picked-label"
        return [parsed]

    with pytest.raises(ValueError, match="source-derived identity"):
        reconcile_parser_output_to_manifest(
            source,
            parser=parser,
            parser_version="provider-parser-v1",
            expected_manifest_observation_ids=["externally-picked-label"],
        )


def test_missing_extra_or_reordered_observations_fail_closed():
    source = b"artifact-v1"
    rows = [_base_row(1), _base_row(2)]
    ids = _ids_for(source, rows)

    with pytest.raises(ValueError, match="exactly match frozen manifest"):
        reconcile_parser_output_to_manifest(
            source,
            parser=_parser_for(rows[:1]),
            parser_version="provider-parser-v1",
            expected_manifest_observation_ids=ids,
        )

    with pytest.raises(ValueError, match="exactly match frozen manifest"):
        reconcile_parser_output_to_manifest(
            source,
            parser=_parser_for(rows),
            parser_version="provider-parser-v1",
            expected_manifest_observation_ids=list(reversed(ids)),
        )


def test_parser_version_and_manifest_contract_fail_closed():
    source = b"artifact-v1"
    rows = [_base_row()]
    parser = _parser_for(rows)
    ids = _ids_for(source, rows)

    with pytest.raises(ValueError, match="parser_version"):
        reconcile_parser_output_to_manifest(
            source,
            parser=parser,
            parser_version=" ",
            expected_manifest_observation_ids=ids,
        )

    with pytest.raises(ValueError, match="duplicate observation_id"):
        reconcile_parser_output_to_manifest(
            source,
            parser=parser,
            parser_version="provider-parser-v1",
            expected_manifest_observation_ids=[ids[0], ids[0]],
        )


def test_non_bytes_or_malformed_parser_output_is_rejected():
    with pytest.raises(ValueError, match="exact bytes"):
        reconcile_parser_output_to_manifest(
            "not-bytes",
            parser=lambda _: [],
            parser_version="provider-parser-v1",
            expected_manifest_observation_ids=["x"],
        )

    with pytest.raises(ValueError, match="mappings"):
        reconcile_parser_output_to_manifest(
            b"artifact-v1",
            parser=lambda _: ["not-a-row"],
            parser_version="provider-parser-v1",
            expected_manifest_observation_ids=["x"],
        )
