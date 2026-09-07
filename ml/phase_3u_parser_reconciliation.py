"""Provider-independent Phase 3U parser-to-manifest reconciliation.

Measurement-only boundary that executes a preregistered external parser on exact
source bytes, derives observation identities from the parser output, and requires
an exact match with the frozen observation manifest. This module deliberately does
not define a provider serialization or score any candidate allocation rule.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence

from ml.phase_3u_observation_identity import derive_observation_ids

PARSER_RECONCILIATION_VERSION = "3u_parser_reconciliation_v2"


def _nonempty_text(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _sha256_bytes(value: object, field: str) -> str:
    if not isinstance(value, bytes):
        raise ValueError(f"{field} must be exact bytes")
    return hashlib.sha256(value).hexdigest()


def _frozen_sha256(value: object, field: str) -> str:
    text = _nonempty_text(value, field).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be a 64-hex SHA-256 digest")
    return text


def reconcile_parser_output_to_manifest(
    source_bytes: bytes,
    *,
    parser: Callable[[bytes], Iterable[Mapping]],
    parser_version: str,
    parser_artifact_bytes: bytes,
    expected_parser_artifact_sha256: str,
    parser_config_bytes: bytes,
    expected_parser_config_sha256: str,
    expected_manifest_observation_ids: Sequence[str],
) -> dict:
    """Execute one frozen parser and reconcile its output to an immutable manifest.

    The exact source bytes are hashed here, before parsing. The parser receives
    those same bytes. Before execution, the exact preregistered parser implementation
    artifact and configuration bytes are independently hashed and must match their
    frozen SHA-256 digests. This makes a human-readable ``parser_version`` label
    insufficient to clear the gate by itself.

    Each parsed row must carry an ``observation_id`` equal to the provider-independent
    deterministic ID derived from immutable row coordinates plus the source SHA.
    Finally, the ordered derived-ID sequence must exactly equal the preregistered
    manifest sequence; missing, extra, reordered, duplicate, or independently
    relabelled observations fail closed.

    This proves executable source->parser-output->identity->manifest binding plus an
    immutable parser-artifact/config provenance contract. It does not prove that an
    arbitrary Python callable is byte-for-byte the supplied parser artifact; the
    provider adapter still needs a reviewed execution wrapper that loads/runs that
    exact frozen artifact before Phase 3U ranking can clear.
    """
    if not isinstance(source_bytes, bytes):
        raise ValueError("source_bytes must be exact bytes")
    if not callable(parser):
        raise ValueError("parser must be callable")
    frozen_parser_version = _nonempty_text(parser_version, "parser_version")

    parser_artifact_sha256 = _sha256_bytes(parser_artifact_bytes, "parser_artifact_bytes")
    expected_artifact_sha = _frozen_sha256(
        expected_parser_artifact_sha256, "expected_parser_artifact_sha256"
    )
    if parser_artifact_sha256 != expected_artifact_sha:
        raise ValueError("parser artifact bytes do not match frozen SHA-256")

    parser_config_sha256 = _sha256_bytes(parser_config_bytes, "parser_config_bytes")
    expected_config_sha = _frozen_sha256(
        expected_parser_config_sha256, "expected_parser_config_sha256"
    )
    if parser_config_sha256 != expected_config_sha:
        raise ValueError("parser config bytes do not match frozen SHA-256")

    expected = [_nonempty_text(value, "manifest observation_id") for value in expected_manifest_observation_ids]
    if not expected:
        raise ValueError("expected manifest must contain at least one observation_id")
    if len(expected) != len(set(expected)):
        raise ValueError("expected manifest contains duplicate observation_id")

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    parsed_rows = list(parser(source_bytes))
    if not parsed_rows:
        raise ValueError("parser produced no transition rows")
    if any(not isinstance(row, Mapping) for row in parsed_rows):
        raise ValueError("parser output rows must be mappings")

    derived_ids = derive_observation_ids(parsed_rows, source_sha256=source_sha256)
    supplied_ids = []
    for row, derived_id in zip(parsed_rows, derived_ids):
        supplied = _nonempty_text(row.get("observation_id"), "observation_id")
        if supplied != derived_id:
            raise ValueError("parsed observation_id does not match source-derived identity")
        supplied_ids.append(supplied)

    if supplied_ids != expected:
        raise ValueError("parsed observation sequence does not exactly match frozen manifest")

    return {
        "reconciliation_version": PARSER_RECONCILIATION_VERSION,
        "parser_version": frozen_parser_version,
        "parser_artifact_sha256": parser_artifact_sha256,
        "parser_config_sha256": parser_config_sha256,
        "source_sha256": source_sha256,
        "row_count": len(parsed_rows),
        "observation_ids": supplied_ids,
        "source_parser_identity_manifest_bound": True,
        "parser_artifact_config_bound": True,
        "execution_provenance_bound": False,
        "ranking_admissible": False,
        "ranking_block_reason": "parser_callable_not_loaded_from_digest_bound_artifact",
        "candidate_scoring_performed": False,
    }
