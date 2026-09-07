"""Provider-independent Phase 3U observation identity contract.

Measurement-only helper for deterministic observation IDs after an external parser
has produced the already-preregistered Phase 3U identity fields. This deliberately
does not define or guess any provider file format and does not score candidate
allocation rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from numbers import Integral
from typing import Iterable, Mapping, Sequence

IDENTITY_CONTRACT_VERSION = "3u_observation_identity_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_text(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _entity_ids(board: object, field: str) -> list[str]:
    if isinstance(board, (str, bytes)) or not isinstance(board, Sequence):
        raise ValueError(f"{field} must be a sequence of body mappings")
    ids: list[str] = []
    seen: set[str] = set()
    for body in board:
        if not isinstance(body, Mapping):
            raise ValueError(f"{field} entries must be mappings")
        if "entity_id" not in body:
            raise ValueError(f"{field} entry missing entity_id")
        entity_id = _nonempty_text(body["entity_id"], "entity_id")
        if entity_id in seen:
            raise ValueError(f"duplicate entity_id in {field}: {entity_id}")
        seen.add(entity_id)
        ids.append(entity_id)
    if not ids:
        raise ValueError(f"{field} must be non-empty")
    return ids


def canonical_observation_identity_payload(
    row: Mapping,
    *,
    source_sha256: str,
) -> dict:
    """Return the canonical immutable identity payload for one parsed transition.

    Identity intentionally excludes ATK/HP/conserved-pool measurements and all
    candidate/simulator outcomes. It binds only the immutable source digest and
    schema-level trajectory/event/body-membership coordinates already required by
    Phase 3U. Board entity order is preserved because source ordering is evidence.
    """
    if not isinstance(row, Mapping):
        raise ValueError("transition row must be a mapping")
    digest = str(source_sha256).lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("source_sha256 must be a 64-hex digest")

    raw_event_index = row.get("event_index")
    if isinstance(raw_event_index, bool) or not isinstance(raw_event_index, Integral):
        raise ValueError("event_index must be an exact integer")
    event_index = int(raw_event_index)
    if event_index < 0:
        raise ValueError("event_index must be non-negative")

    return {
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "source_sha256": digest,
        "game_id": _nonempty_text(row.get("game_id"), "game_id"),
        "player_id": _nonempty_text(row.get("player_id"), "player_id"),
        "event_index": event_index,
        "event_kind": _nonempty_text(row.get("event_kind"), "event_kind"),
        "pre_entity_ids": _entity_ids(row.get("pre_board"), "pre_board"),
        "post_entity_ids": _entity_ids(row.get("post_board"), "post_board"),
    }


def derive_observation_id(row: Mapping, *, source_sha256: str) -> str:
    """Derive a deterministic source-bound ID without provider-specific semantics."""
    payload = canonical_observation_identity_payload(row, source_sha256=source_sha256)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def derive_observation_ids(rows: Iterable[Mapping], *, source_sha256: str) -> list[str]:
    """Derive IDs for a parsed source and fail closed on any collision/duplicate."""
    ids = [derive_observation_id(row, source_sha256=source_sha256) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("derived observation_id collision or duplicate transition identity")
    return ids
