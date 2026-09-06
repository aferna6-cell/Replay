"""Phase 3U independent evidence admission contract.

Measurement-only validation for externally observed body-level transitions used to
identify a conserved-pool allocation rule. This module does not score candidate
rules and does not change simulator behavior.
"""

from __future__ import annotations

import math
import re
from numbers import Integral, Real
from typing import Dict, Iterable, Mapping, Sequence

SCHEMA_VERSION = "3u_evidence_v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_EVENT_FIELDS = (
    "observation_id",
    "game_id",
    "player_id",
    "event_index",
    "event_kind",
    "pre_board",
    "post_board",
)

REQUIRED_BODY_FIELDS = (
    "entity_id",
    "attack",
    "health",
)

# These are downstream simulator outcomes and may not be admitted as labels for
# choosing an allocation rule. They can be reported descriptively elsewhere.
FORBIDDEN_SELECTION_LABELS = {
    "simulated_survival_under_candidate_rule",
    "simulated_damage_under_candidate_rule",
    "placement",
    "mean_game_length",
    "macro_fidelity_gate",
}

ALLOWED_EVENT_KINDS = {
    "play",
    "sell",
    "triple",
    "transform",
    "membership_change",
}


def _nonempty_id(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be an observed numeric value")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _body_map(board: Sequence[Mapping]) -> Dict[str, Mapping]:
    if isinstance(board, (str, bytes)) or not isinstance(board, Sequence):
        raise ValueError("board must be a sequence of body mappings")
    out: Dict[str, Mapping] = {}
    for body in board:
        if not isinstance(body, Mapping):
            raise ValueError("board entries must be body mappings")
        missing = [field for field in REQUIRED_BODY_FIELDS if field not in body]
        if missing:
            raise ValueError(f"body missing required fields: {missing}")
        entity_id = _nonempty_id(body["entity_id"], "entity_id")
        if entity_id in out:
            raise ValueError(f"duplicate entity_id in board: {entity_id}")
        out[entity_id] = body
    return out


def _validate_source_provenance(source: Mapping) -> tuple[str, str]:
    reference = source.get("source_reference")
    digest = source.get("source_sha256")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("external evidence source_reference must be non-empty")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower()):
        raise ValueError("external evidence source_sha256 must be a 64-hex digest")
    return reference.strip(), digest.lower()


def validate_external_transition_evidence(
    rows: Iterable[Mapping],
    *,
    source: Mapping,
    selection_labels: Iterable[str] = (),
) -> Dict:
    """Validate independent real-log transition evidence for Phase 3U.

    Required properties:
    - external/non-simulator provenance, including immutable source reference+digest;
    - stable, unique observation IDs suitable for immutable split manifests;
    - non-empty game/player/body identity and exact integral event ordering;
    - complete pre/post boards with independently observed finite per-body ATK/HP;
    - at least one persistent entity across each transition, so pre/post body
      changes are longitudinal rather than two unrelated snapshots;
    - no simulator-derived candidate-selection labels.

    Conserved-pool reconciliation is deliberately not inferred here: an admitted
    dataset must carry an independently observed or externally reconstructed
    ``conserved_pool`` value on every row before it can advance to the separate
    Phase 3U admission-threshold gate. Schema validity alone never authorizes
    candidate ranking.
    """
    generated_by = source.get("generated_by")
    independent = bool(source.get("independent"))
    if generated_by in {"replay_simulator", "candidate_rule"} or not independent:
        raise ValueError("Phase 3U evidence must be independent of Replay/candidate rules")
    source_reference, source_sha256 = _validate_source_provenance(source)

    forbidden = sorted(set(selection_labels) & FORBIDDEN_SELECTION_LABELS)
    if forbidden:
        raise ValueError(f"forbidden simulator-derived selection labels: {forbidden}")

    materialized = list(rows)
    if not materialized:
        raise ValueError("Phase 3U evidence set must contain at least one transition")

    last_index: Dict[tuple[str, str], int] = {}
    observation_ids: list[str] = []
    seen_observation_ids: set[str] = set()
    persistent_entities = 0
    conserved_pool_rows = 0

    for row in materialized:
        if not isinstance(row, Mapping):
            raise ValueError("transition rows must be mappings")
        missing = [field for field in REQUIRED_EVENT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"transition missing required fields: {missing}")

        observation_id = _nonempty_id(row["observation_id"], "observation_id")
        if observation_id in seen_observation_ids:
            raise ValueError(f"duplicate observation_id: {observation_id}")
        seen_observation_ids.add(observation_id)
        observation_ids.append(observation_id)

        event_kind = str(row["event_kind"])
        if event_kind not in ALLOWED_EVENT_KINDS:
            raise ValueError(f"unsupported membership event_kind: {event_kind}")

        game_id = _nonempty_id(row["game_id"], "game_id")
        player_id = _nonempty_id(row["player_id"], "player_id")
        trajectory = (game_id, player_id)
        raw_event_index = row["event_index"]
        if isinstance(raw_event_index, bool) or not isinstance(raw_event_index, Integral):
            raise ValueError("event_index must be an exact integer")
        event_index = int(raw_event_index)
        if event_index < 0:
            raise ValueError("event_index must be non-negative")
        prior = last_index.get(trajectory)
        if prior is not None and event_index <= prior:
            raise ValueError(
                f"event_index must be strictly increasing for {trajectory}: "
                f"{event_index} after {prior}"
            )
        last_index[trajectory] = event_index

        pre = _body_map(row["pre_board"])
        post = _body_map(row["post_board"])
        if not pre or not post:
            raise ValueError("pre_board and post_board must both be complete/non-empty")

        shared = set(pre) & set(post)
        if not shared:
            raise ValueError("transition has no stable entity_id across pre/post boards")
        persistent_entities += len(shared)

        for board in (pre, post):
            for body in board.values():
                _finite_number(body["attack"], "attack")
                _finite_number(body["health"], "health")

        if "conserved_pool" in row:
            _finite_number(row["conserved_pool"], "conserved_pool")
            conserved_pool_rows += 1

    pool_complete = conserved_pool_rows == len(materialized)
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "row_count": len(materialized),
        "observation_ids": observation_ids,
        "observation_ids_unique": True,
        "trajectory_count": len(last_index),
        "persistent_entity_links": persistent_entities,
        "independent_source": True,
        "source_reference": source_reference,
        "source_sha256": source_sha256,
        "source_provenance_verified": True,
        "per_body_stats_observed": True,
        "per_body_stats_finite": True,
        "event_order_valid": True,
        "complete_pre_post_boards": True,
        "conserved_pool_complete": pool_complete,
        "schema_ready": pool_complete,
        "ranking_ready": False,
        "ranking_blocker": "admission_thresholds_not_yet_satisfied",
        "candidate_scoring_performed": False,
        "simulator_outcome_labels_used": False,
    }
