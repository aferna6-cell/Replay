"""Phase 3U independent evidence admission contract.

Measurement-only validation for externally observed body-level transitions used to
identify a conserved-pool allocation rule. This module does not score candidate
rules and does not change simulator behavior.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

SCHEMA_VERSION = "3u_evidence_v1"

REQUIRED_EVENT_FIELDS = (
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


def _body_map(board: Sequence[Mapping]) -> Dict[str, Mapping]:
    out: Dict[str, Mapping] = {}
    for body in board:
        missing = [field for field in REQUIRED_BODY_FIELDS if field not in body]
        if missing:
            raise ValueError(f"body missing required fields: {missing}")
        entity_id = str(body["entity_id"])
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        if entity_id in out:
            raise ValueError(f"duplicate entity_id in board: {entity_id}")
        out[entity_id] = body
    return out


def validate_external_transition_evidence(
    rows: Iterable[Mapping],
    *,
    source: Mapping,
    selection_labels: Iterable[str] = (),
) -> Dict:
    """Validate independent real-log transition evidence for Phase 3U.

    Required properties:
    - external/non-simulator provenance;
    - stable body entity IDs;
    - strictly increasing event ordering within each game/player trajectory;
    - complete pre/post boards with independently observed per-body ATK/HP;
    - at least one persistent entity across each transition, so pre/post body
      changes are longitudinal rather than two unrelated snapshots;
    - no simulator-derived candidate-selection labels.

    Conserved-pool reconciliation is deliberately not inferred here: an admitted
    dataset must carry an independently observed or externally reconstructed
    ``conserved_pool`` value on every row before it can be used for ranking.
    """
    generated_by = source.get("generated_by")
    independent = bool(source.get("independent"))
    if generated_by in {"replay_simulator", "candidate_rule"} or not independent:
        raise ValueError("Phase 3U evidence must be independent of Replay/candidate rules")

    forbidden = sorted(set(selection_labels) & FORBIDDEN_SELECTION_LABELS)
    if forbidden:
        raise ValueError(f"forbidden simulator-derived selection labels: {forbidden}")

    materialized = list(rows)
    if not materialized:
        raise ValueError("Phase 3U evidence set must contain at least one transition")

    last_index: Dict[tuple[str, str], int] = {}
    persistent_entities = 0
    conserved_pool_rows = 0

    for row in materialized:
        missing = [field for field in REQUIRED_EVENT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"transition missing required fields: {missing}")

        event_kind = str(row["event_kind"])
        if event_kind not in ALLOWED_EVENT_KINDS:
            raise ValueError(f"unsupported membership event_kind: {event_kind}")

        trajectory = (str(row["game_id"]), str(row["player_id"]))
        event_index = int(row["event_index"])
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

        # ATK/HP values must be directly observed numeric values, not absent or
        # symbolic placeholders. bool is rejected because it is an int subtype.
        for board in (pre, post):
            for body in board.values():
                for field in ("attack", "health"):
                    value = body[field]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"{field} must be an observed numeric value")

        if "conserved_pool" in row:
            pool = row["conserved_pool"]
            if isinstance(pool, bool) or not isinstance(pool, (int, float)):
                raise ValueError("conserved_pool must be numeric when provided")
            conserved_pool_rows += 1

    pool_complete = conserved_pool_rows == len(materialized)
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "row_count": len(materialized),
        "trajectory_count": len(last_index),
        "persistent_entity_links": persistent_entities,
        "independent_source": True,
        "per_body_stats_observed": True,
        "event_order_valid": True,
        "complete_pre_post_boards": True,
        "conserved_pool_complete": pool_complete,
        "ranking_ready": pool_complete,
        "candidate_scoring_performed": False,
        "simulator_outcome_labels_used": False,
    }
