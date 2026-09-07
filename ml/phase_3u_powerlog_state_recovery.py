"""Conservative Phase 3U Power.log state-recovery probe (measurement-only).

This probe asks whether canonical GameState records can recover pre-ZONE state from
source-observed log state without simulator or candidate-model assumptions.

Recovery is deliberately conservative. A pre-ZONE state is complete only when zone,
zone position, non-empty card id, and player were observed for the same entity.
FULL_ENTITY blocks may contribute CardID plus directly following ZONE,
ZONE_POSITION, and CONTROLLER tags. CONTROLLER is accepted only when its numeric
value is independently observed as a PlayerID in the same CREATE_GAME-bounded
segment. Entity state and PlayerID validity are reset at every CREATE_GAME boundary,
preventing cross-game identity leakage when numeric IDs are reused.

After a ZONE change, zone position is invalidated until directly observed again.
This probe never defines membership semantics, constructs Phase 3U evidence rows,
reconstructs conserved pools, scores candidates, or authorizes ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict

PROBE_VERSION = "3u_powerlog_state_recovery_v4"

_CANONICAL_POWER_MARKER = "GameState.DebugPrintPower() -"
_CREATE_GAME_RE = re.compile(r"\bCREATE_GAME\b")
_PLAYER_RE = re.compile(r"\bPlayer EntityID=(\d+) PlayerID=(\d+)\b")
_FULL_ENTITY_RE = re.compile(r"\bFULL_ENTITY - Creating ID=(\d+)\s+CardID=(\S*)")
_RAW_TAG_RE = re.compile(r"^\s*tag=([A-Z0-9_]+) value=(\S+)\s*$")
_TAG_CHANGE_RE = re.compile(
    r"\bTAG_CHANGE Entity=(?:(\d+)|\[([^\]]*\bid=(\d+)\b[^\]]*)\]) "
    r"tag=([A-Z0-9_]+) value=(\S+)"
)
_DESCRIPTOR_ZONE_RE = re.compile(r"\bzone=([A-Z_]+)\b")
_DESCRIPTOR_ZONE_POS_RE = re.compile(r"\bzonePos=(-?\d+)\b")
_DESCRIPTOR_CARD_ID_RE = re.compile(r"\bcardId=([^\s\]]*)")
_DESCRIPTOR_PLAYER_RE = re.compile(r"\bplayer=(\d+)\b")


def _coverage(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _empty_state() -> dict:
    return {"zone": None, "zone_pos": None, "card_id": None, "player": None}


def _apply_descriptor(state: dict, descriptor: str) -> None:
    zone = _DESCRIPTOR_ZONE_RE.search(descriptor)
    zone_pos = _DESCRIPTOR_ZONE_POS_RE.search(descriptor)
    card_id = _DESCRIPTOR_CARD_ID_RE.search(descriptor)
    player = _DESCRIPTOR_PLAYER_RE.search(descriptor)
    if zone:
        state["zone"] = zone.group(1)
    if zone_pos:
        state["zone_pos"] = int(zone_pos.group(1))
    if card_id and card_id.group(1):
        state["card_id"] = card_id.group(1)
    if player:
        state["player"] = int(player.group(1))


def _apply_full_entity_tag(
    state: dict,
    *,
    tag: str,
    value: str,
    observed_player_ids: set[int],
) -> bool:
    """Apply only source-observed identity/membership fields from one creation block."""
    if tag == "ZONE":
        state["zone"] = value
        return True
    if tag == "ZONE_POSITION":
        try:
            state["zone_pos"] = int(value)
        except ValueError:
            state["zone_pos"] = None
        return True
    if tag == "CONTROLLER":
        try:
            player = int(value)
        except ValueError:
            return False
        if player in observed_player_ids:
            state["player"] = player
            return True
    return False


def _complete_body_pre_state(state: dict) -> bool:
    return (
        isinstance(state.get("zone"), str)
        and isinstance(state.get("zone_pos"), int)
        and isinstance(state.get("card_id"), str)
        and bool(state.get("card_id"))
        and isinstance(state.get("player"), int)
    )


def _missing_pre_state_fields(state: dict) -> list[str]:
    missing = []
    if not isinstance(state.get("zone"), str):
        missing.append("zone")
    if not isinstance(state.get("zone_pos"), int):
        missing.append("zone_pos")
    if not isinstance(state.get("card_id"), str) or not state.get("card_id"):
        missing.append("card_id")
    if not isinstance(state.get("player"), int):
        missing.append("player")
    return missing


def _canonical_payloads(text: str) -> list[str]:
    payloads = []
    for line in text.splitlines():
        if _CANONICAL_POWER_MARKER in line:
            payloads.append(line.split(_CANONICAL_POWER_MARKER, 1)[1])
    return payloads


def _split_game_segments(payloads: list[str]) -> list[list[str]]:
    """Split canonical payloads at CREATE_GAME without discarding a leading segment."""
    segments: list[list[str]] = []
    current: list[str] = []
    for payload in payloads:
        if _CREATE_GAME_RE.search(payload) and current:
            segments.append(current)
            current = []
        current.append(payload)
    if current:
        segments.append(current)
    return segments


def _audit_game_segment(payloads: list[str], segment_index: int) -> Dict:
    # PlayerID validity is local to this CREATE_GAME-bounded segment. Collect all
    # PlayerIDs in the segment before interpreting FULL_ENTITY CONTROLLER tags so
    # validation is order-invariant without allowing cross-game reuse.
    observed_player_ids: set[int] = set()
    for payload in payloads:
        player_record = _PLAYER_RE.search(payload)
        if player_record:
            observed_player_ids.add(int(player_record.group(2)))

    states: dict[str, dict] = {}
    active_full_entity: str | None = None
    full_entity_records = 0
    full_entity_records_with_card_id = 0
    full_entity_state_tags_applied = 0
    full_entity_entities_with_complete_state = 0
    full_entity_completed_entities: set[str] = set()

    zone_changes = 0
    bracketed_zone_changes = 0
    numeric_zone_changes = 0
    zone_changes_with_recoverable_pre_state = 0
    bracketed_zone_changes_with_recoverable_pre_state = 0
    numeric_zone_changes_with_recoverable_pre_state = 0
    numeric_zone_changes_recovered_from_prior_state = 0
    body_like_zone_changes = 0
    numeric_unrecoverable_missing_zone = 0
    numeric_unrecoverable_missing_zone_pos = 0
    numeric_unrecoverable_missing_card_id = 0
    numeric_unrecoverable_missing_player = 0
    numeric_unrecoverable_missing_field_sets: dict[str, int] = {}

    for payload in payloads:
        player_record = _PLAYER_RE.search(payload)
        if player_record:
            active_full_entity = None
            continue

        full_entity = _FULL_ENTITY_RE.search(payload)
        if full_entity:
            entity_id, card_id = full_entity.groups()
            state = states.setdefault(entity_id, _empty_state())
            full_entity_records += 1
            active_full_entity = entity_id
            if card_id:
                state["card_id"] = card_id
                full_entity_records_with_card_id += 1
            continue

        raw_tag = _RAW_TAG_RE.match(payload)
        if raw_tag and active_full_entity is not None:
            tag, value = raw_tag.groups()
            state = states.setdefault(active_full_entity, _empty_state())
            full_entity_state_tags_applied += bool(
                _apply_full_entity_tag(
                    state,
                    tag=tag,
                    value=value,
                    observed_player_ids=observed_player_ids,
                )
            )
            if (
                _complete_body_pre_state(state)
                and active_full_entity not in full_entity_completed_entities
            ):
                full_entity_completed_entities.add(active_full_entity)
                full_entity_entities_with_complete_state += 1
            continue

        active_full_entity = None
        change = _TAG_CHANGE_RE.search(payload)
        if not change:
            continue

        numeric_entity, descriptor, bracketed_entity, tag, value = change.groups()
        entity_id = numeric_entity or bracketed_entity
        state = states.setdefault(entity_id, _empty_state())
        has_descriptor = descriptor is not None
        if has_descriptor:
            _apply_descriptor(state, descriptor)

        if tag == "ZONE":
            zone_changes += 1
            if has_descriptor:
                bracketed_zone_changes += 1
            else:
                numeric_zone_changes += 1

            recoverable = _complete_body_pre_state(state)
            if recoverable:
                zone_changes_with_recoverable_pre_state += 1
                body_like_zone_changes += 1
                if has_descriptor:
                    bracketed_zone_changes_with_recoverable_pre_state += 1
                else:
                    numeric_zone_changes_with_recoverable_pre_state += 1
                    numeric_zone_changes_recovered_from_prior_state += 1
            elif not has_descriptor:
                missing = _missing_pre_state_fields(state)
                if "zone" in missing:
                    numeric_unrecoverable_missing_zone += 1
                if "zone_pos" in missing:
                    numeric_unrecoverable_missing_zone_pos += 1
                if "card_id" in missing:
                    numeric_unrecoverable_missing_card_id += 1
                if "player" in missing:
                    numeric_unrecoverable_missing_player += 1
                key = "+".join(missing)
                numeric_unrecoverable_missing_field_sets[key] = (
                    numeric_unrecoverable_missing_field_sets.get(key, 0) + 1
                )

            state["zone"] = value
            state["zone_pos"] = None
        elif tag == "ZONE_POSITION":
            try:
                state["zone_pos"] = int(value)
            except ValueError:
                state["zone_pos"] = None

    return {
        "segment_index": segment_index,
        "has_create_game_marker": any(_CREATE_GAME_RE.search(p) for p in payloads),
        "canonical_payloads": len(payloads),
        "observed_player_ids": sorted(observed_player_ids),
        "full_entity_records": full_entity_records,
        "full_entity_records_with_card_id": full_entity_records_with_card_id,
        "full_entity_state_tags_applied": full_entity_state_tags_applied,
        "full_entity_entities_with_complete_state": full_entity_entities_with_complete_state,
        "zone_changes": zone_changes,
        "bracketed_zone_changes": bracketed_zone_changes,
        "numeric_zone_changes": numeric_zone_changes,
        "zone_changes_with_recoverable_pre_state": zone_changes_with_recoverable_pre_state,
        "bracketed_zone_changes_with_recoverable_pre_state": bracketed_zone_changes_with_recoverable_pre_state,
        "numeric_zone_changes_with_recoverable_pre_state": numeric_zone_changes_with_recoverable_pre_state,
        "numeric_zone_changes_recovered_from_prior_state": numeric_zone_changes_recovered_from_prior_state,
        "body_like_zone_changes": body_like_zone_changes,
        "numeric_unrecoverable_missing_zone": numeric_unrecoverable_missing_zone,
        "numeric_unrecoverable_missing_zone_pos": numeric_unrecoverable_missing_zone_pos,
        "numeric_unrecoverable_missing_card_id": numeric_unrecoverable_missing_card_id,
        "numeric_unrecoverable_missing_player": numeric_unrecoverable_missing_player,
        "numeric_unrecoverable_missing_field_sets": dict(sorted(numeric_unrecoverable_missing_field_sets.items())),
    }


def _sum_metric(per_game: list[Dict], key: str) -> int:
    return sum(int(game[key]) for game in per_game)


def audit_powerlog_state_recovery(source_content: bytes) -> Dict:
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")
    try:
        text = source_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Power.log must be valid UTF-8 text") from exc

    payloads = _canonical_payloads(text)
    game_segments = _split_game_segments(payloads)
    per_game = [
        _audit_game_segment(segment, index)
        for index, segment in enumerate(game_segments)
    ]

    observed_player_ids = sorted(
        {player_id for game in per_game for player_id in game["observed_player_ids"]}
    )
    full_entity_records = _sum_metric(per_game, "full_entity_records")
    full_entity_records_with_card_id = _sum_metric(per_game, "full_entity_records_with_card_id")
    full_entity_state_tags_applied = _sum_metric(per_game, "full_entity_state_tags_applied")
    full_entity_entities_with_complete_state = _sum_metric(per_game, "full_entity_entities_with_complete_state")
    zone_changes = _sum_metric(per_game, "zone_changes")
    bracketed_zone_changes = _sum_metric(per_game, "bracketed_zone_changes")
    numeric_zone_changes = _sum_metric(per_game, "numeric_zone_changes")
    zone_changes_with_recoverable_pre_state = _sum_metric(per_game, "zone_changes_with_recoverable_pre_state")
    bracketed_zone_changes_with_recoverable_pre_state = _sum_metric(per_game, "bracketed_zone_changes_with_recoverable_pre_state")
    numeric_zone_changes_with_recoverable_pre_state = _sum_metric(per_game, "numeric_zone_changes_with_recoverable_pre_state")
    numeric_zone_changes_recovered_from_prior_state = _sum_metric(per_game, "numeric_zone_changes_recovered_from_prior_state")
    body_like_zone_changes = _sum_metric(per_game, "body_like_zone_changes")
    numeric_unrecoverable_missing_zone = _sum_metric(per_game, "numeric_unrecoverable_missing_zone")
    numeric_unrecoverable_missing_zone_pos = _sum_metric(per_game, "numeric_unrecoverable_missing_zone_pos")
    numeric_unrecoverable_missing_card_id = _sum_metric(per_game, "numeric_unrecoverable_missing_card_id")
    numeric_unrecoverable_missing_player = _sum_metric(per_game, "numeric_unrecoverable_missing_player")
    numeric_unrecoverable_missing_field_sets: dict[str, int] = {}
    for game in per_game:
        for key, value in game["numeric_unrecoverable_missing_field_sets"].items():
            numeric_unrecoverable_missing_field_sets[key] = (
                numeric_unrecoverable_missing_field_sets.get(key, 0) + int(value)
            )

    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "game_boundary_marker": "CREATE_GAME",
        "game_segments": len(per_game),
        "player_id_collection_passes_per_game": 2,
        "player_id_validation_order_invariant": True,
        "player_id_validation_game_local": True,
        "entity_state_reset_at_game_boundary": True,
        "observed_player_ids": observed_player_ids,
        "per_game": per_game,
        "full_entity_records": full_entity_records,
        "full_entity_records_with_card_id": full_entity_records_with_card_id,
        "full_entity_state_tags_applied": full_entity_state_tags_applied,
        "full_entity_entities_with_complete_state": full_entity_entities_with_complete_state,
        "zone_changes": zone_changes,
        "bracketed_zone_changes": bracketed_zone_changes,
        "numeric_zone_changes": numeric_zone_changes,
        "zone_changes_with_recoverable_pre_state": zone_changes_with_recoverable_pre_state,
        "zone_pre_state_recovery_coverage": _coverage(zone_changes_with_recoverable_pre_state, zone_changes),
        "bracketed_zone_changes_with_recoverable_pre_state": bracketed_zone_changes_with_recoverable_pre_state,
        "bracketed_zone_pre_state_recovery_coverage": _coverage(
            bracketed_zone_changes_with_recoverable_pre_state, bracketed_zone_changes
        ),
        "numeric_zone_changes_with_recoverable_pre_state": numeric_zone_changes_with_recoverable_pre_state,
        "numeric_zone_changes_recovered_from_prior_state": numeric_zone_changes_recovered_from_prior_state,
        "numeric_zone_pre_state_recovery_coverage": _coverage(
            numeric_zone_changes_with_recoverable_pre_state, numeric_zone_changes
        ),
        "zone_changes_without_recoverable_pre_state": zone_changes - zone_changes_with_recoverable_pre_state,
        "numeric_zone_changes_without_recoverable_pre_state": numeric_zone_changes - numeric_zone_changes_with_recoverable_pre_state,
        "numeric_unrecoverable_missing_zone": numeric_unrecoverable_missing_zone,
        "numeric_unrecoverable_missing_zone_pos": numeric_unrecoverable_missing_zone_pos,
        "numeric_unrecoverable_missing_card_id": numeric_unrecoverable_missing_card_id,
        "numeric_unrecoverable_missing_player": numeric_unrecoverable_missing_player,
        "numeric_unrecoverable_missing_field_sets": dict(sorted(numeric_unrecoverable_missing_field_sets.items())),
        "body_like_zone_changes": body_like_zone_changes,
        "full_entity_state_recovery_enabled": True,
        "state_recovery_only": True,
        "membership_semantics_frozen": False,
        "pre_post_board_reconstruction_performed": False,
        "candidate_scoring_performed": False,
        "phase_3u_schema_ready": False,
        "ranking_ready": False,
        "blockers": [
            "membership_event_semantics_not_frozen",
            "complete_pre_post_board_reconstruction_not_implemented",
            "conserved_pool_reconstruction_not_defined",
            "calibration_evaluation_split_not_defined",
            "measurement_error_not_characterized",
            "digest_bound_parser_loader_not_implemented",
        ],
    }


def audit_powerlog_state_recovery_file(source: Path) -> Dict:
    return audit_powerlog_state_recovery(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measurement-only conservative Power.log state-recovery probe"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_powerlog_state_recovery_file(args.source)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
