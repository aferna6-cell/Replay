"""Forward-only Phase 3U Power.log post-ZONE observability probe.

Measurement only. For numeric ZONE transitions whose pre-state is incomplete,
measure whether the canonical log explicitly reveals post-change ZONE_POSITION
before the same entity changes ZONE again. Keep provenance channels separate:
(1) direct ZONE_POSITION TAG_CHANGE and (2) later bracketed entity descriptors
whose descriptor zone matches the pending post-ZONE value. Their union is the
explicit post-position ceiling; contradictions are counted, never reconciled.

Future observations never repair pre-state. State, PlayerID validity, and pending
intervals are CREATE_GAME-local. This probe never defines board-membership
semantics, constructs Phase 3U evidence rows, reconstructs conserved pools,
scores candidates, or authorizes ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

from ml.phase_3u_powerlog_state_recovery import (
    _CANONICAL_POWER_MARKER,
    _CREATE_GAME_RE,
    _DESCRIPTOR_CARD_ID_RE,
    _DESCRIPTOR_PLAYER_RE,
    _DESCRIPTOR_ZONE_POS_RE,
    _DESCRIPTOR_ZONE_RE,
    _FULL_ENTITY_RE,
    _PLAYER_RE,
    _RAW_TAG_RE,
    _TAG_CHANGE_RE,
    _apply_descriptor,
    _apply_full_entity_tag,
    _complete_body_pre_state,
    _empty_state,
    _missing_pre_state_fields,
)

PROBE_VERSION = "3u_powerlog_post_state_v2"


def _coverage(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _canonical_payloads(text: str) -> list[str]:
    return [
        line.split(_CANONICAL_POWER_MARKER, 1)[1]
        for line in text.splitlines()
        if _CANONICAL_POWER_MARKER in line
    ]


def _split_game_segments(payloads: list[str]) -> list[list[str]]:
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


def _descriptor_card_id(descriptor: str | None) -> str | None:
    if descriptor is None:
        return None
    match = _DESCRIPTOR_CARD_ID_RE.search(descriptor)
    return match.group(1) if match and match.group(1) else None


def _descriptor_zone(descriptor: str | None) -> str | None:
    if descriptor is None:
        return None
    match = _DESCRIPTOR_ZONE_RE.search(descriptor)
    return match.group(1) if match else None


def _descriptor_zone_pos(descriptor: str | None) -> int | None:
    if descriptor is None:
        return None
    match = _DESCRIPTOR_ZONE_POS_RE.search(descriptor)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _new_pending_event(
    *, segment_index: int, entity_id: str, ordinal: int, post_zone: str, missing: list[str]
) -> dict:
    return {
        "segment_index": segment_index,
        "entity_id": entity_id,
        "zone_ordinal": ordinal,
        "post_zone": post_zone,
        "missing_pre_state_fields": missing,
        "post_zone_position_observed": False,
        "post_zone_position": None,
        "post_zone_position_distance": None,
        "post_zone_position_source": None,
        "tag_zone_position_observed": False,
        "tag_zone_position": None,
        "tag_zone_position_distance": None,
        "descriptor_zone_position_observed": False,
        "descriptor_zone_position": None,
        "descriptor_zone_position_distance": None,
        "descriptor_zone_mismatch_count": 0,
        "position_contradiction": False,
        "card_id_observed_after_distance": None,
        "closed_by_next_zone": False,
        "next_zone_distance": None,
    }


def _record_tag_position(event: dict, *, zone_pos: int, distance: int) -> None:
    if not event["tag_zone_position_observed"]:
        event["tag_zone_position_observed"] = True
        event["tag_zone_position"] = zone_pos
        event["tag_zone_position_distance"] = distance
    if event["descriptor_zone_position_observed"] and event["descriptor_zone_position"] != zone_pos:
        event["position_contradiction"] = True
    if not event["post_zone_position_observed"]:
        event["post_zone_position_observed"] = True
        event["post_zone_position"] = zone_pos
        event["post_zone_position_distance"] = distance
        event["post_zone_position_source"] = "tag"


def _record_descriptor_position(
    event: dict, *, descriptor: str | None, distance: int
) -> None:
    descriptor_zone = _descriptor_zone(descriptor)
    descriptor_pos = _descriptor_zone_pos(descriptor)
    if descriptor_zone != event["post_zone"]:
        if descriptor_pos is not None:
            event["descriptor_zone_mismatch_count"] += 1
        return
    if descriptor_pos is None:
        return
    if not event["descriptor_zone_position_observed"]:
        event["descriptor_zone_position_observed"] = True
        event["descriptor_zone_position"] = descriptor_pos
        event["descriptor_zone_position_distance"] = distance
    if event["tag_zone_position_observed"] and event["tag_zone_position"] != descriptor_pos:
        event["position_contradiction"] = True
    if not event["post_zone_position_observed"]:
        event["post_zone_position_observed"] = True
        event["post_zone_position"] = descriptor_pos
        event["post_zone_position_distance"] = distance
        event["post_zone_position_source"] = "descriptor"


def _audit_game_segment(payloads: list[str], segment_index: int) -> Dict:
    observed_player_ids: set[int] = set()
    for payload in payloads:
        player_record = _PLAYER_RE.search(payload)
        if player_record:
            observed_player_ids.add(int(player_record.group(2)))

    states: dict[str, dict] = {}
    active_full_entity: str | None = None
    pending: dict[str, dict] = {}
    intervals: list[dict] = []
    unresolved_numeric_zone_events = 0

    for ordinal, payload in enumerate(payloads):
        player_record = _PLAYER_RE.search(payload)
        if player_record:
            active_full_entity = None
            continue

        full_entity = _FULL_ENTITY_RE.search(payload)
        if full_entity:
            entity_id, card_id = full_entity.groups()
            state = states.setdefault(entity_id, _empty_state())
            active_full_entity = entity_id
            if card_id:
                state["card_id"] = card_id
                event = pending.get(entity_id)
                if event is not None and event["card_id_observed_after_distance"] is None:
                    event["card_id_observed_after_distance"] = ordinal - event["zone_ordinal"]
            continue

        raw_tag = _RAW_TAG_RE.match(payload)
        if raw_tag and active_full_entity is not None:
            tag, value = raw_tag.groups()
            state = states.setdefault(active_full_entity, _empty_state())
            _apply_full_entity_tag(
                state,
                tag=tag,
                value=value,
                observed_player_ids=observed_player_ids,
            )
            continue

        active_full_entity = None
        change = _TAG_CHANGE_RE.search(payload)
        if not change:
            continue

        numeric_entity, descriptor, bracketed_entity, tag, value = change.groups()
        entity_id = numeric_entity or bracketed_entity
        state = states.setdefault(entity_id, _empty_state())
        has_descriptor = descriptor is not None

        if tag == "ZONE":
            prior_pending = pending.pop(entity_id, None)
            if prior_pending is not None:
                prior_pending["closed_by_next_zone"] = True
                prior_pending["next_zone_distance"] = ordinal - prior_pending["zone_ordinal"]
                intervals.append(prior_pending)

            if has_descriptor:
                _apply_descriptor(state, descriptor)

            recoverable = _complete_body_pre_state(state)
            missing = [] if recoverable else _missing_pre_state_fields(state)
            if not has_descriptor and not recoverable:
                unresolved_numeric_zone_events += 1
                pending[entity_id] = _new_pending_event(
                    segment_index=segment_index,
                    entity_id=entity_id,
                    ordinal=ordinal,
                    post_zone=value,
                    missing=missing,
                )

            state["zone"] = value
            state["zone_pos"] = None
            continue

        event = pending.get(entity_id)
        if has_descriptor:
            if event is not None:
                distance = ordinal - event["zone_ordinal"]
                _record_descriptor_position(event, descriptor=descriptor, distance=distance)
                if event["card_id_observed_after_distance"] is None:
                    card_id = _descriptor_card_id(descriptor)
                    if card_id:
                        event["card_id_observed_after_distance"] = distance
            _apply_descriptor(state, descriptor)

        if tag == "ZONE_POSITION":
            try:
                zone_pos = int(value)
            except ValueError:
                state["zone_pos"] = None
                continue
            state["zone_pos"] = zone_pos
            if event is not None:
                _record_tag_position(
                    event,
                    zone_pos=zone_pos,
                    distance=ordinal - event["zone_ordinal"],
                )

    intervals.extend(pending.values())
    intervals.sort(key=lambda row: row["zone_ordinal"])

    tag_count = sum(bool(row["tag_zone_position_observed"]) for row in intervals)
    descriptor_count = sum(bool(row["descriptor_zone_position_observed"]) for row in intervals)
    union_count = sum(bool(row["post_zone_position_observed"]) for row in intervals)
    contradiction_count = sum(bool(row["position_contradiction"]) for row in intervals)
    descriptor_mismatch_events = sum(
        bool(row["descriptor_zone_mismatch_count"]) for row in intervals
    )
    missing_card_pre = [row for row in intervals if "card_id" in row["missing_pre_state_fields"]]
    missing_card_pre_with_forward_card = sum(
        row["card_id_observed_after_distance"] is not None for row in missing_card_pre
    )
    closed_before_post_pos = sum(
        bool(row["closed_by_next_zone"]) and not bool(row["post_zone_position_observed"])
        for row in intervals
    )

    return {
        "segment_index": segment_index,
        "canonical_payloads": len(payloads),
        "observed_player_ids": sorted(observed_player_ids),
        "unresolved_numeric_zone_events": unresolved_numeric_zone_events,
        "tag_zone_position_observed_before_next_zone": tag_count,
        "tag_zone_position_coverage": _coverage(tag_count, unresolved_numeric_zone_events),
        "descriptor_zone_position_observed_before_next_zone": descriptor_count,
        "descriptor_zone_position_coverage": _coverage(descriptor_count, unresolved_numeric_zone_events),
        "post_zone_position_observed_before_next_zone": union_count,
        "post_zone_position_coverage": _coverage(union_count, unresolved_numeric_zone_events),
        "position_contradictions": contradiction_count,
        "descriptor_zone_mismatch_events": descriptor_mismatch_events,
        "missing_card_id_pre_events": len(missing_card_pre),
        "missing_card_id_pre_with_forward_card_id": missing_card_pre_with_forward_card,
        "forward_card_id_coverage_for_missing_pre_card": _coverage(
            missing_card_pre_with_forward_card, len(missing_card_pre)
        ),
        "closed_by_next_zone_without_post_position": closed_before_post_pos,
        "intervals": intervals,
    }


def audit_powerlog_post_state(source_content: bytes) -> Dict:
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")
    try:
        text = source_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Power.log must be valid UTF-8 text") from exc

    payloads = _canonical_payloads(text)
    per_game = [
        _audit_game_segment(segment, index)
        for index, segment in enumerate(_split_game_segments(payloads))
    ]
    unresolved = sum(game["unresolved_numeric_zone_events"] for game in per_game)
    tag_count = sum(game["tag_zone_position_observed_before_next_zone"] for game in per_game)
    descriptor_count = sum(
        game["descriptor_zone_position_observed_before_next_zone"] for game in per_game
    )
    union_count = sum(game["post_zone_position_observed_before_next_zone"] for game in per_game)
    contradictions = sum(game["position_contradictions"] for game in per_game)
    descriptor_mismatch_events = sum(game["descriptor_zone_mismatch_events"] for game in per_game)
    missing_card = sum(game["missing_card_id_pre_events"] for game in per_game)
    forward_card = sum(game["missing_card_id_pre_with_forward_card_id"] for game in per_game)
    closed_without_pos = sum(game["closed_by_next_zone_without_post_position"] for game in per_game)

    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "game_boundary_marker": "CREATE_GAME",
        "game_segments": len(per_game),
        "per_game": per_game,
        "unresolved_numeric_zone_events": unresolved,
        "tag_zone_position_observed_before_next_zone": tag_count,
        "tag_zone_position_coverage": _coverage(tag_count, unresolved),
        "descriptor_zone_position_observed_before_next_zone": descriptor_count,
        "descriptor_zone_position_coverage": _coverage(descriptor_count, unresolved),
        "post_zone_position_observed_before_next_zone": union_count,
        "post_zone_position_coverage": _coverage(union_count, unresolved),
        "position_contradictions": contradictions,
        "descriptor_zone_mismatch_events": descriptor_mismatch_events,
        "missing_card_id_pre_events": missing_card,
        "missing_card_id_pre_with_forward_card_id": forward_card,
        "forward_card_id_coverage_for_missing_pre_card": _coverage(forward_card, missing_card),
        "closed_by_next_zone_without_post_position": closed_without_pos,
        "forward_observation_only": True,
        "pre_state_repaired_from_future_events": False,
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


def audit_powerlog_post_state_file(source: Path) -> Dict:
    return audit_powerlog_post_state(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only Power.log post-ZONE observability probe")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_powerlog_post_state_file(args.source)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
