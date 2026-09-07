"""Forward-only Phase 3U Power.log post-ZONE observability probe.

This measurement-only probe asks a narrow question left open by the conservative
pre-state recovery audit: when a numeric ZONE transition lacks recoverable
pre-change state, does the canonical log explicitly reveal the entity's new
ZONE_POSITION (and any previously missing CardID) before that entity changes
ZONE again?

Observations are forward-only. A later ZONE_POSITION is evidence about post-change
state only and is never used to repair or relabel the transition's pre-state.
State, PlayerID validity, and pending intervals are CREATE_GAME-local. The probe
never defines board-membership semantics, constructs Phase 3U evidence rows,
reconstructs conserved pools, scores candidates, or authorizes ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

PROBE_VERSION = "3u_powerlog_post_state_v1"


def _coverage(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _canonical_payloads(text: str) -> list[str]:
    payloads: list[str] = []
    for line in text.splitlines():
        if _CANONICAL_POWER_MARKER in line:
            payloads.append(line.split(_CANONICAL_POWER_MARKER, 1)[1])
    return payloads


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
    if match and match.group(1):
        return match.group(1)
    return None


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
                pending[entity_id] = {
                    "segment_index": segment_index,
                    "entity_id": entity_id,
                    "zone_ordinal": ordinal,
                    "post_zone": value,
                    "missing_pre_state_fields": missing,
                    "post_zone_position_observed": False,
                    "post_zone_position": None,
                    "post_zone_position_distance": None,
                    "card_id_observed_after_distance": None,
                    "closed_by_next_zone": False,
                    "next_zone_distance": None,
                }

            state["zone"] = value
            state["zone_pos"] = None
            continue

        if has_descriptor:
            _apply_descriptor(state, descriptor)
            event = pending.get(entity_id)
            if event is not None and event["card_id_observed_after_distance"] is None:
                card_id = _descriptor_card_id(descriptor)
                if card_id:
                    event["card_id_observed_after_distance"] = ordinal - event["zone_ordinal"]

        if tag == "ZONE_POSITION":
            try:
                zone_pos = int(value)
            except ValueError:
                state["zone_pos"] = None
                continue
            state["zone_pos"] = zone_pos
            event = pending.get(entity_id)
            if event is not None and not event["post_zone_position_observed"]:
                event["post_zone_position_observed"] = True
                event["post_zone_position"] = zone_pos
                event["post_zone_position_distance"] = ordinal - event["zone_ordinal"]

    for event in pending.values():
        intervals.append(event)

    intervals.sort(key=lambda row: row["zone_ordinal"])
    with_post_pos = sum(bool(row["post_zone_position_observed"]) for row in intervals)
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
        "post_zone_position_observed_before_next_zone": with_post_pos,
        "post_zone_position_coverage": _coverage(with_post_pos, unresolved_numeric_zone_events),
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
    with_post_pos = sum(game["post_zone_position_observed_before_next_zone"] for game in per_game)
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
        "post_zone_position_observed_before_next_zone": with_post_pos,
        "post_zone_position_coverage": _coverage(with_post_pos, unresolved),
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
