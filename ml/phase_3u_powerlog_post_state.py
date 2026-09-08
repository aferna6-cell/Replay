"""Forward-only Phase 3U Power.log post-ZONE observability probe.

Measurement only. For numeric ZONE transitions whose pre-state is incomplete,
measure explicit post-change ZONE_POSITION evidence before the same entity changes
ZONE again. Preserve ordered observation provenance. A contradiction exists only
when tag and descriptor positions disagree on the same canonical record; differing
positions observed later are state evolution, not contradictory evidence.

Future observations never repair pre-state. State, PlayerID validity, and pending
intervals are CREATE_GAME-local. This probe never defines board-membership
semantics, constructs Phase 3U evidence rows, scores candidates, or authorizes
ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from ml.phase_3u_powerlog_state_recovery import (
    _CANONICAL_POWER_MARKER, _CREATE_GAME_RE, _DESCRIPTOR_CARD_ID_RE,
    _DESCRIPTOR_ZONE_POS_RE, _DESCRIPTOR_ZONE_RE, _FULL_ENTITY_RE, _PLAYER_RE,
    _RAW_TAG_RE, _TAG_CHANGE_RE, _apply_descriptor, _apply_full_entity_tag,
    _complete_body_pre_state, _empty_state, _missing_pre_state_fields,
)

PROBE_VERSION = "3u_powerlog_post_state_v4"


def _coverage(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def _canonical_payloads(text: str) -> list[str]:
    return [line.split(_CANONICAL_POWER_MARKER, 1)[1] for line in text.splitlines()
            if _CANONICAL_POWER_MARKER in line]


def _split_game_segments(payloads: list[str]) -> list[list[str]]:
    segments, current = [], []
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


def _new_pending_event(*, segment_index: int, entity_id: str, ordinal: int,
                       post_zone: str, missing: list[str]) -> dict:
    return {
        "segment_index": segment_index, "entity_id": entity_id,
        "zone_ordinal": ordinal, "post_zone": post_zone,
        "missing_pre_state_fields": missing,
        "post_zone_position_observed": False, "post_zone_position": None,
        "post_zone_position_distance": None, "post_zone_position_source": None,
        "tag_zone_position_observed": False, "tag_zone_position": None,
        "tag_zone_position_distance": None,
        "descriptor_zone_position_observed": False,
        "descriptor_zone_position": None, "descriptor_zone_position_distance": None,
        "full_entity_tag_zone_position_observed": False,
        "full_entity_tag_zone_position": None,
        "full_entity_tag_zone_position_distance": None,
        "descriptor_zone_mismatch_count": 0,
        "full_entity_zone_mismatch_count": 0,
        "position_observations": [], "same_ordinal_position_conflicts": 0,
        "position_evolution_events": 0, "position_contradiction": False,
        "card_id_observed_after_distance": None,
        "closed_by_next_zone": False, "next_zone_distance": None,
    }


def _append_position(event: dict, *, ordinal: int, source: str, position: int) -> None:
    distance = ordinal - event["zone_ordinal"]
    observation = {"ordinal": ordinal, "distance": distance,
                   "source": source, "position": position}
    event["position_observations"].append(observation)

    same_ordinal = [o for o in event["position_observations"][:-1]
                    if o["ordinal"] == ordinal and o["source"] != source]
    if any(o["position"] != position for o in same_ordinal):
        event["same_ordinal_position_conflicts"] += 1
        event["position_contradiction"] = True

    prior_ordinals = [o for o in event["position_observations"][:-1]
                      if o["ordinal"] < ordinal]
    if prior_ordinals and prior_ordinals[-1]["position"] != position:
        event["position_evolution_events"] += 1

    if source == "tag" and not event["tag_zone_position_observed"]:
        event["tag_zone_position_observed"] = True
        event["tag_zone_position"] = position
        event["tag_zone_position_distance"] = distance
    if source == "descriptor" and not event["descriptor_zone_position_observed"]:
        event["descriptor_zone_position_observed"] = True
        event["descriptor_zone_position"] = position
        event["descriptor_zone_position_distance"] = distance
    if source == "full_entity_tag" and not event["full_entity_tag_zone_position_observed"]:
        event["full_entity_tag_zone_position_observed"] = True
        event["full_entity_tag_zone_position"] = position
        event["full_entity_tag_zone_position_distance"] = distance
    if not event["post_zone_position_observed"]:
        event["post_zone_position_observed"] = True
        event["post_zone_position"] = position
        event["post_zone_position_distance"] = distance
        event["post_zone_position_source"] = source


def _summary(intervals: list[dict]) -> dict:
    unresolved = len(intervals)
    tag = sum(bool(r["tag_zone_position_observed"]) for r in intervals)
    desc = sum(bool(r["descriptor_zone_position_observed"]) for r in intervals)
    full_entity = sum(bool(r["full_entity_tag_zone_position_observed"]) for r in intervals)
    union = sum(bool(r["post_zone_position_observed"]) for r in intervals)
    closed = sum(bool(r["closed_by_next_zone"]) and not r["post_zone_position_observed"]
                 for r in intervals)
    conflicts = sum(r["same_ordinal_position_conflicts"] for r in intervals)
    evolution = sum(r["position_evolution_events"] for r in intervals)
    return {
        "unresolved": unresolved,
        "tag": tag, "descriptor": desc, "full_entity_tag": full_entity, "union": union,
        "closed_without_position": closed,
        "tag_coverage": _coverage(tag, unresolved),
        "descriptor_coverage": _coverage(desc, unresolved),
        "full_entity_tag_coverage": _coverage(full_entity, unresolved),
        "union_coverage": _coverage(union, unresolved),
        "same_ordinal_position_conflicts": conflicts,
        "position_evolution_events": evolution,
    }


def _audit_game_segment(payloads: list[str], segment_index: int) -> Dict:
    observed_player_ids = set()
    for payload in payloads:
        player_record = _PLAYER_RE.search(payload)
        if player_record:
            observed_player_ids.add(int(player_record.group(2)))

    states, pending, intervals = {}, {}, []
    active_full_entity = None
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
            _apply_full_entity_tag(state, tag=tag, value=value,
                                   observed_player_ids=observed_player_ids)
            event = pending.get(active_full_entity)
            if tag == "ZONE_POSITION" and event is not None:
                try:
                    zone_pos = int(value)
                except ValueError:
                    zone_pos = None
                if zone_pos is not None:
                    if state.get("zone") == event["post_zone"]:
                        _append_position(event, ordinal=ordinal,
                                         source="full_entity_tag", position=zone_pos)
                    else:
                        event["full_entity_zone_mismatch_count"] += 1
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
            prior = pending.pop(entity_id, None)
            if prior is not None:
                prior["closed_by_next_zone"] = True
                prior["next_zone_distance"] = ordinal - prior["zone_ordinal"]
                intervals.append(prior)
            if has_descriptor:
                _apply_descriptor(state, descriptor)
            recoverable = _complete_body_pre_state(state)
            missing = [] if recoverable else _missing_pre_state_fields(state)
            if not has_descriptor and not recoverable:
                pending[entity_id] = _new_pending_event(
                    segment_index=segment_index, entity_id=entity_id,
                    ordinal=ordinal, post_zone=value, missing=missing)
            state["zone"] = value
            state["zone_pos"] = None
            continue

        event = pending.get(entity_id)
        descriptor_pos = None
        if has_descriptor:
            if event is not None:
                distance = ordinal - event["zone_ordinal"]
                descriptor_zone = _descriptor_zone(descriptor)
                descriptor_pos = _descriptor_zone_pos(descriptor)
                if descriptor_pos is not None and descriptor_zone != event["post_zone"]:
                    event["descriptor_zone_mismatch_count"] += 1
                elif descriptor_pos is not None:
                    _append_position(event, ordinal=ordinal, source="descriptor",
                                     position=descriptor_pos)
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
                _append_position(event, ordinal=ordinal, source="tag", position=zone_pos)

    intervals.extend(pending.values())
    intervals.sort(key=lambda r: r["zone_ordinal"])
    missing_card = [r for r in intervals if "card_id" in r["missing_pre_state_fields"]]
    forward_card = sum(r["card_id_observed_after_distance"] is not None for r in missing_card)
    by_zone = defaultdict(list)
    for row in intervals:
        by_zone[row["post_zone"]].append(row)
    summary = _summary(intervals)
    return {
        "segment_index": segment_index, "canonical_payloads": len(payloads),
        "observed_player_ids": sorted(observed_player_ids),
        "unresolved_numeric_zone_events": len(intervals),
        "tag_zone_position_observed_before_next_zone": summary["tag"],
        "tag_zone_position_coverage": summary["tag_coverage"],
        "descriptor_zone_position_observed_before_next_zone": summary["descriptor"],
        "descriptor_zone_position_coverage": summary["descriptor_coverage"],
        "full_entity_tag_zone_position_observed_before_next_zone": summary["full_entity_tag"],
        "full_entity_tag_zone_position_coverage": summary["full_entity_tag_coverage"],
        "post_zone_position_observed_before_next_zone": summary["union"],
        "post_zone_position_coverage": summary["union_coverage"],
        "position_contradictions": summary["same_ordinal_position_conflicts"],
        "same_ordinal_position_conflicts": summary["same_ordinal_position_conflicts"],
        "position_evolution_events": summary["position_evolution_events"],
        "descriptor_zone_mismatch_events": sum(bool(r["descriptor_zone_mismatch_count"]) for r in intervals),
        "full_entity_zone_mismatch_events": sum(bool(r["full_entity_zone_mismatch_count"]) for r in intervals),
        "missing_card_id_pre_events": len(missing_card),
        "missing_card_id_pre_with_forward_card_id": forward_card,
        "forward_card_id_coverage_for_missing_pre_card": _coverage(forward_card, len(missing_card)),
        "closed_by_next_zone_without_post_position": summary["closed_without_position"],
        "by_zone": {zone: _summary(rows) for zone, rows in sorted(by_zone.items())},
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
    per_game = [_audit_game_segment(seg, i) for i, seg in enumerate(
        _split_game_segments(_canonical_payloads(text)))]
    all_intervals = [row for game in per_game for row in game["intervals"]]
    summary = _summary(all_intervals)
    by_zone = defaultdict(list)
    for row in all_intervals:
        by_zone[row["post_zone"]].append(row)
    missing_card = [r for r in all_intervals if "card_id" in r["missing_pre_state_fields"]]
    forward_card = sum(r["card_id_observed_after_distance"] is not None for r in missing_card)
    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content), "canonical_stream": "GameState.DebugPrintPower",
        "game_boundary_marker": "CREATE_GAME", "game_segments": len(per_game),
        "per_game": per_game, "unresolved_numeric_zone_events": len(all_intervals),
        "tag_zone_position_observed_before_next_zone": summary["tag"],
        "tag_zone_position_coverage": summary["tag_coverage"],
        "descriptor_zone_position_observed_before_next_zone": summary["descriptor"],
        "descriptor_zone_position_coverage": summary["descriptor_coverage"],
        "full_entity_tag_zone_position_observed_before_next_zone": summary["full_entity_tag"],
        "full_entity_tag_zone_position_coverage": summary["full_entity_tag_coverage"],
        "post_zone_position_observed_before_next_zone": summary["union"],
        "post_zone_position_coverage": summary["union_coverage"],
        "position_contradictions": summary["same_ordinal_position_conflicts"],
        "same_ordinal_position_conflicts": summary["same_ordinal_position_conflicts"],
        "position_evolution_events": summary["position_evolution_events"],
        "descriptor_zone_mismatch_events": sum(g["descriptor_zone_mismatch_events"] for g in per_game),
        "full_entity_zone_mismatch_events": sum(g["full_entity_zone_mismatch_events"] for g in per_game),
        "missing_card_id_pre_events": len(missing_card),
        "missing_card_id_pre_with_forward_card_id": forward_card,
        "forward_card_id_coverage_for_missing_pre_card": _coverage(forward_card, len(missing_card)),
        "closed_by_next_zone_without_post_position": summary["closed_without_position"],
        "by_zone": {zone: _summary(rows) for zone, rows in sorted(by_zone.items())},
        "forward_observation_only": True, "pre_state_repaired_from_future_events": False,
        "membership_semantics_frozen": False, "pre_post_board_reconstruction_performed": False,
        "candidate_scoring_performed": False, "phase_3u_schema_ready": False,
        "ranking_ready": False,
        "blockers": ["membership_event_semantics_not_frozen",
                     "complete_pre_post_board_reconstruction_not_implemented",
                     "conserved_pool_reconstruction_not_defined",
                     "calibration_evaluation_split_not_defined",
                     "measurement_error_not_characterized",
                     "digest_bound_parser_loader_not_implemented"],
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
