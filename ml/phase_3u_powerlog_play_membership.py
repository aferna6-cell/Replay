"""Measurement-only Phase 3U Power.log PLAY-membership observability probe.

Membership, body identity, controller identity, and board position are deliberately
measured as separate observables. Canonical GameState ZONE=PLAY records establish
PLAY membership directly; the next source-observed ZONE for the same entity closes
that interval. State and intervals are CREATE_GAME-local, and mirrored
PowerTaskList records are ignored.

This probe does not infer board order, repair missing evidence from future games,
construct Phase 3U evidence rows, score candidates, or authorize ranking.
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
)

PROBE_VERSION = "3u_powerlog_play_membership_v1"


def _coverage(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


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


def _descriptor_fields(descriptor: str | None) -> tuple[str | None, int | None, str | None, int | None]:
    if descriptor is None:
        return None, None, None, None
    zone_m = _DESCRIPTOR_ZONE_RE.search(descriptor)
    pos_m = _DESCRIPTOR_ZONE_POS_RE.search(descriptor)
    card_m = _DESCRIPTOR_CARD_ID_RE.search(descriptor)
    player_m = _DESCRIPTOR_PLAYER_RE.search(descriptor)
    zone = zone_m.group(1) if zone_m else None
    pos = int(pos_m.group(1)) if pos_m else None
    card = card_m.group(1) if card_m and card_m.group(1) else None
    player = int(player_m.group(1)) if player_m else None
    return zone, pos, card, player


def _empty_state() -> dict:
    return {"zone": None, "card_id": None, "player": None}


def _new_interval(*, segment_index: int, entity_id: str, ordinal: int,
                  source: str, state: dict) -> dict:
    return {
        "segment_index": segment_index,
        "entity_id": entity_id,
        "play_start_ordinal": ordinal,
        "membership_source": source,
        "closed_by_zone": False,
        "play_end_ordinal": None,
        "exit_zone": None,
        "card_id_known_at_entry": state.get("card_id"),
        "player_known_at_entry": state.get("player"),
        "forward_card_id": None,
        "forward_card_id_distance": None,
        "forward_player": None,
        "forward_player_distance": None,
        "explicit_position": None,
        "explicit_position_distance": None,
        "position_source": None,
    }


def _record_card(interval: dict, *, ordinal: int, card_id: str | None) -> None:
    if card_id and interval["forward_card_id"] is None:
        interval["forward_card_id"] = card_id
        interval["forward_card_id_distance"] = ordinal - interval["play_start_ordinal"]


def _record_player(interval: dict, *, ordinal: int, player: int | None,
                   observed_player_ids: set[int]) -> None:
    if player is not None and player in observed_player_ids and interval["forward_player"] is None:
        interval["forward_player"] = player
        interval["forward_player_distance"] = ordinal - interval["play_start_ordinal"]


def _record_position(interval: dict, *, ordinal: int, position: int | None,
                     source: str) -> None:
    if position is not None and interval["explicit_position"] is None:
        interval["explicit_position"] = position
        interval["explicit_position_distance"] = ordinal - interval["play_start_ordinal"]
        interval["position_source"] = source


def _close_interval(pending: dict[str, dict], intervals: list[dict], *,
                    entity_id: str, ordinal: int, exit_zone: str) -> None:
    prior = pending.pop(entity_id, None)
    if prior is None:
        return
    prior["closed_by_zone"] = True
    prior["play_end_ordinal"] = ordinal
    prior["exit_zone"] = exit_zone
    intervals.append(prior)


def _finalize(interval: dict) -> dict:
    card = interval["forward_card_id"] or interval["card_id_known_at_entry"]
    player = interval["forward_player"] or interval["player_known_at_entry"]
    out = dict(interval)
    out["card_id_grounded"] = card is not None
    out["player_grounded"] = player is not None
    out["identity_grounded"] = card is not None and player is not None
    out["position_grounded"] = interval["explicit_position"] is not None
    out["identity_grounded_without_position"] = (
        out["identity_grounded"] and not out["position_grounded"]
    )
    return out


def _summary(intervals: list[dict]) -> dict:
    n = len(intervals)
    tag = sum(r["membership_source"] == "tag_change" for r in intervals)
    full = sum(r["membership_source"] == "full_entity_tag" for r in intervals)
    closed = sum(bool(r["closed_by_zone"]) for r in intervals)
    card = sum(bool(r["card_id_grounded"]) for r in intervals)
    player = sum(bool(r["player_grounded"]) for r in intervals)
    identity = sum(bool(r["identity_grounded"]) for r in intervals)
    pos = sum(bool(r["position_grounded"]) for r in intervals)
    identity_no_pos = sum(bool(r["identity_grounded_without_position"]) for r in intervals)
    return {
        "play_membership_intervals": n,
        "tag_change_membership_intervals": tag,
        "full_entity_tag_membership_intervals": full,
        "closed_by_next_zone": closed,
        "card_id_grounded": card,
        "player_grounded": player,
        "identity_grounded": identity,
        "explicit_position_grounded": pos,
        "identity_grounded_without_position": identity_no_pos,
        "card_id_coverage": _coverage(card, n),
        "player_coverage": _coverage(player, n),
        "identity_coverage": _coverage(identity, n),
        "explicit_position_coverage": _coverage(pos, n),
        "identity_without_position_coverage": _coverage(identity_no_pos, n),
    }


def _audit_segment(payloads: list[str], segment_index: int) -> dict:
    observed_player_ids: set[int] = set()
    for payload in payloads:
        record = _PLAYER_RE.search(payload)
        if record:
            observed_player_ids.add(int(record.group(2)))

    states: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    intervals: list[dict] = []
    active_full_entity: str | None = None

    for ordinal, payload in enumerate(payloads):
        player_record = _PLAYER_RE.search(payload)
        if player_record:
            active_full_entity = None
            continue

        full_entity = _FULL_ENTITY_RE.search(payload)
        if full_entity:
            entity_id, card_id = full_entity.groups()
            active_full_entity = entity_id
            state = states.setdefault(entity_id, _empty_state())
            if card_id:
                state["card_id"] = card_id
                if entity_id in pending:
                    _record_card(pending[entity_id], ordinal=ordinal, card_id=card_id)
            continue

        raw_tag = _RAW_TAG_RE.match(payload)
        if raw_tag and active_full_entity is not None:
            tag, value = raw_tag.groups()
            entity_id = active_full_entity
            state = states.setdefault(entity_id, _empty_state())
            if tag == "ZONE":
                _close_interval(pending, intervals, entity_id=entity_id,
                                ordinal=ordinal, exit_zone=value)
                state["zone"] = value
                if value == "PLAY":
                    pending[entity_id] = _new_interval(
                        segment_index=segment_index, entity_id=entity_id,
                        ordinal=ordinal, source="full_entity_tag", state=state,
                    )
                    _record_card(pending[entity_id], ordinal=ordinal,
                                 card_id=state.get("card_id"))
                    _record_player(pending[entity_id], ordinal=ordinal,
                                   player=state.get("player"),
                                   observed_player_ids=observed_player_ids)
            elif tag == "CONTROLLER":
                try:
                    player = int(value)
                except ValueError:
                    player = None
                if player is not None and player in observed_player_ids:
                    state["player"] = player
                    if entity_id in pending:
                        _record_player(pending[entity_id], ordinal=ordinal, player=player,
                                       observed_player_ids=observed_player_ids)
            elif tag == "ZONE_POSITION" and entity_id in pending and state.get("zone") == "PLAY":
                try:
                    position = int(value)
                except ValueError:
                    position = None
                _record_position(pending[entity_id], ordinal=ordinal,
                                 position=position, source="full_entity_tag")
            continue

        active_full_entity = None
        change = _TAG_CHANGE_RE.search(payload)
        if not change:
            continue
        numeric_entity, descriptor, bracketed_entity, tag, value = change.groups()
        entity_id = numeric_entity or bracketed_entity
        state = states.setdefault(entity_id, _empty_state())
        descriptor_zone, descriptor_pos, descriptor_card, descriptor_player = _descriptor_fields(descriptor)

        # Descriptor fields on a ZONE mutation describe pre-mutation state. CardID and
        # player are entity identity and may be retained at entry; descriptor position
        # is never treated as post-PLAY position on that mutation.
        if descriptor_card:
            state["card_id"] = descriptor_card
        if descriptor_player is not None and descriptor_player in observed_player_ids:
            state["player"] = descriptor_player

        if tag == "ZONE":
            _close_interval(pending, intervals, entity_id=entity_id,
                            ordinal=ordinal, exit_zone=value)
            state["zone"] = value
            if value == "PLAY":
                pending[entity_id] = _new_interval(
                    segment_index=segment_index, entity_id=entity_id,
                    ordinal=ordinal, source="tag_change", state=state,
                )
            continue

        interval = pending.get(entity_id)
        if interval is not None and descriptor is not None and descriptor_zone == "PLAY":
            _record_card(interval, ordinal=ordinal, card_id=descriptor_card)
            _record_player(interval, ordinal=ordinal, player=descriptor_player,
                           observed_player_ids=observed_player_ids)
            if tag != "ZONE_POSITION":
                _record_position(interval, ordinal=ordinal, position=descriptor_pos,
                                 source="descriptor_snapshot")

        if tag == "ZONE_POSITION" and interval is not None and state.get("zone") == "PLAY":
            try:
                position = int(value)
            except ValueError:
                position = None
            _record_position(interval, ordinal=ordinal, position=position,
                             source="tag_change")
        elif tag == "CONTROLLER":
            try:
                player = int(value)
            except ValueError:
                player = None
            if player is not None and player in observed_player_ids:
                state["player"] = player
                if interval is not None:
                    _record_player(interval, ordinal=ordinal, player=player,
                                   observed_player_ids=observed_player_ids)

    intervals.extend(pending.values())
    finalized = [_finalize(row) for row in sorted(intervals, key=lambda r: r["play_start_ordinal"])]
    return {
        "segment_index": segment_index,
        "canonical_payloads": len(payloads),
        "observed_player_ids": sorted(observed_player_ids),
        **_summary(finalized),
        "intervals": finalized,
    }


def audit_powerlog_play_membership(source_content: bytes) -> Dict:
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")
    try:
        text = source_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Power.log must be valid UTF-8 text") from exc

    segments = _split_game_segments(_canonical_payloads(text))
    per_game = [_audit_segment(seg, i) for i, seg in enumerate(segments)]
    intervals = [row for game in per_game for row in game["intervals"]]
    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "game_boundary_marker": "CREATE_GAME",
        "game_segments": len(per_game),
        **_summary(intervals),
        "per_game": per_game,
        "board_order_reconstructed": False,
        "phase_3u_schema_ready": False,
        "ranking_ready": False,
        "candidate_scoring_performed": False,
        "confirmation_seeds_consumed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.source).read_bytes()
    result = audit_powerlog_play_membership(source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_game"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
