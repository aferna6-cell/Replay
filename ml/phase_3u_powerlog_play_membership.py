"""Source-grounded, measurement-only Power.log PLAY-membership probe.

Membership, identity and position are separate observables. Only canonical
GameState.DebugPrintPower records are consumed; CREATE_GAME resets all state.
Repeated ZONE=PLAY assertions for an already-open PLAY interval do not create a
second entry. Descriptor identity on that repeated assertion can still ground the
open interval; descriptor position is not inferred from the ZONE mutation. No
board order is inferred and no Phase 3U admission is authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

from ml.phase_3u_powerlog_state_recovery import (
    _CANONICAL_POWER_MARKER, _CREATE_GAME_RE, _DESCRIPTOR_CARD_ID_RE,
    _DESCRIPTOR_PLAYER_RE, _DESCRIPTOR_ZONE_POS_RE, _DESCRIPTOR_ZONE_RE,
    _FULL_ENTITY_RE, _PLAYER_RE, _RAW_TAG_RE, _TAG_CHANGE_RE,
)

PROBE_VERSION = "3u_powerlog_play_membership_v2"


def _coverage(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def _payloads(text: str) -> list[str]:
    return [line.split(_CANONICAL_POWER_MARKER, 1)[1] for line in text.splitlines()
            if _CANONICAL_POWER_MARKER in line]


def _segments(payloads: list[str]) -> list[list[str]]:
    out, current = [], []
    for payload in payloads:
        if _CREATE_GAME_RE.search(payload) and current:
            out.append(current)
            current = []
        current.append(payload)
    if current:
        out.append(current)
    return out


def _descriptor(descriptor: str | None):
    if descriptor is None:
        return None, None, None, None
    zone = _DESCRIPTOR_ZONE_RE.search(descriptor)
    pos = _DESCRIPTOR_ZONE_POS_RE.search(descriptor)
    card = _DESCRIPTOR_CARD_ID_RE.search(descriptor)
    player = _DESCRIPTOR_PLAYER_RE.search(descriptor)
    return (
        zone.group(1) if zone else None,
        int(pos.group(1)) if pos else None,
        card.group(1) if card and card.group(1) else None,
        int(player.group(1)) if player else None,
    )


def _state() -> dict:
    return {"zone": None, "card_id": None, "player": None}


def _new_interval(segment: int, entity: str, ordinal: int, source: str, state: dict) -> dict:
    return {
        "segment_index": segment, "entity_id": entity,
        "play_start_ordinal": ordinal, "membership_source": source,
        "closed_by_zone": False, "play_end_ordinal": None, "exit_zone": None,
        "card_id_known_at_entry": state.get("card_id"),
        "player_known_at_entry": state.get("player"),
        "forward_card_id": None, "forward_card_id_distance": None,
        "forward_player": None, "forward_player_distance": None,
        "explicit_position": None, "explicit_position_distance": None,
        "position_source": None, "duplicate_play_assertions": 0,
    }


def _card(row: dict, ordinal: int, card: str | None) -> None:
    if card and row["forward_card_id"] is None:
        row["forward_card_id"] = card
        row["forward_card_id_distance"] = ordinal - row["play_start_ordinal"]


def _player(row: dict, ordinal: int, player: int | None, valid: set[int]) -> None:
    if player is not None and player in valid and row["forward_player"] is None:
        row["forward_player"] = player
        row["forward_player_distance"] = ordinal - row["play_start_ordinal"]


def _position(row: dict, ordinal: int, pos: int | None, source: str) -> None:
    if pos is not None and row["explicit_position"] is None:
        row["explicit_position"] = pos
        row["explicit_position_distance"] = ordinal - row["play_start_ordinal"]
        row["position_source"] = source


def _observe_zone(pending: dict[str, dict], completed: list[dict], states: dict[str, dict],
                  *, segment: int, entity: str, ordinal: int, zone: str, source: str) -> None:
    state = states.setdefault(entity, _state())
    current = pending.get(entity)
    if zone == "PLAY" and current is not None:
        current["duplicate_play_assertions"] += 1
        state["zone"] = "PLAY"
        return
    if current is not None:
        current = pending.pop(entity)
        current["closed_by_zone"] = True
        current["play_end_ordinal"] = ordinal
        current["exit_zone"] = zone
        completed.append(current)
    state["zone"] = zone
    if zone == "PLAY":
        pending[entity] = _new_interval(segment, entity, ordinal, source, state)


def _finalize(row: dict) -> dict:
    card = row["forward_card_id"] or row["card_id_known_at_entry"]
    player = row["forward_player"] or row["player_known_at_entry"]
    out = dict(row)
    out["card_id_grounded"] = card is not None
    out["player_grounded"] = player is not None
    out["identity_grounded"] = card is not None and player is not None
    out["position_grounded"] = row["explicit_position"] is not None
    out["identity_grounded_without_position"] = out["identity_grounded"] and not out["position_grounded"]
    return out


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    vals = {
        "tag_change_membership_intervals": sum(r["membership_source"] == "tag_change" for r in rows),
        "full_entity_tag_membership_intervals": sum(r["membership_source"] == "full_entity_tag" for r in rows),
        "closed_by_next_zone": sum(bool(r["closed_by_zone"]) for r in rows),
        "card_id_grounded": sum(bool(r["card_id_grounded"]) for r in rows),
        "player_grounded": sum(bool(r["player_grounded"]) for r in rows),
        "identity_grounded": sum(bool(r["identity_grounded"]) for r in rows),
        "explicit_position_grounded": sum(bool(r["position_grounded"]) for r in rows),
        "identity_grounded_without_position": sum(bool(r["identity_grounded_without_position"]) for r in rows),
        "duplicate_play_assertions": sum(r["duplicate_play_assertions"] for r in rows),
    }
    return {
        "play_membership_intervals": n, **vals,
        "card_id_coverage": _coverage(vals["card_id_grounded"], n),
        "player_coverage": _coverage(vals["player_grounded"], n),
        "identity_coverage": _coverage(vals["identity_grounded"], n),
        "explicit_position_coverage": _coverage(vals["explicit_position_grounded"], n),
        "identity_without_position_coverage": _coverage(vals["identity_grounded_without_position"], n),
    }


def _audit_segment(payloads: list[str], segment: int) -> dict:
    valid_players = {int(m.group(2)) for p in payloads if (m := _PLAYER_RE.search(p))}
    states: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    completed: list[dict] = []
    active_full: str | None = None

    for ordinal, payload in enumerate(payloads):
        if _PLAYER_RE.search(payload):
            active_full = None
            continue
        full = _FULL_ENTITY_RE.search(payload)
        if full:
            entity, card = full.groups()
            active_full = entity
            state = states.setdefault(entity, _state())
            if card:
                state["card_id"] = card
                if entity in pending:
                    _card(pending[entity], ordinal, card)
            continue
        raw = _RAW_TAG_RE.match(payload)
        if raw and active_full is not None:
            tag, value = raw.groups()
            entity = active_full
            state = states.setdefault(entity, _state())
            if tag == "ZONE":
                _observe_zone(pending, completed, states, segment=segment, entity=entity,
                              ordinal=ordinal, zone=value, source="full_entity_tag")
                if value == "PLAY" and entity in pending:
                    _card(pending[entity], ordinal, state.get("card_id"))
                    _player(pending[entity], ordinal, state.get("player"), valid_players)
            elif tag == "CONTROLLER":
                try:
                    controller = int(value)
                except ValueError:
                    controller = None
                if controller in valid_players:
                    state["player"] = controller
                    if entity in pending:
                        _player(pending[entity], ordinal, controller, valid_players)
            elif tag == "ZONE_POSITION" and entity in pending and state.get("zone") == "PLAY":
                try:
                    pos = int(value)
                except ValueError:
                    pos = None
                _position(pending[entity], ordinal, pos, "full_entity_tag")
            continue

        active_full = None
        change = _TAG_CHANGE_RE.search(payload)
        if not change:
            continue
        numeric, descriptor, bracketed, tag, value = change.groups()
        entity = numeric or bracketed
        state = states.setdefault(entity, _state())
        d_zone, d_pos, d_card, d_player = _descriptor(descriptor)
        if d_card:
            state["card_id"] = d_card
        if d_player in valid_players:
            state["player"] = d_player

        if tag == "ZONE":
            # A repeated descriptor-bearing ZONE=PLAY assertion is still direct
            # same-record identity evidence for the already-open interval. Capture
            # only CardID/player from that descriptor; zonePos is the pre-mutation
            # descriptor side and must not be promoted to post-state position.
            duplicate_play_row = pending.get(entity) if value == "PLAY" else None
            _observe_zone(pending, completed, states, segment=segment, entity=entity,
                          ordinal=ordinal, zone=value, source="tag_change")
            if duplicate_play_row is not None:
                _card(duplicate_play_row, ordinal, d_card)
                _player(duplicate_play_row, ordinal, d_player, valid_players)
            continue

        row = pending.get(entity)
        if row is not None and descriptor is not None and d_zone == "PLAY":
            _card(row, ordinal, d_card)
            _player(row, ordinal, d_player, valid_players)
            # On a ZONE_POSITION mutation the descriptor position is pre-mutation;
            # only unrelated descriptor snapshots can establish post-state position.
            if tag != "ZONE_POSITION":
                _position(row, ordinal, d_pos, "descriptor_snapshot")
        if tag == "ZONE_POSITION" and row is not None and state.get("zone") == "PLAY":
            try:
                pos = int(value)
            except ValueError:
                pos = None
            _position(row, ordinal, pos, "tag_change")
        elif tag == "CONTROLLER":
            try:
                controller = int(value)
            except ValueError:
                controller = None
            if controller in valid_players:
                state["player"] = controller
                if row is not None:
                    _player(row, ordinal, controller, valid_players)

    completed.extend(pending.values())
    rows = [_finalize(r) for r in sorted(completed, key=lambda r: r["play_start_ordinal"])]
    return {"segment_index": segment, "canonical_payloads": len(payloads),
            "observed_player_ids": sorted(valid_players), **_summary(rows), "intervals": rows}


def audit_powerlog_play_membership(source_content: bytes) -> Dict:
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")
    try:
        text = source_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Power.log must be valid UTF-8 text") from exc
    per_game = [_audit_segment(seg, i) for i, seg in enumerate(_segments(_payloads(text)))]
    rows = [row for game in per_game for row in game["intervals"]]
    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "game_boundary_marker": "CREATE_GAME",
        "game_segments": len(per_game),
        **_summary(rows),
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
    result = audit_powerlog_play_membership(Path(args.source).read_bytes())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_game"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
