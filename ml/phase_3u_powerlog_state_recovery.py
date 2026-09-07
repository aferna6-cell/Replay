"""Conservative Phase 3U Power.log state-recovery probe (measurement-only).

This module measures a narrower question than the main observability probe: when a
canonical GameState TAG_CHANGE lacks an inline entity descriptor, can its pre-ZONE
state still be recovered from *previously observed log state* without simulator or
candidate-model assumptions?

Recovery is deliberately conservative. A pre-ZONE state is considered complete
only when zone, zone position, non-empty card id, and player were all observed for
the same entity. After a ZONE change, zone position is invalidated until it is
observed again, because the destination position is not implied by the destination
zone. This probe never defines membership semantics, constructs Phase 3U evidence
rows, reconstructs conserved pools, scores candidates, or authorizes ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict

PROBE_VERSION = "3u_powerlog_state_recovery_v1"

_CANONICAL_POWER_MARKER = "GameState.DebugPrintPower() -"
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


def _complete_body_pre_state(state: dict) -> bool:
    return (
        isinstance(state.get("zone"), str)
        and isinstance(state.get("zone_pos"), int)
        and isinstance(state.get("card_id"), str)
        and bool(state.get("card_id"))
        and isinstance(state.get("player"), int)
    )


def audit_powerlog_state_recovery(source_content: bytes) -> Dict:
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")
    try:
        text = source_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Power.log must be valid UTF-8 text") from exc

    states: dict[str, dict] = {}
    zone_changes = 0
    bracketed_zone_changes = 0
    numeric_zone_changes = 0
    zone_changes_with_recoverable_pre_state = 0
    bracketed_zone_changes_with_recoverable_pre_state = 0
    numeric_zone_changes_with_recoverable_pre_state = 0
    body_like_zone_changes = 0

    for line in text.splitlines():
        if _CANONICAL_POWER_MARKER not in line:
            continue
        change = _TAG_CHANGE_RE.search(line)
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

            # Destination zone is observed, but destination zone position is not.
            # Invalidate zone_pos rather than carrying a stale pre-transition value.
            state["zone"] = value
            state["zone_pos"] = None
        elif tag == "ZONE_POSITION":
            try:
                state["zone_pos"] = int(value)
            except ValueError:
                state["zone_pos"] = None

    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "zone_changes": zone_changes,
        "bracketed_zone_changes": bracketed_zone_changes,
        "numeric_zone_changes": numeric_zone_changes,
        "zone_changes_with_recoverable_pre_state": zone_changes_with_recoverable_pre_state,
        "zone_pre_state_recovery_coverage": _coverage(
            zone_changes_with_recoverable_pre_state, zone_changes
        ),
        "bracketed_zone_changes_with_recoverable_pre_state": (
            bracketed_zone_changes_with_recoverable_pre_state
        ),
        "bracketed_zone_pre_state_recovery_coverage": _coverage(
            bracketed_zone_changes_with_recoverable_pre_state, bracketed_zone_changes
        ),
        "numeric_zone_changes_with_recoverable_pre_state": (
            numeric_zone_changes_with_recoverable_pre_state
        ),
        "numeric_zone_pre_state_recovery_coverage": _coverage(
            numeric_zone_changes_with_recoverable_pre_state, numeric_zone_changes
        ),
        "zone_changes_without_recoverable_pre_state": (
            zone_changes - zone_changes_with_recoverable_pre_state
        ),
        "body_like_zone_changes": body_like_zone_changes,
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
