"""Phase 3U Power.log observability probe (measurement-only).

This module answers a deliberately narrow question: does a concrete Hearthstone
Power.log contain the raw primitives needed to justify building a real external
measurement adapter? It does *not* convert log lines into Phase 3U transition
rows, does not score allocation candidates, and never authorizes ranking.

The probe is grounded in the committed Power.log syntax (DebugPrintPower,
FULL_ENTITY, Player, and TAG_CHANGE records). Scientifically meaningful
membership-event semantics, complete pre/post board reconstruction, conserved-
pool reconstruction, calibration/evaluation splitting, measurement-error
characterization, and digest-bound parser execution remain separate gates.

Power.log mirrors many power events through both GameState.DebugPrintPower() and
PowerTaskList.DebugPrintPower(). Primitive counts therefore use only the
GameState stream so the same event is not counted twice. Entity IDs are
normalized from either numeric Entity=25 or bracketed Entity=[... id=25 ...]
forms before changed-entity counting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict

PROBE_VERSION = "3u_powerlog_feasibility_v4"

_CANONICAL_POWER_MARKER = "GameState.DebugPrintPower() -"
_NONCANONICAL_POWER_MARKER = "PowerTaskList.DebugPrintPower() -"
_CREATE_GAME_RE = re.compile(r"\bCREATE_GAME\b")
_PLAYER_RE = re.compile(r"\bPlayer EntityID=(\d+) PlayerID=(\d+)\b")
_FULL_ENTITY_RE = re.compile(r"\bFULL_ENTITY - Creating ID=(\d+)\b")
_ENTITY_ID_RE = re.compile(r"\btag=ENTITY_ID value=(\d+)\b")
_ATK_RE = re.compile(r"\btag=ATK value=(-?\d+)\b")
_HEALTH_RE = re.compile(r"\btag=HEALTH value=(-?\d+)\b")
_ZONE_RE = re.compile(r"\btag=ZONE value=([A-Z_]+)\b")
_TAG_CHANGE_RE = re.compile(
    r"\bTAG_CHANGE Entity=(?:(\d+)|\[([^\]]*\bid=(\d+)\b[^\]]*)\]) "
    r"tag=([A-Z0-9_]+) value=(\S+)"
)
_DESCRIPTOR_ZONE_RE = re.compile(r"\bzone=([A-Z_]+)\b")
_DESCRIPTOR_ZONE_POS_RE = re.compile(r"\bzonePos=(-?\d+)\b")
_DESCRIPTOR_CARD_ID_RE = re.compile(r"\bcardId=([^\s\]]*)")
_DESCRIPTOR_PLAYER_RE = re.compile(r"\bplayer=(\d+)\b")


def _coverage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def audit_powerlog_observability(source_content: bytes) -> Dict:
    """Report raw Power.log observability while keeping Phase 3U fail-closed."""
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")

    try:
        text = source_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Power.log must be valid UTF-8 text") from exc

    canonical_power_lines = 0
    ignored_noncanonical_power_lines = 0
    create_game_count = 0
    player_records = 0
    full_entity_records = 0
    entity_id_tags = 0
    attack_tags = 0
    health_tags = 0
    zone_tags = 0
    tag_changes = 0
    attack_changes = 0
    health_changes = 0
    zone_changes = 0
    changed_entities: set[str] = set()

    bracketed_tag_changes = 0
    numeric_tag_changes = 0
    zone_changes_with_descriptor = 0
    zone_changes_with_pre_zone = 0
    zone_changes_with_zone_position = 0
    zone_changes_with_card_id = 0
    zone_changes_with_player = 0
    zone_changes_with_complete_descriptor = 0
    attack_changes_with_descriptor = 0
    health_changes_with_descriptor = 0

    for line in text.splitlines():
        if _NONCANONICAL_POWER_MARKER in line:
            ignored_noncanonical_power_lines += 1
        if _CANONICAL_POWER_MARKER not in line:
            continue

        canonical_power_lines += 1
        create_game_count += bool(_CREATE_GAME_RE.search(line))
        player_records += bool(_PLAYER_RE.search(line))
        full_entity_records += bool(_FULL_ENTITY_RE.search(line))
        entity_id_tags += bool(_ENTITY_ID_RE.search(line))
        attack_tags += bool(_ATK_RE.search(line))
        health_tags += bool(_HEALTH_RE.search(line))
        zone_tags += bool(_ZONE_RE.search(line))

        change = _TAG_CHANGE_RE.search(line)
        if change:
            numeric_entity, descriptor, bracketed_entity, tag, _value = change.groups()
            entity_id = numeric_entity or bracketed_entity
            has_descriptor = descriptor is not None

            tag_changes += 1
            changed_entities.add(entity_id)
            numeric_tag_changes += not has_descriptor
            bracketed_tag_changes += has_descriptor
            attack_changes += tag == "ATK"
            health_changes += tag == "HEALTH"
            zone_changes += tag == "ZONE"

            if tag == "ATK" and has_descriptor:
                attack_changes_with_descriptor += 1
            if tag == "HEALTH" and has_descriptor:
                health_changes_with_descriptor += 1
            if tag == "ZONE" and has_descriptor:
                has_pre_zone = bool(_DESCRIPTOR_ZONE_RE.search(descriptor))
                has_zone_position = bool(_DESCRIPTOR_ZONE_POS_RE.search(descriptor))
                has_card_id = bool(_DESCRIPTOR_CARD_ID_RE.search(descriptor))
                has_player = bool(_DESCRIPTOR_PLAYER_RE.search(descriptor))
                zone_changes_with_descriptor += 1
                zone_changes_with_pre_zone += has_pre_zone
                zone_changes_with_zone_position += has_zone_position
                zone_changes_with_card_id += has_card_id
                zone_changes_with_player += has_player
                zone_changes_with_complete_descriptor += all(
                    (has_pre_zone, has_zone_position, has_card_id, has_player)
                )

    stable_identity_primitives = (
        player_records > 0 and full_entity_records > 0 and entity_id_tags > 0
    )
    per_body_stat_primitives = attack_tags > 0 and health_tags > 0
    ordered_change_primitives = tag_changes > 0
    membership_change_primitives = zone_tags > 0 and zone_changes > 0
    raw_observability_candidate = all(
        (
            canonical_power_lines > 0,
            create_game_count > 0,
            stable_identity_primitives,
            per_body_stat_primitives,
            ordered_change_primitives,
            membership_change_primitives,
        )
    )
    all_zone_changes_have_complete_descriptor = (
        zone_changes > 0 and zone_changes_with_complete_descriptor == zone_changes
    )

    blockers = [
        "membership_event_semantics_not_frozen",
        "complete_pre_post_board_reconstruction_not_implemented",
        "conserved_pool_reconstruction_not_defined",
        "calibration_evaluation_split_not_defined",
        "measurement_error_not_characterized",
        "digest_bound_parser_loader_not_implemented",
    ]
    if not raw_observability_candidate:
        blockers.insert(0, "required_powerlog_primitives_missing")

    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "canonical_power_lines": canonical_power_lines,
        "ignored_noncanonical_power_lines": ignored_noncanonical_power_lines,
        "create_game_count": create_game_count,
        "player_records": player_records,
        "full_entity_records": full_entity_records,
        "entity_id_tags": entity_id_tags,
        "attack_tags": attack_tags,
        "health_tags": health_tags,
        "zone_tags": zone_tags,
        "tag_changes": tag_changes,
        "numeric_tag_changes": numeric_tag_changes,
        "bracketed_tag_changes": bracketed_tag_changes,
        "attack_changes": attack_changes,
        "attack_changes_with_descriptor": attack_changes_with_descriptor,
        "attack_descriptor_coverage": _coverage(attack_changes_with_descriptor, attack_changes),
        "health_changes": health_changes,
        "health_changes_with_descriptor": health_changes_with_descriptor,
        "health_descriptor_coverage": _coverage(health_changes_with_descriptor, health_changes),
        "zone_changes": zone_changes,
        "zone_changes_with_descriptor": zone_changes_with_descriptor,
        "zone_changes_with_pre_zone": zone_changes_with_pre_zone,
        "zone_changes_with_zone_position": zone_changes_with_zone_position,
        "zone_changes_with_card_id": zone_changes_with_card_id,
        "zone_changes_with_player": zone_changes_with_player,
        "zone_changes_with_complete_descriptor": zone_changes_with_complete_descriptor,
        "zone_descriptor_coverage": _coverage(zone_changes_with_descriptor, zone_changes),
        "zone_complete_descriptor_coverage": _coverage(
            zone_changes_with_complete_descriptor, zone_changes
        ),
        "all_zone_changes_have_complete_descriptor": all_zone_changes_have_complete_descriptor,
        "changed_entity_count": len(changed_entities),
        "stable_identity_primitives_observed": stable_identity_primitives,
        "per_body_stat_primitives_observed": per_body_stat_primitives,
        "ordered_change_primitives_observed": ordered_change_primitives,
        "membership_change_primitives_observed": membership_change_primitives,
        "raw_observability_candidate": raw_observability_candidate,
        "phase_3u_schema_ready": False,
        "ranking_ready": False,
        "candidate_scoring_performed": False,
        "blockers": blockers,
    }


def audit_powerlog_file(source: Path) -> Dict:
    """Audit one exact on-disk Power.log artifact without changing admission."""
    return audit_powerlog_observability(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measurement-only Phase 3U Power.log observability probe"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = audit_powerlog_file(args.source)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
