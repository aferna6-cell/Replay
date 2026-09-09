"""Measurement-only census of PLAY intervals still lacking causal literal CARDTYPE.

Runs against the current Phase 3U entity-type parser, including canonical
SHOW_ENTITY raw CARDTYPE support. This probe never infers type from CardID, names,
or card databases and never performs ranking, admission, tuning, or seed use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ml.phase_3u_powerlog_play_membership import _payloads, _segments, _audit_segment
from ml.phase_3u_powerlog_entity_type import _type_observations

PROBE_VERSION = "3u_powerlog_residual_untyped_census_v1"


def _mentions_entity(payload: str, entity: str) -> bool:
    return bool(re.search(rf"(?:\bEntity=|\bID=|\bid=){re.escape(entity)}\b", payload))


def _family(payload: str) -> str:
    for family in ("FULL_ENTITY", "SHOW_ENTITY", "CHANGE_ENTITY", "TAG_CHANGE", "HIDE_ENTITY"):
        if re.search(rf"\b{family}\b", payload):
            return family
    return "OTHER"


def _causal_type_observed(row: dict, observations: list[dict]) -> bool:
    start = int(row["play_start_ordinal"])
    end = row.get("play_end_ordinal")
    for obs in observations:
        ordinal = int(obs["ordinal"])
        if ordinal <= start:
            return True
        if end is None and ordinal > start:
            return True
        if end is not None and start < ordinal < int(end):
            return True
    return False


def audit_residual_untyped(source_content: bytes) -> dict:
    if not isinstance(source_content, bytes) or not source_content:
        raise ValueError("source_content must be non-empty bytes")
    segments = _segments(_payloads(source_content.decode("utf-8")))
    total = 0
    residual = []
    family_counts = Counter()
    tag_counts = Counter()

    for segment_index, payloads in enumerate(segments):
        membership = _audit_segment(payloads, segment_index)
        observations = _type_observations(payloads)
        for row in membership["intervals"]:
            total += 1
            entity = str(row["entity_id"])
            if _causal_type_observed(row, observations.get(entity, [])):
                continue
            start = int(row["play_start_ordinal"])
            end = row.get("play_end_ordinal")
            lo = max(0, start - 40)
            hi = len(payloads) if end is None else min(len(payloads), int(end) + 40)
            mentions = []
            for ordinal in range(lo, hi):
                payload = payloads[ordinal]
                if not _mentions_entity(payload, entity):
                    continue
                family = _family(payload)
                family_counts[family] += 1
                if family == "TAG_CHANGE":
                    match = re.search(r"\btag=([A-Z0-9_]+)\b", payload)
                    if match:
                        tag_counts[match.group(1)] += 1
                mentions.append({
                    "ordinal": ordinal,
                    "relative": ordinal - start,
                    "family": family,
                    "payload": payload.strip(),
                })
            residual.append({
                "segment_index": segment_index,
                "entity_id": entity,
                "play_start_ordinal": start,
                "play_end_ordinal": end,
                "membership_source": row.get("membership_source"),
                "card_id_grounded": row.get("card_id_grounded"),
                "player_grounded": row.get("player_grounded"),
                "position_grounded": row.get("position_grounded"),
                "nearby_entity_mentions": mentions,
            })

    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "play_membership_intervals": total,
        "causally_untyped_intervals": len(residual),
        "record_family_mentions": dict(sorted(family_counts.items())),
        "tag_change_tags_on_residual_entities": dict(sorted(tag_counts.items())),
        "intervals": residual,
        "cardtype_inference_performed": False,
        "candidate_scoring_performed": False,
        "ranking_ready": False,
        "phase_3u_schema_ready": False,
        "confirmation_seeds_consumed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit_residual_untyped(Path(args.source).read_bytes())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
