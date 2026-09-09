"""Measurement-only record-family census for causally untyped PLAY intervals.

This probe does not infer CARDTYPE. It asks which canonical Power.log record forms
mention entities whose PLAY interval lacks causal literal CARDTYPE under the
existing Phase 3U contract. The purpose is to find source-observable record forms
that the CARDTYPE parser may be omitting before declaring the 31-interval type
ceiling intrinsic. No card database, CardID/name semantics, ranking, or seeds are
used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ml.phase_3u_powerlog_play_membership import _payloads, _segments, _audit_segment
from ml.phase_3u_powerlog_entity_type import _type_observations

PROBE_VERSION = "3u_powerlog_untyped_record_census_v1"
_SHOW_ENTITY_RE = re.compile(r"\bSHOW_ENTITY\b")
_CHANGE_ENTITY_RE = re.compile(r"\bCHANGE_ENTITY\b")
_FULL_ENTITY_RE = re.compile(r"\bFULL_ENTITY\b")
_TAG_CHANGE_RE = re.compile(r"\bTAG_CHANGE\b")
_HIDE_ENTITY_RE = re.compile(r"\bHIDE_ENTITY\b")


def _mentions_entity(payload: str, entity: str) -> bool:
    return bool(re.search(rf"(?:\bEntity=|\bID=|\bid=){re.escape(entity)}\b", payload))


def _family(payload: str) -> str:
    if _FULL_ENTITY_RE.search(payload):
        return "FULL_ENTITY"
    if _SHOW_ENTITY_RE.search(payload):
        return "SHOW_ENTITY"
    if _CHANGE_ENTITY_RE.search(payload):
        return "CHANGE_ENTITY"
    if _TAG_CHANGE_RE.search(payload):
        return "TAG_CHANGE"
    if _HIDE_ENTITY_RE.search(payload):
        return "HIDE_ENTITY"
    return "OTHER"


def _causal_type_observed(row: dict, observations: list[dict]) -> bool:
    start = int(row["play_start_ordinal"])
    end = row.get("play_end_ordinal")
    for obs in observations:
        ordinal = int(obs["ordinal"])
        if ordinal <= start:
            return True
        if end is None:
            if ordinal > start:
                return True
        elif start < ordinal < int(end):
            return True
    return False


def audit_powerlog_untyped_census(source_content: bytes) -> dict:
    if not isinstance(source_content, bytes) or not source_content:
        raise ValueError("source_content must be non-empty bytes")
    text = source_content.decode("utf-8")
    segments = _segments(_payloads(text))
    total_intervals = 0
    untyped = []
    family_counts = Counter()
    tag_change_tags = Counter()
    per_entity_families: dict[str, Counter] = defaultdict(Counter)

    for segment_index, payloads in enumerate(segments):
        membership = _audit_segment(payloads, segment_index)
        observations = _type_observations(payloads)
        for row in membership["intervals"]:
            total_intervals += 1
            entity = str(row["entity_id"])
            if _causal_type_observed(row, observations.get(entity, [])):
                continue
            start = int(row["play_start_ordinal"])
            end = row.get("play_end_ordinal")
            lo = max(0, start - 25)
            hi = len(payloads) if end is None else min(len(payloads), int(end) + 25)
            mentions = []
            for ordinal in range(lo, hi):
                payload = payloads[ordinal]
                if not _mentions_entity(payload, entity):
                    continue
                family = _family(payload)
                family_counts[family] += 1
                per_entity_families[entity][family] += 1
                if family == "TAG_CHANGE":
                    m = re.search(r"\btag=([A-Z0-9_]+)\b", payload)
                    if m:
                        tag_change_tags[m.group(1)] += 1
                mentions.append({"ordinal": ordinal, "relative": ordinal - start,
                                 "family": family, "payload": payload.strip()})
            untyped.append({
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
        "play_membership_intervals": total_intervals,
        "causally_untyped_intervals": len(untyped),
        "record_family_mentions": dict(sorted(family_counts.items())),
        "tag_change_tags_on_untyped_entities": dict(sorted(tag_change_tags.items())),
        "untyped_entities_with_show_entity": sum(bool(c.get("SHOW_ENTITY")) for c in per_entity_families.values()),
        "untyped_entities_with_change_entity": sum(bool(c.get("CHANGE_ENTITY")) for c in per_entity_families.values()),
        "untyped_entities_with_full_entity": sum(bool(c.get("FULL_ENTITY")) for c in per_entity_families.values()),
        "intervals": untyped,
        "cardtype_inference_performed": False,
        "candidate_scoring_performed": False,
        "ranking_ready": False,
        "phase_3u_schema_ready": False,
        "confirmation_seeds_consumed": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = audit_powerlog_untyped_census(Path(args.source).read_bytes())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "intervals"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
