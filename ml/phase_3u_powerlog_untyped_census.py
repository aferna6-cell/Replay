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
from ml.phase_3u_powerlog_state_recovery import _RAW_TAG_RE

PROBE_VERSION = "3u_powerlog_untyped_record_census_v2"
_SHOW_ENTITY_RE = re.compile(r"\bSHOW_ENTITY\b")
_SHOW_ENTITY_ID_RE = re.compile(r"\bSHOW_ENTITY - Updating Entity=(?:\d+|\[[^\]]*\bid=(\d+)\b[^\]]*\])")
_SHOW_ENTITY_NUMERIC_RE = re.compile(r"\bSHOW_ENTITY - Updating Entity=(\d+)\b")
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


def _show_entity_id(payload: str) -> str | None:
    numeric = _SHOW_ENTITY_NUMERIC_RE.search(payload)
    if numeric:
        return numeric.group(1)
    descriptor = _SHOW_ENTITY_ID_RE.search(payload)
    if descriptor:
        return descriptor.group(1)
    return None


def _show_entity_type_observations(payloads: list[str]) -> dict[str, list[dict]]:
    """Collect only literal CARDTYPE raw tags directly attached to SHOW_ENTITY blocks."""
    observations: dict[str, list[dict]] = defaultdict(list)
    active_show: str | None = None
    for ordinal, payload in enumerate(payloads):
        show_entity = _show_entity_id(payload)
        if show_entity is not None:
            active_show = show_entity
            continue
        raw = _RAW_TAG_RE.match(payload)
        if raw and active_show is not None:
            tag, value = raw.groups()
            if tag == "CARDTYPE":
                observations[active_show].append(
                    {"ordinal": ordinal, "value": value, "source": "show_entity_tag"}
                )
            continue
        active_show = None
    return observations


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
    show_entity_cardtype_values = Counter()
    show_entity_cardtype_intervals = 0
    show_entity_explicit_minion_intervals = 0
    per_entity_families: dict[tuple[int, str], Counter] = defaultdict(Counter)

    for segment_index, payloads in enumerate(segments):
        membership = _audit_segment(payloads, segment_index)
        observations = _type_observations(payloads)
        show_type_observations = _show_entity_type_observations(payloads)
        for row in membership["intervals"]:
            total_intervals += 1
            entity = str(row["entity_id"])
            if _causal_type_observed(row, observations.get(entity, [])):
                continue
            start = int(row["play_start_ordinal"])
            end = row.get("play_end_ordinal")
            show_causal = [
                obs for obs in show_type_observations.get(entity, [])
                if int(obs["ordinal"]) <= start
                or (end is None and int(obs["ordinal"]) > start)
                or (end is not None and start < int(obs["ordinal"]) < int(end))
            ]
            if show_causal:
                show_entity_cardtype_intervals += 1
                show_value = str(show_causal[-1]["value"])
                show_entity_cardtype_values[show_value] += 1
                if show_value.upper() == "MINION":
                    show_entity_explicit_minion_intervals += 1
            lo = max(0, start - 25)
            hi = len(payloads) if end is None else min(len(payloads), int(end) + 25)
            mentions = []
            key = (segment_index, entity)
            for ordinal in range(lo, hi):
                payload = payloads[ordinal]
                if not _mentions_entity(payload, entity):
                    continue
                family = _family(payload)
                family_counts[family] += 1
                per_entity_families[key][family] += 1
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
                "show_entity_causal_cardtype": show_causal[-1]["value"] if show_causal else None,
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
        "untyped_intervals_with_causal_show_entity_cardtype": show_entity_cardtype_intervals,
        "untyped_intervals_with_causal_show_entity_explicit_minion": show_entity_explicit_minion_intervals,
        "causal_show_entity_cardtype_value_counts": dict(sorted(show_entity_cardtype_values.items())),
        "show_entity_cardtype_is_source_literal_only": True,
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
