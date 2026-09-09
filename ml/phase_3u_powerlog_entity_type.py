"""Measurement-only source-grounded entity-type observability for PLAY intervals.

This probe asks a narrower question than board reconstruction: does canonical
Power.log itself ever explicitly identify the CARDTYPE of an entity whose
ZONE=PLAY membership was already observed? It does not infer type from CardID
prefixes, the card database, names, stats, position, or gameplay semantics.

Only exact source-observed CARDTYPE values are reported. Literal CARDTYPE tags
attached to canonical FULL_ENTITY and SHOW_ENTITY blocks are both accepted, as
are canonical TAG_CHANGE CARDTYPE records. The string value ``MINION`` is
counted separately only when it is literally present in Power.log; numeric/other
values remain uninterpreted distributions. Evidence is bounded by CREATE_GAME.
Causal/at-entry grounding remains bounded by the PLAY interval's next ZONE
closure. Separately, v3 measures whether later same-game observations are
internally consistent enough to form a *retrospective candidate*; those
candidates are never promoted to causal grounding or board reconstruction.
No ranking, candidate scoring, board order, schema admission, or confirmation
seeds are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from ml.phase_3u_powerlog_play_membership import (
    _payloads,
    _segments,
    audit_powerlog_play_membership,
)
from ml.phase_3u_powerlog_state_recovery import (
    _FULL_ENTITY_RE,
    _RAW_TAG_RE,
    _TAG_CHANGE_RE,
)

PROBE_VERSION = "3u_powerlog_entity_type_v3"
_SHOW_ENTITY_NUMERIC_RE = re.compile(r"\bSHOW_ENTITY - Updating Entity=(\d+)\b")
_SHOW_ENTITY_DESCRIPTOR_RE = re.compile(
    r"\bSHOW_ENTITY - Updating Entity=\[[^\]]*\bid=(\d+)\b[^\]]*\]"
)


def _coverage(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def _show_entity_id(payload: str) -> str | None:
    numeric = _SHOW_ENTITY_NUMERIC_RE.search(payload)
    if numeric:
        return numeric.group(1)
    descriptor = _SHOW_ENTITY_DESCRIPTOR_RE.search(payload)
    return descriptor.group(1) if descriptor else None


def _type_observations(payloads: list[str]) -> dict[str, list[dict]]:
    observations: dict[str, list[dict]] = defaultdict(list)
    active_entity: str | None = None
    active_source: str | None = None

    for ordinal, payload in enumerate(payloads):
        full = _FULL_ENTITY_RE.search(payload)
        if full:
            active_entity = full.group(1)
            active_source = "full_entity_tag"
            continue

        show_entity = _show_entity_id(payload)
        if show_entity is not None:
            active_entity = show_entity
            active_source = "show_entity_tag"
            continue

        raw = _RAW_TAG_RE.match(payload)
        if raw and active_entity is not None:
            tag, value = raw.groups()
            if tag == "CARDTYPE":
                observations[active_entity].append(
                    {"ordinal": ordinal, "value": value, "source": active_source}
                )
            continue

        active_entity = None
        active_source = None
        change = _TAG_CHANGE_RE.search(payload)
        if not change:
            continue
        numeric, _descriptor, bracketed, tag, value = change.groups()
        if tag == "CARDTYPE":
            entity = numeric or bracketed
            observations[entity].append(
                {"ordinal": ordinal, "value": value, "source": "tag_change"}
            )

    return observations


def _annotate_interval(row: dict, observations: list[dict]) -> dict:
    start = int(row["play_start_ordinal"])
    end = row.get("play_end_ordinal")
    before_or_at_entry = [obs for obs in observations if obs["ordinal"] <= start]
    if end is None:
        forward = [obs for obs in observations if obs["ordinal"] > start]
        after_exit: list[dict] = []
    else:
        end_ordinal = int(end)
        forward = [obs for obs in observations if start < obs["ordinal"] < end_ordinal]
        after_exit = [obs for obs in observations if obs["ordinal"] >= end_ordinal]

    chosen = before_or_at_entry[-1] if before_or_at_entry else (forward[0] if forward else None)
    distinct_same_game_values = sorted({str(obs["value"]) for obs in observations})
    same_game_consistent = len(distinct_same_game_values) <= 1
    retrospective = None
    if chosen is None and after_exit and same_game_consistent:
        retrospective = after_exit[0]

    out = dict(row)
    out["cardtype_known_at_entry"] = before_or_at_entry[-1]["value"] if before_or_at_entry else None
    out["forward_cardtype"] = forward[0]["value"] if forward else None
    out["forward_cardtype_distance"] = (
        forward[0]["ordinal"] - start if forward else None
    )
    out["cardtype_grounded"] = chosen is not None
    out["cardtype_value"] = chosen["value"] if chosen else None
    out["cardtype_source"] = chosen["source"] if chosen else None
    out["explicit_minion_grounded"] = bool(
        chosen is not None and str(chosen["value"]).upper() == "MINION"
    )
    out["same_game_cardtype_observation_count"] = len(observations)
    out["same_game_cardtype_distinct_values"] = distinct_same_game_values
    out["same_game_cardtype_consistent"] = same_game_consistent
    out["post_exit_cardtype"] = after_exit[0]["value"] if after_exit else None
    out["post_exit_cardtype_distance"] = (
        after_exit[0]["ordinal"] - int(end) if after_exit and end is not None else None
    )
    out["retrospective_cardtype_candidate"] = retrospective["value"] if retrospective else None
    out["retrospective_explicit_minion_candidate"] = bool(
        retrospective is not None and str(retrospective["value"]).upper() == "MINION"
    )
    out["retrospective_candidate_is_causal_grounding"] = False
    return out


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    grounded = sum(bool(row["cardtype_grounded"]) for row in rows)
    explicit_minion = sum(bool(row["explicit_minion_grounded"]) for row in rows)
    identity_and_type = sum(
        bool(row.get("identity_grounded")) and bool(row["cardtype_grounded"])
        for row in rows
    )
    minion_identity = sum(
        bool(row.get("identity_grounded")) and bool(row["explicit_minion_grounded"])
        for row in rows
    )
    rows_with_any_same_game_type = sum(
        int(row["same_game_cardtype_observation_count"]) > 0 for row in rows
    )
    rows_with_same_game_type_conflict = sum(
        len(row["same_game_cardtype_distinct_values"]) > 1 for row in rows
    )
    retrospective = sum(
        row["retrospective_cardtype_candidate"] is not None for row in rows
    )
    retrospective_minion = sum(
        bool(row["retrospective_explicit_minion_candidate"]) for row in rows
    )
    counts = Counter(
        str(row["cardtype_value"])
        for row in rows
        if row["cardtype_grounded"]
    )
    retrospective_counts = Counter(
        str(row["retrospective_cardtype_candidate"])
        for row in rows
        if row["retrospective_cardtype_candidate"] is not None
    )
    return {
        "play_membership_intervals": n,
        "cardtype_grounded": grounded,
        "cardtype_coverage": _coverage(grounded, n),
        "explicit_minion_grounded": explicit_minion,
        "explicit_minion_coverage": _coverage(explicit_minion, n),
        "identity_and_cardtype_grounded": identity_and_type,
        "identity_and_cardtype_coverage": _coverage(identity_and_type, n),
        "identity_and_explicit_minion_grounded": minion_identity,
        "identity_and_explicit_minion_coverage": _coverage(minion_identity, n),
        "cardtype_value_counts": dict(sorted(counts.items())),
        "same_game_type_observed_intervals": rows_with_any_same_game_type,
        "same_game_type_observed_coverage": _coverage(rows_with_any_same_game_type, n),
        "same_game_type_conflict_intervals": rows_with_same_game_type_conflict,
        "retrospective_cardtype_candidates": retrospective,
        "retrospective_cardtype_candidate_coverage": _coverage(retrospective, n),
        "retrospective_explicit_minion_candidates": retrospective_minion,
        "retrospective_cardtype_value_counts": dict(sorted(retrospective_counts.items())),
    }


def audit_powerlog_entity_type(source_content: bytes) -> Dict:
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")
    try:
        text = source_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Power.log must be valid UTF-8 text") from exc

    membership = audit_powerlog_play_membership(source_content)
    segments = _segments(_payloads(text))
    if len(segments) != len(membership["per_game"]):
        raise AssertionError("membership/type segment mismatch")

    per_game = []
    all_rows: list[dict] = []
    for segment_index, (payloads, membership_game) in enumerate(
        zip(segments, membership["per_game"], strict=True)
    ):
        observations = _type_observations(payloads)
        rows = [
            _annotate_interval(row, observations.get(row["entity_id"], []))
            for row in membership_game["intervals"]
        ]
        all_rows.extend(rows)
        per_game.append(
            {"segment_index": segment_index, **_summary(rows), "intervals": rows}
        )

    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "game_boundary_marker": "CREATE_GAME",
        "type_observable": "literal CARDTYPE from FULL_ENTITY/SHOW_ENTITY raw tags or TAG_CHANGE only",
        "minion_definition": "literal CARDTYPE value MINION only; no CardID/name/database inference",
        "retrospective_contract": (
            "post-exit same-game CARDTYPE is reported only as a non-causal candidate when "
            "all source-observed CARDTYPE values for that entity in the game agree"
        ),
        "game_segments": len(per_game),
        **_summary(all_rows),
        "per_game": per_game,
        "board_set_reconstructed": False,
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
    result = audit_powerlog_entity_type(Path(args.source).read_bytes())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_game"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
