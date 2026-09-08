"""Measurement-only source-grounded entity-type observability for PLAY intervals.

This probe asks a narrower question than board reconstruction: does canonical
Power.log itself ever explicitly identify the CARDTYPE of an entity whose
ZONE=PLAY membership was already observed? It does not infer type from CardID
prefixes, the card database, names, stats, position, or gameplay semantics.

Only exact source-observed CARDTYPE values are reported. The string value
``MINION`` is counted separately when it is literally present in Power.log;
numeric/other values remain uninterpreted distributions. Evidence is bounded by
CREATE_GAME and by the PLAY interval's next ZONE closure. No ranking, candidate
scoring, board order, schema admission, or confirmation seeds are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

PROBE_VERSION = "3u_powerlog_entity_type_v1"


def _coverage(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def _type_observations(payloads: list[str]) -> dict[str, list[dict]]:
    observations: dict[str, list[dict]] = defaultdict(list)
    active_full: str | None = None

    for ordinal, payload in enumerate(payloads):
        full = _FULL_ENTITY_RE.search(payload)
        if full:
            active_full = full.group(1)
            continue

        raw = _RAW_TAG_RE.match(payload)
        if raw and active_full is not None:
            tag, value = raw.groups()
            if tag == "CARDTYPE":
                observations[active_full].append(
                    {"ordinal": ordinal, "value": value, "source": "full_entity_tag"}
                )
            continue

        active_full = None
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
    else:
        forward = [obs for obs in observations if start < obs["ordinal"] < int(end)]

    chosen = before_or_at_entry[-1] if before_or_at_entry else (forward[0] if forward else None)
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
    counts = Counter(
        str(row["cardtype_value"])
        for row in rows
        if row["cardtype_grounded"]
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
        "type_observable": "literal CARDTYPE only",
        "minion_definition": "literal CARDTYPE value MINION only; no CardID/name/database inference",
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
