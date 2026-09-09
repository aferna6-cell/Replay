"""Measurement-only Phase 3U position/order observability census.

This probe composes the already source-grounded PLAY-membership and literal
CARDTYPE probes. It does not infer board order, minion identity, or type from
CardID/name/database semantics. A position is counted only when the membership
probe has already accepted an explicit source-observed ZONE_POSITION or an
unrelated descriptor snapshot while the PLAY interval is active.

The purpose is to quantify the ceiling for future board reconstruction. It does
not reconstruct boards, rank candidates, change simulator behavior, or consume
confirmation seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

from ml.phase_3u_powerlog_entity_type import audit_powerlog_entity_type

PROBE_VERSION = "3u_powerlog_position_census_v1"


def _coverage(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    typed = [r for r in rows if r.get("cardtype_grounded")]
    minions = [r for r in rows if r.get("explicit_minion_grounded")]
    minion_positioned = [r for r in minions if r.get("position_grounded")]
    minion_position_at_entry = [
        r for r in minion_positioned
        if r.get("explicit_position_distance") == 0
    ]
    minion_position_forward = [
        r for r in minion_positioned
        if isinstance(r.get("explicit_position_distance"), int)
        and r["explicit_position_distance"] > 0
    ]
    typed_positioned = [r for r in typed if r.get("position_grounded")]
    return {
        "play_membership_intervals": n,
        "cardtype_grounded": len(typed),
        "cardtype_and_position_grounded": len(typed_positioned),
        "cardtype_position_coverage": _coverage(len(typed_positioned), len(typed)),
        "explicit_minion_grounded": len(minions),
        "explicit_minion_position_grounded": len(minion_positioned),
        "explicit_minion_position_coverage": _coverage(len(minion_positioned), len(minions)),
        "explicit_minion_position_at_entry": len(minion_position_at_entry),
        "explicit_minion_position_forward": len(minion_position_forward),
        "explicit_minion_without_position": len(minions) - len(minion_positioned),
    }


def audit_powerlog_position_census(source_content: bytes) -> Dict:
    if not isinstance(source_content, bytes):
        raise TypeError("source_content must be exact bytes")
    if not source_content:
        raise ValueError("Power.log source_content must be non-empty")

    typed = audit_powerlog_entity_type(source_content)
    rows = [row for game in typed["per_game"] for row in game["intervals"]]
    per_game = []
    for game in typed["per_game"]:
        per_game.append({
            "segment_index": game["segment_index"],
            **_summary(game["intervals"]),
        })

    return {
        "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_bytes": len(source_content),
        "canonical_stream": "GameState.DebugPrintPower",
        "position_contract": (
            "explicit source-observed PLAY position accepted by the existing membership probe only; "
            "no pre-mutation descriptor position promotion"
        ),
        "minion_contract": "literal causal CARDTYPE=MINION only",
        **_summary(rows),
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
    result = audit_powerlog_position_census(Path(args.source).read_bytes())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_game"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
