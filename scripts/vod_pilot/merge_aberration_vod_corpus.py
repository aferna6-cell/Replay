#!/usr/bin/env python3
"""Merge label JSONLs into train-ready expert_aberration_vods.jsonl.

SKIP: wigwPzucfXc (mislabeled pre-patch per Aidan 2026-09-22 hunt).
Naga: forever excluded (tribes/cards/notes/quarantine file).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels"
OUTS = [
    ROOT / "train_ready" / "expert_aberration_vods.jsonl",
    Path("/workspace/hsreplay-tier7/data/expert_aberration_vods.jsonl"),
    Path("/tmp/Replay-hsreplay/data/expert_aberration_vods.jsonl"),
    Path("/tmp/Replay-hsreplay/data/expert/expert_aberration_vods.jsonl"),
]

SKIP_VIDEO_IDS = {"wigwPzucfXc"}
SKIP_FILES = {"naga_quarantine.jsonl", "PROGRESS.md", "README.md"}
NAGA_RE = re.compile(r"\bnaga\b|archlich\s+kel", re.I)


def is_naga(row: dict) -> bool:
    if row.get("exclude_from_train"):
        return True
    blob = json.dumps(row, ensure_ascii=False)
    if NAGA_RE.search(blob):
        return True
    state = row.get("state") or row.get("state_before") or {}
    tribes = state.get("tribes") or []
    if any(str(t).lower() == "naga" for t in tribes):
        return True
    return False


def video_id(row: dict) -> str | None:
    vod = row.get("vod") or {}
    return vod.get("video_id")


def main():
    rows = []
    skipped_vid = skipped_naga = 0
    for path in sorted(LABELS.glob("*.jsonl")):
        if path.name in SKIP_FILES:
            continue
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = video_id(row)
            if vid in SKIP_VIDEO_IDS:
                skipped_vid += 1
                continue
            if is_naga(row):
                skipped_naga += 1
                continue
            # force patch note for today hunt shorts already tagged; leave early_access as-is
            rows.append(row)

    # stable-ish dedupe on (video_id, t_sec, record_type, decision.type/plan)
    seen = set()
    deduped = []
    for r in rows:
        vod = r.get("vod") or {}
        dec = r.get("decision") or {}
        key = (
            vod.get("video_id"),
            vod.get("t_sec"),
            r.get("record_type"),
            dec.get("type"),
            dec.get("plan") or dec.get("subtype"),
            r.get("game_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in deduped)
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} rows={len(deduped)}")

    # stats
    from collections import Counter

    rt = Counter(r.get("record_type") for r in deduped)
    dt = Counter((r.get("decision") or {}).get("type") for r in deduped if r.get("record_type") == "decision")
    gifts = sum(
        1
        for r in deduped
        if r.get("record_type") == "decision"
        and (
            (r.get("decision") or {}).get("gift_over_body")
            or "gift" in str((r.get("decision") or {}).get("plan", "")).lower()
            or (r.get("decision") or {}).get("dark_gift")
            or "gift" in str((r.get("decision") or {}).get("subtype", "")).lower()
        )
    )
    buys = dt.get("buy", 0)
    placeable = sum(
        1
        for r in deduped
        if r.get("placement") is not None
        and len(((r.get("state") or {}).get("board") or [])) >= 2
    )
    print(
        json.dumps(
            {
                "rows": len(deduped),
                "record_types": dict(rt),
                "decision_types": {k: v for k, v in dt.items() if k},
                "gift_decisions": gifts,
                "buy_decisions": buys,
                "placeable_boards_ge2": placeable,
                "skipped_wigw": skipped_vid,
                "skipped_naga": skipped_naga,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
