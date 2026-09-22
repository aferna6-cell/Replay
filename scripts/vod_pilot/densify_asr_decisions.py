#!/usr/bin/env python3
"""Append ASR-derived decision rows (gift/buy/sell/freeze/HP/level/roll/pivot) into label JSONLs.

Reuses existing record schema from labels/README.md. Skips near-duplicate timestamps.
Does NOT invent card_ids. gift>body preference encoded via plan/subtype.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels"
VIDEOS = ROOT / "videos"
SAMPLES = ROOT / "samples"

# Priority hunt VODs for densify (SKIP wigwPzucfXc)
TARGETS = {
    "z9PfvDgMqAU": {
        "source": "jeef_vod",
        "expert_player": "JeefHS",
        "channel": "JeefHS",
        "patch_note": "36.6.1_early_access",
        "url": "https://www.youtube.com/watch?v=z9PfvDgMqAU",
        "hero_name": "Kith'ix",
        "deity": "Y'Shaarj",
    },
    "g8kCF5Z-UIs": {
        "source": "jeef_vod",
        "expert_player": "JeefHS",
        "channel": "JeefHS",
        "patch_note": "36.6.1_early_access",
        "url": "https://www.youtube.com/watch?v=g8kCF5Z-UIs",
        "hero_name": "Drest'agath",
        "deity": None,
    },
    "V0ZyTYC9gr8": {
        "source": "jeef_vod",
        "expert_player": "JeefHS",
        "channel": "JeefHS",
        "patch_note": "36.6.1",
        "url": "https://www.youtube.com/watch?v=V0ZyTYC9gr8",
        "hero_name": "Magni Bronzebeard",
        "deity": None,
    },
    "0Srr-tgf5Vo": {
        "source": "rdu_vod",
        "expert_player": "RDU",
        "channel": "RduHS",
        "patch_note": "36.6.1_early_access",
        "url": "https://www.youtube.com/watch?v=0Srr-tgf5Vo",
        "hero_name": None,
        "deity": None,
    },
    "Bpr_xTUjktA": {
        "source": "rdu_vod",
        "expert_player": "RDU",
        "channel": "RduHS",
        "patch_note": "36.6.1_early_access",
        "url": "https://www.youtube.com/watch?v=Bpr_xTUjktA",
        "hero_name": None,
        "deity": None,
    },
    "Ry-Zn2sPP2k": {
        "source": "rdu_vod",
        "expert_player": "RDU",
        "channel": "RduHS",
        "patch_note": "36.6.1_early_access",
        "url": "https://www.youtube.com/watch?v=Ry-Zn2sPP2k",
        "hero_name": None,
        "deity": None,
    },
    "1BZtAVX8n50": {
        "source": "rdu_vod",
        "expert_player": "RDU",
        "channel": "RduHS",
        "patch_note": "36.6.1_early_access",
        "url": "https://www.youtube.com/watch?v=1BZtAVX8n50",
        "hero_name": None,
        "deity": None,
    },
    "pzYwWAnjJ54": {
        "source": "shadybunny_vod",
        "expert_player": "Shadybunny",
        "channel": "Shadybunny",
        "patch_note": "36.6.1",
        "url": "https://www.youtube.com/watch?v=pzYwWAnjJ54",
        "hero_name": None,
        "deity": None,
    },
}

# (regex, decision.type, subtype/plan, conf)
RULES = [
    (r"should(?:\s+\w+){0,4}\s+dark\s+gift|just\s+dark\s+gift|dark\s+gifted", "discover", "dark_gift_over_level", 0.85),
    (r"dark\s+gift\s*\+|dark\s+gift\s+plus|gift\s*\+\s*chef|gift\s*\+\s*hero", "discover", "dark_gift_chef_hp", 0.9),
    (r"dark\s+gift(?:ing|s)?\s+(?:are|is|on|for|really)|keep\s+my\s+dark\s+gift|\bdark\s+gifting\b", "discover", "dark_gift", 0.8),
    (r"\bdark\s+gift", "discover", "dark_gift", 0.75),
    (r"\bi(?:'m| am)?\s+buy(?:ing)?\b|\bbought\s+(?:it|this|that)\b|\bbuy\s+(?:this|that|it|like)\b|\btake\s+(?:this|that|it)\b.*(?:shop|minion)?", "buy", "shop_buy", 0.7),
    (r"\bcould\s+(?:have\s+)?sold|\bsell\s+(?:this|that|you|trash)\b|\bsold\s+(?:this|that|trash)\b", "sell", "sell_chaff", 0.65),
    (r"\bfreeze|\bfroze|\bfrozen\b", "freeze", "freeze_shop", 0.7),
    (r"\b(?:re)?roll(?:ing|ed)?\b", "roll", "roll_shop", 0.65),
    (r"\blevel(?:ing|ed)?\b.*(?:dumb|dumb\.|kind of dumb)|should(?:n't)? have.*level", "level_up", "level_regret_prefer_gift", 0.8),
    (r"\bnext\s+turn\s+level|\blevel\s+incubate|\blevel\s+up\b|\bi(?:'m)?\s+level", "level_up", "level_timing", 0.7),
    (r"hero\s+power|\bclick(?:ing)?\s+(?:the\s+)?hp\b", "hero_power", "hero_power_timing", 0.7),
    (r"\bactivate\s*(?:zero|0|one|1)?\b", "other", "activate", 0.75),
    (r"\bpivot(?:ing)?\s+to|\bdon'?t\s+you\s+dare\s+pivot|\bgo\s+undead\b|\bplay\s+aberration", "other", "tribe_direction", 0.7),
]


def parse_srt(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) < 2:
            continue
        m = re.search(
            r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
            "\n".join(lines),
        )
        if not m:
            continue
        t = (
            int(m.group(1)) * 3600
            + int(m.group(2)) * 60
            + int(m.group(3))
            + int(m.group(4)[:3]) / 1000
        )
        body_lines = lines[2:] if re.match(r"^\d+$", lines[0]) else lines[1:]
        body = re.sub(r"<[^>]+>", "", " ".join(body_lines)).strip()
        if body:
            cues.append((t, body))
    return cues


def parse_vtt(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    cues = []
    # YouTube auto-VTT often has align/position extras on the timing line.
    for m in re.finditer(
        r"(?:^|\n)(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*(\d+):(\d+):(\d+)\.(\d+)[^\n]*\n(.*?)(?=\n\d+:\d+:\d+\.\d+\s*-->|\Z)",
        text,
        re.S,
    ):
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)[:3]) / 1000
        body = re.sub(r"<[^>]+>", "", m.group(9))
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            cues.append((t, body))
    return cues


def load_captions(vid: str):
    for base in (VIDEOS, SAMPLES):
        for ext, parser in ((".en.srt", parse_srt), (".en.vtt", parse_vtt), (".srt", parse_srt)):
            p = base / f"{vid}{ext}"
            if p.exists():
                return parser(p), str(p)
    return [], None


def existing_decision_times(path: Path):
    times = defaultdict(list)
    if not path.exists():
        return times
    for line in path.open():
        r = json.loads(line)
        if r.get("record_type") != "decision":
            continue
        t = (r.get("vod") or {}).get("t_sec")
        dtype = (r.get("decision") or {}).get("type")
        if t is not None and dtype:
            times[dtype].append(float(t))
    return times


def near(existing, t, window=4.0):
    return any(abs(t - e) <= window for e in existing)


def make_row(vid, meta, t, body, dtype, plan, conf):
    gift_pref = "gift" in plan or plan.startswith("dark_gift")
    return {
        "source": meta["source"],
        "weight_hint": 2.5 if gift_pref else 2.0,
        "game_id": f"{meta['source'].split('_')[0]}_{vid}_g0",
        "expert_player": meta["expert_player"],
        "patch_note": meta["patch_note"],
        "composition_id": None,
        "mmr": None,
        "placement": None,
        "record_type": "decision",
        "vod": {
            "url": meta["url"],
            "video_id": vid,
            "channel": meta["channel"],
            "t_sec": round(t, 1),
            "frame_path": None,
        },
        "state_before": {
            "hero": None,
            "hero_name": meta.get("hero_name"),
            "tavern_tier": None,
            "turn": None,
            "gold": None,
            "board": [],
            "shop": [],
            "hand": [],
            "deity": meta.get("deity"),
        },
        "decision": {
            "type": dtype,
            "subtype": plan,
            "targets": [],
            "targets_name": [],
            "freeze": dtype == "freeze",
            "roll": dtype == "roll",
            "hero_power": dtype == "hero_power",
            "level_up": dtype == "level_up",
            "plan": plan,
            "gift_over_body": bool(gift_pref),
        },
        "rationale_asr": body[:240],
        "label_confidence": conf,
        "labeler": "asr_densify_36_6_1",
        "needs_review": ["asr_only", "no_card_id"],
    }


def densify_one(vid: str, meta: dict) -> int:
    cues, src = load_captions(vid)
    out_path = LABELS / f"{vid}.jsonl"
    existing = existing_decision_times(out_path)
    added = 0
    seen_plans = []
    for t, body in cues:
        low = body.lower()
        # skip naga chatter as training signal
        if re.search(r"\bnaga", low):
            continue
        for pat, dtype, plan, conf in RULES:
            if not re.search(pat, low):
                continue
            if near(existing.get(dtype, []), t) or near([x[0] for x in seen_plans if x[1] == dtype], t, 3.0):
                break
            # avoid duplicate identical plan within 8s
            if any(abs(t - st) < 8 and sp == plan for st, _, sp in seen_plans):
                break
            row = make_row(vid, meta, t, body, dtype, plan, conf)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen_plans.append((t, dtype, plan))
            existing[dtype].append(t)
            added += 1
            break
    print(f"{vid}: captions={len(cues)} from {src}; appended {added}")
    return added


def main():
    total = 0
    for vid, meta in TARGETS.items():
        total += densify_one(vid, meta)
    print(f"TOTAL appended={total}")


if __name__ == "__main__":
    main()
