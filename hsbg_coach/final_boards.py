"""Real winning-board extraction from Firestone comp data.

Firestone's comp endpoint embeds **actual final boards** from high-MMR games
under ``compStats[].heroStats[].finalBoards[]`` — each a real winning lobby's
board with every minion's card, position, stats, tribe, and keywords. This is
"what the winning board looks like," population-scale.

We turn that into two things:
1. **Real core cards per comp** — by how often each card appears across the
   comp's winning boards (replaces the earlier tribe-strength approximation).
2. **Example boards** — a few slim board snapshots per comp, usable as a target
   to compare your board against, for positioning reference, and as ML data.

Card ids are aggregated by *name* so golden/triple copies merge with their base.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .firestone_stats import RACE_NAMES, _archetype_tribe

# Board-tag keywords worth surfacing.
_KW_TAGS = ["REBORN", "DIVINE_SHIELD", "TAUNT", "DEATHRATTLE", "POISONOUS",
           "WINDFURY", "MEGA_WINDFURY", "STEALTH"]


def _minion(m: dict, names: Dict[str, str]) -> dict:
    t = m.get("tags", {}) or {}
    cid = m.get("cardID") or m.get("cardId") or ""
    return {
        "cardId": cid,
        "name": names.get(cid, cid),
        "pos": t.get("ZONE_POSITION"),
        "atk": t.get("ATK"),
        "health": t.get("HEALTH"),
        "tribe": RACE_NAMES.get(t.get("CARDRACE")),
        "keywords": [k for k in _KW_TAGS if t.get(k) == 1],
    }


def _iter_boards(comp: dict, min_mmr: int):
    for h in comp.get("heroStats", []):
        for fb in h.get("finalBoards", []):
            if (fb.get("mmr") or 0) < min_mmr:
                continue
            board = (fb.get("finalComp") or {}).get("board") or []
            if board:
                yield fb.get("mmr"), board


def extract(comp_raw: dict, names: Dict[str, str], min_mmr: int = 0,
            top_core: int = 12, examples: int = 3,
            min_boards: int = 30) -> List[dict]:
    """Per-comp core cards (by winning-board frequency) + example boards.

    If the MMR filter leaves a comp with < min_boards examples, fall back to all
    of that comp's boards so thin comps still get core cards (keeps "top players"
    where the data supports it, without dropping comps)."""
    out = []
    for c in comp_raw.get("compStats", []):
        boards = list(_iter_boards(c, min_mmr))
        if len(boards) < min_boards:
            boards = list(_iter_boards(c, 0))     # fallback: all MMRs
        if not boards:
            continue

        freq: Counter = Counter()
        positions: Dict[str, List[int]] = defaultdict(list)
        example_boards = []
        for mmr, board in boards:
            minions = [_minion(m, names) for m in board]
            for nm in {mi["name"] for mi in minions}:   # presence, once per board
                freq[nm] += 1
            for mi in minions:
                if mi["pos"]:
                    positions[mi["name"]].append(mi["pos"])
            if len(example_boards) < examples:
                example_boards.append({"mmr": mmr, "minions": minions})

        n = len(boards)
        core = []
        for nm, f in freq.most_common(top_core):
            ps = positions.get(nm, [])
            core.append({
                "name": nm,
                "frequency": round(f / n, 3),         # share of winning boards with it
                "commonPosition": max(set(ps), key=ps.count) if ps else None,
            })
        out.append({
            "archetype": c.get("archetype", ""),
            "name": c.get("archetype", "").replace("_", " ").title(),
            "tribe": _archetype_tribe(c.get("archetype", "")),
            "boardCount": n,
            "coreCards": core,
            "examples": example_boards,
        })
    return out


def core_cards_by_archetype(extracted: List[dict], top: int = 6) -> Dict[str, List[str]]:
    """archetype -> top core card names, for joining into comp stats."""
    return {e["archetype"]: [c["name"] for c in e["coreCards"][:top]] for e in extracted}
