"""Frozen current-active Bob's Tavern purchasable-minion manifest (Phase 2N-D).

The general card KB retains historical / token / generated minions for knowledge
and effect lookup. ``build_pool()`` must intersect against this manifest so the
shared shop catalogue matches the **current** Tavern population — not every
HearthstoneJSON card that happens to carry a ``techLevel``.

Source of truth for the freeze: HearthstoneJSON ``isBattlegroundsPoolMinion``
(non-golden, solo-sim eligible = not Duos-exclusive, tier 1–6).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Set

_CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cards")
ACTIVE_TAVERN_POOL_PATH = os.path.join(_CARDS_DIR, "active_tavern_pool.json")

_CACHE: Optional[Dict] = None


def active_tavern_pool_path() -> str:
    return ACTIVE_TAVERN_POOL_PATH


def load_active_tavern_pool(path: Optional[str] = None,
                            *, force_reload: bool = False) -> Dict:
    """Load the frozen active-Tavern manifest (cached)."""
    global _CACHE
    path = path or ACTIVE_TAVERN_POOL_PATH
    if _CACHE is not None and not force_reload and path == ACTIVE_TAVERN_POOL_PATH:
        return _CACHE
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if path == ACTIVE_TAVERN_POOL_PATH:
        _CACHE = data
    return data


def active_tavern_card_ids(path: Optional[str] = None) -> Set[str]:
    data = load_active_tavern_pool(path)
    return {m["card_id"] for m in data.get("minions", [])}


def active_tavern_names(path: Optional[str] = None) -> Set[str]:
    data = load_active_tavern_pool(path)
    return {m["name"] for m in data.get("minions", [])}


def build_active_tavern_pool_manifest(
        cards_source: Optional[str] = None) -> Dict:
    """Rebuild the frozen manifest from HearthstoneJSON (refresh helper)."""
    from .firestone_stats import CARDS_URL, _fetch_json

    cards = _fetch_json(cards_source or CARDS_URL)
    rows: List[Dict] = []
    for c in cards:
        if not c.get("isBattlegroundsPoolMinion"):
            continue
        if c.get("type") != "MINION":
            continue
        if c.get("battlegroundsNormalDbfId"):
            continue  # golden / premium copy
        rows.append({
            "card_id": c["id"],
            "name": c.get("name", c["id"]),
            "tier": c.get("techLevel"),
            "attack": c.get("attack"),
            "health": c.get("health"),
            "duos_exclusive": bool(c.get("isBattlegroundsDuosExclusive")),
            "buddy": bool(c.get("isBattlegroundsBuddy")),
        })

    active = [r for r in rows
              if not r["duos_exclusive"] and r["tier"] in (1, 2, 3, 4, 5, 6)]
    duos = [r for r in rows if r["duos_exclusive"]]
    t7 = [r for r in rows if r["tier"] == 7]
    from collections import Counter
    return {
        "_source": "HearthstoneJSON",
        "_flag": "isBattlegroundsPoolMinion",
        "_cards_url": cards_source or CARDS_URL,
        "_patch_context": (
            "Season 14 / current HSJSON latest. KB may retain "
            "historical/token cards; build_pool intersects this manifest."
        ),
        "_rules": {
            "include": (
                "isBattlegroundsPoolMinion && type==MINION "
                "&& !battlegroundsNormalDbfId"),
            "solo_sim_eligible": "not duos_exclusive && 1 <= tier <= 6",
            "max_tier_sim": 6,
            "exclude_from_solo_build_pool": [
                "duos_exclusive", "tier>=7", "tokens",
                "removed", "generated-only",
            ],
        },
        "n_pool_minions_all": len(rows),
        "n_solo_sim_eligible": len(active),
        "n_duos_exclusive": len(duos),
        "n_tier7": len(t7),
        "tier_counts_solo": dict(sorted(Counter(r["tier"] for r in active).items())),
        "minions": sorted(active, key=lambda r: (r["tier"] or 0, r["name"])),
        "excluded_duos_exclusive": sorted(
            [{"card_id": r["card_id"], "name": r["name"], "tier": r["tier"]}
             for r in duos],
            key=lambda r: r["name"]),
        "excluded_tier7": sorted(
            [{"card_id": r["card_id"], "name": r["name"], "tier": r["tier"],
              "duos_exclusive": r["duos_exclusive"]} for r in t7],
            key=lambda r: r["name"]),
        "known_non_pool_examples": [
            {"card_id": "BG36_200t", "name": "Foraging Bat",
             "reason": "token_not_tavern_pool"},
        ],
    }


def save_active_tavern_pool(manifest: Dict,
                            path: Optional[str] = None) -> str:
    path = path or ACTIVE_TAVERN_POOL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    global _CACHE
    if path == ACTIVE_TAVERN_POOL_PATH:
        _CACHE = manifest
    return path
