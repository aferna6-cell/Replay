"""Live-meta playstyle prior for Battlegrounds patch 36.6.1 (Aberrations).

Data: ``data/playstyle_prior.json`` — HSReplay visible comps (Naga dropped) +
HearthstoneJSON ``isBattlegroundsPoolMinion`` shop pool (Naga hard-excluded) +
Aidan's S-tribe lock (Beast / Elemental / Demon / Undead).

Recommend path MUST call ``is_out_of_pool`` / ``bias_shop_scores`` so the coach
never boosts dead cards (Archlich Kel'Thuzad, Monstrous Macaw, Naga, …).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PRIOR_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "playstyle_prior.json")
)
_POOL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "cards", "bg_live_pool_36_6_1.json")
)

_TRIBE_ALIASES = {
    "beasts": "Beast", "beast": "Beast",
    "elementals": "Elemental", "elemental": "Elemental",
    "demons": "Demon", "demon": "Demon",
    "undead": "Undead",
    "murlocs": "Murloc", "murloc": "Murloc",
    "dragons": "Dragon", "dragon": "Dragon",
    "mechs": "Mech", "mech": "Mech", "mechanical": "Mech",
    "pirates": "Pirate", "pirate": "Pirate",
    "quilboar": "Quilboar", "quilboars": "Quilboar",
    "naga": "Naga", "nagas": "Naga",
    "aberration": "Aberration", "aberrations": "Aberration",
    "neutral": "Neutral", "all": "Neutral",
}


def _norm_tribe(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    return _TRIBE_ALIASES.get(str(t).strip().lower()) or str(t).strip().title()


@lru_cache(maxsize=1)
def load_playstyle_prior(path: Optional[str] = None) -> dict:
    with open(path or _PRIOR_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def live_pool_names(path: Optional[str] = None) -> frozenset:
    p = path or _POOL_PATH
    if os.path.isfile(p):
        data = json.load(open(p, encoding="utf-8"))
        names = data.get("names") or [m["name"] for m in data.get("minions", [])]
        return frozenset(names)
    prior = load_playstyle_prior()
    keys = {
        k
        for comps in (prior.get("comp_tiers") or {}).values()
        for c in comps
        for k in (c.get("key_minions") or [])
    }
    keys |= set(prior.get("aberration_soft_key_minions") or [])
    return frozenset(keys)


@lru_cache(maxsize=1)
def out_of_pool_names() -> frozenset:
    prior = load_playstyle_prior()
    oop = set(prior.get("out_of_pool_cards") or [])
    if os.path.isfile(_POOL_PATH):
        doc = json.load(open(_POOL_PATH, encoding="utf-8"))
        oop |= set(doc.get("naga_excluded_names") or [])
    # Hard bans Aidan called out explicitly
    oop.update({
        "Archlich Kel'Thuzad",
        "Monstrous Macaw",
        "Young Murk-Eye",
        "Old Murk-Eye",
        "Zapp Slywick",
        "Darkgaze Elder",
    })
    return frozenset(oop)


def is_out_of_pool(name: Optional[str]) -> bool:
    if not name:
        return False
    n = str(name).strip()
    if n in out_of_pool_names():
        return True
    pool = live_pool_names()
    if pool and n in pool:
        return False
    # Name present in OOP list variants already handled; unknown names are not
    # auto-OOP (tokens / beetles / deities). Only flag if explicitly listed.
    return False


def tribe_weight(tribe: Optional[str]) -> float:
    prior = load_playstyle_prior()
    weights = prior.get("tribe_weights") or {}
    t = _norm_tribe(tribe)
    if t == "Naga" or (t == "Naga"):
        return 0.0
    if not t:
        return float(weights.get("other", 0.25))
    if t in weights:
        return float(weights[t])
    tiers = prior.get("tribe_tiers") or {}
    for bucket in ("S", "A", "B"):
        if t in (tiers.get(bucket) or []):
            return float(weights.get(bucket, 0.25))
    return float(weights.get("other", 0.25))


def _comp_index(prior: dict) -> Dict[str, Tuple[str, frozenset]]:
    pool = live_pool_names()
    oop = out_of_pool_names()
    out: Dict[str, Tuple[str, frozenset]] = {}
    for tier, comps in (prior.get("comp_tiers") or {}).items():
        for c in comps:
            if "naga" in (c.get("name") or "").lower():
                continue
            if _norm_tribe(c.get("tribe")) == "Naga":
                continue
            keys = [
                k for k in (c.get("key_minions") or [])
                if k not in oop and (not pool or k in pool)
            ]
            if len(keys) < 2:
                continue
            out[c["name"]] = (tier, frozenset(keys))
    return out


def matching_comps(board_names: Sequence[str]) -> List[Tuple[str, str, int]]:
    prior = load_playstyle_prior()
    board = {str(n) for n in board_names if n}
    scored = []
    for name, (tier, keys) in _comp_index(prior).items():
        hit = len(board & keys)
        if hit:
            scored.append((name, tier, hit))
    scored.sort(key=lambda x: (-x[2], {"S": 0, "A": 1, "B": 2}.get(x[1], 9), x[0]))
    return scored


def comp_weight(board_names: Sequence[str] = (), comp_name: Optional[str] = None) -> float:
    prior = load_playstyle_prior()
    weights = prior.get("comp_weights") or {"S": 1.0, "A": 0.55, "B": 0.15, "other": 0.25}
    if not prior.get("use_comp_tiers", True):
        return float(weights.get("other", 0.25))
    idx = _comp_index(prior)
    if comp_name and comp_name in idx:
        return float(weights.get(idx[comp_name][0], weights.get("other", 0.25)))
    matches = matching_comps(board_names)
    if not matches:
        return float(weights.get("other", 0.25))
    return float(weights.get(matches[0][1], weights.get("other", 0.25)))


def score_board_playstyle(
    board_names: Sequence[str],
    tribe: Optional[str] = None,
) -> float:
    """``final = 0.5*tribe_w + 0.5*comp_w``; Naga/heavy-OOP board → 0."""
    names = [n for n in board_names if n]
    if _norm_tribe(tribe) == "Naga":
        return 0.0
    oop_n = sum(1 for n in names if is_out_of_pool(n))
    if names and oop_n >= max(2, (len(names) + 1) // 2):
        return 0.0
    return 0.5 * tribe_weight(tribe) + 0.5 * comp_weight(names)


def key_minion_boost(name: Optional[str]) -> Tuple[float, Optional[str]]:
    """Negative = promote (placement units)."""
    if not name or is_out_of_pool(name):
        return 0.0, None
    prior = load_playstyle_prior()
    for tier, boost in (("S", -0.55), ("A", -0.28), ("B", -0.08)):
        for c in (prior.get("comp_tiers") or {}).get(tier) or []:
            if name in (c.get("key_minions") or []):
                return boost, f"HSReplay {tier}-comp key ({c['name']})"
    if name in (prior.get("aberration_soft_key_minions") or []):
        return -0.12, "Aberration soft key (no HSReplay comp yet)"
    return 0.0, None


def bias_shop_scores(
    shop: Iterable[dict],
    *,
    board_names: Sequence[str] = (),
    direction: Optional[str] = None,
) -> List[Tuple[dict, float, Optional[str]]]:
    """[(card, placement_delta, reason)] — positive delta demotes the buy."""
    out: List[Tuple[dict, float, Optional[str]]] = []
    idx = _comp_index(load_playstyle_prior())
    top = matching_comps(board_names)[:2]
    for card in shop:
        name = card.get("name") if isinstance(card, dict) else getattr(card, "name", None)
        tribes = []
        if isinstance(card, dict):
            tribes = card.get("tribes") or card.get("races") or []
        else:
            tribes = getattr(card, "tribes", None) or []
        if isinstance(tribes, str):
            tribes = [tribes]
        tribes_n = [_norm_tribe(t) for t in tribes]

        if "Naga" in tribes_n or is_out_of_pool(name):
            out.append((card, 2.5, f"{name} out of pool / Naga — never buy"))
            continue

        delta = 0.0
        reason: Optional[str] = None
        boost, why = key_minion_boost(name)
        if boost:
            delta += boost
            reason = why

        for t in tribes_n:
            tw = tribe_weight(t)
            if tw <= 0.01:
                delta += 2.5
                reason = f"{t} excluded"
            elif tw >= 0.99:
                delta += -0.18
                reason = reason or f"S-tribe {t} lean"
            elif tw <= 0.20 and direction and _norm_tribe(direction) != t:
                delta += 0.12
                reason = reason or f"B-tribe {t} soft demote off-direction"

        for cname, tier, _hit in top:
            if tier in ("S", "A") and cname in idx and name in idx[cname][1]:
                delta += -0.35 if tier == "S" else -0.18
                reason = f"finish {tier} {cname}"
                break

        out.append((card, delta, reason))
    return out


# Aliases matching scoring_hook.functions
load_playstyle_prior  # noqa: keep
score_board_playstyle = score_board_playstyle  # explicit export name


__all__ = [
    "load_playstyle_prior",
    "live_pool_names",
    "out_of_pool_names",
    "is_out_of_pool",
    "tribe_weight",
    "comp_weight",
    "score_board_playstyle",
    "bias_shop_scores",
    "key_minion_boost",
    "matching_comps",
]
