"""Firestone final-board composition reference (Phase 2W).

Joins ``data/stats/firestone_final_boards.json`` example minions to
``data/cards/bg_cards.json`` and the active Tavern pool for printed tier,
golden, tribe, keywords, and printed/base atk+hp.

Firestone rows are **final-board** snapshots (high-MMR, past-seven). They
are not turn-labeled mid-game boards. Weighting: each example board in an
archetype carries ``boardCount / n_examples`` so popular comps count more.
Unweighted (equal per example) is retained for reconciliation.
"""

from __future__ import annotations

import json
import os
import statistics as st
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.active_tavern_pool import (
    active_tavern_card_ids,
    active_tavern_names,
)
from hsbg_coach.cards import CardKnowledge, load_kb

_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIRESTONE_FINAL_BOARDS = os.path.join(
    _ROOT, "data", "stats", "firestone_final_boards.json"
)

TIERS = (1, 2, 3, 4, 5, 6)
HIGH_TIER_MIN = 4


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _pctl(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    idx = q * (len(xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def is_golden_card_id(card_id: str) -> bool:
    cid = str(card_id or "")
    return cid.endswith("_G") or "_G_" in cid


def base_card_id(card_id: str) -> str:
    """Strip the Firestone/HSJSON golden suffix for KB / pool lookup."""
    cid = str(card_id or "")
    if cid.endswith("_G"):
        return cid[:-2]
    return cid


def resolve_card(
    card_id: str,
    name: str,
    kb_id: Dict[str, CardKnowledge],
    kb_name: Dict[str, CardKnowledge],
) -> Tuple[Optional[CardKnowledge], bool, str]:
    """Return (kb_row, golden, resolve_path)."""
    cid = str(card_id or "")
    golden = is_golden_card_id(cid)
    if cid in kb_id:
        return kb_id[cid], golden, "card_id"
    base = base_card_id(cid)
    if base and base in kb_id:
        return kb_id[base], golden or False, "base_card_id"
    if name and name in kb_name:
        return kb_name[name], golden, "name"
    return None, golden, "unresolved"


def load_lookup() -> Dict:
    kb = load_kb()
    kb_name = {}
    for ck in kb.values():
        kb_name.setdefault(ck.name, ck)
    return {
        "kb_id": kb,
        "kb_name": kb_name,
        "pool_ids": active_tavern_card_ids(),
        "pool_names": active_tavern_names(),
    }


def join_minion(raw: Dict, lookup: Dict) -> Dict:
    """Join one Firestone example minion to printed KB / pool fields."""
    card_id = str(raw.get("cardId") or raw.get("card_id") or "")
    name = str(raw.get("name") or "")
    ck, golden, path = resolve_card(
        card_id, name, lookup["kb_id"], lookup["kb_name"]
    )
    joined = ck is not None
    printed_atk = int(ck.attack or 0) if ck else None
    printed_hp = int(ck.health or 0) if ck else None
    printed_tier = int(ck.tier) if ck and ck.tier is not None else None
    factor = 2 if golden else 1
    printed_raw = None
    if printed_atk is not None and printed_hp is not None:
        printed_raw = float((printed_atk + printed_hp) * factor)
    tribes = list(ck.tribes) if ck else (
        [raw["tribe"]] if raw.get("tribe") else []
    )
    keywords = list(ck.keywords) if ck else list(raw.get("keywords") or [])
    base = base_card_id(card_id)
    in_pool = bool(
        (base and base in lookup["pool_ids"])
        or (card_id in lookup["pool_ids"])
        or (name in lookup["pool_names"])
    )
    return {
        "card_id": card_id,
        "base_card_id": base,
        "name": name or (ck.name if ck else ""),
        "golden": bool(golden),
        "joined": joined,
        "resolve_path": path,
        "in_active_pool": in_pool,
        "printed_tier": printed_tier,
        "printed_attack": printed_atk,
        "printed_health": printed_hp,
        "printed_raw": printed_raw,
        "combat_attack": raw.get("atk"),
        "combat_health": raw.get("health"),
        "tribes": tribes,
        "archetype": tribes[0] if tribes else "tribeless",
        "keywords": keywords,
        "kb_card_id": ck.card_id if ck else None,
    }


def load_joined_boards(
    path: Optional[str] = None,
    lookup: Optional[Dict] = None,
) -> Dict:
    """Load Firestone examples, join, and attach boardCount weights."""
    path = path or FIRESTONE_FINAL_BOARDS
    lookup = lookup or load_lookup()
    raw = json.load(open(path, encoding="utf-8"))
    boards: List[Dict] = []
    minions: List[Dict] = []
    n_unresolved = 0
    n_minions = 0
    unique_ids = set()
    unique_joined = set()
    for arch in raw.get("boards") or []:
        examples = list(arch.get("examples") or [])
        n_ex = max(1, len(examples))
        board_count = float(arch.get("boardCount") or 0)
        w = board_count / n_ex
        for ex in examples:
            rows = []
            for m in ex.get("minions") or []:
                n_minions += 1
                jm = join_minion(m, lookup)
                jm["weight"] = w
                jm["unweighted"] = 1.0
                jm["arch_key"] = arch.get("archetype")
                jm["arch_name"] = arch.get("name")
                jm["arch_tribe"] = arch.get("tribe")
                jm["board_count"] = board_count
                rows.append(jm)
                minions.append(jm)
                unique_ids.add(jm["card_id"])
                if jm["joined"]:
                    unique_joined.add(jm["kb_card_id"] or jm["base_card_id"])
                else:
                    n_unresolved += 1
            boards.append({
                "archetype": arch.get("archetype"),
                "name": arch.get("name"),
                "tribe": arch.get("tribe"),
                "board_count": board_count,
                "weight": w,
                "mmr": ex.get("mmr"),
                "n_minions": len(rows),
                "minions": rows,
            })
    return {
        "meta": {
            "source": raw.get("_source"),
            "fetched": raw.get("_fetched"),
            "mmr": raw.get("_mmr"),
            "period": raw.get("_period"),
            "hero_data_points": raw.get("_heroDataPoints"),
            "n_archetypes": len(raw.get("boards") or []),
            "is_final_board_data": True,
            "compare_to": (
                "simulated last/alive late-game boards (plus T12–T14), "
                "not early turns"
            ),
        },
        "boards": boards,
        "minions": minions,
        "n_example_boards": len(boards),
        "n_minions": n_minions,
        "n_unresolved": n_unresolved,
        "n_unique_card_ids": len(unique_ids),
        "n_unique_joined_cards": len(unique_joined),
        "sum_board_count": float(sum(
            float(a.get("boardCount") or 0) for a in (raw.get("boards") or [])
        )),
        "sum_example_weight": float(sum(b["weight"] for b in boards)),
    }


def _weighted_share(rows: Sequence[Dict], pred, *, weighted: bool) -> Optional[float]:
    num = 0.0
    den = 0.0
    for r in rows:
        w = float(r["weight"] if weighted else r.get("unweighted") or 1.0)
        den += w
        if pred(r):
            num += w
    if den <= 1e-12:
        return None
    return num / den


def _weighted_mean(rows: Sequence[Dict], key: str, *, weighted: bool) -> Optional[float]:
    num = 0.0
    den = 0.0
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        w = float(r["weight"] if weighted else r.get("unweighted") or 1.0)
        num += w * float(v)
        den += w
    if den <= 1e-12:
        return None
    return num / den


def _tier_hist(rows: Sequence[Dict], *, weighted: bool) -> Dict[str, float]:
    acc = {str(t): 0.0 for t in TIERS}
    acc["7plus"] = 0.0
    den = 0.0
    for r in rows:
        tier = r.get("printed_tier")
        if tier is None:
            continue
        w = float(r["weight"] if weighted else r.get("unweighted") or 1.0)
        den += w
        t = int(tier)
        if t >= 7:
            acc["7plus"] += w
        else:
            acc[str(min(6, max(1, t)))] += w
    if den <= 1e-12:
        return {k: 0.0 for k in acc}
    return {k: v / den for k, v in acc.items()}


def _raw_distribution(rows: Sequence[Dict], *, weighted: bool) -> Dict:
    pairs = []
    for r in rows:
        v = r.get("printed_raw")
        if v is None:
            continue
        w = float(r["weight"] if weighted else r.get("unweighted") or 1.0)
        pairs.append((float(v), w))
    if not pairs:
        return {
            "mean": None, "median": None, "p25": None, "p75": None, "n": 0,
        }
    # Expand lightly for percentiles: repeat by rounded relative weight.
    # Keep compact — use unweighted values for percentile shape, weighted mean.
    values = [v for v, _ in pairs]
    wmean = sum(v * w for v, w in pairs) / sum(w for _, w in pairs)
    return {
        "mean": float(wmean),
        "median": _pctl(values, 0.5),
        "p25": _pctl(values, 0.25),
        "p75": _pctl(values, 0.75),
        "n": len(values),
    }


def _tribe_mix(rows: Sequence[Dict], *, weighted: bool) -> Dict[str, float]:
    acc: Counter = Counter()
    den = 0.0
    for r in rows:
        w = float(r["weight"] if weighted else r.get("unweighted") or 1.0)
        acc[str(r.get("archetype") or "tribeless")] += w
        den += w
    if den <= 1e-12:
        return {}
    return {k: v / den for k, v in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))}


def _card_freq(rows: Sequence[Dict], *, weighted: bool, top_n: int = 20) -> List[Dict]:
    acc: Counter = Counter()
    den = 0.0
    for r in rows:
        if not r.get("joined"):
            continue
        w = float(r["weight"] if weighted else r.get("unweighted") or 1.0)
        name = r.get("name") or r.get("kb_card_id")
        if not name:
            continue
        acc[name] += w
        den += w
    if den <= 1e-12:
        return []
    return [
        {"name": k, "share": v / den, "weight": v}
        for k, v in acc.most_common(top_n)
    ]


def summarize_reference(joined: Dict, *, weighted: bool = True) -> Dict:
    rows = [m for m in joined["minions"] if m.get("joined") and m.get("printed_tier")]
    hist = _tier_hist(rows, weighted=weighted)
    t4 = sum(hist.get(str(t), 0.0) for t in range(4, 7)) + hist.get("7plus", 0.0)
    t5 = sum(hist.get(str(t), 0.0) for t in range(5, 7)) + hist.get("7plus", 0.0)
    t6 = hist.get("6", 0.0) + hist.get("7plus", 0.0)
    sizes = [float(b["n_minions"]) for b in joined["boards"]]
    size_w = (
        sum(float(b["n_minions"]) * float(b["weight"]) for b in joined["boards"])
        / max(1e-12, sum(float(b["weight"]) for b in joined["boards"]))
    )
    n_joined = sum(1 for m in joined["minions"] if m.get("joined"))
    n_pool = sum(1 for m in joined["minions"] if m.get("in_active_pool"))
    n_all = max(1, joined["n_minions"])
    return {
        "weighting": "boardCount_per_example" if weighted else "equal_per_example",
        "n_example_boards": joined["n_example_boards"],
        "n_minions": joined["n_minions"],
        "n_joined": n_joined,
        "join_rate": n_joined / n_all,
        "pool_id_or_name_rate": n_pool / n_all,
        "n_unique_card_ids": joined["n_unique_card_ids"],
        "n_unique_joined_cards": joined["n_unique_joined_cards"],
        "sum_board_count": joined["sum_board_count"],
        "sum_example_weight": joined["sum_example_weight"],
        "weight_reconcile": abs(
            joined["sum_board_count"] - joined["sum_example_weight"]
        ) < 1e-6,
        "tier_histogram": hist,
        "t4_plus_share": t4,
        "t5_plus_share": t5,
        "t6_plus_share": t6,
        "t6_share": hist.get("6", 0.0),
        "t7_plus_share": hist.get("7plus", 0.0),
        "mean_printed_tier": _weighted_mean(rows, "printed_tier", weighted=weighted),
        "mean_printed_raw": _weighted_mean(rows, "printed_raw", weighted=weighted),
        "printed_raw": _raw_distribution(rows, weighted=weighted),
        "golden_share": _weighted_share(rows, lambda r: r.get("golden"), weighted=weighted),
        "mean_board_size": float(size_w) if weighted else _mean(sizes),
        "tribe_mix": _tribe_mix(rows, weighted=weighted),
        "card_frequency_top20": _card_freq(rows, weighted=weighted),
        "in_active_pool_share": _weighted_share(
            rows, lambda r: r.get("in_active_pool"), weighted=weighted
        ),
    }


def build_firestone_reference(path: Optional[str] = None) -> Dict:
    joined = load_joined_boards(path)
    weighted = summarize_reference(joined, weighted=True)
    unweighted = summarize_reference(joined, weighted=False)
    coverage = {
        "join_rate": weighted["join_rate"],
        "n_example_boards": joined["n_example_boards"],
        "n_unique_joined_cards": joined["n_unique_joined_cards"],
        "n_unique_card_ids": joined["n_unique_card_ids"],
        "n_minions": joined["n_minions"],
        "n_unresolved": joined["n_unresolved"],
        "pool_name_rate": sum(
            1 for m in joined["minions"] if m.get("name") in load_lookup()["pool_names"]
        ) / max(1, joined["n_minions"]),
        "pool_id_or_name_rate": weighted["pool_id_or_name_rate"],
        "weight_reconcile": weighted["weight_reconcile"],
        "is_final_board_data": True,
        "examples_per_archetype": 3,
        "note": (
            "57 example boards (3 per archetype) joined at card-ID/name. "
            "Adequate for tier / printed-raw / golden / tribe mix. "
            "Card-frequency overlap is noisier (3 boards/arch)."
        ),
    }
    return {
        "meta": joined["meta"],
        "coverage": coverage,
        "weighted": weighted,
        "unweighted": unweighted,
        "reconciliation": {
            "n_minions": joined["n_minions"],
            "n_joined": weighted["n_joined"],
            "n_unresolved": joined["n_unresolved"],
            "join_plus_unresolved": (
                weighted["n_joined"] + joined["n_unresolved"]
            ),
            "sum_board_count": joined["sum_board_count"],
            "sum_example_weight": joined["sum_example_weight"],
            "weight_delta": (
                joined["sum_example_weight"] - joined["sum_board_count"]
            ),
        },
    }
