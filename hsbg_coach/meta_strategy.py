"""Soft meta priors: HSReplay/Firestone comps + lobby tribe strength.

Playstyle Aidan wants for the live #1 move:
  1. Prefer comps from the Firestone/HSReplay guide list (meta comps).
  2. Soft-bias toward the lobby's strongest tribe (avg-placement prior —
     Firestone exports averagePosition, which tracks first-place rate).
  3. Do NOT hard-lock — if shop/board clearly supports another *listed*
     tribe's viable comp, pivot.

Returns placement adjustments (negative = better finish) for game_value.rank_actions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .stats import StatsDB, CompStats

# Module-level cache for the default StatsDB tribe table (StatsDB is unhashable,
# so we cannot @lru_cache a function that takes db=).
_DEFAULT_TRIBE_TABLE: Optional[Dict[str, float]] = None


@dataclass
class MetaPrior:
    preferred_tribe: Optional[str]
    reason: str
    tribe_scores: Dict[str, float]          # lower avg placement = stronger
    listed_comps: List[str]                 # archetype/comp names considered


def _tribe_key(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    return str(t).strip().lower() or None


def _compute_tribe_table(comps: List[CompStats]) -> Dict[str, float]:
    """Tribe -> popularity-weighted average placement (lower = stronger / more firsts)."""
    buckets: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for c in comps:
        tr = _tribe_key(c.tribe)
        if not tr:
            continue
        pop = max(0.01, float(c.popularity or 0.01))
        buckets[tr].append((float(c.average_position), pop))
    out: Dict[str, float] = {}
    for tr, rows in buckets.items():
        w = sum(p for _, p in rows)
        out[tr] = sum(pos * p for pos, p in rows) / w
    return out


def tribe_placement_table(db: Optional[StatsDB] = None) -> Dict[str, float]:
    """Tribe -> average placement across listed meta comps (lower = stronger).

    Weighted by popularity so a rare good niche doesn't outrank a popular winner.
    This is the deck-tracker-style 'which tribes take first' prior from
    Firestone/HSReplay averagePosition (proxy for first-place%).
    """
    global _DEFAULT_TRIBE_TABLE
    if db is None:
        if _DEFAULT_TRIBE_TABLE is not None:
            return _DEFAULT_TRIBE_TABLE
        db = StatsDB.load()
        _DEFAULT_TRIBE_TABLE = _compute_tribe_table(db.comps)
        return _DEFAULT_TRIBE_TABLE
    return _compute_tribe_table(db.comps)


def lobby_tribes(snapshot) -> List[str]:
    """Tribes seen on opponent profiles + our board (lobby context)."""
    counts: Dict[str, int] = defaultdict(int)
    profiles = (snapshot.get("opponent_profiles") if isinstance(snapshot, dict)
                else getattr(snapshot, "opponent_profiles", None)) or []
    for p in profiles:
        tr = _tribe_key(p.get("tribe") if isinstance(p, dict) else None)
        if tr:
            counts[tr] += 2  # lobby opponents weigh more
    board = (snapshot.get("board") if isinstance(snapshot, dict)
             else getattr(snapshot, "board", None)) or []
    for m in board:
        tags = m.get("tags") if isinstance(m, dict) else getattr(m, "tags", {}) or {}
        if isinstance(m, dict):
            race = m.get("tribe") or tags.get("CARDRACE")
        else:
            race = getattr(m, "tribe", None) or tags.get("CARDRACE")
        tr = _tribe_key(race)
        if tr:
            counts[tr] += 1
    return sorted(counts, key=counts.get, reverse=True)


def board_tribe_commitment(board) -> Tuple[Optional[str], int]:
    counts: Dict[str, int] = defaultdict(int)
    for m in board or []:
        tags = m.get("tags") if isinstance(m, dict) else getattr(m, "tags", {}) or {}
        race = (m.get("tribe") if isinstance(m, dict) else getattr(m, "tribe", None))
        race = race or tags.get("CARDRACE")
        tr = _tribe_key(race)
        if tr and tr not in ("all", "neutral"):
            counts[tr] += 1
    if not counts:
        return None, 0
    top = max(counts, key=counts.get)
    return top, counts[top]


def choose_meta_prior(snapshot, hero_ctx=None, db: Optional[StatsDB] = None) -> MetaPrior:
    """Pick a soft preferred tribe from meta comps + lobby, without hard-locking."""
    db = db or StatsDB.load()
    scores = tribe_placement_table(db)
    lobby = lobby_tribes(snapshot)
    board = (snapshot.get("board") if isinstance(snapshot, dict)
             else getattr(snapshot, "board", None)) or []
    committed, n = board_tribe_commitment(board)

    # Hero's best tribes from population stats (HSReplay/Firestone hero sheet).
    hero_tribes: List[str] = []
    if hero_ctx and getattr(hero_ctx, "target_tribe", None):
        hero_tribes.append(_tribe_key(hero_ctx.target_tribe))
    try:
        hero_name = getattr(hero_ctx, "hero", None) if hero_ctx else None
        if hero_name:
            h = db.hero(hero_name)
            if h:
                hero_tribes.extend(_tribe_key(t) for t in (h.best_tribes or []))
    except Exception:
        pass
    hero_tribes = [t for t in hero_tribes if t]

    # Rank tribes: stronger meta placement first; boost if present in lobby/hero.
    def rank(tr: str) -> Tuple[float, float]:
        base = scores.get(tr, 4.5)
        lobby_boost = -0.15 if tr in lobby else 0.0
        hero_boost = -0.10 if tr in hero_tribes else 0.0
        return (base + lobby_boost + hero_boost, base)

    candidates = list(scores.keys()) or hero_tribes or lobby
    preferred = min(candidates, key=rank) if candidates else None

    listed = [c.name for c in db.comps if _tribe_key(c.tribe)]
    # If board is already committed (≥2 of a tribe) and that tribe has a listed
    # viable comp, stick with it — pivot only when another listed tribe is clearly
    # open in shop (handled in meta_buy_adjust).
    if committed and n >= 2 and committed in scores:
        best = scores.get(preferred or committed, 4.5)
        if scores[committed] <= best + 0.6:
            preferred = committed
            return MetaPrior(preferred, f"committed {committed} (listed meta)",
                             scores, listed)

    reason = "meta tribe prior"
    if preferred:
        bits = [f"{preferred} avg {scores.get(preferred, 4.5):.2f}"]
        if preferred in lobby:
            bits.append("in lobby")
        if preferred in hero_tribes:
            bits.append("hero lean")
        reason = " · ".join(bits)
    return MetaPrior(preferred, reason, scores, listed)


def candidate_tribe(minion, kb=None) -> Optional[str]:
    if minion is None:
        return None
    if isinstance(minion, dict):
        tr = minion.get("tribe") or (minion.get("tags") or {}).get("CARDRACE")
        if tr:
            return _tribe_key(tr)
        name = minion.get("name")
    else:
        tags = getattr(minion, "tags", {}) or {}
        tr = getattr(minion, "tribe", None) or tags.get("CARDRACE")
        if tr:
            return _tribe_key(tr)
        name = getattr(minion, "name", None)
    if kb is not None and name:
        try:
            from .cards import by_name
            ck = by_name(kb).get(name)
            tribes = getattr(ck, "tribes", None) or []
            if tribes:
                return _tribe_key(tribes[0])
        except Exception:
            pass
    return None


def is_listed_comp_piece(card_name: Optional[str], tribe: Optional[str],
                         db: Optional[StatsDB] = None) -> bool:
    """True if this card is a core piece of a listed meta comp (optionally on-tribe)."""
    if not card_name:
        return False
    db = db or StatsDB.load()
    name = card_name.lower()
    for c in db.comps:
        if tribe and _tribe_key(c.tribe) not in (None, tribe):
            continue
        cores = [x.lower() for x in (c.core_cards or [])]
        if name in cores:
            return True
    return False


def meta_buy_adjust(action, snapshot, hero_ctx=None, kb=None,
                    db: Optional[StatsDB] = None) -> Tuple[float, Optional[str]]:
    """Placement delta for a BUY toward Aidan's meta playstyle.

    Soft: promote preferred-tribe + listed-comp cores; allow pivot to another
    listed tribe when the shop piece is a core of that tribe's viable comp and
    the board isn't hard-committed elsewhere.
    """
    kind = getattr(action, "kind", None) or ""
    from . import actions as acts
    if kind not in ("buy", "BUY", acts.BUY):
        return 0.0, None
    db = db or StatsDB.load()
    prior = choose_meta_prior(snapshot, hero_ctx=hero_ctx, db=db)
    minion = (action.detail or {}).get("minion")
    ctribe = candidate_tribe(minion, kb=kb)
    name = getattr(action, "target", None) or (
        minion.get("name") if isinstance(minion, dict) else None
    )
    board = (snapshot.get("board") if isinstance(snapshot, dict)
             else getattr(snapshot, "board", None)) or []
    committed, n = board_tribe_commitment(board)

    listed_piece = is_listed_comp_piece(name, ctribe, db=db)
    listed_any = is_listed_comp_piece(name, None, db=db)

    # Preferred tribe core → strong promote.
    if ctribe and prior.preferred_tribe and ctribe == prior.preferred_tribe and listed_piece:
        return -0.55, f"meta {ctribe} core — {prior.reason}"
    if ctribe and prior.preferred_tribe and ctribe == prior.preferred_tribe:
        return -0.25, f"on preferred tribe ({ctribe})"

    # Pivot: another listed tribe's core, board not hard-committed to something else.
    if listed_piece and ctribe and ctribe != prior.preferred_tribe:
        if n < 2 or committed == ctribe:
            return -0.40, f"pivot to listed {ctribe} comp (shop opened it)"
        if committed and committed != ctribe and n >= 3:
            return 0.20, f"off your {committed} commitment"
        return -0.15, f"listed {ctribe} piece (soft pivot)"

    # Random synergy / unlisted: mild demote once a preferred tribe exists.
    if prior.preferred_tribe and ctribe and ctribe != prior.preferred_tribe and not listed_any:
        if n >= 2 and committed == prior.preferred_tribe:
            return 0.25, f"off-meta vs your {prior.preferred_tribe} line"
        return 0.10, "not a listed meta-comp piece"

    if listed_any:
        return -0.20, "listed meta-comp piece"
    return 0.0, None
