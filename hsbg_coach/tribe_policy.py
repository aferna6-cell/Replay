"""Lobby-tribe direction + buy policy for the live coach.

Primary rule (patch 36.6.1 Aberrations):
  * Pick a **general tribal direction** from tribes that are **in this lobby**
    (board signals ∩ available lobby tribes). Never Naga — quarantined.
  * Mid/late: only recommend **good on-direction** buys (or true flex keys).
  * Demote/forbid off-direction junk and low-tavern chaff vs current tavern tier.

``available_tribes`` on the snapshot is the lobby set when known; otherwise we
infer from shop/board card knowledge and still hard-exclude quarantined tribes.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Officially rotated out for 36.6.1 — HSJSON pool flags may still list them.
QUARANTINED_TRIBES = frozenset({"naga"})

# Lobby-legal tribes this patch (Aberration guaranteed early weeks).
PATCH_TRIBES = frozenset({
    "aberration", "beast", "demon", "dragon", "elemental",
    "mech", "murloc", "pirate", "quilboar", "undead",
})

# BACON_SUBSET_* / CARDRACE-ish tokens -> canonical tribe display name.
SUBSET_TO_TRIBE = {
    "MECH": "Mech", "MECHANICAL": "Mech",
    "MURLOC": "Murloc",
    "DEMON": "Demon",
    "DRAGON": "Dragon",
    "BEAST": "Beast",
    "PIRATE": "Pirate",
    "ELEMENTAL": "Elemental", "ELEMENTALS": "Elemental",
    "UNDEAD": "Undead",
    "QUILBOAR": "Quilboar",
    "NAGA": "Naga",
    "ABERRATION": "Aberration",
}

# Universal flex keys — OK to buy off-tribe when truly premium.
FLEX_KEY_NAMES = frozenset({
    "brann bronzebeard", "king mukla", "lightfang enforcer",
    "baron rivendare", "titus rivendare", "nomi, kitchen nightmare",
    "kalecgos", "razorgore, the untamed", "young murk-eye",
    "amalgadon", "one-off", # placeholder; real flex set below
})
# Trim accidental placeholder
FLEX_KEY_NAMES = frozenset({
    "brann bronzebeard", "lightfang enforcer",
    "baron rivendare", "titus rivendare",
    "nomi, kitchen nightmare", "kalecgos",
    "amalgadon", "elder taggawag", "kangor's apprentice",
})


def _norm(tribe: Optional[str]) -> str:
    return (tribe or "").strip().lower()


def canonicalize(tribe: Optional[str]) -> Optional[str]:
    if not tribe:
        return None
    t = _norm(tribe)
    if t in QUARANTINED_TRIBES:
        return None
    for display in ("Aberration", "Beast", "Demon", "Dragon", "Elemental",
                    "Mech", "Murloc", "Pirate", "Quilboar", "Undead", "All"):
        if display.lower() == t:
            return display if display != "All" else "All"
    # subset token?
    mapped = SUBSET_TO_TRIBE.get(tribe.strip().upper())
    if mapped and _norm(mapped) not in QUARANTINED_TRIBES:
        return mapped
    return None


def filter_lobby_tribes(tribes: Optional[Iterable[str]]) -> List[str]:
    """Keep patch-legal, non-quarantined tribes; stable title-case order."""
    out: List[str] = []
    seen: Set[str] = set()
    for raw in tribes or []:
        c = canonicalize(raw)
        if c is None or c == "All":
            continue
        if _norm(c) not in PATCH_TRIBES:
            continue
        key = _norm(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def subset_token_to_tribe(token: str) -> Optional[str]:
    return canonicalize(SUBSET_TO_TRIBE.get(token.upper().replace("BACON_SUBSET_", "")))


def _raw_tribe_strings(m, kb=None) -> List[str]:
    """Unfiltered tribe strings (may include quarantined)."""
    raw = None
    if isinstance(m, dict):
        raw = m.get("tribes") or m.get("tribe")
        name = m.get("name")
        cid = m.get("card_id") or m.get("cardId")
    else:
        raw = getattr(m, "tribes", None)
        name = getattr(m, "name", None)
        cid = getattr(m, "card_id", None)
    if isinstance(raw, str):
        raw = [raw]
    if raw:
        return [str(x) for x in raw]
    ck = None
    if kb is not None:
        if cid and cid in kb:
            ck = kb[cid]
        elif name:
            from .cards import by_name
            ck = by_name(kb).get(name)
    if ck is not None:
        return [str(x) for x in (ck.tribes or [])]
    # Cards outside the live-pool KB (e.g. Defiant Shipwright, Coldlight
    # Diver, dual Demon/Naga Ominous Seer): HearthstoneJSON fallback.
    from .cards import fallback_card
    fb = fallback_card(cid, name) if (cid or name) else None
    return [str(x) for x in (fb or {}).get("tribes") or []]


def _minion_tribes(m, kb=None) -> List[str]:
    return [t for t in (canonicalize(x) for x in _raw_tribe_strings(m, kb)) if t]


def _is_quarantined_minion(m, kb=None) -> bool:
    """Only-Naga minions are out of the pool. A dual-tribe card (Ominous Seer
    is Demon/Naga) still plays as its live tribe."""
    raw = [_norm(x) for x in _raw_tribe_strings(m, kb)]
    return bool(raw) and any(x in QUARANTINED_TRIBES for x in raw) and all(
        x in QUARANTINED_TRIBES for x in raw if x != "all")


def board_tribe_counts(board, kb=None,
                       available: Optional[Sequence[str]] = None) -> Counter:
    allowed = {_norm(t) for t in filter_lobby_tribes(available)} if available else None
    counts: Counter = Counter()
    for m in board or []:
        for t in _minion_tribes(m, kb):
            if t == "All":
                continue
            if allowed is not None and _norm(t) not in allowed:
                continue
            counts[t] += 1
    return counts


def infer_direction(board, available_tribes: Optional[Sequence[str]] = None,
                    kb=None, hero_target: Optional[str] = None,
                    shop=None) -> Optional[str]:
    """General tribal direction for this lobby.

    Prefer a tribe that is both lobby-legal and already on the board; fall back to
    a hero prior ∩ lobby; then the strongest shop lean. Never returns Naga.
    """
    lobby = filter_lobby_tribes(available_tribes)
    lobby_set = {_norm(t) for t in lobby} if lobby else set(PATCH_TRIBES)

    counts = board_tribe_counts(board, kb, lobby or list(PATCH_TRIBES))
    if counts:
        best, n = counts.most_common(1)[0]
        if n >= 2 and _norm(best) in lobby_set:
            return best
        if n >= 1 and _norm(best) in lobby_set and len(board or []) >= 3:
            return best

    ht = canonicalize(hero_target)
    if ht and _norm(ht) in lobby_set:
        return ht

    if counts:
        best = counts.most_common(1)[0][0]
        if _norm(best) in lobby_set:
            return best

    if shop:
        sc = board_tribe_counts(shop, kb, lobby or list(PATCH_TRIBES))
        if sc:
            best = sc.most_common(1)[0][0]
            if _norm(best) in lobby_set:
                return best
    return None


def is_flex_key(name: Optional[str], card_id: Optional[str] = None,
                kb=None) -> bool:
    if name and name.strip().lower() in FLEX_KEY_NAMES:
        return True
    if kb and card_id and card_id in kb:
        ck = kb[card_id]
        tribes = [canonicalize(t) for t in (ck.tribes or [])]
        if any(t == "All" for t in tribes if t):
            return True
        # Named premium neutrals often have empty tribes + strong text.
        if ck.name and ck.name.strip().lower() in FLEX_KEY_NAMES:
            return True
    return False


def is_on_direction(minion, direction: Optional[str], kb=None) -> bool:
    if not direction:
        return True
    d = _norm(direction)
    for t in _minion_tribes(minion, kb):
        if t == "All" or _norm(t) == d:
            return True
    name = minion.get("name") if isinstance(minion, dict) else getattr(minion, "name", None)
    cid = (minion.get("card_id") if isinstance(minion, dict)
           else getattr(minion, "card_id", None))
    return is_flex_key(name, cid, kb)


def card_tier(minion, kb=None) -> Optional[int]:
    if isinstance(minion, dict):
        t = minion.get("tier") or (minion.get("tags") or {}).get("TECH_LEVEL")
        if t is not None:
            try:
                return int(t)
            except (TypeError, ValueError):
                pass
        name = minion.get("name")
        cid = minion.get("card_id")
    else:
        t = getattr(minion, "tier", None)
        if t is not None:
            return int(t)
        name = getattr(minion, "name", None)
        cid = getattr(minion, "card_id", None)
    if kb is None:
        return None
    ck = kb.get(cid) if cid else None
    if ck is None and name:
        from .cards import by_name
        ck = by_name(kb).get(name)
    return int(ck.tier) if ck and ck.tier else None


def plan_tribe(snapshot) -> Optional[str]:
    """Tribe of the lobby playbook's locked PLAN comp, if committed."""
    pb = (snapshot.get("playbook") if isinstance(snapshot, dict)
          else getattr(snapshot, "playbook", None))
    if isinstance(pb, dict) and pb.get("phase") == "commit" and pb.get("plan"):
        return canonicalize(pb.get("plan_tribe"))
    return None


def has_playbook(snapshot) -> bool:
    pb = (snapshot.get("playbook") if isinstance(snapshot, dict)
          else getattr(snapshot, "playbook", None))
    return isinstance(pb, dict) and bool(pb.get("phase"))


def playbook_direction(snapshot, kb=None) -> Optional[str]:
    """The lobby playbook's steering tribe — the one direction every buy
    heuristic follows. Locked PLAN → its tribe; before the lock, the strong
    lobby tribe your board already supports most (ties: playbook order), or
    None while the board holds none of them (stay flexible — FILL buys
    solids). Callers check ``has_playbook`` so a None here never falls back to
    the legacy board/shop inference."""
    locked = plan_tribe(snapshot)
    if locked:
        return locked
    pb = (snapshot.get("playbook") if isinstance(snapshot, dict)
          else getattr(snapshot, "playbook", None))
    strong = [canonicalize(t) for t in ((pb or {}).get("strong") or [])
              if isinstance(pb, dict)]
    strong = [t for t in strong if t]
    board = (snapshot.get("board") if isinstance(snapshot, dict)
             else getattr(snapshot, "board", None)) or []
    counts = board_tribe_counts(board, kb, strong) if strong else {}
    best = max(strong, key=lambda t: (counts.get(t, 0), -strong.index(t)), default=None)
    return best if best and counts.get(best, 0) >= 1 else None


def direction_buy_penalty(minion, snapshot, kb=None,
                          direction: Optional[str] = None) -> Tuple[float, Optional[str]]:
    """Placement penalty (higher = worse) for a BUY that fights tribal direction.

    Returns (penalty, reason). 0 when the buy is on-direction / flex / early flex.
    """
    board = snapshot.get("board") if isinstance(snapshot, dict) else getattr(snapshot, "board", [])
    board = board or []
    tavern = (snapshot.get("tavern_tier") if isinstance(snapshot, dict)
              else getattr(snapshot, "tavern_tier", None)) or 1
    available = (snapshot.get("available_tribes") if isinstance(snapshot, dict)
                 else getattr(snapshot, "available_tribes", None))
    hero_target = None
    if isinstance(snapshot, dict):
        hero_target = snapshot.get("target_tribe") or snapshot.get("build_tribe")

    # The lobby playbook IS the direction (locked PLAN, else its strong tribe
    # the board supports most); legacy inference only without a playbook.
    if direction is None and has_playbook(snapshot):
        direction = playbook_direction(snapshot, kb)
    elif direction is None:
        direction = infer_direction(
            board, available, kb=kb, hero_target=hero_target,
            shop=(snapshot.get("shop") if isinstance(snapshot, dict) else None))

    # Quarantine: never buy Naga (even if HSJSON still flags it in-pool).
    if _is_quarantined_minion(minion, kb):
        return 2.0, "Naga is out of the pool — never buy"

    name = minion.get("name") if isinstance(minion, dict) else getattr(minion, "name", None)

    # Live-pool scrub (playstyle_prior): Archlich Kel'Thuzad, Monstrous Macaw,
    # Young Murk-Eye, rotated Naga uniques, etc. — never treat as strong buys.
    try:
        from .playstyle_prior import is_out_of_pool, key_minion_boost
        if is_out_of_pool(name):
            return 2.5, f"{name} is out of the live 36.6.1 pool — never buy"
    except Exception:
        is_out_of_pool = None  # type: ignore
        key_minion_boost = None  # type: ignore
    cid = (minion.get("card_id") if isinstance(minion, dict)
           else getattr(minion, "card_id", None))
    if is_flex_key(name, cid, kb):
        return 0.0, None

    mt = card_tier(minion, kb)
    tier_pen = 0.0
    tier_reason = None
    if mt is not None and tavern:
        gap = int(tavern) - int(mt)
        # Tavern 5+ should not buy T2 (gap>=3); tavern 4+ demotes gap>=2 hard.
        if int(tavern) >= 5 and gap >= 3:
            tier_pen = 1.6
            tier_reason = f"T{mt} is chaff at tavern {tavern} — roll for on-tier"
        elif int(tavern) >= 4 and gap >= 2:
            tier_pen = 1.1 if gap >= 3 else 0.75
            tier_reason = f"T{mt} too low for tavern {tavern}"
        elif gap >= 3:
            tier_pen = 0.9
            tier_reason = f"T{mt} far below tavern {tavern}"

    if not direction:
        return tier_pen, tier_reason

    on = is_on_direction(minion, direction, kb)
    if on:
        # On-direction but low-tier: still demote unless it's a real piece
        # (tier within 1 of tavern, or Activate / named).
        if tier_pen and mt is not None and int(tavern) - int(mt) >= 3:
            return tier_pen * 0.7, tier_reason
        # Soft promote live HSReplay S/A key minions.
        try:
            from .playstyle_prior import key_minion_boost
            boost, why = key_minion_boost(name)
            if boost:
                return boost, why
        except Exception:
            pass
        return 0.0, None

    # Off-direction. Soft lobby prior must NOT lock buys before information:
    # only hard-demote once the board has committed (2+ on-direction) or mid/late.
    on_count = sum(1 for m in board if is_on_direction(m, direction, kb))
    committed = on_count >= 2
    mid_late = int(tavern) >= 3 or len(board) >= 4
    if committed and mid_late:
        pen = 1.35 + (0.25 if len(board) >= 7 else 0.0)
        reason = f"off-{direction} — doesn't advance your lobby direction"
        return max(pen, tier_pen), reason if pen >= tier_pen else tier_reason
    if mid_late and on_count >= 1:
        pen = 0.85
        reason = f"off-{direction} scatter — prefer on-direction pieces"
        return max(pen, tier_pen), reason if pen >= tier_pen else tier_reason
    if mid_late and on_count == 0:
        # Soft prior only — demote gently so a strong falling-in-lap tribe can win.
        pen = 0.35
        reason = f"soft {direction} prior — still flexible if a better tribe shows"
        return max(pen, tier_pen * 0.8), reason if pen >= (tier_pen * 0.8) else tier_reason
    return tier_pen, tier_reason


def build_direction_note(board, available_tribes=None, kb=None,
                         hero_target=None, shop=None) -> Optional[str]:
    d = infer_direction(board, available_tribes, kb=kb,
                        hero_target=hero_target, shop=shop)
    if not d:
        return None
    n = board_tribe_counts(board, kb, available_tribes).get(d, 0)
    return f"building: {d}" + (f" ({n} on board)" if n else "")


# ---------------------------------------------------------------------------
# Soft lobby lean (HSReplay / Firestone tribe placement → prior, not a lock)
# ---------------------------------------------------------------------------

def tribe_win_priors(db=None, available: Optional[Sequence[str]] = None,
                    manual: Optional[Dict[str, float]] = None
                    ) -> List[Tuple[str, float]]:
    """Rank lobby tribes by strength prior (lower placement / higher first% = better).

    ``manual`` maps tribe -> first-place rate or win-ish score in [0,1] (higher better),
    used when HSReplay lacks a tribe (e.g. early Aberration). Remaining lobby tribes
    without a prior share leftover mass uniformly so the lean stays soft.

    Returns [(Tribe, score)] best-first. Score is a soft weight in [0,1], NOT a lock.
    """
    lobby = filter_lobby_tribes(available) if available is not None else [
        "Aberration", "Beast", "Demon", "Dragon", "Elemental",
        "Mech", "Murloc", "Pirate", "Quilboar", "Undead",
    ]
    if not lobby:
        return []

    # manual first-place-ish rates (higher better)
    manual_n = {}
    if manual:
        parsed = {}
        for k, v in manual.items():
            c = canonicalize(k)
            if not c or _norm(c) not in {_norm(x) for x in lobby}:
                continue
            try:
                parsed[c] = float(v)
            except (TypeError, ValueError):
                continue
        # Ranking mode: small integers 1..N (1=best) → strength.
        vals = list(parsed.values())
        if vals and all(v == int(v) and 1 <= v <= 10 for v in vals):
            hi = max(vals)
            manual_n = {c: (hi + 1 - v) / hi for c, v in parsed.items()}
        else:
            # first% / weight in 0..1 (or any positive score)
            manual_n = parsed

    # From comps: average_position (lower better) → convert to strength.
    place: Dict[str, List[float]] = {}
    try:
        if db is None:
            from .stats import StatsDB
            db = StatsDB.load()
        for comp in getattr(db, "comps", []) or []:
            tribe = canonicalize(getattr(comp, "tribe", None))
            if not tribe or _norm(tribe) not in {_norm(x) for x in lobby}:
                continue
            ap = getattr(comp, "average_position", None)
            if ap is None:
                ap = getattr(comp, "avg_placement", None)
            if ap is None:
                continue
            place.setdefault(tribe, []).append(float(ap))
    except Exception:
        pass

    # Unified "goodness" weights (higher = stronger lean). Placement and first%
    # live on different scales — map both into a common weight before normalizing.
    #   placement avg 3.2 → ~1.8; avg 4.5 → ~0.5
    #   manual first% 0.15 → ~1.2; 0.55 → ~4.4 (dominates until board pivots)
    raw: Dict[str, float] = {}
    for t in lobby:
        if t in manual_n:
            p = max(0.0, float(manual_n[t]))
            raw[t] = max(0.05, p * 8.0) if p <= 1.0 else p
            continue
        vals = place.get(t) or []
        if vals:
            avg = sum(vals) / len(vals)
            raw[t] = max(0.05, 5.0 - avg)
        else:
            raw[t] = None  # type: ignore

    missing = [t for t, v in raw.items() if v is None]
    if missing:
        fill = 0.40  # mild — below a measured ~4.0 placement tribe
        for t in missing:
            raw[t] = fill

    ranked = sorted(((t, float(raw[t])) for t in lobby), key=lambda x: -x[1])
    # Normalize to soft weights summing to 1.
    s = sum(v for _, v in ranked) or 1.0
    return [(t, v / s) for t, v in ranked]


def soft_lean_tribe(board, available_tribes: Optional[Sequence[str]] = None,
                    kb=None, db=None, manual: Optional[Dict[str, float]] = None,
                    shop=None, turn: Optional[int] = None,
                    hero_target: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Soft default lean — NOT a hard lock.

    * Early (empty/thin board, turn<=2): return top lobby prior only as a *lean*
      label; buyers stay flexible (no forced direction).
    * Once board shows 2+ of another lobby tribe, **pivot** to that tribe.
    * If top prior never appears and another good tribe does, play the one you have.
    """
    priors = tribe_win_priors(db=db, available=available_tribes, manual=manual)
    lobby = [t for t, _ in priors] or filter_lobby_tribes(available_tribes)
    counts = board_tribe_counts(board, kb, lobby)
    top_prior = priors[0][0] if priors else None

    # Pivot: board commitment beats the prior.
    if counts:
        best, n = counts.most_common(1)[0]
        if n >= 2:
            return best, f"board lean {best} ({n}) — pivoted from lobby prior"
        # Single piece of a non-prior tribe mid-game with shop support → soft pivot
        if n >= 1 and (turn or 0) >= 3 and best != top_prior:
            shop_c = board_tribe_counts(shop, kb, lobby) if shop else Counter()
            if shop_c.get(best, 0) >= 1 or n >= 1 and len(board or []) >= 4:
                return best, f"opportunistic {best} — playing what showed"

    # No board info yet — soft prior only (caller must not hard-gate buys on this).
    early = (turn is None or int(turn) <= 2) and len(board or []) < 3
    if top_prior and early:
        return top_prior, f"soft lobby prior {top_prior} (not locked)"
    if top_prior and counts.get(top_prior, 0) >= 1:
        return top_prior, f"soft prior {top_prior} with board support"
    if top_prior:
        return top_prior, f"soft lobby prior {top_prior} — pivot if something better shows"
    return infer_direction(board, available_tribes, kb=kb, hero_target=hero_target,
                           shop=shop), "inferred"
