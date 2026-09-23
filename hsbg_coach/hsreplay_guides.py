"""HSReplay guide loader + hard coach hooks (Aidan-approved).

Reads committed JSON under ``data/hsreplay_guides/`` from
``scripts/ingest_hsreplay_guides.py``. Never invents strategy text — cite
HSReplay fields only.

Hard behaviors (not soft ±0.3):
  1. Fill board first — sparse + solid shop buy ⇒ ROLL cannot be NEXT
  2. Then lock an HSReplay comp — prefer key/enabler buys; overlay PLAN →
  3. Hero guides drive HP / cycle NEXT when the guide says so
  4. Trinket ranking folds in guide_text + board fit
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "hsreplay_guides")
)

_TIER_LABEL = {1: "S", 2: "A", 3: "B", 4: "C", 5: "D"}
_CARD_MARK = re.compile(r"\[\[([^\]|]+)(?:\|\|\d+)?\]\]")

_HP_NOW = re.compile(
    r"(hero power|heropower|use (?:your )?hp|\bhp (?:every|the|on|turn)|"
    r"click(?:ing)? (?:your )?hero|press(?:ing)? (?:your )?hero|"
    r"rally units are really nice with this hero power)",
    re.I,
)
_HP_SKIP = re.compile(
    r"(don'?t|do not|never|skip|avoid|no need to).{0,40}"
    r"(hero power|heropower|\bhp\b)",
    re.I,
)


def _guides_dir() -> str:
    return os.environ.get("HSREPLAY_GUIDES_DIR") or _DIR


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _get(snap, key, default=None):
    if isinstance(snap, dict):
        return snap.get(key, default)
    return getattr(snap, key, default)


@lru_cache(maxsize=1)
def load_heroes() -> dict:
    path = os.path.join(_guides_dir(), "heroes.json")
    if not os.path.isfile(path):
        return {"heroes": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_comps() -> dict:
    path = os.path.join(_guides_dir(), "comps.json")
    if not os.path.isfile(path):
        return {"comps": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_trinkets() -> dict:
    path = os.path.join(_guides_dir(), "trinkets.json")
    if not os.path.isfile(path):
        return {"trinkets": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def clear_guide_caches() -> None:
    load_heroes.cache_clear()
    load_comps.cache_clear()
    load_trinkets.cache_clear()
    _hero_index.cache_clear()
    _trinket_index.cache_clear()
    _comp_key_index.cache_clear()


@lru_cache(maxsize=1)
def _hero_index() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for h in load_heroes().get("heroes") or []:
        for key in (
            h.get("card_id"), h.get("slug"), h.get("name"),
            str(h.get("dbf_id") or ""), str(h.get("id") or ""),
        ):
            if key:
                out[_norm(str(key))] = h
    return out


@lru_cache(maxsize=1)
def _trinket_index() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for t in load_trinkets().get("trinkets") or []:
        for key in (
            t.get("card_id"), t.get("id"), t.get("name"),
            str(t.get("dbf_id") or ""),
        ):
            if key:
                out[_norm(str(key))] = t
    return out


@lru_cache(maxsize=1)
def _comp_key_index() -> Dict[str, Tuple[str, str, frozenset]]:
    """comp_name -> (tier_label, tribe, frozenset(key_names))."""
    out: Dict[str, Tuple[str, str, frozenset]] = {}
    for c in load_comps().get("comps") or []:
        if c.get("naga") or c.get("hidden"):
            continue
        keys = [n for n in (c.get("key_names") or []) if n]
        if not keys:
            keys = [
                x.get("name") for x in (c.get("core_cards") or [])
                if isinstance(x, dict) and x.get("name")
            ]
        if not keys:
            continue
        tier = _TIER_LABEL.get(int(c.get("tier") or 99), "C")
        out[c["name"]] = (tier, c.get("tribe") or "", frozenset(keys))
    return out


def lookup_hero(snapshot) -> Optional[dict]:
    hero = _get(snapshot, "hero") or _get(snapshot, "hero_card_id")
    hname = _get(snapshot, "hero_name")
    dbf = _get(snapshot, "hero_dbf_id")
    idx = _hero_index()
    for cand in (hero, hname, str(dbf) if dbf else None):
        if not cand:
            continue
        hit = idx.get(_norm(str(cand)))
        if hit:
            return hit
    return None


def lookup_trinket(name_or_id: Optional[str]) -> Optional[dict]:
    if not name_or_id:
        return None
    return _trinket_index().get(_norm(str(name_or_id)))


def cards_mentioned(text: str) -> List[str]:
    return [m.group(1).strip() for m in _CARD_MARK.finditer(text or "")]


def live_comps() -> List[dict]:
    return list(load_comps().get("comps") or [])


def match_comps(board_names: Sequence[str]) -> List[Tuple[str, str, int, dict]]:
    """[(name, tier_label, hit_count, comp_dict)] best-first."""
    board = {str(n) for n in board_names if n}
    by_name = {c["name"]: c for c in live_comps()}
    scored = []
    for name, (tier, _tribe, keys) in _comp_key_index().items():
        hit = len(board & keys)
        if hit:
            scored.append((name, tier, hit, by_name.get(name) or {}))
    scored.sort(key=lambda x: (-x[2], {"S": 0, "A": 1, "B": 2}.get(x[1], 9), x[0]))
    return scored


def _board_names(snapshot) -> List[str]:
    board = list(_get(snapshot, "board", []) or [])
    names = []
    for m in board:
        if isinstance(m, dict):
            names.append(m.get("name"))
        else:
            names.append(getattr(m, "name", None))
    return [n for n in names if n]


def locked_comp(snapshot, kb=None) -> Optional[dict]:
    """The HSReplay comp NEXT is building toward.

    The lobby playbook's PLAN lock (enabler hit, see ``lobby_playbook``) wins;
    otherwise fall back to the #95 filled-board key match.
    """
    pb = _get(snapshot, "playbook")
    plan = pb.get("plan") if isinstance(pb, dict) else None
    plan = plan or _get(snapshot, "playbook_plan")
    if plan:
        for c in live_comps():
            if c.get("name") == plan:
                return c
    if not isinstance(pb, dict) and isinstance(snapshot, dict):
        try:
            from .lobby_playbook import evaluate
            st = evaluate(snapshot, kb=kb)
            if st.committed:
                for c in live_comps():
                    if c.get("name") == st.plan:
                        return c
        except Exception:
            pass
    return _legacy_locked_comp(snapshot)


def _legacy_locked_comp(snapshot) -> Optional[dict]:
    """#95: after the board is filled enough, lock the best HSReplay key match."""
    board = list(_get(snapshot, "board", []) or [])
    if len(board) < 5:
        return None
    matches = match_comps(_board_names(snapshot))
    if not matches:
        return None
    _name, _tier, hit, comp = matches[0]
    if hit >= 2 or (hit >= 1 and len(board) >= 6):
        return comp
    return None


def plan_line(snapshot, kb=None) -> Optional[str]:
    """Overlay line once a comp is locked: ``PLAN → {comp}``."""
    comp = locked_comp(snapshot, kb=kb)
    if not comp:
        return None
    return f"PLAN → {comp.get('name')}"


def key_or_enabler_boost(
    card_name: Optional[str], snapshot, kb=None
) -> Tuple[float, Optional[str]]:
    """HARD promote buys of locked-comp keys/enablers. Negative = better."""
    if not card_name:
        return 0.0, None
    comp = locked_comp(snapshot, kb=kb)
    if comp:
        keys = set(comp.get("key_names") or [])
        if card_name in keys:
            return -1.25, f"HSReplay PLAN key — {comp.get('name')}"
        return 0.0, None
    # Pre-lock: still hard-boost S/A keys so a direction can form after fill.
    for name, (tier, _tr, keys) in _comp_key_index().items():
        if card_name in keys:
            boost = {"S": -0.85, "A": -0.55, "B": -0.25}.get(tier, -0.15)
            return boost, f"HSReplay {tier}-comp key ({name})"
    return 0.0, None


def hero_guide_hp_adjust(snapshot, cost: int = 0) -> Tuple[float, Optional[str]]:
    """HP placement from HSReplay hero guide. Negative = better / can be NEXT."""
    hero = lookup_hero(snapshot)
    if not hero:
        return 0.0, None
    hp_info = _get(snapshot, "hero_power") or {}
    gold = int(_get(snapshot, "gold") or 0)
    if isinstance(hp_info, dict) and hp_info.get("usable") is False:
        return 0.0, None
    if gold < int(cost or 0):
        return 0.0, None

    from .hero_power_verdict import usable_hp_bullets, weak_quote
    weak = weak_quote(hero)
    if weak:
        cite = weak[:110] + ("…" if len(weak) > 110 else "")
        return 0.5, f"HSReplay: weak hero power — {cite}"
    guide = hero.get("guide_text") or ""
    structured = hero.get("structured") or {}
    all_bullets = list(structured.get("hp") or [])
    hp_bullets = usable_hp_bullets(hero)
    blob = " ".join([guide] + all_bullets)
    if not blob.strip():
        return 0.0, None
    if all_bullets and not hp_bullets:
        return 0.0, None          # the guide only says when NOT to press it
    from .hero_power_verdict import restricted_now
    if restricted_now(hero, _get(snapshot, "tavern_tier"), _get(snapshot, "turn")):
        return 0.0, None          # "do not hero power on tavern 2" — not now

    if _HP_SKIP.search(blob) and not (hp_bullets and _HP_NOW.search(" ".join(hp_bullets))):
        return 0.35, f"HSReplay hero guide — skip HP ({hero.get('name')})"

    if hp_bullets or _HP_NOW.search(blob):
        cite = (hp_bullets[0] if hp_bullets else guide).strip()
        cite = cite[:120] + ("…" if len(cite) > 120 else "")
        return -0.95, f"HSReplay hero guide HP — {cite}"
    return 0.0, None


def hero_guide_lines(snapshot, kb=None) -> List[str]:
    """Advice lines from the hero's HSReplay guide that can lead NEXT."""
    hero = lookup_hero(snapshot)
    if not hero:
        return []
    lines: List[str] = []
    structured = hero.get("structured") or {}
    hp_info = _get(snapshot, "hero_power") or {}
    usable = True
    cost = 0
    if isinstance(hp_info, dict):
        usable = hp_info.get("usable", True) is not False
        cost = int(hp_info.get("cost") or 0)
    gold = int(_get(snapshot, "gold") or 0)

    from .hero_power_verdict import usable_hp_bullets, weak_quote
    from .hero_power_verdict import restricted_now
    if (usable and gold >= cost and not weak_quote(hero)
            and not restricted_now(hero, _get(snapshot, "tavern_tier"), _get(snapshot, "turn"))):
        for bullet in usable_hp_bullets(hero)[:1]:
            lines.append(f"Hero Power — HSReplay: {bullet}")
        if (not lines and not (structured.get("hp") or [])
                and _HP_NOW.search(hero.get("guide_text") or "")):
            lines.append(f"Hero Power — HSReplay guide for {hero.get('name')}")

    shop = list(_get(snapshot, "shop", []) or [])
    shop_names = set()
    for m in shop:
        if isinstance(m, dict):
            shop_names.add(m.get("name"))
        else:
            shop_names.add(getattr(m, "name", None))
    shop_names.discard(None)

    for bullet in (structured.get("buy_prefs") or [])[:1]:
        mentioned = cards_mentioned(bullet)
        hit = shop_names & set(mentioned)
        if hit:
            name = next(iter(hit))
            lines.append(f"Buy {name} — HSReplay hero guide buy pref")
    return lines


def hero_guide_notes(snapshot) -> List[str]:
    """Non-action HSReplay hero-guide context for the overlay header.

    Cycle prose and "look for X" are guidance, not a move — they used to lead
    NEXT (e.g. "Cycle — HSReplay: Way too expensive…"), which hid the real
    action. They now render as a ``HERO →`` phase line instead.
    """
    hero = lookup_hero(snapshot)
    if not hero:
        return []
    structured = hero.get("structured") or {}
    bits: List[str] = []
    for bullet in (structured.get("buy_prefs") or [])[:1]:
        mentioned = [n for n in cards_mentioned(bullet) if n]
        if mentioned:
            bits.append("look for " + ", ".join(mentioned[:2]))
    for bullet in (structured.get("cycle") or [])[:1]:
        bits.append(_CARD_MARK.sub(lambda m: m.group(1), bullet)[:90])
    if not bits:
        return []
    return [f"HERO → {hero.get('name')}: " + " · ".join(bits)]


def trinket_guide_score(
    name_or_id: Optional[str],
    board_tribes: Optional[Dict[str, int]] = None,
    board_names: Optional[Sequence[str]] = None,
    target_tribe: Optional[str] = None,
) -> Tuple[float, List[str]]:
    """Placement delta (negative=better) + reason bits from HSReplay guide."""
    t = lookup_trinket(name_or_id)
    if not t:
        return 0.0, []
    guide = (t.get("guide_text") or "").strip()
    bits: List[str] = []
    delta = 0.0
    board_tribes = board_tribes or {}
    board_set = {str(n) for n in (board_names or []) if n}
    target = (target_tribe or "").lower()

    fav = [str(x).lower() for x in (t.get("favorable_tribes") or [])]

    if not guide:
        for tr in fav:
            if board_tribes.get(tr) or tr == target:
                delta -= 0.35
                bits.append(f"HSReplay fav tribe {tr}")
                break
        return delta, bits

    low = guide.lower()
    if re.search(r"\b(never pick|don'?t pick|do not pick|avoid picking|skip this)\b", low):
        return 1.40, ["HSReplay guide: avoid"]

    mentioned = cards_mentioned(guide)
    hits = [n for n in mentioned if n in board_set]
    if hits:
        delta -= min(1.50, 0.70 + 0.35 * len(hits))
        bits.append(f"HSReplay guide enables {hits[0]}")
    elif mentioned and len(board_set) >= 4:
        delta += 0.45
        bits.append(f"HSReplay guide wants {mentioned[0]} (missing)")

    tribes = (
        "beast", "demon", "dragon", "elemental", "mech", "murloc",
        "pirate", "quilboar", "undead", "naga", "aberration",
    )
    for tr in list(fav) + [x for x in tribes if x in low]:
        if board_tribes.get(tr) or tr == target:
            delta -= 0.55
            bits.append(f"HSReplay guide fits {tr}")
            break
        if tr in fav and board_tribes and tr not in board_tribes and (
            not target or tr != target
        ):
            delta += 0.40
            bits.append(f"HSReplay guide off-tribe ({tr})")
            break

    if bits:
        bits.append("HSReplay trinket guide")
    return delta, bits


def sync_playstyle_comp_tiers() -> dict:
    """Build playstyle-prior-shaped comp_tiers from ingested comps.json."""
    buckets: Dict[str, list] = {"S": [], "A": [], "B": []}
    for c in live_comps():
        tier = _TIER_LABEL.get(int(c.get("tier") or 99), "C")
        if tier not in buckets:
            continue
        keys = list(c.get("key_names") or [])
        if not keys:
            keys = [
                x.get("name") for x in (c.get("core_cards") or [])
                if isinstance(x, dict) and x.get("name")
            ]
        tribe = c.get("tribe") or ""
        # "Beasts" -> "Beast" for prior tribe_weight keys
        if tribe.endswith("s") and tribe.lower() not in ("undead",):
            tribe = tribe[:-1]
        buckets[tier].append({
            "name": c["name"],
            "tribe": tribe or None,
            "key_minions": keys,
            "summary": c.get("summary") or "",
            "how_to_play": c.get("how_to_play") or "",
            "source_url": c.get("source_url"),
        })
    return buckets
