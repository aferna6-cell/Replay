"""Meta pack — the LLM Turn Director's compact, current-patch context bundle.

Phase 2 of ``specs/llm-turn-director_spec.md``. The Director never trusts its
own memory for "what's strong right now" (req 8) — this module assembles that
context from the committed data sources into one small dict the Director's
context window can actually hold, and stamps it with a patch id so stale data
gets refused, not silently served (req 4, req 13).

Two data sources, one priority order (req 3): **Tier7** (lobby-conditioned
hero picks + first-place comps, ``tier7.py``) is PRIMARY when the caller has
already queried it this lobby; the committed **Firestone** snapshot
(population-wide, always available offline, ``stats.py``/``firestone_stats.py``)
is the fallback. Which one actually filled each field is recorded in the
``sources`` dict — that's how the Director (and this module's own tests) can
tell "this pack is lobby-conditioned" from "this pack is generic population
data" without re-deriving it.

Card-level "enabler" derivation is shared with ``playbooks.py``
(``playbooks.enablers_for_tribe``) so the compact lobby-prompt slice here and
the full per-comp playbook agree on what an enabler is.

Curve (req 9) carries two numbers on purpose: the classic guide-language
baseline ("tier 2 on turn 2, 4-on-5, 5-on-7-ish" — the owner's own words in
the spec) alongside the measured top-10% pace from ``pace.py``. They're
usually not the same curve — the measured one runs ~0.2-0.25 tiers ahead —
and the gap is the point: it's what "top players level faster than folk
wisdom says" looks like as data.

# CALIBRATE: `patch_build` has no autodetector here — Power.log build-number
extraction (req 13's "Power.log/build number... flags meta pack behind the
patch") is live-game plumbing that belongs to a later phase (overlay/live
wiring), out of this module's lane. Callers pass `patch_build` explicitly
(or leave it None, in which case `check_patch_drift` always reports "fine").

Runs fully offline, stdlib only.
"""

import datetime
import json
import os
from typing import Dict, List, Optional

from . import cards as cards_mod
from . import pace as pace_mod
from . import playbooks as playbooks_mod
from .firestone_stats import _tier as _placement_tier
from .stats import CompStats, StatsDB, SAMPLE_COMPS, SAMPLE_HEROES

DEFAULT_META_PACK_PATH = "data/stats/meta_pack.json"

HERO_TOP_N = 10
COMP_TOP_N = 15
TRINKET_TOP_N = 8

# The owner's own words (spec req 9): the classic guide-language leveling
# line, kept ONLY as a contrast baseline against the measured pace below —
# never used for advice on its own.
FOLK_WISDOM_TAVERN_TIER = {
    1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 5, 9: 6, 10: 6, 11: 6, 12: 6,
}


def _read_fetched(*paths: str) -> str:
    for p in paths:
        if p and os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                v = d.get("_fetched")
                if v and v != "from-local":
                    return v
            except (OSError, ValueError):
                continue
    return datetime.date.today().isoformat()


def _group_tribes(comps: List[CompStats]) -> List[dict]:
    """Lobby tribes ranked by their comps' avg placement (req 5's "rank the
    lobby's tribes"), each with its own top comps."""
    by_tribe: Dict[str, List[CompStats]] = {}
    for c in comps:
        if not c.tribe:
            continue
        by_tribe.setdefault(c.tribe, []).append(c)
    out = []
    for tribe, group in by_tribe.items():
        group.sort(key=lambda c: c.average_position)
        avg = sum(c.average_position for c in group) / len(group)
        out.append({
            "tribe": tribe,
            "avg_placement": round(avg, 3),
            "top_comps": [c.name for c in group[:3]],
        })
    out.sort(key=lambda t: t["avg_placement"])
    return out


def _comp_entry(name: str, tribe: Optional[str], tier: str,
                avg_placement: Optional[float], core_cards: List[str],
                kb: dict, source: str) -> dict:
    return {
        "name": name,
        "tribe": tribe,
        "tier": tier,
        "avg_placement": avg_placement,
        "core_cards": list(core_cards),
        "enabler_cards": playbooks_mod.enablers_for_tribe(tribe, kb, exclude=core_cards),
        "source": source,
    }


def _comps_from_firestone(stats_db: StatsDB, kb: dict, limit: int) -> List[dict]:
    ranked = sorted(stats_db.comps, key=lambda c: c.average_position)[:limit]
    return [_comp_entry(c.name, c.tribe, c.tier, c.average_position,
                        c.core_cards, kb, "firestone") for c in ranked]


def _infer_tribe(core_cards: List[str], kb_by_name: dict) -> Optional[str]:
    """Best-guess tribe for a Tier7 comp row (the endpoint doesn't tag one) —
    the most common tribe among its named key minions."""
    counts: Dict[str, int] = {}
    for name in core_cards:
        ck = kb_by_name.get(name)
        for tr in (ck.tribes if ck else []):
            if tr == "All":
                continue
            counts[tr] = counts.get(tr, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _comps_from_tier7(rows: List[dict], kb: dict, kb_by_name: dict,
                      limit: int) -> List[dict]:
    ranked = sorted(
        rows, key=lambda r: r.get("avg_final_placement")
        if r.get("avg_final_placement") is not None else 9.0)[:limit]
    out = []
    for r in ranked:
        core = list(r.get("key_minions") or [])
        tribe = _infer_tribe(core, kb_by_name)
        avg = r.get("avg_final_placement")
        tier = _placement_tier(avg) if avg is not None else "?"
        out.append(_comp_entry(r.get("name", ""), tribe, tier, avg, core, kb,
                               "tier7"))
    return out


def _heroes_from_firestone(stats_db: StatsDB, limit: int) -> List[dict]:
    ranked = sorted(stats_db.heroes, key=lambda h: h.average_position)[:limit]
    return [{
        "name": h.name, "tier": None, "avg_placement": h.average_position,
        "pick_rate": h.pick_rate, "best_tribes": list(h.best_tribes),
        "playstyle": h.playstyle, "top_comps": [], "source": "firestone",
    } for h in ranked]


def _heroes_from_tier7(rows: List[dict]) -> List[dict]:
    return [{
        "name": r.get("name", ""), "tier": r.get("tier"),
        "avg_placement": r.get("avg_placement"), "pick_rate": r.get("pick_rate"),
        "best_tribes": [], "playstyle": None,
        "top_comps": list(r.get("top_comps") or []), "source": "tier7",
    } for r in rows]


def _merge_heroes(base: List[dict], overlay: List[dict], limit: int) -> List[dict]:
    """"top N + any requested" (interface wording): keep the population top N
    as the base, then let any Tier7-queried heroes (this lobby's actual offer)
    replace their Firestone row or get appended if they're outside the top N."""
    by_name = {h["name"].lower(): h for h in base}
    for h in overlay:
        by_name[h["name"].lower()] = h
    merged = list(by_name.values())
    merged.sort(key=lambda h: h["avg_placement"]
                if h["avg_placement"] is not None else 9.0)
    return merged[:max(limit, len(overlay))]


def _trinkets(stats_db: StatsDB, limit: int) -> List[dict]:
    return [{
        "name": t.name, "tier": t.tier, "avg_placement": t.average_position,
        "text": t.text,
    } for t in stats_db.best_trinkets(limit)]


def build_meta_pack(stats_dir: Optional[str] = None, kb: Optional[dict] = None,
                    tier7_rows: Optional[Dict[str, List[dict]]] = None,
                    patch_build: Optional[str] = None) -> dict:
    """Assemble the current meta pack.

    `tier7_rows` (optional): ``{"heroes": [...], "comps": [...]}`` — rows
    already normalized by ``tier7.normalize_hero_pick`` / ``tier7.normalize_comps``
    respectively for this lobby. Either key may be omitted; Firestone covers
    whichever one is missing.
    """
    stats_dir = stats_dir or os.path.join(
        os.path.dirname(__file__), "..", "data", "stats")
    hero_src = os.path.join(stats_dir, "firestone_hero_stats.json")
    comp_src = os.path.join(stats_dir, "firestone_comp_stats.json")
    trinket_src = os.path.join(stats_dir, "firestone_trinket_stats.json")
    pace_src = os.path.join(stats_dir, "firestone_pace.json")

    stats_db = StatsDB.load(
        hero_src if os.path.isfile(hero_src) else SAMPLE_HEROES,
        comp_src if os.path.isfile(comp_src) else SAMPLE_COMPS,
        trinket_src if os.path.isfile(trinket_src) else None,
    )
    pace_data = pace_mod.load_pace(pace_src if os.path.isfile(pace_src) else None)
    kb = kb if kb is not None else cards_mod.load_kb()
    kb_by_name = cards_mod.by_name(kb)

    tier7_rows = tier7_rows or {}
    tier7_heroes = tier7_rows.get("heroes") or []
    tier7_comps = tier7_rows.get("comps") or []

    heroes_base = _heroes_from_firestone(stats_db, HERO_TOP_N)
    if tier7_heroes:
        heroes = _merge_heroes(heroes_base, _heroes_from_tier7(tier7_heroes),
                               HERO_TOP_N)
        hero_source = "tier7+firestone"
    else:
        heroes = heroes_base
        hero_source = "firestone"

    if tier7_comps:
        comps = _comps_from_tier7(tier7_comps, kb, kb_by_name, COMP_TOP_N)
        comp_source = "tier7"
    else:
        comps = _comps_from_firestone(stats_db, kb, COMP_TOP_N)
        comp_source = "firestone"

    playbook_index = [c["name"] for c in comps
                      if playbooks_mod.load_playbook(c["name"]) is not None]

    return {
        "patch_build": patch_build,
        "fetched": _read_fetched(comp_src, hero_src),
        "sources": {"heroes": hero_source, "comps": comp_source},
        "tribes": _group_tribes(stats_db.comps),
        "heroes": heroes,
        "comps": comps,
        "trinkets": _trinkets(stats_db, TRINKET_TOP_N),
        "curve": {
            "standard": dict(FOLK_WISDOM_TAVERN_TIER),
            "measured": {int(t): v for t, v in
                        sorted(pace_data.get("tavern_tier", {}).items())},
        },
        "playbook_index": playbook_index,
    }


def save_meta_pack(pack: dict, path: str = DEFAULT_META_PACK_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(pack, fh, indent=1)
    return path


def load_meta_pack(path: str = DEFAULT_META_PACK_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_patch_drift(pack: dict, live_build: Optional[str]) -> Optional[str]:
    """None when the pack is current (or when either side is unknown — req 13
    says refuse *stale* data, not data we can't compare). A warning string
    when the pack was built on a different patch than the live game."""
    packed = pack.get("patch_build")
    if not packed or not live_build:
        return None
    if str(packed) != str(live_build):
        return (f"Meta pack is behind the patch: pack built for "
                f"{packed!r}, live game is {live_build!r}. Run refresh-meta "
                f"before trusting comp/curve/enabler advice this game.")
    return None


def pack_age_warning(pack: dict, max_days: int = 7) -> Optional[str]:
    """A warning when the pack's snapshot data is older than `max_days` —
    the meta moves weekly even without a patch, and target boards / comp
    ranks should track it (owner: 'the more recent the data the better')."""
    fetched = pack.get("fetched")
    if not fetched:
        return None
    try:
        age = (datetime.date.today()
               - datetime.date.fromisoformat(str(fetched)[:10])).days
    except ValueError:
        return None
    if age > max_days:
        return (f"Meta pack snapshot is {age} days old (fetched {fetched}). "
                f"Run `python -m hsbg_coach refresh-meta` for current "
                f"comps/curve/target boards. (Tier7 pick queries are always "
                f"live and unaffected.)")
    return None


def _fmt_curve_row(turn: int, standard: Dict[int, float],
                   measured: Dict[int, float]) -> str:
    s = standard.get(turn)
    m = measured.get(turn)
    s_txt = f"{s:.1f}" if s is not None else "?"
    m_txt = f"{m:.1f}" if m is not None else "?"
    return f"  T{turn}: folk~{s_txt} measured~{m_txt}"


def pack_for_prompt(pack: dict, lobby_tribes: List[str],
                    max_chars: int = 6000) -> str:
    """Compact text slice for the Director's context: only this lobby's
    tribes' comps/enablers, the curve table, and the top trinkets. Built as
    priority-ordered blocks and truncated at a block boundary so the result
    is always <= max_chars and deterministic for the same pack + tribes."""
    lobby = {t.lower() for t in lobby_tribes}
    blocks = []

    header = f"Meta pack — patch {pack.get('patch_build') or 'unknown'}, " \
             f"fetched {pack.get('fetched')}\nLobby tribes: " \
             f"{', '.join(lobby_tribes)}\n"
    blocks.append(header)

    standard = pack.get("curve", {}).get("standard", {})
    measured = pack.get("curve", {}).get("measured", {})
    turns = sorted(set(standard) | set(measured))
    if turns:
        sample_turns = [t for t in turns if t in (2, 4, 5, 6, 7, 8, 9, 10, 12)] or turns
        curve_lines = ["Curve (tavern tier by turn — folk wisdom vs measured "
                       "top-10% pace):"]
        curve_lines += [_fmt_curve_row(t, standard, measured) for t in sample_turns]
        blocks.append("\n".join(curve_lines) + "\n")

    for comp in pack.get("comps", []):
        tribe = (comp.get("tribe") or "").lower()
        if tribe not in lobby:
            continue
        lines = [f"Comp: {comp['name']} ({comp.get('tribe')}, tier "
                 f"{comp.get('tier')}, avg {comp.get('avg_placement')})"]
        if comp.get("core_cards"):
            lines.append("  core: " + ", ".join(comp["core_cards"]))
        if comp.get("enabler_cards"):
            lines.append("  enablers: " + ", ".join(comp["enabler_cards"]))
        blocks.append("\n".join(lines) + "\n")

    trinkets = pack.get("trinkets", [])
    if trinkets:
        t_lines = ["Top trinkets:"]
        for t in trinkets:
            text = (t.get("text") or "")[:70]
            t_lines.append(f"  {t['name']} (tier {t.get('tier')}, avg "
                           f"{t.get('avg_placement')}) — {text}")
        blocks.append("\n".join(t_lines) + "\n")

    out = ""
    for block in blocks:
        if len(out) + len(block) > max_chars:
            break
        out += block
    return out.rstrip("\n")
