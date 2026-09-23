"""Aidan's locked playbook as a HARD phase machine for the live coach.

    1. HERO      — HSReplay hero guide (hsreplay_guides / hero_scripts, #95)
    2. LOBBY     — rank the lobby's tribes from HSReplay comp tiers
    3. ENABLERS  — preload per-tribe enablers from HSReplay comp guides
    4. FILL      — until committed: solid bodies, no garbage, no hard roll
    5. COMMIT    — first clear enabler hit (shop / hand / board / discover)
                   locks ``PLAN → {comp}``; buys/sells then follow that comp

Everything is derived from ``data/hsreplay_guides/comps.json`` (HSReplay tiers,
``when_to_commit``, enabler cards, key/core/addon cards) filtered to the live
36.6.1 pool. Naga and out-of-pool cards never appear. No comps are invented.

The gates here REORDER ``rank_actions`` output (they change NEXT); they are not
placement nudges. The playbook is internal: the live overlay shows only NEXT
and short alternates, never the phase state. ``evaluate`` is pure given (snapshot, locked plan); the live
coach keeps the lock across snapshots for the rest of the game.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .tribe_policy import (
    PATCH_TRIBES, QUARANTINED_TRIBES, canonicalize, filter_lobby_tribes,
)

PHASE_LOBBY = "lobby"        # tribes not detected yet (fallback ranking)
PHASE_FILL = "fill"          # lobby ranked + enablers preloaded; hunting
PHASE_COMMIT = "commit"      # PLAN locked to one HSReplay comp

_TIER_LABEL = {1: "S", 2: "A", 3: "B"}
_CARD_MARK = re.compile(r"\[\[([^\]|]+)(?:\|\|\d+)?\]\]")
_POOL_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "cards", "bg_live_pool_36_6_1.json"))

# How many lobby tribes count as "strong" (enablers preloaded / commit-eligible).
_MAX_STRONG = 3
_MIN_STRONG = 2
_ENABLERS_PER_TRIBE = 4       # overlay + hunt list cap per tribe


def _get(snap, key, default=None):
    return snap.get(key, default) if isinstance(snap, dict) else getattr(snap, key, default)


def _mname(m) -> Optional[str]:
    if isinstance(m, dict):
        return m.get("name")
    if isinstance(m, str):
        return m
    return getattr(m, "name", None)


def _singular(tribe: Optional[str]) -> Optional[str]:
    """HSReplay comp tribe ("Beasts", "Mechs", "Undead") -> canonical tribe."""
    if not tribe:
        return None
    t = str(tribe).strip()
    c = canonicalize(t)
    if c is None and t.lower().endswith("s"):
        c = canonicalize(t[:-1])
    return c if c and c != "All" else None


# ---------------------------------------------------------------------------
# Data: live pool + HSReplay comps
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _pool_tribes() -> Dict[str, Tuple[str, ...]]:
    """Live-pool minion name (lower) -> canonical tribes ("All" kept)."""
    out: Dict[str, Tuple[str, ...]] = {}
    if not os.path.isfile(_POOL_PATH):
        return out
    with open(_POOL_PATH, encoding="utf-8") as fh:
        doc = json.load(fh)
    for m in doc.get("minions") or []:
        tribes = []
        for r in m.get("races") or []:
            if str(r).upper() == "ALL":
                tribes.append("All")
                continue
            c = canonicalize(r)
            if c:
                tribes.append(c)
        out[str(m.get("name", "")).lower()] = tuple(tribes)
    return out


def in_live_pool(name: Optional[str]) -> bool:
    if not name:
        return False
    try:
        from .playstyle_prior import is_out_of_pool, live_pool_names
        if is_out_of_pool(name):
            return False
        pool = live_pool_names()
        if pool:
            return name in pool
    except Exception:
        pass
    return name.lower() in _pool_tribes()


def card_tribes(m, kb=None) -> Tuple[str, ...]:
    """Tribes for a minion/name: snapshot field → kb → live-pool JSON."""
    raw = None
    if isinstance(m, dict):
        raw = m.get("tribes") or m.get("tribe")
    elif not isinstance(m, str):
        raw = getattr(m, "tribes", None)
    if isinstance(raw, str):
        raw = [raw]
    if raw:
        out = []
        for r in raw:
            if str(r).strip().lower() in QUARANTINED_TRIBES:
                out.append("Naga")
                continue
            c = canonicalize(r)
            if c:
                out.append(c)
        return tuple(out)
    name = _mname(m)
    if kb is not None and name:
        try:
            from .cards import by_name
            ck = by_name(kb).get(name)
            if ck is not None and ck.tribes:
                return tuple(t for t in (canonicalize(x) for x in ck.tribes) if t)
        except Exception:
            pass
    return _pool_tribes().get(str(name or "").lower(), ())


@dataclass(frozen=True)
class CompInfo:
    name: str
    tribe: str                      # canonical ("Beast")
    tier: int                       # HSReplay tier (1 best)
    triggers: Tuple[str, ...]       # when_to_commit ∪ enabler cards (live pool)
    cards: Tuple[str, ...]          # key ∪ core ∪ addon ∪ triggers (live pool)
    keys: Tuple[str, ...]           # key_names (live pool)
    core: Tuple[str, ...] = ()      # post-commit hunt: key ∪ core ∪ triggers

    @property
    def tier_label(self) -> str:
        return _TIER_LABEL.get(self.tier, "C")


@lru_cache(maxsize=1)
def live_comp_infos() -> Tuple[CompInfo, ...]:
    """Live-pool HSReplay comps (Naga / hidden / out-of-pool cards dropped)."""
    from .hsreplay_guides import live_comps
    cased = {n.lower(): n for n in _pool_names_cased()}
    out: List[CompInfo] = []

    def live(name: Optional[str]) -> Optional[str]:
        """Live-pool spelling of an HSReplay card name, else None (OOP/Naga)."""
        if not name:
            return None
        n = str(name).strip()
        if in_live_pool(n):
            return n
        real = cased.get(n.lower())          # HSReplay mis-casing ("Gem rat")
        return real if real and in_live_pool(real) else None

    for c in live_comps():
        if c.get("naga") or c.get("hidden"):
            continue
        tribe = _singular(c.get("tribe"))
        if not tribe or tribe.lower() in QUARANTINED_TRIBES:
            continue
        trig: List[str] = []
        for n in _CARD_MARK.findall(c.get("when_to_commit") or ""):
            ln = live(n)
            if ln and ln not in trig:
                trig.append(ln)
        for x in c.get("enabler_cards") or []:
            if isinstance(x, dict) and x.get("in_live_pool") is False:
                continue
            ln = live(x.get("name") if isinstance(x, dict) else x)
            if ln and ln not in trig:
                trig.append(ln)
        keys = [k for k in (live(n) for n in c.get("key_names") or []) if k]
        core: List[str] = list(keys)
        for group in ("key_cards", "core_cards"):
            for x in c.get(group) or []:
                ln = live(x.get("name") if isinstance(x, dict) else x)
                if ln and ln not in core:
                    core.append(ln)
        for n in trig:
            if n not in core:
                core.append(n)
        cards: List[str] = list(core)
        for x in c.get("addon_cards") or []:
            ln = live(x.get("name") if isinstance(x, dict) else x)
            if ln and ln not in cards:
                cards.append(ln)
        if not trig and not cards:
            continue
        out.append(CompInfo(
            name=c["name"], tribe=tribe, tier=int(c.get("tier") or 9),
            triggers=tuple(trig), cards=tuple(cards), keys=tuple(keys),
            core=tuple(core)))
    out.sort(key=lambda ci: (ci.tier, ci.name))
    return tuple(out)


@lru_cache(maxsize=1)
def _pool_names_cased() -> Tuple[str, ...]:
    if not os.path.isfile(_POOL_PATH):
        return ()
    with open(_POOL_PATH, encoding="utf-8") as fh:
        doc = json.load(fh)
    return tuple(m.get("name") for m in doc.get("minions") or [] if m.get("name"))


def comp_by_name(name: Optional[str]) -> Optional[CompInfo]:
    if not name:
        return None
    for ci in live_comp_infos():
        if ci.name == name:
            return ci
    return None


# ---------------------------------------------------------------------------
# 2. LOBBY — rank the lobby's tribes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TribeRank:
    tribe: str
    best_tier: Optional[int]        # best HSReplay comp tier for this tribe
    tier_counts: Tuple[int, int, int]   # (#S, #A, #B) live comps
    comps: Tuple[str, ...]

    @property
    def label(self) -> str:
        return _TIER_LABEL.get(self.best_tier or 0, "-")


def rank_lobby_tribes(available: Optional[Sequence[str]]) -> List[TribeRank]:
    """Rank the lobby's tribes by HSReplay comp strength (best first).

    Order: best comp tier → #S comps → #A comps → #comps → name. Tribes with no
    live HSReplay comp (e.g. Aberration this patch) sort last. Naga never.
    """
    lobby = filter_lobby_tribes(available) if available else []
    rows = []
    for t in lobby:
        comps = [ci for ci in live_comp_infos() if ci.tribe == t]
        counts = tuple(sum(1 for ci in comps if ci.tier == k) for k in (1, 2, 3))
        best = min((ci.tier for ci in comps), default=None)
        rows.append(TribeRank(t, best, counts, tuple(ci.name for ci in comps)))
    rows.sort(key=lambda r: (
        r.best_tier if r.best_tier is not None else 99,
        -r.tier_counts[0], -r.tier_counts[1], -len(r.comps), r.tribe))
    return rows


def strong_tribes(ranked: Sequence[TribeRank]) -> List[str]:
    """Commit-eligible tribes: every S-tier lobby tribe, topped up with the next
    best to at least two, capped at three. Only tribes with a live comp."""
    with_comps = [r for r in ranked if r.best_tier is not None]
    strong = [r.tribe for r in with_comps if r.best_tier == 1]
    for r in with_comps:
        if len(strong) >= _MIN_STRONG:
            break
        if r.tribe not in strong:
            strong.append(r.tribe)
    return strong[:_MAX_STRONG]


# ---------------------------------------------------------------------------
# 3. ENABLERS — preload per strong tribe
# ---------------------------------------------------------------------------

def preload_enablers(tribes: Sequence[str]) -> Dict[str, List[str]]:
    """{tribe: [enabler names]} — HSReplay commit triggers, best comp first."""
    out: Dict[str, List[str]] = {}
    for t in tribes:
        names: List[str] = []
        for ci in live_comp_infos():          # already tier-sorted
            if ci.tribe != t:
                continue
            for n in ci.triggers:
                if n not in names:
                    names.append(n)
        out[t] = names
    return out


def _comps_for_enabler(name: str, tribes: Sequence[str]) -> List[CompInfo]:
    return [ci for ci in live_comp_infos()
            if ci.tribe in tribes and name in ci.triggers]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class PlaybookState:
    phase: str
    lobby: List[str] = field(default_factory=list)          # ranked tribes
    lobby_labels: Dict[str, str] = field(default_factory=dict)
    lobby_known: bool = True
    strong: List[str] = field(default_factory=list)
    enablers: Dict[str, List[str]] = field(default_factory=dict)
    plan: Optional[str] = None           # locked HSReplay comp name
    plan_tribe: Optional[str] = None
    trigger: Optional[str] = None        # enabler that locked the plan
    trigger_zone: Optional[str] = None   # shop / hand / board / discover / memory
    shop_enablers: List[str] = field(default_factory=list)   # hunt hits in shop

    def to_dict(self) -> dict:
        return {
            "phase": self.phase, "lobby": list(self.lobby),
            "lobby_labels": dict(self.lobby_labels),
            "lobby_known": self.lobby_known, "strong": list(self.strong),
            "enablers": {k: list(v) for k, v in self.enablers.items()},
            "plan": self.plan, "plan_tribe": self.plan_tribe,
            "trigger": self.trigger, "trigger_zone": self.trigger_zone,
            "shop_enablers": list(self.shop_enablers),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookState":
        return cls(**{k: d.get(k) for k in (
            "phase", "lobby", "lobby_labels", "lobby_known", "strong",
            "enablers", "plan", "plan_tribe", "trigger", "trigger_zone",
            "shop_enablers") if k in d})

    @property
    def committed(self) -> bool:
        return self.phase == PHASE_COMMIT and bool(self.plan)

    def all_enablers(self) -> List[str]:
        out: List[str] = []
        for t in self.strong:
            for n in self.enablers.get(t, []):
                if n not in out:
                    out.append(n)
        return out


def _board_tribe_counts(board, kb=None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in board or []:
        for t in card_tribes(m, kb):
            if t in ("All", "Naga"):
                continue
            counts[t] = counts.get(t, 0) + 1
    return counts


def resolve_enabler(name: str, strong: Sequence[str], board, kb=None,
                    minion=None) -> Optional[CompInfo]:
    """The comp a *clear* enabler hit commits to, else None.

    Clear = the card's own tribe is the comp's tribe (Ravaging Scorpid → Beasts),
    or a cross-tribe/neutral enabler (Brann, Sky Admiral Rogers) whose comp
    tribe already has 2+ bodies on the board. Among several comps of the same
    tribe, prefer the one the board already overlaps most, then HSReplay tier.
    """
    cands = _comps_for_enabler(name, strong)
    if not cands:
        return None
    own = set(card_tribes(minion if minion is not None else name, kb))
    counts = _board_tribe_counts(board, kb)
    clear = [ci for ci in cands if ci.tribe in own or "All" in own
             or counts.get(ci.tribe, 0) >= 2]
    if not clear:
        return None
    board_names = {_mname(m) for m in board or []}

    def rank(ci: CompInfo):
        overlap = len(board_names & set(ci.cards))
        return (-overlap, -counts.get(ci.tribe, 0), ci.tier,
                0 if name in ci.triggers[:1] else 1, ci.name)
    return sorted(clear, key=rank)[0]


def evaluate(snapshot, kb=None, locked_plan: Optional[str] = None,
             discover: Optional[Sequence[str]] = None) -> PlaybookState:
    """Phase state for this snapshot. ``locked_plan`` (live memory or the
    snapshot's ``playbook_plan``) keeps a commit for the rest of the game."""
    available = _get(snapshot, "available_tribes") or []
    lobby_known = bool(filter_lobby_tribes(available))
    ranked = rank_lobby_tribes(available if lobby_known else sorted(PATCH_TRIBES))
    strong = strong_tribes(ranked)
    enablers = preload_enablers(strong)
    st = PlaybookState(
        phase=PHASE_FILL if lobby_known else PHASE_LOBBY,
        lobby=[r.tribe for r in ranked],
        lobby_labels={r.tribe: r.label for r in ranked},
        lobby_known=lobby_known, strong=strong, enablers=enablers)

    board = list(_get(snapshot, "board", []) or [])
    hand = list(_get(snapshot, "hand", []) or [])
    shop = list(_get(snapshot, "shop", []) or [])
    hunt = set(st.all_enablers())
    st.shop_enablers = [n for n in (_mname(m) for m in shop)
                        if n and n in hunt and in_live_pool(n)]

    locked_plan = locked_plan or _get(snapshot, "playbook_plan")
    ci = comp_by_name(locked_plan) if locked_plan else None
    if ci is not None:
        st.phase, st.plan, st.plan_tribe = PHASE_COMMIT, ci.name, ci.tribe
        st.trigger = _get(snapshot, "playbook_trigger")
        st.trigger_zone = "memory"
        return st

    # First clear enabler hit: owned (board/hand) beats offered (discover/shop).
    zones = [("board", board), ("hand", hand),
             ("discover", [{"name": n} for n in discover or []]), ("shop", shop)]
    for zone, cards in zones:
        best = None
        for m in cards:
            n = _mname(m)
            if not n or n not in hunt or not in_live_pool(n):
                continue
            hit = resolve_enabler(n, strong, board, kb,
                                  minion=m if isinstance(m, dict) and (
                                      m.get("tribes") or m.get("tribe")) else None)
            if hit is None:
                continue
            key = (strong.index(hit.tribe), hit.tier)
            if best is None or key < best[0]:
                best = (key, hit, n)
        if best is not None:
            _, hit, n = best
            st.phase, st.plan, st.plan_tribe = PHASE_COMMIT, hit.name, hit.tribe
            st.trigger, st.trigger_zone = n, zone
            return st

    # Legacy #95 lock (filled board already matching HSReplay keys).
    try:
        from .hsreplay_guides import _legacy_locked_comp
        legacy = _legacy_locked_comp(snapshot)
    except Exception:
        legacy = None
    if legacy:
        ci = comp_by_name(legacy.get("name"))
        if ci is not None:
            st.phase, st.plan, st.plan_tribe = PHASE_COMMIT, ci.name, ci.tribe
            st.trigger_zone = "board keys"
    return st


def ensure(snapshot, kb=None):
    """Return a snapshot dict carrying ``playbook`` (computed if missing).
    Non-dict snapshots are returned unchanged."""
    if not isinstance(snapshot, dict):
        return snapshot
    if isinstance(snapshot.get("playbook"), dict):
        return snapshot
    try:
        st = evaluate(snapshot, kb=kb)
    except Exception:
        return snapshot
    return dict(snapshot, playbook=st.to_dict())


def state_of(snapshot, kb=None) -> Optional[PlaybookState]:
    pb = _get(snapshot, "playbook")
    if isinstance(pb, dict):
        try:
            return PlaybookState.from_dict(pb)
        except Exception:
            return None
    try:
        return evaluate(snapshot, kb=kb)
    except Exception:
        return None


def plan_comp(snapshot, kb=None) -> Optional[CompInfo]:
    st = state_of(snapshot, kb)
    return comp_by_name(st.plan) if st and st.committed else None


# ---------------------------------------------------------------------------
# On-plan helpers (post-commit buys / sells)
# ---------------------------------------------------------------------------

def is_on_plan(m, ci: CompInfo, kb=None) -> bool:
    """HSReplay card of the locked comp, or a body of the comp's tribe.

    Neutral "flex" keys and other tribes' enablers are NOT on-plan once
    committed — Aidan: after the lock, hunt the comp, don't wander.
    """
    n = _mname(m)
    if n and n in ci.cards:
        return True
    tribes = card_tribes(m, kb)
    return ci.tribe in tribes or "All" in tribes


def is_core_hunt(name: Optional[str], ci: CompInfo) -> bool:
    """HSReplay core / key / commit-trigger card of the locked comp."""
    return bool(name and name in ci.core)


def is_plan_piece(m, ci: CompInfo) -> bool:
    """Named HSReplay card of the locked comp (never sell for room)."""
    n = _mname(m)
    return bool(n and n in ci.cards)


def keep_bonus(m, snapshot, kb=None) -> float:
    """Extra keep value once PLAN is locked: comp cards ≫ comp tribe."""
    ci = plan_comp(snapshot, kb)
    if ci is None:
        return 0.0
    if is_plan_piece(m, ci):
        return 40.0
    if ci.tribe in card_tribes(m, kb):
        return 12.0
    return 0.0


# ---------------------------------------------------------------------------
# Hard NEXT gates (applied to rank_actions output)
# ---------------------------------------------------------------------------

def _promote(recs, idx: int, reason: str, base: float):
    """Move recs[idx] strictly above the current top; keep placement axis sane."""
    from .game_value import WholeGameRec
    top_p = min(r.placement for r in recs)
    r = recs[idx]
    new_p = round(min(r.placement, top_p - 0.05), 2)
    recs[idx] = WholeGameRec(r.action, new_p, reason, round(base - new_p, 2))
    recs.sort(key=lambda x: x.placement)
    return recs


def _relabel(recs, idx: int, reason: str):
    from .game_value import WholeGameRec
    r = recs[idx]
    recs[idx] = WholeGameRec(r.action, r.placement, reason, r.gain)
    return recs


def _is_triple(r) -> bool:
    return "TRIPLE" in (r.reason or "")


def _is_tech(m) -> bool:
    """Matchup tech (Tunnel Blaster …) is ranked by the combat sim, not as fill."""
    try:
        from .card_roles import is_tech
        cid = m.get("card_id") if isinstance(m, dict) else getattr(m, "card_id", None)
        return bool(is_tech(cid, _mname(m)))
    except Exception:
        return False


def _buy_target(r) -> Optional[str]:
    return getattr(r.action, "target", None)


def _buy_minion(r):
    detail = getattr(r.action, "detail", None) or {}
    return detail.get("minion") if isinstance(detail, dict) else None


def apply_next_gates(recs, snapshot, kb=None, base: float = 4.5):
    """Reorder ranked recs so NEXT follows the playbook phase.

    FILL (not committed):
      * a preloaded enabler buyable in shop ⇒ Buy it is NEXT (commit trigger);
        unaffordable ⇒ Freeze
      * garbage buy (not an acceptable fill) can't be NEXT when a fill buy exists
      * Roll can't be NEXT on sparse + solid shop (#93 gate, kept upstream)
    COMMIT: see ``_committed_gates`` — hunt only the comp's core/key cards.
    """
    from .actions import BUY
    if not recs:
        return recs
    st = state_of(snapshot, kb)
    if st is None:
        return recs
    recs = sorted(recs, key=lambda r: r.placement)

    def find_buy(names):
        best = None
        for i, r in enumerate(recs):
            if r.action.kind == BUY and _buy_target(r) in names:
                if best is None or r.placement < recs[best].placement:
                    best = i
        return best

    ci = comp_by_name(st.plan) if st.committed else None

    if ci is not None:
        return _committed_gates(recs, snapshot, st, ci, kb, base)
    if _is_triple(recs[0]):
        return recs

    # FILL: a preloaded enabler in shop → Buy it is NEXT (the commit trigger).
    if st.shop_enablers:
        i = find_buy(st.shop_enablers)
        if i is not None:
            n = _buy_target(recs[i])
            hit = _comps_for_enabler(n, st.strong)
            label = hit[0].name if hit else "HSReplay comp"
            why = f"ENABLER {n} — HSReplay commit trigger ({label})"
            recs = _relabel(recs, 0, why) if i == 0 else _promote(recs, i, why, base)
        else:
            recs = _freeze_for(recs, snapshot, st.shop_enablers, base)

    # FILL: no garbage — a buy that isn't an acceptable fill can't lead
    # while an acceptable-fill buy is on offer.
    top = recs[0]
    if top.action.kind == BUY and not _is_triple(top):
        try:
            from .jeef_priors import shop_unit_is_acceptable_fill
            m = _buy_minion(top)
            if m is not None and not _is_tech(m) and \
                    not shop_unit_is_acceptable_fill(m, snapshot, kb):
                alt = [k for k, r in enumerate(recs) if r.action.kind == BUY
                       and _buy_minion(r) is not None
                       and shop_unit_is_acceptable_fill(_buy_minion(r), snapshot, kb)]
                if alt:
                    recs = _promote(recs, alt[0],
                                    f"FILL — solid body over garbage "
                                    f"{_buy_target(top)}", base)
        except Exception:
            pass
    return recs


def _freeze_for(recs, snapshot, names: Sequence[str], base: float):
    """Wanted card in shop but unaffordable ⇒ Freeze leads (don't roll it away)."""
    from .actions import BUY, FREEZE
    if not names or any(r.action.kind == BUY and _buy_target(r) in names for r in recs):
        return recs
    if int(_get(snapshot, "gold") or 0) >= 3:
        return recs
    fi = next((k for k, r in enumerate(recs) if r.action.kind == FREEZE), None)
    if fi is not None and fi != 0:
        recs = _promote(recs, fi, f"freeze — keep {names[0]} for next turn "
                        f"(HSReplay enabler/key)", base)
    return recs


def _committed_gates(recs, snapshot, st: "PlaybookState", ci: CompInfo, kb, base):
    """PLAN locked: hunt ONLY the comp's HSReplay core / key / trigger cards.

      1. a core/key/trigger buy in shop is NEXT (over roll, level, HP, fills)
      2. unaffordable core card in shop ⇒ Freeze is NEXT
      3. off-plan buys (other tribes, neutral flex, other comps' enablers) are
         pushed below every other action — never NEXT, never the first alt
      4. selling a named PLAN card is never NEXT
    """
    from .actions import BUY, SELL
    from .game_value import WholeGameRec

    owned = {_mname(m) for m in (_get(snapshot, "board", []) or [])}
    owned |= {_mname(m) for m in (_get(snapshot, "hand", []) or [])}
    order: List[str] = []
    for n in [st.trigger] + list(ci.triggers) + list(ci.core):
        if n and n in ci.core and n not in order:
            order.append(n)
    # Unowned pieces first; a second copy (triple progress) after.
    order = [n for n in order if n not in owned] + [n for n in order if n in owned]
    rank = {n: k for k, n in enumerate(order)}

    hunt = [k for k, r in enumerate(recs)
            if r.action.kind == BUY and _buy_target(r) in rank]
    if hunt and not _is_triple(recs[0]):
        i = min(hunt, key=lambda k: (rank[_buy_target(recs[k])], recs[k].placement))
        n = _buy_target(recs[i])
        tag = "commit trigger" if n in ci.triggers else "core/key"
        why = f"PLAN {ci.name} — HSReplay {tag}"
        recs = _relabel(recs, 0, why) if i == 0 else _promote(recs, i, why, base)
    elif not hunt:
        shop_names = {_mname(m) for m in (_get(snapshot, "shop", []) or [])}
        recs = _freeze_for(recs, snapshot, [n for n in order if n in shop_names], base)

    # Off-plan buys sink below every other action.
    def off_plan(r) -> bool:
        if r.action.kind != BUY or _is_triple(r):
            return False
        return not is_on_plan(_buy_minion(r) or _buy_target(r), ci, kb)

    others = [r.placement for r in recs if not off_plan(r)]
    if others:
        floor = max(others)
        out = []
        for r in recs:
            if off_plan(r):
                p = round(max(r.placement, floor + 0.05), 2)
                out.append(WholeGameRec(r.action, p,
                                        f"off-plan — PLAN {ci.name} hunts its core",
                                        round(base - p, 2)))
            else:
                out.append(r)
        recs = sorted(out, key=lambda r: r.placement)

    # Never lead with selling a named PLAN card.
    top = recs[0]
    if top.action.kind == SELL and len(recs) > 1:
        m = (getattr(top.action, "detail", None) or {}).get("minion") or _buy_target(top)
        if is_plan_piece(m, ci):
            p = round(recs[1].placement + 0.05, 2)
            recs = [WholeGameRec(top.action, p, f"keep {_mname(m)} — PLAN {ci.name} piece",
                                 round(base - p, 2))] + recs[1:]
            recs.sort(key=lambda r: r.placement)
    return recs


def discover_pick(offered: Sequence[str], snapshot, kb=None) -> Optional[Tuple[str, str]]:
    """(card, comp) when a Discover offers a clear enabler of a strong tribe
    (or a card of the locked PLAN)."""
    st = state_of(snapshot, kb)
    if st is None:
        return None
    board = list(_get(snapshot, "board", []) or [])
    if st.committed:
        ci = comp_by_name(st.plan)
        if ci is None:
            return None
        for n in list(ci.triggers) + list(ci.keys) + list(ci.cards):
            if n in offered and in_live_pool(n):
                return n, ci.name
        return None
    hunt = st.all_enablers()
    best = None
    for n in offered:
        if n not in hunt or not in_live_pool(n):
            continue
        hit = resolve_enabler(n, st.strong, board, kb)
        if hit is None:
            continue
        key = (st.strong.index(hit.tribe), hit.tier)
        if best is None or key < best[0]:
            best = (key, n, hit.name)
    return (best[1], best[2]) if best else None


# ---------------------------------------------------------------------------
# Overlay lines
# ---------------------------------------------------------------------------

def _short(name: str) -> str:
    return name.split(",")[0]


def overlay_lines(snapshot, kb=None, st: Optional[PlaybookState] = None) -> List[str]:
    """Phase summary (LOBBY / ENABLERS / FILL or PLAN / NEED) for debugging.

    NOT shown on the live overlay — Aidan wants the playbook to change NEXT
    silently, with no strategy chrome. Kept for tests / ad-hoc inspection.
    """
    st = st or state_of(snapshot, kb)
    if st is None:
        return []
    lines: List[str] = []
    board = list(_get(snapshot, "board", []) or [])
    if st.committed:
        ci = comp_by_name(st.plan)
        tier = ci.tier_label if ci else "?"
        why = f" · hit {_short(st.trigger)}" if st.trigger else ""
        if st.trigger_zone == "board keys":
            why = " · board keys"
        lines.append(f"PLAN → {st.plan} (HSReplay {tier}{why})")
        if ci is not None:
            owned = {_mname(m) for m in board}
            need = [_short(n) for n in ci.keys if n not in owned][:4]
            if need:
                lines.append("NEED → " + ", ".join(need))
        return lines

    ranked = " > ".join(f"{t}({st.lobby_labels.get(t, '-')})" for t in st.lobby)
    if st.lobby_known:
        lines.append(f"LOBBY → {ranked}")
    else:
        lines.append(f"LOBBY → tribes not detected — HSReplay best: "
                     f"{', '.join(st.strong)}")
    parts = []
    for t in st.strong:
        names = st.enablers.get(t) or []
        if names:
            parts.append(f"{t}: " + ", ".join(_short(n) for n in names[:_ENABLERS_PER_TRIBE]))
    if parts:
        lines.append("ENABLERS → " + " · ".join(parts))
    if st.shop_enablers:
        lines.append(f"COMMIT → buy {_short(st.shop_enablers[0])} ⇒ lock HSReplay comp")
    else:
        lines.append(f"FILL → board {len(board)}/7 — buy solids, no garbage, "
                     f"no roll while a solid buy is up")
    return lines
