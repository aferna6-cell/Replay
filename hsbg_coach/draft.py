"""Draft/choice recommender: rank a set of *offered* options and pick the best.

Distinct from the action recommender (`advisor.py`, which ranks buy/sell/roll on
your board). A draft is "choose 1 of N offered": hero select, trinket pick,
Discover. Each has its own best signal:

  * **hero**    — population average placement (lower = better), at your MMR.
  * **trinket** — population average placement.
  * **discover**— board fit: how much adding this minion raises your board's
    expected finish (the eval net + card2vec synergy on your *live* board).

So hero/trinket lean on the meta stats; Discover leans on the deep brain + your
current board. Everything is ranked best-first.
"""

from dataclasses import dataclass
from typing import List, Optional

from .board_value import get_scorer, _name as _minion_name
from .cards import by_name
from .economy import HeroContext
from .stats import StatsDB, HeroStats, TrinketStats
from .synergy import score_card, load_embeddings


@dataclass
class Choice:
    name: str
    rank_value: float        # sort key (best = smallest); internal
    reason: str
    metric: str              # what rank_value means, for display

    def line(self) -> str:
        return f"  {self.name} — {self.reason}"


def _match(name: str, pool):
    """Find a stats row for an offered option by name or card id (fuzzy)."""
    key = (name or "").strip().lower()
    for s in pool:
        if s.name.lower() == key or getattr(s, "card_id", "").lower() == key:
            return s
    for s in pool:                                  # partial / substring
        if key and (key in s.name.lower() or s.name.lower() in key):
            return s
    return None


# Hero-select offer size. Perks / paid players see 4 heroes (and typically 1
# hero-select reroll). Free-to-play sees 2. Default is 4 (Aidan's setup); set
# HSBG_HERO_CHOICES=2 to truncate to the free pair. Do NOT confuse with the
# tavern Refresh button (TB_BaconShop_8p_Reroll_Button) — that is shop roll.
import os as _os
F2P_HERO_CHOICES = int(_os.environ.get("HSBG_HERO_CHOICES", "4"))


def rank_heroes(offered: List[str], db: StatsDB,
                max_choices: int = F2P_HERO_CHOICES) -> List[Choice]:
    # Cap to the configured offer size (default 4). max_choices=0 means no cap.
    if max_choices and len(offered) > max_choices:
        offered = offered[:max_choices]
    out = []
    for nm in offered:
        h: Optional[HeroStats] = _match(nm, db.heroes)
        if h:
            tribes = ("favors " + "/".join(h.best_tribes)) if h.best_tribes else "flexible tribes"
            sample = " · all-MMR sample" if getattr(h, "broad", False) else ""
            out.append(Choice(h.name, h.average_position,
                              f"avg {h.average_position:.2f} · {tribes} · {h.playstyle}{sample}",
                              "avg placement"))
        else:
            out.append(Choice(nm, 4.5, "no stats for this hero (defaulting to average)",
                              "avg placement"))
    out.sort(key=lambda c: c.rank_value)            # lower placement = better
    return out


# 36.6.1 pool: no Naga. Aberration is in.
_TRIBES = ["beast", "mech", "murloc", "dragon", "demon", "elemental",
           "pirate", "undead", "quilboar", "aberration"]
_KEYWORDS = ["battlecry", "deathrattle", "divine shield", "reborn", "taunt",
             "windfury", "venomous", "magnetic", "avenge"]
# Keyword trinkets need board support — don't take Battlecry premium on a
# non-Battlecry board (soft demotion below).
_PLAN_KEYWORDS = ("battlecry", "deathrattle", "divine shield", "reborn",
                  "avenge", "magnetic")


def _board_profile(board, kb):
    """Tribe counts + keyword set on the board, from card knowledge."""
    idx = by_name(kb) if kb is not None else {}
    tribes, keywords = {}, set()
    for m in board or []:
        ck = idx.get(_minion_name(m))
        if not ck:
            continue
        for tr in ck.tribes:
            tribes[tr.lower()] = tribes.get(tr.lower(), 0) + 1
        for kw in getattr(ck, "keywords", []) or []:
            keywords.add(kw.lower())
    return tribes, keywords


def _trinket_fit(text: str, board_tribes: dict, target_tribe: Optional[str],
                 board_keywords: set, lobby_tribes=None):
    """Placement adjustment (negative = better) from how the trinket's effect
    matches your board / lobby lean / plan. Soft only — meta placement still
    matters; this stops random premium picks we won't play into.

    Returns (delta, reason_bits).
    """
    t = (text or "").lower()
    if not t:
        return 0.0, []
    bonus, bits = 0.0, []
    target = (target_tribe or "").lower()
    lobby = {x.lower() for x in (lobby_tribes or []) if x}
    mentioned = [tr for tr in _TRIBES if tr in t]
    for tr in mentioned:
        if board_tribes.get(tr) or tr == target or (tr in lobby and not board_tribes):
            bonus -= 0.45                                   # buffs a tribe you run/lean
            bits.append(f"matches your {tr.capitalize()}s")
            break
    else:
        # Mentions a tribe, but not one you're on, and you have a clear identity.
        dom = max(board_tribes, key=board_tribes.get) if board_tribes else None
        if mentioned and dom and dom not in mentioned and (
                not target or target not in mentioned):
            bonus += 0.45
            bits.append(f"off-tribe ({mentioned[0]}) for your {dom.capitalize()} board")
        elif mentioned and lobby and not any(tr in lobby for tr in mentioned):
            bonus += 0.35
            bits.append(f"off-lobby tribe ({mentioned[0]})")

    # Keyword plan alignment — Battlecry trinket only if we can be Battlecry-heavy.
    plan_hits = [kw for kw in _PLAN_KEYWORDS if kw in t]
    if plan_hits:
        supported = [kw for kw in plan_hits if kw in board_keywords]
        if supported:
            bonus -= min(0.35, 0.15 * len(supported))
            bits.append(f"plays into your {supported[0]}")
        else:
            # Board has no matching keyword density — demote even premium text.
            # Thin boards (<2 minions) stay flexible (can pivot into the plan).
            n_board = sum(board_tribes.values()) if board_tribes else 0
            if n_board >= 3:
                bonus += 0.55
                bits.append(f"no {plan_hits[0]} plan on board — don't force it")
            elif n_board >= 1:
                bonus += 0.25
                bits.append(f"thin {plan_hits[0]} support — risky premium")
    else:
        kw_hits = [kw for kw in _KEYWORDS if kw in t and kw in board_keywords]
        if kw_hits:
            bonus -= min(0.2, 0.1 * len(kw_hits))
            bits.append(f"synergizes with {kw_hits[0]}")
    return bonus, bits


def _build_target_tribe(board) -> Optional[str]:
    """The tribe of the winning comp this board is heading toward (build-path),
    so choices favor your *direction*, not just the minions you happen to hold."""
    try:
        from .build_path import infer_target
        fit = infer_target(board or [])
        if fit and fit.have >= 1 and fit.arch.tribe:
            return fit.arch.tribe.lower()
    except Exception:
        pass
    return None


def rank_trinkets(offered: List[str], db: StatsDB, board=None, kb=None,
                  hero_ctx: Optional[HeroContext] = None,
                  available_tribes=None) -> List[Choice]:
    """Rank trinkets by meta placement, adjusted for board / lobby lean / plan.

    Soft rule (Aidan): don't take a Battlecry (etc.) premium unless the board
    can play into that plan; prefer tribe/keyword fit over raw meta avg.
    """
    board_tribes, board_kw = _board_profile(board, kb)
    # Also count keywords from raw board dicts when kb misses.
    for m in board or []:
        if isinstance(m, dict):
            for kw in m.get("keywords") or []:
                board_kw.add(str(kw).lower().replace("_", " "))
            for tr in m.get("tribes") or []:
                board_tribes[str(tr).lower()] = board_tribes.get(str(tr).lower(), 0) + 1
    # Direction = hero target tribe, else build-path, else soft lobby lean.
    target = (hero_ctx.target_tribe if hero_ctx else None) or _build_target_tribe(board)
    lobby = available_tribes or getattr(hero_ctx, "available_tribes", None) if hero_ctx else available_tribes
    if target is None and lobby:
        try:
            from .tribe_policy import soft_lean_tribe
            target, _ = soft_lean_tribe(board, available_tribes=lobby, kb=kb)
            if target:
                target = target.lower()
        except Exception:
            pass
    out = []
    for nm in offered:
        t: Optional[TrinketStats] = _match(nm, db.trinkets)
        if not t:
            out.append(Choice(nm, 4.5, "no stats for this trinket", "avg placement"))
            continue
        fit, bits = _trinket_fit(t.text, board_tribes, target, board_kw, lobby)
        eff = t.average_position + fit                     # lower = better
        reason = f"avg {t.average_position:.2f} · tier {t.tier}"
        if bits:
            reason += " · " + "; ".join(bits) + f" ({fit:+.2f})"
        out.append(Choice(t.name, eff, reason, "board-adjusted placement"))
    out.sort(key=lambda c: c.rank_value)
    return out


def _minion_from_name(ck, name):
    if ck is None:
        return {"name": name, "attack": 3, "health": 3}
    return {"name": ck.name, "attack": ck.attack, "health": ck.health}


def rank_discover(offered: List[str], board, kb, scorer=None,
                  hero_ctx: Optional[HeroContext] = None, tier=None,
                  gift_by_name=None, available_tribes=None) -> List[Choice]:
    """Rank Discover options (incl. triple-discovers) by how much each improves
    your live board AND advances the winning comp you're building toward. Both
    signals are conditioned on the current board state.

    Sort key blends two board-aware terms:
      * eval-net equity delta — does adding it strengthen the board now?
      * build-path value — does it advance a reachable winning archetype?
    """
    scorer = scorer or get_scorer()
    # Dark Gift discovers: gift effect >> base minion body.
    if gift_by_name:
        from .dark_gift import enrich_discover_for_gifts, rank_dark_gift_options
        from .tribe_policy import soft_lean_tribe
        opts = enrich_discover_for_gifts(list(offered), gift_by_name, kb)
        avail = available_tribes or (getattr(hero_ctx, "available_tribes", None) if hero_ctx else None)
        direction, _ = soft_lean_tribe(
            board, avail, kb=kb,
            hero_target=getattr(hero_ctx, "target_tribe", None) if hero_ctx else None)
        ranked_g = rank_dark_gift_options(opts, board=board, kb=kb,
                                          available_tribes=avail, direction=direction)
        out = []
        for opt, val, reason in ranked_g:
            out.append(Choice(opt.name, val, f"Dark Gift — {reason}", "gift > body"))
        return out

    emb = load_embeddings()
    idx = by_name(kb) if kb is not None else {}
    hero_id = hero_ctx.hero if hero_ctx and hero_ctx.hero else "UNKNOWN"
    target = hero_ctx.target_tribe if hero_ctx else None
    board = list(board or [])
    base = scorer.equity(board, hero_id)
    board_cks = [idx.get(_minion_name(m)) for m in board if idx.get(_minion_name(m))]
    board_tribes, _ = _board_profile(board, kb)
    dominant = max(board_tribes, key=board_tribes.get) if board_tribes else None
    committed = dominant and board_tribes.get(dominant, 0) >= 2

    from .build_path import path_value
    from .effect_synergy import board_synergy
    out = []
    for nm in offered:
        ck = idx.get(nm)
        cand = _minion_from_name(ck, nm)
        delta = scorer.equity(board + [cand], hero_id) - base
        ctribes = [t.lower() for t in (getattr(ck, "tribes", None) or [])]
        ctribe = ctribes[0] if ctribes else None
        padj, preason = path_value(board, nm, tier, candidate_tribe=ctribe, emb=emb)

        # Tribe fit relative to your committed comp — a Naga discovered into a
        # Murloc board should lose to an on-tribe Murloc even if its raw stats win.
        tribe_adj, tribe_bit = 0.0, None
        if committed and ctribes:
            if dominant in ctribes or "all" in ctribes:     # 'all' = Amalgam-style
                tribe_adj, tribe_bit = -0.07, f"on-tribe ({dominant.capitalize()})"
            else:
                tribe_adj, tribe_bit = 0.10, f"off-tribe for your {dominant.capitalize()}s"

        # Mechanical combo from card text (produces/wants), board-aware.
        syn, syn_bits = (board_synergy(ck, board_cks) if ck is not None else (0.0, []))

        # One sort key (smaller = better). Build-path weighted harder than before
        # (÷5, not ÷7) so the comp direction beats raw stats.
        rank_value = -delta + (padj / 5.0) + tribe_adj - syn * 0.03

        reason = f"{delta:+.0%} equity"
        extra = [b for b in (tribe_bit, preason) if b] + (syn_bits[:1] if syn else [])
        if extra:
            reason += " — " + "; ".join(extra)
        out.append(Choice(nm, rank_value, reason, "board fit + build-path"))
    out.sort(key=lambda c: c.rank_value)
    return out


def hero_draft_plan(offered: List[str], db: StatsDB,
                    rerolls_available: int = 1,
                    max_choices: int = F2P_HERO_CHOICES) -> dict:
    """Plan hero-select: which hero to reroll, then how to rank the rest.

    Advisory only. Hero-select reroll is a mulligan-phase action (Power.log tags
    BACON_MULLIGAN_HERO_REROLL_ACTIVE / BACON_NUM_MAX_REROLL_PER_HERO) — not the
    tavern Refresh button (TB_BaconShop_8p_Reroll_Button).

    Returns ``{reroll: Choice|None, picks: List[Choice], lines: List[str]}``.
    """
    ranked = rank_heroes(offered, db, max_choices=max_choices)
    reroll = None
    picks = list(ranked)

    if len(ranked) >= 3 and rerolls_available > 0:
        worst = ranked[-1]
        # Skip only when every option is equally unknown — no signal to prefer.
        all_unknown = all(
            abs(c.rank_value - 4.5) < 1e-9 and "no stats" in c.reason
            for c in ranked
        )
        if not all_unknown:
            # With 4 offers + a reroll, always name the weakest target (even if it
            # is within ~0.15 of the 3rd-best — Aidan wants a clear reroll).
            # With 3 offers, still name the worst when a reroll is available.
            reroll = worst
            picks = ranked[:-1]

    lines: List[str] = []
    if reroll is not None:
        n = len(ranked)
        lines.append(
            f"Reroll: {reroll.name} — avg {reroll.rank_value:.2f} "
            f"(weakest of {n})"
        )
        lines.append("Then pick (best first):")
        for i, c in enumerate(picks, 1):
            lines.append(f"  {i}. {c.name} — {c.reason}")
    else:
        lines.append("Hero ranking — pick the best one that isn't locked:")
        for i, c in enumerate(picks, 1):
            lines.append(f"  {i}. {c.name} — {c.reason}")
    return {"reroll": reroll, "picks": picks, "lines": lines}


def recommend_choice(kind: str, offered: List[str], *, db: Optional[StatsDB] = None,
                     board=None, kb=None, scorer=None,
                     hero_ctx: Optional[HeroContext] = None, tier=None,
                     gift_by_name=None, available_tribes=None) -> List[Choice]:
    """Dispatch to the right ranker. kind: 'hero' | 'trinket' | 'discover'.

    Heroes: rank up to HSBG_HERO_CHOICES (default 4). For the live overlay's
    reroll-then-pick UX, use ``hero_draft_plan`` instead."""
    if kind == "hero":
        return rank_heroes(offered, db or StatsDB.load())
    if kind == "trinket":
        return rank_trinkets(offered, db or StatsDB.load(), board=board, kb=kb,
                             hero_ctx=hero_ctx)
    if kind == "discover":
        return rank_discover(offered, board or [], kb, scorer=scorer,
                             hero_ctx=hero_ctx, tier=tier,
                             gift_by_name=gift_by_name,
                             available_tribes=available_tribes)
    if kind in ("hero_power", "quest"):
        return rank_pick(offered, kind)
    raise ValueError(f"unknown choice kind: {kind}")


def rank_pick(offered: List[str], kind: str) -> List[Choice]:
    """Surface a hero-power / quest choice (e.g. Nguyen, Sire Denathrius) so the
    coach prompts the pick instead of mis-ranking it as a board discover.

    HONEST SCOPE: we don't yet have per-option stats/effects for hero powers or
    quests, so this lists the options (best-effort, offer order) with a clear note
    rather than faking a confident ranking. A real game log of these heroes + an
    effect table is what's needed to rank them well — same pattern as spells/tech."""
    label = "hero power" if kind == "hero_power" else "quest"
    return [Choice(nm, float(i),
                   f"{label} option — pick what fits your plan (no per-option stats yet)",
                   "pick")
            for i, nm in enumerate(offered)]
