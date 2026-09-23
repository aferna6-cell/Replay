"""Draft/choice recommender: rank a set of *offered* options and pick the best.

Distinct from the action recommender (`advisor.py`, which ranks buy/sell/roll on
your board). A draft is "choose 1 of N offered": hero select, trinket pick,
Discover. Each has its own best signal:

  * **hero**    — HSReplay average placement (lower = better), Firestone fallback,
    plus a small lobby-fit nudge from the hero's HSReplay guide.
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
                max_choices: int = F2P_HERO_CHOICES,
                available_tribes: Optional[List[str]] = None) -> List[Choice]:
    """Rank offered heroes by HSReplay 1st-place rate (most 1sts first).

    rank_value = -(1st % − lobby-fit points) + avg/1000, so the hero that wins
    the most lobbies leads and average placement only breaks ties. Falls back
    to the StatsDB (Firestone) average when HSReplay has no stats for the
    hero, converted to an estimated 1st % (see hero_pick.estimate_first).
    """
    from .hero_pick import estimate_first, hsreplay_row, lobby_fit, placement_line
    # Cap to the configured offer size (default 4). max_choices=0 means no cap.
    if max_choices and len(offered) > max_choices:
        offered = offered[:max_choices]
    out = []
    for nm in offered:
        adj, note = lobby_fit(nm, available_tribes)
        tail = f" · {note}" if note else ""
        row = hsreplay_row(nm)
        first, avg, line = placement_line(row) if row else (None, None, "")
        if first is not None:
            out.append(Choice(row.get("name") or nm,
                              -(first - adj) + (avg if avg is not None else 4.5) / 1000,
                              line + tail, "HSReplay 1st-place rate"))
            continue
        h: Optional[HeroStats] = _match(nm, db.heroes)
        if h:
            tribes = ("favors " + "/".join(h.best_tribes)) if h.best_tribes else "flexible tribes"
            sample = " · all-MMR sample" if getattr(h, "broad", False) else ""
            est = estimate_first(h.average_position)
            out.append(Choice(h.name, -(est - adj) + h.average_position / 1000,
                              f"avg {h.average_position:.2f} · 1st ~{est:.0f}% (est.) · "
                              f"{tribes} · {h.playstyle}{sample}{tail}",
                              "est. 1st-place rate"))
        else:
            out.append(Choice(nm, -estimate_first(4.5) + 4.5 / 1000,
                              "no stats for this hero (defaulting to average)",
                              "est. 1st-place rate"))
    out.sort(key=lambda c: c.rank_value)            # most 1st places first
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


def _normalize_kw(kw) -> str:
    return str(kw or "").lower().replace("_", " ").strip()


def _resolve_ck(m, kb, by_nm):
    """Resolve CardKnowledge: card_id first (live MinionView), then name."""
    if kb is None or m is None:
        return None
    cid = None
    if isinstance(m, dict):
        cid = m.get("card_id") or m.get("id")
    else:
        cid = getattr(m, "card_id", None) or getattr(m, "id", None)
    if cid and cid in kb:
        return kb[cid]
    nm = _minion_name(m) if not isinstance(m, str) else m
    if nm and by_nm:
        return by_nm.get(nm)
    return None


def _board_profile(board, kb):
    """Tribe counts + keyword set on the board, from card knowledge.

    Live board dicts carry card_id (MinionView.__dict__). Looking up by name
    alone misses many KB entries → empty tribes/keywords → trinket fit no-ops
    and ranking collapses to raw meta avg (feels like "not scoring effects").
    """
    by_nm = by_name(kb) if kb is not None else {}
    tribes, keywords = {}, set()
    for m in board or []:
        ck = _resolve_ck(m, kb, by_nm)
        if ck is not None:
            for tr in ck.tribes or []:
                tribes[tr.lower()] = tribes.get(tr.lower(), 0) + 1
            for kw in getattr(ck, "keywords", []) or []:
                keywords.add(_normalize_kw(kw))
            continue
        # Fallback: raw board dict keywords/tribes when KB misses entirely.
        if isinstance(m, dict):
            for tr in m.get("tribes") or []:
                tribes[str(tr).lower()] = tribes.get(str(tr).lower(), 0) + 1
            for kw in m.get("keywords") or []:
                keywords.add(_normalize_kw(kw))
    return tribes, keywords


def _trinket_fit(text: str, board_tribes: dict, target_tribe: Optional[str],
                 board_keywords: set, lobby_tribes=None):
    """Placement adjustment (negative = better) from effect vs board/lobby/plan.

    When the board has a clear plan, fit MUST be able to beat raw meta avg
    (typical S-vs-B gap ~1.0). Soft-only demotions of 0.15–0.55 were drowned by
    average_position and made picks feel like "always the premium / rightmost".

    Returns (delta, reason_bits).
    """
    t = (text or "").lower()
    if not t:
        return 0.0, []
    bonus, bits = 0.0, []
    target = (target_tribe or "").lower()
    lobby = {x.lower() for x in (lobby_tribes or []) if x}
    n_board = sum(board_tribes.values()) if board_tribes else 0
    clear_plan = n_board >= 3 or bool(board_keywords)
    mentioned = [tr for tr in _TRIBES if tr in t]
    for tr in mentioned:
        if board_tribes.get(tr) or tr == target or (tr in lobby and not board_tribes):
            bonus -= 0.90 if clear_plan else 0.50
            bits.append(f"matches your {tr.capitalize()}s")
            break
    else:
        dom = max(board_tribes, key=board_tribes.get) if board_tribes else None
        if mentioned and dom and dom not in mentioned and (
                not target or target not in mentioned):
            bonus += 0.90 if clear_plan else 0.45
            bits.append(f"off-tribe ({mentioned[0]}) for your {dom.capitalize()} board")
        elif mentioned and lobby and not any(tr in lobby for tr in mentioned):
            bonus += 0.55
            bits.append(f"off-lobby tribe ({mentioned[0]})")

    # Keyword plan: Battlecry premium only if board can play into it.
    plan_hits = [kw for kw in _PLAN_KEYWORDS if kw in t]
    if plan_hits:
        supported = [kw for kw in plan_hits if kw in board_keywords]
        if supported:
            # Strong enough to overcome ~1.0 meta gap when plan is real.
            density = sum(1 for _ in supported)
            swing = 1.15 if (clear_plan and n_board >= 3) else 0.70
            bonus -= min(1.40, swing + 0.20 * (density - 1))
            bits.append(f"plays into your {supported[0]}")
        else:
            if n_board >= 3:
                bonus += 1.20
                bits.append(f"no {plan_hits[0]} plan on board — don't force it")
            elif n_board >= 1:
                bonus += 0.55
                bits.append(f"thin {plan_hits[0]} support — risky premium")
    else:
        kw_hits = [kw for kw in _KEYWORDS if kw in t and kw in board_keywords]
        if kw_hits:
            bonus -= min(0.55, 0.25 * len(kw_hits))
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
                  available_tribes=None,
                  card_ids=None, snapshot=None) -> List[Choice]:
    """Rank trinkets by 1st-place rate, adjusted by effect/board/direction fit.

    Each trinket's HSReplay guide is followed like the hero guide: with PLAN
    locked, a trinket whose guide is for that comp/tribe is promoted and one
    whose guide wants another comp is demoted; before the lock the lobby's
    strong tribes decide; Naga-only guides are dead (trinket_comps).
    Pass the live ``snapshot`` so the locked PLAN is known.

    Base strength is the trinket's HSReplay 1st-place % (real when the ingest
    has HSReplay's placement distribution, else estimated from HSReplay's avg,
    else from the Firestone avg — see first_place). Fit deltas are placement
    units and convert at PTS_PER_PLACE, so a clear plan can still beat a
    trinket that only wins more lobbies in general.
    Aidan: strategy fit must beat raw avg when the board/lobby has a plan.
    Pass card_ids (parallel to offered names) so live MagicItem ids resolve
    even when entityName mismatches the stats DB.
    """
    from .first_place import PTS_PER_PLACE, estimate_first, first_label, first_rate
    from .hsreplay_guides import lookup_trinket
    board_tribes, board_kw = _board_profile(board, kb)
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
    clear_plan = (sum(board_tribes.values()) >= 3) or bool(board_kw)
    from .trinket_comps import pick_adjust
    plan_snap = snapshot
    if plan_snap is None and (board or lobby):
        plan_snap = {"board": list(board or []), "available_tribes": list(lobby or [])}
    out = []
    ids = list(card_ids or [])
    for i, nm in enumerate(offered):
        cid = ids[i] if i < len(ids) else None
        t: Optional[TrinketStats] = None
        if cid:
            t = _match(cid, db.trinkets)
        if t is None:
            t = _match(nm, db.trinkets)
        if not t:
            # Unknown: neutral 4.5 — do NOT let offer order decide #1 on ties;
            # stable-sort would pick Entities[0]. Slight index penalty keeps
            # unknowns below any real fit hit without preferring rightmost.
            out.append(Choice(nm, -estimate_first(4.5) + 0.001 * i,
                              "no stats for this trinket", "est. 1st-place rate"))
            continue
        fit, bits = _trinket_fit(t.text, board_tribes, target, board_kw, lobby)
        try:
            from .hsreplay_guides import trinket_guide_score
            bnames = []
            for m in (board or []):
                if isinstance(m, dict):
                    bnames.append(m.get("name"))
                else:
                    bnames.append(getattr(m, "name", None))
            gdelta, gbits = trinket_guide_score(
                t.name or nm, board_tribes=board_tribes,
                board_names=bnames, target_tribe=target,
            )
            # Also try by card id if present
            if gdelta == 0.0 and getattr(t, "card_id", None):
                gdelta, gbits = trinket_guide_score(
                    t.card_id, board_tribes=board_tribes,
                    board_names=bnames, target_tribe=target,
                )
            fit += gdelta
            bits.extend(gbits)
        except Exception:
            pass
        try:
            padj, pbits = pick_adjust(cid or getattr(t, "card_id", None) or t.name,
                                      plan_snap)
        except Exception:
            padj, pbits = 0.0, []
        if padj == 0.0 and not pbits and cid:
            padj, pbits = pick_adjust(t.name, plan_snap)
        fit += padj
        bits[:0] = pbits
        # When board has a plan, amplify fit so strategy outranks raw meta.
        if clear_plan and fit != 0.0:
            fit = fit * 1.15
        hs = lookup_trinket(getattr(t, "card_id", None)) or lookup_trinket(t.name or nm)
        first, avg, est = first_rate((hs or {}).get("stats"))
        if first is None:
            avg, first, est = t.average_position, estimate_first(t.average_position), True
        eff = -(first - fit * PTS_PER_PLACE)
        label = first_label(first, est)
        # Effect-first reason for overlay (1st rate + avg are context).
        if bits:
            reason = "; ".join(bits) + f" · {label} · avg {avg:.2f}"
        else:
            reason = f"{label} · avg {avg:.2f} · tier {t.tier}"
        out.append(Choice(t.name, eff, reason, "board-adjusted 1st-place rate"))
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
                    max_choices: int = F2P_HERO_CHOICES,
                    available_tribes: Optional[List[str]] = None) -> dict:
    """Plan hero-select: which hero to reroll, then how to rank the rest.

    Advisory only. Hero-select reroll is a mulligan-phase action (Power.log tags
    BACON_MULLIGAN_HERO_REROLL_ACTIVE / BACON_NUM_MAX_REROLL_PER_HERO) — not the
    tavern Refresh button (TB_BaconShop_8p_Reroll_Button).

    Returns ``{reroll: Choice|None, picks: List[Choice], lines: List[str]}``.
    """
    ranked = rank_heroes(offered, db, max_choices=max_choices,
                         available_tribes=available_tribes)
    reroll = None
    picks = list(ranked)

    if len(ranked) >= 3 and rerolls_available > 0:
        worst = ranked[-1]
        # Skip only when every option is equally unknown — no signal to prefer.
        all_unknown = all("no stats" in c.reason for c in ranked)
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
            f"Reroll: {reroll.name} — {reroll.reason.split(' · ')[0]} "
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
                     gift_by_name=None, available_tribes=None,
                     card_ids=None, snapshot=None) -> List[Choice]:
    """Dispatch to the right ranker. kind: 'hero' | 'trinket' | 'discover'.

    Heroes: rank up to HSBG_HERO_CHOICES (default 4). For the live overlay's
    reroll-then-pick UX, use ``hero_draft_plan`` instead."""
    if kind == "hero":
        return rank_heroes(offered, db or StatsDB.load())
    if kind == "trinket":
        return rank_trinkets(offered, db or StatsDB.load(), board=board, kb=kb,
                             hero_ctx=hero_ctx, available_tribes=available_tribes,
                             card_ids=card_ids, snapshot=snapshot)
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
