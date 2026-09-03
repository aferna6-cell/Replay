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


# Free-to-play players choose from 2 heroes; the paid Perks add 2 more (shown to
# the right of the offer). Recommend only the free pair so the pick is one you can
# actually take. Set HSBG_HERO_CHOICES=4 to include the paid extras.
import os as _os
F2P_HERO_CHOICES = int(_os.environ.get("HSBG_HERO_CHOICES", "2"))


def rank_heroes(offered: List[str], db: StatsDB,
                max_choices: int = F2P_HERO_CHOICES) -> List[Choice]:
    # The first `max_choices` offered are the free heroes (offer order = display
    # order, free pair first); the rest are the paid Perks heroes.
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


_TRIBES = ["beast", "mech", "murloc", "dragon", "demon", "elemental",
           "pirate", "naga", "undead", "quilboar"]
_KEYWORDS = ["divine shield", "deathrattle", "reborn", "taunt", "windfury",
             "venomous", "magnetic"]


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
                 board_keywords: set):
    """Placement adjustment (negative = better) from how the trinket's effect
    matches your board's tribes/keywords. Returns (delta, reason_bits)."""
    t = (text or "").lower()
    if not t:
        return 0.0, []
    bonus, bits = 0.0, []
    target = (target_tribe or "").lower()
    mentioned = [tr for tr in _TRIBES if tr in t]
    for tr in mentioned:
        if board_tribes.get(tr) or tr == target:
            bonus -= 0.4                                    # buffs a tribe you run
            bits.append(f"matches your {tr.capitalize()}s")
            break
    else:
        # Mentions a tribe, but not one you're on, and you have a clear identity.
        dom = max(board_tribes, key=board_tribes.get) if board_tribes else None
        if mentioned and dom and dom not in mentioned and target and target not in mentioned:
            bonus += 0.3
            bits.append(f"off-tribe ({mentioned[0]}) for your {dom.capitalize()} board")
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
                  hero_ctx: Optional[HeroContext] = None) -> List[Choice]:
    """Rank trinkets by meta placement, adjusted for how well each fits your
    board (tribe/keyword match) AND the comp you're building toward (build-path),
    so the pick is conditioned on board state."""
    board_tribes, board_kw = _board_profile(board, kb)
    # Direction = hero target tribe, else the build-path comp's tribe.
    target = (hero_ctx.target_tribe if hero_ctx else None) or _build_target_tribe(board)
    out = []
    for nm in offered:
        t: Optional[TrinketStats] = _match(nm, db.trinkets)
        if not t:
            out.append(Choice(nm, 4.5, "no stats for this trinket", "avg placement"))
            continue
        fit, bits = _trinket_fit(t.text, board_tribes, target, board_kw)
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
    # Tavern spells (in the KB since 2026-08-20) have no stats — keep the
    # eval features numeric.
    return {"name": ck.name, "attack": ck.attack if ck.attack is not None else 0,
            "health": ck.health if ck.health is not None else 1}


def rank_discover(offered: List[str], board, kb, scorer=None,
                  hero_ctx: Optional[HeroContext] = None, tier=None) -> List[Choice]:
    """Rank Discover options (incl. triple-discovers) by how much each improves
    your live board AND advances the winning comp you're building toward. Both
    signals are conditioned on the current board state.

    Sort key blends two board-aware terms:
      * eval-net equity delta — does adding it strengthen the board now?
      * build-path value — does it advance a reachable winning archetype?
    """
    scorer = scorer or get_scorer()
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


def recommend_choice(kind: str, offered: List[str], *, db: Optional[StatsDB] = None,
                     board=None, kb=None, scorer=None,
                     hero_ctx: Optional[HeroContext] = None, tier=None) -> List[Choice]:
    """Dispatch to the right ranker. kind: 'hero' | 'trinket' | 'discover'.

    Heroes: rank ALL offered (no positional cap). Paywalled heroes can't be told
    apart from owned ones in the log yet, so the overlay shows the full ranking and
    caveats the user to skip any padlocked one — better than silently capping to
    the wrong two."""
    if kind == "hero":
        return rank_heroes(offered, db or StatsDB.load(), max_choices=0)
    if kind == "trinket":
        return rank_trinkets(offered, db or StatsDB.load(), board=board, kb=kb,
                             hero_ctx=hero_ctx)
    if kind in ("discover", "dark_gift"):
        # Dark gifts rank like discovers — board fit + build direction. The
        # WHEN-to-press timing call is the Director's (LLM) job; here we only
        # rank the offered gifts against the live board (spec req 10).
        return rank_discover(offered, board or [], kb, scorer=scorer,
                             hero_ctx=hero_ctx, tier=tier)
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
