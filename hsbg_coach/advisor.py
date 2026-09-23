"""The move recommender: rank every legal action by how much it helps.

For each action from `actions.legal_actions`, estimate the value of the resulting
state and rank them:

  * **buy / sell** — the resulting board is scored by `board_value` (the deep
    eval net if trained, else the heuristic). This is real one-ply lookahead:
    "does adding this minion raise my expected finish?" A buy onto a full board
    is modelled as sell-the-weakest-then-buy. Reasons come from the synergy tags.
  * **tier up / roll / freeze** — economy actions don't change the board now, so
    they're scored by heuristics (pace-vs-curve, surplus gold, unaffordable shop
    gems). Honest: these are heuristic, not lookahead — we don't simulate future
    shops.
  * **reposition** — scored by the combat sim against a known opponent.

Output is one ranked list (every action, best first) plus a single `best` move.
One-ply: it ranks each immediate action, not full multi-buy turn sequences.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import copy

from .actions import (
    legal_actions, Action, BUY, BUY_SPELL, SELL, ROLL, LEVEL, REPOSITION, FREEZE,
    HERO_POWER, ACTIVATE, DARK_GIFT, END, MAX_BOARD, BUY_COST, SELL_VALUE, ROLL_COST,
)
from .board_value import get_scorer, _val, _name
from .cards import by_name
from .economy import HeroContext, EconomyConfig
from .synergy import score_card, load_embeddings

_PRIO_SCALE = 7.0          # equity delta -> priority slope
_BUY_MEANINGFUL = 0.01     # equity gain that counts as a real improvement


def _get(snap, key, default=None):
    return snap.get(key, default) if isinstance(snap, dict) else getattr(snap, key, default)


def _clamp(x, lo=0.0, hi=0.97):
    return max(lo, min(hi, x))


@dataclass
class ScoredAction:
    action: Action
    priority: float
    reason: str
    equity: Optional[float] = None
    delta: Optional[float] = None

    def line(self) -> str:
        d = f"  ({self.delta:+.0%} equity)" if self.delta is not None else ""
        return f"  {self.priority:4.2f}  {self.action.describe()}{d} — {self.reason}"


@dataclass
class ActionPlan:
    scorer: str
    current_equity: float
    ranked: List[ScoredAction]
    best: ScoredAction
    caveats: List[str] = field(default_factory=list)

    def of_kind(self, kind: str) -> List[ScoredAction]:
        return [a for a in self.ranked if a.action.kind == kind]

    def summary(self) -> str:
        lines = [f"Best move: {self.best.action.describe()} — {self.best.reason}",
                 f"(board equity {self.current_equity:.0%}, scorer: {self.scorer})",
                 "", "All actions, ranked:"]
        lines += [a.line() for a in self.ranked]
        if self.caveats:
            lines += ["", "Caveats:"] + [f"  - {c}" for c in self.caveats]
        return "\n".join(lines)


def advise_actions(snapshot, kb=None, hero_ctx: Optional[HeroContext] = None,
                   scorer=None, enemy_boards=None, pace=None,
                   config: Optional[EconomyConfig] = None,
                   include_reposition: bool = True) -> ActionPlan:
    cfg = config or EconomyConfig()
    scorer = scorer or get_scorer()
    board = list(_get(snapshot, "board", []) or [])
    gold = _get(snapshot, "gold") or 0
    tier = _get(snapshot, "tavern_tier") or 1
    health = _get(snapshot, "hero_health")
    hero_id = (hero_ctx.hero if hero_ctx and hero_ctx.hero else "UNKNOWN")
    target_tribe = hero_ctx.target_tribe if hero_ctx else None

    emb = load_embeddings()
    idx = by_name(kb) if kb is not None else {}
    board_cks = [idx.get(_name(m)) for m in board if idx.get(_name(m))]
    base = scorer.equity(board, hero_id)

    scored: List[ScoredAction] = []
    buys: List[ScoredAction] = []
    for act in legal_actions(snapshot, kb):
        if act.kind == BUY:
            sa = _score_buy(act, board, base, scorer, hero_id, idx, board_cks,
                            target_tribe, emb)
            buys.append(sa)
            scored.append(sa)
        elif act.kind == BUY_SPELL:
            scored.append(_score_spell(act, gold))
        elif act.kind == HERO_POWER:
            from .jeef_priors import hero_power_adjust
            adj, reason = hero_power_adjust(snapshot, act.cost)
            # priority: higher is better; placement adj is negative-better.
            # Stronger baseline so usable HP can be NEXT in midgame windows.
            prio = _clamp(0.62 - (adj if adj is not None else -0.55))
            scored.append(ScoredAction(
                act, prio, reason or "hero power is available — using it is usually value"))
        elif act.kind == ACTIVATE:
            from .jeef_priors import activate_adjust
            adj, reason = activate_adjust(snapshot, act.cost)
            prio = _clamp(0.55 - (adj or 0.0))
            scored.append(ScoredAction(
                act, prio,
                reason or f"activate {act.target} ({act.cost}g)"))
        elif act.kind == DARK_GIFT:
            from .jeef_priors import dark_gift_adjust
            adj, reason = dark_gift_adjust(snapshot, act.cost)
            prio = _clamp(0.55 - (adj or 0.0))
            scored.append(ScoredAction(
                act, prio,
                reason or f"Dark Gift ({act.cost}g) — discover a gifted minion"))
        elif act.kind == SELL:
            scored.append(_score_sell(act, board, base, scorer, hero_id))

    best_buy_delta = max((b.delta for b in buys), default=0.0)

    # Economy + positioning depend on the buy landscape, so score them after.
    for act in legal_actions(snapshot, kb):
        if act.kind == LEVEL:
            scored.append(_score_level(act, snapshot, pace, health, cfg,
                                       hero_ctx, best_buy_delta))
        elif act.kind == ROLL:
            scored.append(_score_roll(act, gold, best_buy_delta, target_tribe,
                                      snapshot=snapshot))
        elif act.kind == FREEZE:
            scored.append(_score_freeze(act, snapshot, gold, idx, board_cks,
                                        target_tribe, emb))
        elif act.kind == REPOSITION and include_reposition:
            # Skipped in the live overlay path: optimize_vs_field runs hundreds of
            # thousands of combat sims (~1.5s) and we don't display reposition
            # advice, so computing it just makes the panel laggy.
            scored.append(_score_reposition(act, board, enemy_boards))
        elif act.kind == END:
            scored.append(ScoredAction(act, 0.15, "pass the turn"))

    scored.sort(key=lambda a: a.priority, reverse=True)
    # Hard rule: sparse/board<5 + gold>=3 + acceptable fill → best kind != ROLL.
    try:
        from .jeef_priors import roll_must_not_be_next
        if scored and scored[0].action.kind == ROLL and roll_must_not_be_next(snapshot):
            buys = [a for a in scored if a.action.kind == BUY]
            if buys:
                best_buy = max(buys, key=lambda a: a.priority)
                for i, a in enumerate(scored):
                    if a.action.kind == ROLL:
                        scored[i] = ScoredAction(
                            a.action, min(a.priority, best_buy.priority - 0.05),
                            a.reason, a.equity, a.delta)
                scored.sort(key=lambda a: a.priority, reverse=True)
    except Exception:
        pass
    caveats = [
        "Buy/sell use board lookahead; roll/level/freeze are heuristic "
        "(future shops aren't simulated).",
    ]
    if enemy_boards is None:
        caveats.append("No opponent board given — reposition advice is generic.")
    if scorer.name == "heuristic":
        caveats.append("Deep eval net not loaded — using the stdlib heuristic "
                       "scorer (train ml/eval_net.pt for sharper buys).")
    return ActionPlan(scorer.name, base, scored, scored[0], caveats)


# --- per-action scorers ------------------------------------------------------
def _keep_rank(m, board, idx, target_tribe, emb) -> float:
    """Lower = sell first. Synergy-aware, not stats-only: on a committed comp we
    keep on-tribe pieces and sell off-comp bodies first, so a fat off-tribe vanilla
    goes before a small comp piece."""
    ck = idx.get(_name(m)) if idx else None
    if ck is None:
        return _val(m)
    rest = [idx.get(_name(x)) for x in board if x is not m and idx.get(_name(x))]
    tribes = {}
    for c in rest:
        for t in (getattr(c, "tribes", None) or []):
            tribes[t.lower()] = tribes.get(t.lower(), 0) + 1
    dom = max(tribes, key=tribes.get) if tribes else None
    committed = bool(dom) and tribes.get(dom, 0) >= 3
    ctr = [t.lower() for t in (ck.tribes or [])]
    syn = score_card(ck, rest, target_tribe=(target_tribe or dom), embeddings=emb).score
    rank = _val(m) + max(0.0, syn) * 2.5
    if committed:
        if dom in ctr or "all" in ctr:
            rank += 8.0                      # an on-tribe comp piece — keep it
        else:
            from .effect_synergy import board_synergy
            es, _ = board_synergy(ck, rest)
            if es <= 0:
                rank -= 6.0                  # off-comp body, no combo — sell first
    return rank


def _score_buy(act, board, base, scorer, hero_id, idx, board_cks, target_tribe, emb):
    minion = act.detail.get("minion")
    sold = None
    if len(board) >= MAX_BOARD:
        # Sell the least valuable to KEEP (stats + synergy), not the lowest stats,
        # so the eval net values keeping your comp pieces.
        weakest = min(board, key=lambda x: _keep_rank(x, board, idx, target_tribe, emb))
        cand = [m for m in board if m is not weakest] + [minion]
        sold = _name(weakest)
    else:
        cand = board + [minion]
    eq = scorer.equity(cand, hero_id)
    delta = eq - base

    bits = []
    ck = idx.get(act.target)
    if ck is not None:
        verdict = score_card(ck, board_cks, target_tribe=target_tribe, embeddings=emb)
        bits = verdict.reasons[:2]
    reason = "; ".join(bits) if bits else "adds board strength"

    # Situational tech (Tunnel Blaster, Deadly Spore, …) reads as a strong buy on
    # raw stats/keywords. Note that here; whole-game ranking (game_value) does the
    # matchup-aware promote/demote against the live opponent board.
    from .card_roles import tech_note
    note = tech_note(getattr(minion, "card_id", None) if not isinstance(minion, dict)
                     else minion.get("card_id"), act.target)
    if note:
        reason = note
    prio = _clamp(0.5 + delta * _PRIO_SCALE)
    if sold:
        reason = f"sell {sold} for room; " + reason
    return ScoredAction(act, prio, reason, equity=eq, delta=delta)


def _score_spell(act, gold):
    """Score a tavern-spell buy. Value/notes come from spell_roles (whole-game
    ranking applies the placement bonus); here we set a reasonable priority and
    the reason so it shows up as a genuine option."""
    from .spell_roles import spell_value
    from .jeef_priors import spell_prior_adjust
    spell = act.detail.get("spell") or {}
    cid = spell.get("card_id") if isinstance(spell, dict) else getattr(spell, "card_id", None)
    bonus, reason = spell_value(cid, act.target, act.cost, gold)
    jadj, jreason = spell_prior_adjust(cid, act.target)
    if jadj:
        bonus = bonus + jadj
        reason = jreason or reason
    prio = _clamp(0.45 - bonus)        # better (more negative) bonus -> higher prio
    return ScoredAction(act, prio, reason)


def _score_sell(act, board, base, scorer, hero_id):
    minion = act.detail.get("minion")
    cand = [m for m in board if m is not minion]
    eq = scorer.equity(cand, hero_id)
    delta = eq - base
    if delta > 0:
        reason = "removing it actually improves the board (dead weight)"
    else:
        reason = "frees a slot / 1 gold, but weakens the board"
    prio = _clamp(0.3 + delta * _PRIO_SCALE)
    return ScoredAction(act, prio, reason, equity=eq, delta=delta)


def _score_level(act, snapshot, pace, health, cfg, hero_ctx, best_buy_delta):
    prio = 0.45
    reason = f"tier up to {act.detail.get('to_tier')} for a stronger pool"
    if pace:
        from .pace import pace_advice
        v = pace_advice(snapshot, pace)
        if v.behind_leveling:
            gap = (v.bench_tier or 0) - (v.your_tier or 0)
            prio = 0.6 + gap * 0.15
            reason = v.notes[0] if v.notes else reason
        elif v.your_tier and v.bench_tier and v.your_tier >= v.bench_tier:
            prio = 0.4
            reason = "on/ahead of the tier curve — leveling is optional"
    if health is not None and health < cfg.low_health:
        prio -= 0.2
        reason += " (low HP — staying alive may matter more)"
    elif health is None or health >= cfg.safe_health:
        prio += 0.08
    if hero_ctx:
        prio += hero_ctx.level_aggression
    prio -= min(0.2, max(best_buy_delta, 0) * 3)      # a strong buy outranks leveling
    return ScoredAction(act, _clamp(prio), reason)


def _score_roll(act, gold, best_buy_delta, target_tribe, snapshot=None):
    prio = 0.5 - best_buy_delta * _PRIO_SCALE          # good buys make rolling worse
    if gold < BUY_COST + ROLL_COST:
        prio -= 0.25                                   # rolling would starve the buy
    tribe = f" for {target_tribe} pieces" if target_tribe else ""
    if best_buy_delta < _BUY_MEANINGFUL:
        reason = f"shop has no strong buy — roll{tribe}"
    else:
        reason = "a buy beats rolling this turn"
    # Board-fill / anti-stuck: when adj>0, Roll must not be NEXT.
    # Apply FULL demotion (no *0.5 soft-cap), min 0.6, and cap prio <= 0.05
    # so typical buys (~0.5+) always outrank roll.
    if snapshot is not None:
        try:
            from .jeef_priors import board_fill_roll_adjust, anti_stuck_roll_adjust
            demote = 0.0
            for adj_fn in (board_fill_roll_adjust, anti_stuck_roll_adjust):
                adj, adj_reason = adj_fn(snapshot)
                if adj:
                    demote = max(demote, max(0.6, float(adj)))
                    if adj_reason:
                        reason = adj_reason
            if demote:
                prio -= demote
                prio = min(prio, 0.05)
        except Exception:
            pass
    return ScoredAction(act, _clamp(prio, hi=0.8), reason)


_FREEZE_STRONG = 3.5       # high synergy alone is enough
_FREEZE_GEM = 5.0          # a single card so good it's worth freezing alone (rare)
# Early-curve / named keeps worth freezing even with modest synergy scores.
_FREEZE_KEEP_NAMES = frozenset({
    "naga battlemage", "floating watcher", "brann bronzebeard",
    "lightfang enforcer", "baron rivendare", "kangor's apprentice",
    "kalecgos", "razorgore", "n'zoth", "y'shaarj", "titus rivendare",
})


def _freeze_keepable(m, ck, board_cks, target_tribe, tavern_tier, emb):
    """Is this shop minion *clearly* worth freezing (not random T2 chaff)?

    Returns (keep_score, clearly_keepable, label). clearly_keepable is True when
    any strong signal fires: high synergy, on-tribe, high tier relative, named
    keep, or strong meta pick.
    """
    name = (ck.name if ck is not None else _name(m)) or "?"
    syn = 0.0
    if ck is not None:
        syn = score_card(ck, board_cks, target_tribe=target_tribe, embeddings=emb).score
    flags = []
    # On-tribe with a committed board (not a lone random tribe hit).
    tribes = [t.lower() for t in (getattr(ck, "tribes", None) or [])] if ck else []
    on_tribe = bool(target_tribe and target_tribe.lower() in tribes and len(board_cks) >= 2)
    if on_tribe:
        flags.append("on-tribe")
    # High tier relative to current tavern (upgrade, not same-tier filler).
    card_tier = getattr(ck, "tier", None) if ck is not None else None
    if card_tier is None and isinstance(m, dict):
        card_tier = m.get("tier") or (m.get("tags") or {}).get("TECH_LEVEL")
        try:
            card_tier = int(card_tier) if card_tier is not None else None
        except (TypeError, ValueError):
            card_tier = None
    high_tier = (card_tier is not None and tavern_tier is not None
                 and int(card_tier) > int(tavern_tier))
    if high_tier:
        flags.append(f"T{card_tier}")
    named = name.lower() in _FREEZE_KEEP_NAMES
    if named:
        flags.append("named keep")
    meta_strong = False
    try:
        from .card_quality import placement, _STRONG
        cid = None
        if isinstance(m, dict):
            cid = m.get("card_id")
        if ck is not None:
            cid = cid or ck.card_id
        ap = placement(cid, name)
        if ap is not None and ap <= _STRONG:
            meta_strong = True
            flags.append(f"meta {ap:.1f}")
    except Exception:
        pass
    strong_syn = syn >= _FREEZE_STRONG
    if strong_syn:
        flags.append(f"syn {syn:.1f}")
    clearly = strong_syn or on_tribe or high_tier or named or meta_strong
    # Ranking score for gem / multi-strong checks (synergy dominates when present).
    keep = syn
    if on_tribe:
        keep += 1.5
    if high_tier:
        keep += 1.5
    if named:
        keep += 2.0
    if meta_strong:
        keep += 1.5
    label = name if not flags else f"{name} ({', '.join(flags)})"
    return keep, clearly, label


def _score_freeze(act, snapshot, gold, idx, board_cks, target_tribe, emb):
    """Freeze when you've spent down and the shop is *clearly* worth keeping.

    Bar is intentionally high: at 0 gold, freeze only for a strong early curve /
    on-tribe / high-tier-relative / named-keep / meta-strong unit — not random
    T2 chaff. Prefer End Turn / Sell when the shop is mediocre.
    """
    shop = list(_get(snapshot, "shop", []) or [])
    if gold >= BUY_COST:                          # you can buy — don't freeze
        return ScoredAction(act, 0.05, "you can act this turn — no need to freeze")
    tavern_tier = _get(snapshot, "tavern_tier") or 1
    # Without an explicit comp target, infer it from the board's dominant tribe so
    # on-tribe shop cards score as the upgrades they are (else freeze never fires).
    if target_tribe is None and board_cks:
        tribes = {}
        for ck in board_cks:
            for t in (getattr(ck, "tribes", None) or []):
                tribes[t.lower()] = tribes.get(t.lower(), 0) + 1
        if tribes:
            target_tribe = max(tribes, key=tribes.get)
    strong = []
    best_name, best_score, best_clear = None, 0.0, False
    for m in shop:
        ck = idx.get(_name(m))
        v, clearly, label = _freeze_keepable(
            m, ck, board_cks, target_tribe, tavern_tier, emb)
        if v > best_score:
            best_name, best_score, best_clear = label, v, clearly
        elif clearly and not best_clear:
            best_name, best_score, best_clear = label, v, clearly
        if clearly:
            strong.append(label.split(" (")[0])
    # An "insane" shop you can't afford: 2+ clearly-keepable cards.
    if len(strong) >= 2:
        return ScoredAction(act, 0.6,
                            f"freeze — insane shop: {', '.join(strong[:3])} (can't afford yet)")
    if best_score >= _FREEZE_GEM:
        return ScoredAction(act, 0.55, f"freeze — {best_name} is a perfect fit you can't afford yet")
    # Out of gold with a clearly keepable card: freeze for next turn.
    if gold <= 0 and best_clear:
        return ScoredAction(act, 0.5,
                            f"freeze — out of gold; keep {best_name} for next turn")
    return ScoredAction(act, 0.05, "freeze only when out of gold with a card worth keeping")


def plan_turn(snapshot, kb=None, hero_ctx: Optional[HeroContext] = None,
              scorer=None, max_steps: int = 8, min_delta: float = 0.005) -> List[str]:
    """Greedy whole-turn plan: repeatedly apply the best board-improving action,
    simulating the result, until nothing improves — then the trailing moves.

    This is "play the turn by the recommender": each step is the locally best
    move on the state the previous step produced. Greedy (one-ply), not globally
    optimal turn search, but it sequences a full turn of buys/sells."""
    scorer = scorer or get_scorer()
    state = _as_state(snapshot)
    steps: List[str] = []
    for _ in range(max_steps):
        plan = advise_actions(state, kb=kb, hero_ctx=hero_ctx, scorer=scorer)
        move = next((a for a in plan.ranked
                     if a.action.kind in (BUY, SELL) and (a.delta or 0) > min_delta), None)
        if move is None:
            break
        steps.append(f"{move.action.describe()} ({move.delta:+.0%})")
        _apply(state, move.action)
    if state.get("gold", 0) >= 4 and state.get("shop"):
        steps.append("Roll — surplus gold and no improving buys left")
    if len(state.get("board", [])) >= 2:
        steps.append("Reposition for combat (see positioning)")
    steps.append("End turn")
    return steps


def _as_state(snapshot) -> dict:
    """A mutable dict copy with just the fields the planner simulates."""
    return {
        "board": [dict(m) if isinstance(m, dict) else _min_to_dict(m)
                  for m in (_get(snapshot, "board", []) or [])],
        "shop": [dict(m) if isinstance(m, dict) else _min_to_dict(m)
                 for m in (_get(snapshot, "shop", []) or [])],
        "gold": _get(snapshot, "gold") or 0,
        "tavern_tier": _get(snapshot, "tavern_tier") or 1,
        "hero_health": _get(snapshot, "hero_health"),
        "turn": _get(snapshot, "turn"),
    }


def _min_to_dict(m):
    return {"name": getattr(m, "name", None), "card_id": getattr(m, "card_id", None),
            "attack": getattr(m, "attack", None), "health": getattr(m, "health", None)}


def _apply(state, action):
    """Mutate the simulated state by an action (buy/sell)."""
    board, shop = state["board"], state["shop"]
    if action.kind == BUY:
        if len(board) >= MAX_BOARD:                     # sell-weakest-for-room
            board.remove(min(board, key=_val))
        for i, m in enumerate(shop):                    # bought minion leaves shop
            if _name(m) == action.target:
                board.append(shop.pop(i))
                break
        else:
            board.append(action.detail.get("minion") or {"name": action.target})
        state["gold"] -= BUY_COST
    elif action.kind == SELL:
        for i, m in enumerate(board):
            if _name(m) == action.target:
                board.pop(i)
                break
        state["gold"] += SELL_VALUE


def _score_reposition(act, board, enemy_boards):
    if not enemy_boards or len(board) < 2:
        return ScoredAction(act, 0.2,
                            "order matters for attack sequence / taunts (no opponent to simulate)")
    from .position import optimize_vs_field
    ranked = optimize_vs_field(board, enemy_boards, top=len(board))
    if not ranked:
        return ScoredAction(act, 0.2, "couldn't simulate positioning")
    best = ranked[0]
    current = next((r for r in ranked if r.is_current), None)
    gain = best.win_pct - (current.win_pct if current else best.win_pct)
    if best.is_current or gain < 0.03:
        return ScoredAction(act, 0.2, f"current order is ~best ({best.win_pct:.0%} win)",
                            delta=0.0)
    # Name the minions in their recommended slots so the advice is actionable
    # ("put Deflect-o-Bot first"), not just index numbers.
    names = [_name(board[i]) for i in best.ordering]
    order_desc = " → ".join(n for n in names if n) or \
        " ".join(str(i + 1) for i in best.ordering)
    return ScoredAction(act, _clamp(0.4 + gain * 2),
                        f"reorder to: {order_desc} (+{gain:.0%} win, {best.win_pct:.0%})",
                        delta=gain)
