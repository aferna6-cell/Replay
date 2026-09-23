"""Whole-game value: score a state by expected FINAL placement, not just the
current board.

Every move the recommender suggests should account for the rest of the game. So
instead of ranking actions by immediate board strength, we rank them by
`expected_placement(state_after_move)` — a blend of three signals:

  1. **Learned board value** — the eval net (trained on real final boards →
     placement) reads the board composition. This is already a whole-game signal:
     "how well does this kind of board *finish*."
  2. **Trajectory** — the multi-turn planner projects where this state is heading
     (tier / board-vs-curve / HP). Being ahead of the pace curve lowers expected
     placement; a line that dies raises it.
  3. **Survival** — low HP pushes expected placement up (closer to going out).

Because each action's resulting state is valued this way, leveling, rolling and
buying are finally compared on ONE axis — expected finish — with the future of
the game baked in. It's an honest blend (the board term is learned; trajectory +
HP are the heuristic economy model), not a perfect oracle.
"""

import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import multiturn
from .actions import (
    BUY, BUY_SPELL, SELL, LEVEL, ROLL, REPOSITION, FREEZE, HERO_POWER, ACTIVATE,
    DARK_GIFT, BUY_COST, SELL_VALUE, MAX_BOARD, tavern_up_cost,
)
from .advisor import advise_actions, _as_state, Action
from .board_value import get_scorer, _val, _name
from .pace import load_pace, _at as _curve_at

_LOW_HP = 15.0
_SCORER_CAP = 0.25     # max placement swing the scorer alone puts on a buy/sell
_K_TRAJ = 1.5          # being ahead/behind the pace curve, in placement units
_K_HP = 1.5           # HP risk, in placement units
_DEATH_PEN = 2.0      # a projected death is a big placement hit
_K_POS = 3.0          # combat win% gain from repositioning -> placement units

_ECON = "unloaded"     # cached economy value model (or None if unavailable)


def _econ_model():
    """The learned economy trajectory value (ml/econ_value.pt), if trained."""
    global _ECON
    if _ECON == "unloaded":
        _ECON = None
        try:
            import os
            from ml.econ_value import EconValue
            p = os.path.join(os.path.dirname(__file__), "..", "ml", "econ_value.pt")
            if os.path.isfile(p):
                _ECON = EconValue.load(p)
        except Exception:
            _ECON = None
    return _ECON


_SELL_TEMPO_PEN = 0.6         # a naked sell loses a body — you want 7 minions
_SELL_FULL_BOARD = 7          # full Battlegrounds board


def _get(s, k, d=None):
    return s.get(k, d) if isinstance(s, dict) else getattr(s, k, d)


_K_SIM_TECH = 2.2          # combat win% delta -> placement units for sim-grounded tech


def _tech_adjust(action, opponent_board, player_board=None):
    """(placement_adjustment, reason|None) for a tech-card BUY, matchup-aware.
    Negative promotes the card; positive demotes it. (0, None) for non-tech.

    Preferred path: simulate the player's board with vs without the tech against
    the last opponent and value it by the real combat-win delta — this 'recognizes
    the situation' for ANY opponent board (the sim now models Tunnel Blaster's AOE
    deathrattle and Venomous). Falls back to the keyword heuristic with no opponent."""
    from .card_roles import is_tech, tech_assessment
    minion = action.detail.get("minion")
    cid = (minion.get("card_id") if isinstance(minion, dict)
           else getattr(minion, "card_id", None))
    if not is_tech(cid, action.target):
        return 0.0, None
    if opponent_board and player_board and minion is not None:
        simmed = _sim_tech(player_board, minion, opponent_board)
        if simmed is not None:
            return simmed
    return tech_assessment(cid, action.target, opponent_board) or (0.0, None)


def _sim_tech(board, candidate, opponent):
    """Placement adjustment from the simulated combat-win delta of adding `candidate`
    to `board` against `opponent`. None if the sim can't run."""
    try:
        from .sim import simulate, Combatant
        opp = [Combatant.from_minion(m) for m in opponent]
        if not opp:
            return None
        # Fixed seed: a stable, reproducible read (and non-flaky) — the win% delta
        # is an estimate, so we don't need fresh randomness each call.
        base = simulate([Combatant.from_minion(m) for m in board], opp,
                        runs=200, seed=1234).win_pct
        with_tech = simulate([Combatant.from_minion(m) for m in board]
                             + [Combatant.from_minion(candidate)], opp,
                             runs=200, seed=1234).win_pct
        delta = with_tech - base
        adj = -max(-_MAX_TECH, min(_MAX_TECH, delta * _K_SIM_TECH))
        if delta > 0.03:
            return adj, f"+{delta:.0%} combat win vs their board (sim)"
        return adj, "no combat swing vs their board (sim) — situational"
    except Exception:
        return None


_MAX_TECH = 0.8


def _build_path_adjust(action, snapshot):
    """(placement_adjustment, reason) for how a BUY advances a reachable winning
    archetype. Negative = advances the build; positive = scatters it mid-game."""
    from .build_path import path_value
    minion = action.detail.get("minion")
    tribe = None
    if isinstance(minion, dict):
        tribe = minion.get("tribe") or (minion.get("tags") or {}).get("tribe")
    try:
        return path_value(_get(snapshot, "board", []) or [], action.target,
                          _get(snapshot, "tavern_tier"), candidate_tribe=tribe)
    except Exception:
        return 0.0, None


_K_SYN = 0.30              # effect-synergy points -> placement units (weighted up:
_MAX_SYN = 1.1            # effects/combo should outweigh a raw stat line)


def _effect_synergy_adjust(action, snapshot, kb):
    """(placement_adjustment, reason) from how a BUY's effects mesh with the board
    — generalizes combos from card text (produces/wants), not co-occurrence data."""
    if kb is None:
        return 0.0, None
    try:
        from .effect_synergy import board_synergy
        from .cards import by_name
        idx = by_name(kb)
        cand = idx.get(action.target)
        if cand is None:
            return 0.0, None
        board = [idx.get(_name(m)) for m in (_get(snapshot, "board", []) or [])]
        board = [c for c in board if c]
        score, reasons = board_synergy(cand, board)
        if score <= 0:
            return 0.0, None
        return -min(_MAX_SYN, score * _K_SYN), ("; ".join(reasons) if reasons else None)
    except Exception:
        return 0.0, None


def _completes_triple(action, snapshot) -> bool:
    """True if buying this shop minion would give you a 3rd copy (a triple). Uses
    the game's own BACON_TRIPLE_CANDIDATE flag when present, else counts copies of
    the same minion across your board + hand."""
    minion = action.detail.get("minion") or {}
    tags = minion.get("tags") if isinstance(minion, dict) else getattr(minion, "tags", None)
    if tags and str(tags.get("BACON_TRIPLE_CANDIDATE", "")) == "1":
        return True
    name = action.target
    if not name:
        return False
    owned = list(_get(snapshot, "board", []) or []) + list(_get(snapshot, "hand", []) or [])
    return sum(1 for m in owned if _name(m) == name) >= 2


def _late_scaling_adjust(action, snapshot):
    """(placement_adjustment, reason) rewarding a late-game buy that meaningfully
    scales the board — a minion well above your current board average. Top boards
    snowball hard (the scaling pace curve ~4x per turn late), so treading water
    loses; this pushes the coach toward the biggest power upgrades. Negative=better.

    Only fires in the late game (turn >= 9 or tier >= 5) and only for a genuine
    upgrade, so it complements (not overrides) synergy/comp direction."""
    turn = _get(snapshot, "turn") or 0
    tier = _get(snapshot, "tavern_tier") or 1
    if turn < 9 and tier < 5:
        return 0.0, None
    minion = action.detail.get("minion")
    board = _get(snapshot, "board", []) or []
    if minion is None or not board:
        return 0.0, None
    avg = sum(_val(m) for m in board) / len(board)
    if avg <= 0:
        return 0.0, None
    ratio = _val(minion) / avg
    if ratio <= 1.15:                       # not a real power upgrade
        return 0.0, None
    # Smaller than the effect/synergy terms — stats are a tiebreaker, not the driver.
    return (-min(0.4, (ratio - 1.0) * 0.3),
            "scales your board up — a real power upgrade for the late game")


def _quality_buy_adjust(action):
    """(placement_adjustment, reason, is_strong) from real top-MMR card stats.
    Works for both minion buys and spell buys (different detail keys)."""
    card = action.detail.get("minion") or action.detail.get("spell") or {}
    cid = (card.get("card_id") if isinstance(card, dict)
           else getattr(card, "card_id", None))
    try:
        from .card_quality import buy_adjust
        return buy_adjust(cid, action.target)
    except Exception:
        return 0.0, None, False


def _lobby_tech_adjust(action, snapshot):
    """(placement_adjustment, reason) promoting a tech card when the whole lobby
    runs what it answers — e.g. AOE/Divine-Shield-pop vs a lobby full of Divine
    Shields. Generalizes the sim (which only sees the last board) to all opponents."""
    from .opponents import threats
    t = threats(_get(snapshot, "opponent_profiles", None) or [])
    if not t:
        return 0.0, None
    from .card_roles import is_tech
    minion = action.detail.get("minion") or {}
    cid = (minion.get("card_id") if isinstance(minion, dict)
           else getattr(minion, "card_id", None))
    if not is_tech(cid, action.target):
        return 0.0, None
    ds = t["keywords"].get("DIVINE_SHIELD", 0)
    if ds >= 3:
        return -min(0.6, ds * 0.12), f"the lobby runs {ds} Divine Shields — this tech answers them"
    return 0.0, None


def _anomaly_buy_adjust(action, snapshot):
    """(placement_adjustment, reason) from the active anomaly for this buy. Keyed on
    minion tags (e.g. the Timewarped supercharge flag), which are self-identifying,
    so it works even if the anomaly's name wasn't captured."""
    minion = action.detail.get("minion") or {}
    tags = minion.get("tags") if isinstance(minion, dict) else getattr(minion, "tags", None)
    try:
        from .anomaly import buy_adjust
        return buy_adjust(tags or {})
    except Exception:
        return 0.0, None


def _keep_value(minion, board, kb) -> float:
    """How much a minion is worth KEEPING = raw stats + synergy with the rest of
    the board (shared tribe + effect combos). Used so the coach never sells a
    synergistic comp piece just because its stats are low — selling/room decisions
    rank on this, not on bare stats."""
    v = _val(minion)
    if kb is None or minion is None:
        return v
    try:
        from .cards import by_name
        from .effect_synergy import board_synergy
        idx = by_name(kb)
        ck = idx.get(_name(minion))
        if ck is None:
            return v
        rest = [idx.get(_name(m)) for m in (board or []) if _name(m) != _name(minion)]
        rest = [c for c in rest if c]
        tribes = {}
        for c in rest:
            for t in (c.tribes or []):
                tribes[t.lower()] = tribes.get(t.lower(), 0) + 1
        bonus, ctr = 0.0, [t.lower() for t in (ck.tribes or [])]
        if "all" in ctr and tribes:
            bonus += 4.0                              # Amalgam-style: fits any tribe
        elif any(tribes.get(t, 0) >= 2 for t in ctr):
            bonus += 8.0                              # on-tribe core — do NOT sell for stats
        elif any(tribes.get(t, 0) >= 1 for t in ctr) and max(tribes.values(), default=0) >= 3:
            bonus += 5.0                              # on-direction piece of committed build
        score, _ = board_synergy(ck, rest)
        bonus += min(6.0, max(0.0, score) * 0.8)      # mechanical combo (produces/wants)
        return v + bonus
    except Exception:
        return v


def _sell_synergy_penalty(action, snapshot, kb) -> float:
    """Extra placement penalty for selling a minion that synergizes with your comp
    — the larger its keep-value over its bare stats, the worse selling it is."""
    minion = action.detail.get("minion")
    if minion is None or kb is None:
        return 0.0
    board = _get(snapshot, "board", []) or []
    synergy = _keep_value(minion, board, kb) - _val(minion)
    return min(3.0, max(0.0, synergy) * 0.4)


def _sell_penalty(state) -> float:
    """A standalone sell shrinks your board; you almost always want a full 7.
    Penalize it (scaled up when you have few minions) so the recommender won't
    suggest selling unless the board genuinely improves enough to overcome it, or
    it's a buy making room (handled inside the buy action's 'sell X for room')."""
    n = len(state.get("board", []) or [])
    short = max(0, _SELL_FULL_BOARD - n)            # how far below a full board
    return _SELL_TEMPO_PEN + short * 0.08


def expected_placement(snapshot, scorer=None, pace=None, horizon: int = 4) -> float:
    """Expected final placement (1=1st … 8=last; lower is better) for a state."""
    scorer = scorer or get_scorer()
    pace = pace if pace is not None else load_pace()
    board = list(_get(snapshot, "board", []) or [])
    hero_id = _get(snapshot, "hero", None) or "UNKNOWN"

    # Pass the whole state so a context-trained eval net values the board given the
    # economy / survival / lobby around it (ignored by the heuristic + board-only nets).
    state = snapshot if isinstance(snapshot, dict) else None
    equity = scorer.equity(board, hero_id, state=state)   # 0..1, higher = better
    board_placement = 8.0 - equity * 7.0                 # composition value (learned)

    turn = _get(snapshot, "turn", None) or 8
    tier = _get(snapshot, "tavern_tier", None) or 1
    hp = _get(snapshot, "hero_health", None)

    econ = _econ_model()
    if econ is not None:
        # Learned trajectory value (trained on self-play lobbies): blend the
        # board-composition read with the economy/tempo/HP outlook.
        from ml.econ_env import alive_at
        strength = sum(_val(m) for m in board)
        curve = _curve_at(pace.get("scaling", {}), turn) or max(1.0, strength)
        ratio = strength / curve if curve else 1.0
        econ_pl = econ.predict(turn, tier, strength, ratio,
                               hp if hp is not None else 30.0,
                               players_left=alive_at(turn))
        return max(1.0, min(8.0, 0.5 * board_placement + 0.5 * econ_pl))

    # Heuristic fallback (no econ model trained yet).
    placement = board_placement
    plan = multiturn.best_plan(snapshot, pace, horizon)
    if plan and plan.projection:
        term = plan.projection[-1]
        placement += -(term.ratio - 1.0) * _K_TRAJ       # ahead of curve -> better
        if plan.died:
            placement += _DEATH_PEN
    if hp is not None and hp < _LOW_HP:
        placement += (_LOW_HP - hp) / _LOW_HP * _K_HP
    return max(1.0, min(8.0, placement))


def _apply(state: dict, action: Action) -> dict:
    """Resulting state after an action (buy/sell/level/roll). Roll can't see the
    next shop, so it's modelled as 'spend 1 gold, board unchanged' — its value
    comes from the trajectory/HP terms, not a simulated draw."""
    s = copy.deepcopy(state)
    board, shop = s["board"], s["shop"]
    if action.kind == BUY:
        if len(board) >= MAX_BOARD:
            board.remove(min(board, key=_val))
        for i, m in enumerate(shop):
            if _name(m) == action.target:
                board.append(shop.pop(i))
                break
        else:
            board.append(action.detail.get("minion") or {"name": action.target})
        s["gold"] -= BUY_COST
    elif action.kind == SELL:
        for i, m in enumerate(board):
            if _name(m) == action.target:
                board.pop(i)
                break
        s["gold"] += SELL_VALUE
    elif action.kind == LEVEL:
        # Prefer the action's live (discounted) cost; fall back to the base.
        cost = action.cost or tavern_up_cost(s["tavern_tier"]) or 0
        s["tavern_tier"] = min(6, s["tavern_tier"] + 1)
        s["gold"] -= cost
    elif action.kind == ROLL:
        s["gold"] -= 1
    return s


@dataclass
class WholeGameRec:
    action: Action
    placement: float        # expected final placement after this move (lower better)
    reason: str
    gain: float             # base_placement - placement (positive = improves finish)

    def line(self) -> str:
        return (f"  finish {self.placement:.1f} ({self.gain:+.2f})  "
                f"{self.action.describe()} — {self.reason}")


def rank_actions(snapshot, kb=None, scorer=None, pace=None, hero_ctx=None,
                 horizon: int = 4, include_reposition: bool = True
                 ) -> Tuple[List[WholeGameRec], float]:
    """Rank every legal action by the expected final placement of its result.

    Reuses `advise_actions` for the action set + synergy reasons, then re-scores
    each by whole-game value so leveling/rolling/buying are directly comparable."""
    scorer = scorer or get_scorer()
    pace = pace if pace is not None else load_pace()
    enemy_boards = _get(snapshot, "opponents_seen", None) or None
    plan = advise_actions(snapshot, kb=kb, hero_ctx=hero_ctx, scorer=scorer,
                          enemy_boards=enemy_boards,
                          include_reposition=include_reposition)
    base = expected_placement(snapshot, scorer, pace, horizon)
    state = _as_state(snapshot)

    enemy0 = enemy_boards[0] if enemy_boards else None
    recs: List[WholeGameRec] = []
    for sa in plan.ranked:
        a = sa.action
        reason = sa.reason
        if a.kind in (BUY, SELL, LEVEL, ROLL):
            v = expected_placement(_apply(state, a), scorer, pace, horizon)
            if a.kind in (BUY, SELL):
                # Design rule: card effects/synergy/meta-quality DRIVE buys;
                # raw board strength is a tiebreaker. The heuristic scorer's
                # one-ply deltas are naturally small, but a learned scorer can
                # swing several placements on a stat body (the env-trained set
                # net especially — its world is stats+keywords). Squash the
                # scorer's per-action delta to tiebreaker scale so the
                # knowledge adjustments below stay decisive with ANY scorer.
                v = base + _SCORER_CAP * math.tanh((v - base) / _SCORER_CAP)
            if a.kind == BUY:                            # matchup-aware tech read
                # Triple! Buying a 3rd copy golds it and Discovers a higher-tier
                # minion — one of the strongest tempo plays, almost always beats
                # rolling. Big bonus + it takes the line.
                if _completes_triple(a, snapshot):
                    v = max(1.0, v - 1.2)
                    recs.append(WholeGameRec(
                        a, round(v, 2),
                        f"completes a TRIPLE — golds it + Discover a higher-tier minion",
                        round(base - v, 2)))
                    continue
                adj, tech_reason = _tech_adjust(a, enemy0,
                                                _get(snapshot, "board", []) or [])
                v = max(1.0, min(8.0, v + adj))
                if tech_reason:
                    reason = tech_reason
                # Lobby-wide tech read: if the whole lobby runs Divine Shields /
                # Poisonous, value the answer (AOE, Divine-Shield pop) even before
                # we fight that board — not just vs the last opponent.
                ladj, lreason = _lobby_tech_adjust(a, snapshot)
                if ladj:
                    v = max(1.0, v + ladj)
                    reason = lreason
                # Anomaly-aware: the active anomaly (e.g. Timewarped) can make a
                # specific shop minion a priority buy — factor it in, don't ignore it.
                aadj, areason = _anomaly_buy_adjust(a, snapshot)
                if aadj:
                    v = max(1.0, v + aadj)
                    reason = areason
                # Meta quality: real top-MMR placement for this card. Strong cards
                # get bought instead of rolled past; a strong card is also exempt
                # from the filler/off-comp penalties below (it's worth buying).
                qadj, qreason, q_strong = _quality_buy_adjust(a)
                if qadj:
                    v = max(1.0, min(8.0, v + qadj))
                    if qreason and not tech_reason:
                        reason = qreason
                # Effects + synergy + the comp you're building toward are weighted
                # to MATTER, not gated behind the stat-driven eval net — a card's
                # combo with your board and your target comp is the point, not just
                # its stat line. Build-path and effect-synergy apply always; a card
                # with real synergy is then exempt from the stat-based filler guard.
                padj, preason = _build_path_adjust(a, snapshot)
                if padj:
                    v = max(1.0, min(8.0, v + padj))
                    if preason and not tech_reason:
                        reason = preason
                sadj, sreason = _effect_synergy_adjust(a, snapshot, kb)
                if sadj:
                    v = max(1.0, min(8.0, v + sadj))
                    if sreason and not tech_reason and not preason:
                        reason = sreason
                # A genuine synergy/comp piece (strong effect combo or a build-path
                # core/enabler) — don't let the low-stat filler guard bury it.
                synergy_buy = (sadj <= -0.3) or (padj <= -0.3)
                # Late-game scaling stays a stat-improvement signal: only when the
                # buy actually raises board power.
                if v <= base + 0.01 or tech_reason:
                    scadj, screason = _late_scaling_adjust(a, snapshot)
                    if scadj:
                        v = max(1.0, min(8.0, v + scadj))
                        if screason and not tech_reason and not preason:
                            reason = screason
                # Filler / low-tier guard: the eval net overrates filling a slot, so
                # a minion far weaker than your board OR well below your tavern tier
                # (a tier-1 at tier 6) reads as an "upgrade". Penalize so you roll
                # for a real one instead of settling — unless it's a real synergy
                # piece or a known-strong card.
                fpen = max(_filler_penalty(a, _get(snapshot, "board", []) or []),
                           _low_tier_penalty(a, _get(snapshot, "tavern_tier"), kb))
                ocpen = _off_comp_penalty(a, snapshot, kb)
                dpen, dreason = _direction_buy_penalty(a, snapshot, kb)
                # Direction / tier-sanity is the PRIMARY buy gate (lobby tribes).
                pen = max(fpen, ocpen, dpen)
                if pen and not tech_reason and not (q_strong and dpen < 0.5) and not (synergy_buy and dpen < 0.5):
                    v = min(8.0, v + pen)
                    if dpen >= max(fpen, ocpen) and dpen > 0.2 and dreason:
                        reason = dreason
                    elif ocpen >= fpen and ocpen > 0.2:
                        reason = "off-comp — doesn't fit your build; roll for a piece that fits"
                    elif fpen > 0.2:
                        reason = "too weak/low-tier for this stage — roll for a real upgrade"
                # Soft board-fill prior (Aidan): when thin, prefer stabilize buys
                # over hard rolling — including direction-only mediocre bodies.
                try:
                    from .jeef_priors import board_fill_buy_adjust
                    fadj, freason = board_fill_buy_adjust(a, snapshot, kb)
                    if fadj:
                        v = max(1.0, min(8.0, v + fadj))
                        # Placement nudge only — never clobber tech/anomaly/combat why.
                        if freason and not tech_reason and reason == sa.reason:
                            reason = freason
                except Exception:
                    pass
                # Soft mid-game solid prior: take on-direction solid over endless roll.
                try:
                    from .jeef_priors import midgame_solid_buy_adjust
                    madj, mreason = midgame_solid_buy_adjust(a, snapshot, kb)
                    if madj:
                        v = max(1.0, min(8.0, v + madj))
                        if mreason and not tech_reason and reason == sa.reason:
                            reason = mreason
                except Exception:
                    pass
                # Play into hero power / hero plan when HP is part of the game.
                try:
                    from .jeef_priors import hero_power_buy_adjust
                    hadj, hreason = hero_power_buy_adjust(
                        a, snapshot, kb, hero_ctx=hero_ctx)
                    if hadj:
                        v = max(1.0, min(8.0, v + hadj))
                        if hreason and not tech_reason and reason == sa.reason:
                            reason = hreason
                except Exception:
                    pass
                # Gallywix (and similar): bias BUY for cycle churn.
                try:
                    from .hero_scripts import gallywix_buy_adjust
                    gadj, greason = gallywix_buy_adjust(a, snapshot, kb)
                    if gadj:
                        v = max(1.0, min(8.0, v + gadj))
                        if greason and not tech_reason and reason == sa.reason:
                            reason = greason
                except Exception:
                    pass

                try:
                    from .hsreplay_guides import key_or_enabler_boost
                    tgt = getattr(a, "target", None)
                    mname = None
                    detail = getattr(a, "detail", None) or {}
                    if isinstance(detail, dict):
                        mn = detail.get("minion") or detail.get("card") or {}
                        if isinstance(mn, dict):
                            mname = mn.get("name")
                        elif isinstance(mn, str):
                            mname = mn
                    if not mname and isinstance(tgt, str):
                        mname = tgt
                    kadj, kreason = key_or_enabler_boost(mname, snapshot, kb)
                    if kadj:
                        v = max(1.0, min(8.0, v + kadj))
                        if kreason and reason == sa.reason:
                            reason = kreason
                except Exception:
                    pass
                # Soft trinket play-into prior: boost buys that match equipped
                # trinket synergies (battlecry, deathrattle, tribe, …).
                try:
                    from .comp_signals import minion_trinket_buy_adjust, off_trinket_buy_penalty
                    tadj, treason = minion_trinket_buy_adjust(
                        a.detail.get("minion"), snapshot, kb)
                    if tadj:
                        v = max(1.0, min(8.0, v + tadj))
                        if treason and not tech_reason:
                            reason = treason
                    op, oreason = off_trinket_buy_penalty(
                        a.detail.get("minion"), snapshot, kb)
                    if op:
                        v = min(8.0, v + op)
                        if oreason and not tech_reason and not tadj:
                            reason = oreason
                except Exception:
                    pass
                # Full board: a buy needs a sell first. Always name the minion to
                # sell (the weakest), even when a synergy/tech reason took the line.
                board_now = _get(snapshot, "board", []) or []
                if len(board_now) >= MAX_BOARD and "sell " not in reason.lower():
                    # A reason that replaced the advisor's lost the 'sell for room'
                    # note — re-add it, choosing the least valuable to KEEP (stats +
                    # synergy) so a comp piece isn't dumped for a fatter vanilla.
                    weakest = min(board_now, key=lambda m: _keep_value(m, board_now, kb))
                    reason = f"sell {_name(weakest)} for room — {reason}"
            elif a.kind == LEVEL:                         # aggressive leveling pace
                adj, lreason = _aggressive_level_adj(snapshot)
                if adj:
                    v = max(1.0, v + adj)
                    reason = lreason
            elif a.kind == SELL:                         # you want a full board of 7
                # …and never sell a synergistic comp piece for its low stats.
                v = min(8.0, v + _sell_penalty(state)
                        + _sell_synergy_penalty(a, snapshot, kb))
                # Soft: demote selling pieces that enable equipped trinkets.
                try:
                    from .comp_signals import sell_trinket_penalty
                    v = min(8.0, v + sell_trinket_penalty(
                        a.detail.get("minion"), snapshot, kb))
                except Exception:
                    pass
                # Soft direction cut: once committed, prefer selling useless
                # off-direction chaff over holding it while rolling.
                try:
                    from .jeef_priors import direction_cut_sell_adjust
                    cadj, creason = direction_cut_sell_adjust(a, snapshot, kb)
                    if cadj:
                        v = max(1.0, min(8.0, v + cadj))
                        if creason:
                            reason = creason
                except Exception:
                    pass
            elif a.kind == ROLL:
                # Board-fill / anti-stuck: full-strength demotion (min +0.6), no
                # *0.5 soft-cap — sparse+fill or stable+solid must keep Roll off NEXT.
                try:
                    from .jeef_priors import board_fill_roll_adjust, anti_stuck_roll_adjust
                    demote = 0.0
                    for adj_fn in (board_fill_roll_adjust, anti_stuck_roll_adjust):
                        adj, adj_reason = adj_fn(snapshot, kb)
                        if adj:
                            demote = max(demote, max(0.6, float(adj)))
                            if adj_reason:
                                reason = adj_reason
                    if demote:
                        v = min(8.0, v + demote)
                except Exception:
                    pass
                try:
                    from .hero_scripts import gallywix_roll_adjust
                    gadj, greason = gallywix_roll_adjust(snapshot, kb)
                    if gadj:
                        v = min(8.0, v + gadj)
                        if greason:
                            reason = greason
                except Exception:
                    pass
        elif a.kind == BUY_SPELL:
            # Spells don't change the board composition the eval net reads, so we
            # value them off base via spell_roles' placement bonus + the reason.
            from .spell_roles import spell_value
            spell = a.detail.get("spell") or {}
            cid = (spell.get("card_id") if isinstance(spell, dict)
                   else getattr(spell, "card_id", None))
            bonus, sreason = spell_value(cid, a.target, a.cost,
                                         _get(snapshot, "gold") or 0)
            v = max(1.0, base + bonus)
            reason = sreason
            # Meta quality for spells too — a strong tavern spell is a priority buy.
            qadj, qreason, _ = _quality_buy_adjust(a)
            if qadj:
                v = max(1.0, v + qadj)
                if qreason:
                    reason = qreason
            # Comp lean: a spell-reward trinket means buy MORE tavern spells than
            # usual — fold that bias in so spell comps actually buy spells.
            try:
                from .comp_signals import buy_bias
                sb = buy_bias(snapshot).get("spell")
                if sb:
                    v = max(1.0, v + sb)
                    reason = (reason or "tavern spell") + " — your trinket rewards spells"
            except Exception:
                pass
            # Jeef VOD soft prior (Corrupted Coin etc.); unknown spells stay demoted.
            try:
                from .jeef_priors import spell_prior_adjust
                jadj, jreason = spell_prior_adjust(cid, a.target)
                if jadj:
                    v = max(1.0, v + jadj)
                    if jreason:
                        reason = jreason
            except Exception:
                pass
        elif a.kind == HERO_POWER:
            from .jeef_priors import hero_power_adjust
            adj, preason = hero_power_adjust(snapshot, a.cost)
            if adj is None:
                adj = -0.55
            v = max(1.0, base + adj)
            if preason:
                reason = preason
        elif a.kind == ACTIVATE:
            from .jeef_priors import activate_adjust
            adj, preason = activate_adjust(snapshot, a.cost)
            v = max(1.0, base + (adj if adj else -0.25))
            if preason:
                reason = preason
        elif a.kind == DARK_GIFT:
            from .jeef_priors import dark_gift_adjust
            adj, preason = dark_gift_adjust(snapshot, a.cost)
            v = max(1.0, base + (adj if adj else -0.25))
            if preason:
                reason = preason
        elif a.kind == FREEZE:
            # Rare by design: only good when the shop has a gem you can't afford
            # yet (the advisor flags that via priority >= 0.5). Mediocre shops
            # must rank *below* End/Sell so the overlay never leads with Freeze
            # on random T2 chaff (END is often hidden from the panel).
            v = (max(1.0, base - 0.2) if (sa.priority or 0) >= 0.5
                 else min(8.0, base + 1.6))
        elif a.kind == REPOSITION and sa.delta:
            # Reposition doesn't change board composition, so placement is flat —
            # but a better attack order raises combat win%. Convert that win-rate
            # gain (carried in sa.delta) into a small placement improvement so good
            # positioning can surface among the ranked moves.
            v = max(1.0, base - sa.delta * _K_POS)
        else:
            v = base                                     # freeze/end: neutral here
        recs.append(WholeGameRec(a, round(v, 2), reason, round(base - v, 2)))

    # NOTE: we do NOT boost roll when buys look "small" on the 1-8 placement axis
    # (that buried real buys). Board-fill / anti-stuck *demote* roll (full strength)
    # when sparse+fill or stable+solid. Roll still wins on a full/strong board,
    # late high-roll tempo, or when the shop is all trash.
    recs.sort(key=lambda r: r.placement)
    # Hard rule (Aidan): board <5 / sparse + gold>=3 + acceptable fill → NEXT != ROLL.
    try:
        from .jeef_priors import roll_must_not_be_next
        if recs and recs[0].action.kind == ROLL and roll_must_not_be_next(snapshot, kb):
            buys = [r for r in recs if r.action.kind == BUY]
            if buys:
                best_buy_p = min(r.placement for r in buys)
                fixed = []
                for r in recs:
                    if r.action.kind == ROLL:
                        new_p = max(r.placement, best_buy_p + 0.15)
                        fixed.append(WholeGameRec(
                            r.action, round(new_p, 2), r.reason,
                            round(base - new_p, 2)))
                    else:
                        fixed.append(r)
                fixed.sort(key=lambda r: r.placement)
                recs = fixed
    except Exception:
        pass
    return recs, base


def _direction_buy_penalty(action, snapshot, kb) -> tuple:
    """Primary buy policy: lobby-tribe direction + tier sanity.

    Returns (placement_penalty, reason). Heavy penalty means the buy must not
    rank as NEXT — off-direction junk and T2-at-tavern-5 chaff get buried.
    """
    if action.kind != BUY:
        return 0.0, None
    minion = action.detail.get("minion")
    if minion is None:
        return 0.0, None
    try:
        from .tribe_policy import direction_buy_penalty
        # Snapshot may be a dict already (live path).
        snap = snapshot if isinstance(snapshot, dict) else (
            snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot)
        # Attach target tribe from hero context if rank_actions stashed it.
        return direction_buy_penalty(minion, snap, kb=kb)
    except Exception:
        return 0.0, None


def _filler_penalty(action, board) -> float:
    """Placement penalty for buying a minion much weaker than your board — it's
    slot-filler, not an upgrade. 0 for a competitive buy. Scales with how far below
    the board's average the minion is, so on a board of giants a small minion is
    heavily demoted (roll for a real one)."""
    if not board:
        return 0.0
    minion = action.detail.get("minion")
    if minion is None:
        return 0.0
    vals = [_val(m) for m in board]
    avg = sum(vals) / len(vals) if vals else 0.0
    if avg <= 0:
        return 0.0
    ratio = _val(minion) / avg
    # Stats are only a minor tiebreaker now (effects/synergy/quality drive buys), so
    # this fires only for a body MUCH smaller than the board, and gently.
    if ratio >= 0.45:
        return 0.0
    return min(0.5, (0.45 - ratio) * 1.1)


def _off_comp_penalty(action, snapshot, kb) -> float:
    """Penalty for buying a minion that doesn't fit a committed comp — off your
    dominant tribe AND no mechanical synergy. The eval net rates it on raw stats
    ('adds board strength'), but on a committed board it just dilutes the comp and,
    on a full board, sells a real piece for a body. Bigger penalty when full."""
    if kb is None:
        return 0.0
    minion = action.detail.get("minion")
    if minion is None:
        return 0.0
    board = _get(snapshot, "board", []) or []
    if len(board) < 3:                       # still flexing — allow best body
        return 0.0
    try:
        from .cards import by_name
        from .effect_synergy import board_synergy
        idx = by_name(kb)
        tribes = {}
        for m in board:
            ck = idx.get(_name(m))
            for t in (getattr(ck, "tribes", None) or []):
                tribes[t.lower()] = tribes.get(t.lower(), 0) + 1
        if not tribes:
            return 0.0
        dom = max(tribes, key=tribes.get)
        if tribes[dom] < 2:                  # no real commitment → no penalty
            return 0.0
        cand = idx.get(action.target)
        if cand is None:
            return 0.0
        ctr = [t.lower() for t in (cand.tribes or [])]
        if dom in ctr or "all" in ctr:       # on-tribe — fine
            return 0.0
        board_cks = [idx.get(_name(m)) for m in board if idx.get(_name(m))]
        score, _ = board_synergy(cand, board_cks)
        if score > 0:                        # off-tribe but real mechanical combo — ok
            return 0.0
        pen = 0.7                            # off-comp filler — do not sprinkle stats
        if tribes[dom] >= 3:
            pen += 0.35
        if len(board) >= MAX_BOARD:
            pen += 0.4                       # …and it'd sell a comp piece for room
        return pen
    except Exception:
        return 0.0


def _low_tier_penalty(action, tavern_tier, kb) -> float:
    """Placement penalty for buying a minion well below your tavern tier — a
    tier-1 minion at tier 6 is almost never right. Catches buffed low-tier minions
    the stat-based filler guard misses (high stats, still a weak base card)."""
    if kb is None or not tavern_tier:
        return 0.0
    try:
        from .cards import by_name
        idx = by_name(kb)
        minion = action.detail.get("minion") or {}
        cid = minion.get("card_id") if isinstance(minion, dict) else None
        ck = (kb.get(cid) if cid else None) or idx.get(action.target)
        mt = getattr(ck, "tier", None) if ck else None
        if not mt:
            return 0.0
        gap = tavern_tier - mt
        # Tavern 5 must not lead with T2 chaff; gap>=2 at tavern>=4 is already bad
        # unless tribe_policy exempts a key synergy piece.
        if int(tavern_tier) >= 5 and gap >= 3:
            return 1.6
        if int(tavern_tier) >= 4 and gap >= 2:
            return min(1.4, 0.55 * gap)
        if gap >= 3:
            return min(1.2, (gap - 2) * 0.55)
    except Exception:
        pass
    return 0.0


# Aggressive tavern-tier target by turn — push the lobby's pace, not the
# conservative "level when you have spare gold" line. Reaching breakpoints early
# (tier 2 on turn 2, tier 3 on turn 4, tier 4 on turn 5-6) opens a stronger pool
# before opponents. Eased off only when you're low and need to survive.
_AGGRO_TIER = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 6,
               11: 6, 12: 6}
_AGGRO_LEVEL_HP = 12       # below this HP, don't push tempo-greedy leveling


def _aggressive_level_adj(snapshot):
    """(placement_adjustment, reason) promoting a tier-up that keeps you on the
    aggressive curve. Negative = better. Fires when you're below the turn's target
    tier and healthy enough to invest."""
    turn = _get(snapshot, "turn") or 0
    tier = _get(snapshot, "tavern_tier") or 1
    hp = _get(snapshot, "hero_health")
    if not turn or tier >= 6:
        return 0.0, None
    target = _AGGRO_TIER.get(turn, 6)
    if tier < target and (hp is None or hp >= _AGGRO_LEVEL_HP):
        to_tier = tier + 1
        return (-min(1.0, (target - tier) * 0.6),
                f"aggressive leveling — hit tier {to_tier} ahead of the lobby")
    return 0.0, None


