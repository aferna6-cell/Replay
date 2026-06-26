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
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import multiturn
from .actions import (
    BUY, BUY_SPELL, SELL, LEVEL, ROLL, REPOSITION, FREEZE, HERO_POWER, BUY_COST,
    SELL_VALUE, MAX_BOARD, tavern_up_cost,
)
from .advisor import advise_actions, _as_state, Action
from .board_value import get_scorer, _val, _name
from .pace import load_pace, _at as _curve_at

_LOW_HP = 15.0
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


_K_SYN = 0.18              # effect-synergy points -> placement units
_MAX_SYN = 0.6


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

    equity = scorer.equity(board, hero_id)              # 0..1, higher = better
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
                 horizon: int = 4) -> Tuple[List[WholeGameRec], float]:
    """Rank every legal action by the expected final placement of its result.

    Reuses `advise_actions` for the action set + synergy reasons, then re-scores
    each by whole-game value so leveling/rolling/buying are directly comparable."""
    scorer = scorer or get_scorer()
    pace = pace if pace is not None else load_pace()
    enemy_boards = _get(snapshot, "opponents_seen", None) or None
    plan = advise_actions(snapshot, kb=kb, hero_ctx=hero_ctx, scorer=scorer,
                          enemy_boards=enemy_boards)
    base = expected_placement(snapshot, scorer, pace, horizon)
    state = _as_state(snapshot)

    enemy0 = enemy_boards[0] if enemy_boards else None
    recs: List[WholeGameRec] = []
    for sa in plan.ranked:
        a = sa.action
        reason = sa.reason
        if a.kind in (BUY, SELL, LEVEL, ROLL):
            v = expected_placement(_apply(state, a), scorer, pace, horizon)
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
                # CRITICAL: a buy is only worth comp/synergy credit if it actually
                # strengthens the board. A weak comp-piece (e.g. a tier-1 minion on
                # a board of giants, or a buy that forces selling a giant) doesn't —
                # don't let the build-path bonus rescue a buy the eval net rejects.
                improves = v <= base + 0.01 or tech_reason
                if improves:
                    # Build-path: does this buy advance a reachable winning comp?
                    padj, preason = _build_path_adjust(a, snapshot)
                    if padj:
                        v = max(1.0, min(8.0, v + padj))
                        if preason and not tech_reason:
                            reason = preason
                    # Effect-text synergy: mechanical combo with the board.
                    sadj, sreason = _effect_synergy_adjust(a, snapshot, kb)
                    if sadj:
                        v = max(1.0, min(8.0, v + sadj))
                        if sreason and not tech_reason and not preason:
                            reason = sreason
                else:
                    reason = "doesn't improve your board — skip"
                # Filler / low-tier guard: the eval net overrates filling a slot, so
                # a minion far weaker than your board OR well below your tavern tier
                # (a tier-1 at tier 6) reads as an "upgrade". Penalize so you roll
                # for a real one instead of settling.
                fpen = max(_filler_penalty(a, _get(snapshot, "board", []) or []),
                           _low_tier_penalty(a, _get(snapshot, "tavern_tier"), kb))
                if fpen and not tech_reason:
                    v = min(8.0, v + fpen)
                    if fpen > 0.2:
                        reason = "too weak/low-tier for this stage — roll for a real upgrade"
                # Full board: a buy needs a sell first. Always name the minion to
                # sell (the weakest), even when a synergy/tech reason took the line.
                board_now = _get(snapshot, "board", []) or []
                if len(board_now) >= MAX_BOARD:
                    weakest = min(board_now, key=_val)
                    reason = f"sell {_name(weakest)} for room — {reason}"
            elif a.kind == LEVEL:                         # aggressive leveling pace
                adj, lreason = _aggressive_level_adj(snapshot)
                if adj:
                    v = max(1.0, v + adj)
                    reason = lreason
            elif a.kind == SELL:                         # you want a full board of 7
                v = min(8.0, v + _sell_penalty(state))
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
        elif a.kind == HERO_POWER:
            v = max(1.0, base - 0.15)        # using the hero power is generally +EV
        elif a.kind == FREEZE:
            # Rare by design: only good when the shop has a gem you can't afford
            # yet (the advisor flags that via priority). Otherwise bury it.
            v = (max(1.0, base - 0.2) if (sa.priority or 0) >= 0.5
                 else min(8.0, base + 0.4))
        elif a.kind == REPOSITION and sa.delta:
            # Reposition doesn't change board composition, so placement is flat —
            # but a better attack order raises combat win%. Convert that win-rate
            # gain (carried in sa.delta) into a small placement improvement so good
            # positioning can surface among the ranked moves.
            v = max(1.0, base - sa.delta * _K_POS)
        else:
            v = base                                     # freeze/end: neutral here
        recs.append(WholeGameRec(a, round(v, 2), reason, round(base - v, 2)))

    _favor_roll_over_mediocre_buys(recs, snapshot, base)
    recs.sort(key=lambda r: r.placement)
    return recs, base


_WEAK_BUY_GAIN = 0.30      # placement gain below this = an "okay" minion, not a gem


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
    if ratio >= 0.6:                    # competitive with your board — fine
        return 0.0
    return min(1.0, (0.6 - ratio) * 1.6)


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
        if gap >= 3:                    # e.g. a tier-1/2 minion when you're tier 4+
            return min(1.0, (gap - 2) * 0.45)
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


def _favor_roll_over_mediocre_buys(recs, snapshot, base) -> None:
    """Mid-game, don't settle: if the best available buy only marginally improves
    the board (an 'okay' minion), prefer rolling for a real upgrade — but only when
    rolling is genuinely the best use of the turn. Rewrites the roll rec in place.

    Deliberately rare (the user kept getting stuck on 'roll'):
      * tiers 3-5 only — at tier 6 don't manufacture a preference (triples /
        hand-plays / real upgrades surface instead);
      * gold >= BUY_COST + 2 — real headroom to roll AND still buy a result with a
        coin to spare (at 4 gold you can't fish, so don't suggest it);
      * board already developed (>= 5 minions) — with an open board you want a body,
        not a roll.
    Outside that window roll keeps its honest EV and only shows if it earns it."""
    gold = _get(snapshot, "gold") or 0
    tier = _get(snapshot, "tavern_tier") or 1
    board_n = len(_get(snapshot, "board", []) or [])
    best_buy_gain = max((r.gain for r in recs if r.action.kind == BUY), default=0.0)
    roll = next((r for r in recs if r.action.kind == ROLL), None)
    if roll is None:
        return
    if (3 <= tier <= 5 and gold >= BUY_COST + 2 and board_n >= 5
            and best_buy_gain < _WEAK_BUY_GAIN):
        roll.placement = round(max(1.0, base - (_WEAK_BUY_GAIN + 0.02)), 2)
        roll.gain = round(base - roll.placement, 2)
        roll.reason = "shop is only okay — roll for a stronger minion"
