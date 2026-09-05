"""Phase 3P — observational 2S paint-equation / allocation-input attribution.

Reuses the 3O T5/T6 class-(3) walk on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For every winner starting body records the exact 2S paint inputs at
combat start (abstract pool, recruit weight, board denominator, board
size/tier composition, largest-remainder contribution, last
sell/play/triple membership event, pre/post-reallocation synth) and
decomposes treatment−control body synth into pool magnitude, weight
composition, reallocation timing/membership, integer rounding, and
residual.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    A_BUY0,
    A_PLAY0,
    A_SELL0,
    BGEnv,
    N_BUY,
    N_PLAY,
    N_SELL,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    minion_recruit_stats,
    minion_synthetic_delta,
    recruit_value_stats_enabled,
)
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.elimination_chain_diagnostic import compare_chain
from ml.elimination_timing_diagnostic import compare_elimination
from ml.hp_divergence_diagnostic import compare_first_divergence
from ml.matched_state_damage_diagnostic import (
    compare_matched_state_damage,
    iter_class3_events,
)
from ml.pairing_who_wins_diagnostic import _index_seat_fights
from ml.phase_3p_prereg import (
    ALLOCATION_COMPONENTS,
    NESTED_ALLOCATION_IDENTITY,
    PAINT_EQUATION_IDENTITY,
    PAINT_RECONCILE_IDENTITY,
    PHASE_3N_CLASS3,
    PHASE_3N_CLASS3_T5,
    PHASE_3N_CLASS3_T6,
    PHASE_3N_WITHIN_TIER_B,
    PHASE_3O_PRIMARY_N,
    PHASE_3O_SHARE_START_STATS,
    PHASE_3O_SHARE_SYNTH,
    PHASE_3O_T1_SYNTH_CONTROL,
    PHASE_3O_T1_SYNTH_TREATMENT,
    PHASE_3O_T3_SYNTH_CONTROL,
    PHASE_3O_T3_SYNTH_TREATMENT,
    PHASE_3O_T5T6_B,
    PRIMARY_TURNS,
    diagnose_phase_3p,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.survivor_composition_diagnostic import TIERS, clamp_tier, tier_histogram
from ml.survivor_mechanic_diagnostic import (
    SurvivorMechanicTracer,
    _fight_for_event,
    _primary_turn,
    _stamp_start_minions,
    collect_class3_minions,
    compare_survivor_mechanics,
)
from ml.synthetic_allocation_diagnostic import (
    _safe_div,
    largest_remainder_shares,
)

METHODOLOGY_VERSION = "3p_v1"
_N_EXAMPLES = 8
_SYNTH_REPRO_TOL = 0.15
_PART_ABS_TOL = 1e-9


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _safe_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def paint_weight(recruit_raw) -> int:
    """2S paint weight: recruit raw, floored at 1 so a 0-raw body still paints."""
    try:
        w = int(recruit_raw)
    except (TypeError, ValueError):
        w = 0
    return w if w > 0 else 1


def painted_pool_from_abstract(abstract_pool) -> int:
    """Exact 2S conservation: painted pool = round(abstract_pool)."""
    if abstract_pool is None:
        return 0
    try:
        return int(round(float(abstract_pool)))
    except (TypeError, ValueError):
        return 0


def exact_proportional_shares(
    weights: Sequence[int],
    pool_int: int,
) -> List[float]:
    """Real-valued paint before largest-remainder integer split."""
    if not weights:
        return []
    if int(pool_int) <= 0:
        return [0.0] * len(weights)
    w = [paint_weight(x) for x in weights]
    total_w = sum(w)
    if total_w <= 0:
        return [0.0] * len(weights)
    return [float(pool_int) * float(wi) / float(total_w) for wi in w]


def decode_recruit_action(action: int) -> str:
    if A_BUY0 <= action < A_BUY0 + N_BUY:
        return "buy"
    if A_PLAY0 <= action < A_PLAY0 + N_PLAY:
        return "play"
    if A_SELL0 <= action < A_SELL0 + N_SELL:
        return "sell"
    return "other"


def classify_membership_event(
    action: int,
    ids_before: Sequence[int],
    ids_after: Sequence[int],
) -> Optional[str]:
    """sell / play / triple when board object-identity changed; else None."""
    before = list(ids_before)
    after = list(ids_after)
    if before == after:
        return None
    kind = decode_recruit_action(action)
    removed = [i for i in before if i not in after]
    if kind == "sell":
        return "triple" if len(removed) >= 3 else "sell"
    if kind == "play":
        return "triple" if removed else "play"
    if kind == "buy":
        return "triple"
    if removed or after:
        return "play" if not removed else "sell"
    return None


def _slot_snap(m, slot: int, *, synth: Optional[int] = None) -> Dict:
    recruit = int(minion_recruit_stats(m))
    if synth is None:
        synth = int(minion_synthetic_delta(m))
    return {
        "slot": int(slot),
        "name": str(getattr(m, "name", "") or ""),
        "card_id": str(getattr(m, "card_id", "") or ""),
        "tier": int(getattr(m, "tier", 1) or 1),
        "golden": bool(getattr(m, "golden", False)),
        "recruit_raw": recruit,
        "synthetic_share": int(synth),
        "obj_id": id(m),
    }


def reconstruct_board_paint(
    rows: Sequence[Dict],
    abstract_pool,
) -> Dict:
    """Rebuild the 2S paint equation for one combat-start board."""
    stamped = list(rows or [])
    weights = [paint_weight(r.get("recruit_raw")) for r in stamped]
    denom = int(sum(weights))
    share_sum = int(sum(int(r.get("synthetic_share") or 0) for r in stamped))
    if abstract_pool is not None and abs(float(abstract_pool or 0.0)) > 1e-9:
        painted = painted_pool_from_abstract(abstract_pool)
        pool_source = "abstract_pool"
    else:
        painted = share_sum
        pool_source = "implicit_on_body"
    expected = largest_remainder_shares(
        [int(r.get("recruit_raw") or 0) for r in stamped], painted
    )
    exact = exact_proportional_shares(
        [int(r.get("recruit_raw") or 0) for r in stamped], painted
    )
    hist = tier_histogram(stamped)
    out_rows = []
    for i, r in enumerate(stamped):
        w = weights[i] if i < len(weights) else paint_weight(r.get("recruit_raw"))
        share = (float(w) / float(denom)) if denom else None
        lr = expected[i] if i < len(expected) else 0
        ex = exact[i] if i < len(exact) else 0.0
        actual = int(r.get("synthetic_share") or 0)
        painted_row = dict(r)
        painted_row.update({
            "paint_weight": w,
            "board_recruit_denom": denom,
            "weight_share": share,
            "abstract_pool": None if abstract_pool is None else float(abstract_pool),
            "painted_pool": painted,
            "pool_source": pool_source,
            "exact_proportional_share": ex,
            "largest_remainder_share": lr,
            "rounding_contribution": float(lr) - float(ex),
            "paint_mismatch": actual - int(lr),
            "board_size": int(r.get("board_size") or len(stamped)),
            "board_tier_hist": dict(hist),
        })
        out_rows.append(painted_row)
    return {
        "rows": out_rows,
        "painted_pool": painted,
        "board_recruit_denom": denom,
        "board_size": len(stamped),
        "board_tier_hist": hist,
        "synthetic_shares_sum": share_sum,
        "shares_sum_to_painted_pool": share_sum == painted,
        "expected_synthetic_shares": expected,
        "exact_proportional_shares": exact,
        "painted_matches_expected": (
            [int(r.get("synthetic_share") or 0) for r in stamped] == expected
            if stamped else True
        ),
        "pool_source": pool_source,
    }


def decompose_synth_pair(control: Dict, treatment: Dict) -> Dict:
    """Exact five-way split of treatment−control body synth.

    Identity (residual ~ 0):
      ΔS = (pool_t − pool_c)·share_t
         + pool_c·(share_t − share_c)
         + (LR_c − S_c) + (S_t − LR_t)
         + (LR_t − exact_t) − (LR_c − exact_c)
    """
    s_t = float(treatment.get("synthetic_share") or 0)
    s_c = float(control.get("synthetic_share") or 0)
    pool_t = float(treatment.get("painted_pool") or 0)
    pool_c = float(control.get("painted_pool") or 0)
    share_t = treatment.get("weight_share")
    share_c = control.get("weight_share")
    share_t = 0.0 if share_t is None else float(share_t)
    share_c = 0.0 if share_c is None else float(share_c)
    exact_t = float(treatment.get("exact_proportional_share") or 0.0)
    exact_c = float(control.get("exact_proportional_share") or 0.0)
    lr_t = float(treatment.get("largest_remainder_share") or 0.0)
    lr_c = float(control.get("largest_remainder_share") or 0.0)
    pool = (pool_t - pool_c) * share_t
    composition = pool_c * (share_t - share_c)
    timing = (lr_c - s_c) + (s_t - lr_t)
    rounding = (lr_t - exact_t) - (lr_c - exact_c)
    delta = s_t - s_c
    residual = delta - (pool + composition + timing + rounding)
    return {
        "delta_synth": delta,
        "pool_magnitude": pool,
        "weight_composition": composition,
        "timing_membership": timing,
        "integer_rounding": rounding,
        "residual": residual,
        "control_last_event": control.get("last_membership_event"),
        "treatment_last_event": treatment.get("last_membership_event"),
        "event_kind_mismatch": (
            control.get("last_membership_event")
            != treatment.get("last_membership_event")
        ),
    }


def _match_last_event_slot(row: Dict, last: Optional[Dict]) -> Dict:
    """Attach last-membership pre/post synth for this combat-start slot."""
    empty = {
        "last_membership_turn": None,
        "last_membership_event": None,
        "pre_reallocation_synth": None,
        "post_reallocation_synth": None,
    }
    if not last:
        return empty
    slot = _safe_int(row.get("board_slot"), 0)
    post_slots = list(last.get("post_slots") or [])
    pre_slots = list(last.get("pre_slots") or [])
    post = next((s for s in post_slots if int(s.get("slot", -1)) == slot), None)
    if post is None:
        key = (
            str(row.get("name") or ""),
            str(row.get("card_id") or ""),
            int(row.get("recruit_raw") or 0),
        )
        post = next(
            (
                s for s in post_slots
                if (
                    str(s.get("name") or ""),
                    str(s.get("card_id") or ""),
                    int(s.get("recruit_raw") or 0),
                ) == key
            ),
            None,
        )
    pre = None
    if post is not None:
        oid = post.get("obj_id")
        pre = next((s for s in pre_slots if s.get("obj_id") == oid), None)
        if pre is None:
            pre = next(
                (s for s in pre_slots if int(s.get("slot", -1)) == slot),
                None,
            )
    return {
        "last_membership_turn": last.get("turn"),
        "last_membership_event": last.get("event"),
        "pre_reallocation_synth": (
            None if pre is None else int(pre.get("synthetic_share") or 0)
        ),
        "post_reallocation_synth": (
            None if post is None else int(post.get("synthetic_share") or 0)
        ),
    }


def _stamp_paint_inputs(rec: Dict, fight: Dict, env: Optional[BGEnv] = None,
                        last_events: Optional[Dict] = None) -> None:
    if not rec.get("start_minions"):
        _stamp_start_minions(rec, fight, env)
    rows = list(rec.get("start_minions") or [])
    pool_field = rec.get("winner_abstract_pool_field")
    if pool_field is None and env is not None and fight.get("winner_seat") is not None:
        for p in env.players:
            if int(p.idx) == int(fight["winner_seat"]):
                pool_field = float(getattr(p, "abstract_pool", 0.0) or 0.0)
                break
    paint = reconstruct_board_paint(rows, pool_field)
    winner = fight.get("winner_seat")
    last = None
    if last_events is not None and winner is not None:
        last = last_events.get(int(winner))
    elif rec.get("last_membership_event_record"):
        last = rec.get("last_membership_event_record")
    stamped = []
    for r in paint["rows"]:
        extra = _match_last_event_slot(r, last)
        r.update(extra)
        stamped.append(r)
    rec.update({
        "start_minions": stamped,
        "winner_abstract_pool_field": pool_field,
        "winner_player_pool": paint["painted_pool"],
        "painted_pool": paint["painted_pool"],
        "board_recruit_denom": paint["board_recruit_denom"],
        "board_tier_hist": paint["board_tier_hist"],
        "synthetic_shares_sum": paint["synthetic_shares_sum"],
        "shares_sum_to_pool": paint["shares_sum_to_painted_pool"],
        "shares_sum_to_painted_pool": paint["shares_sum_to_painted_pool"],
        "expected_synthetic_shares": paint["expected_synthetic_shares"],
        "painted_matches_expected": paint["painted_matches_expected"],
        "last_membership_event_record": last,
        "paint_equation_ok": paint["shares_sum_to_painted_pool"],
    })


class AllocationInputTracer(SurvivorMechanicTracer):
    """3O mechanic rows plus last membership-change / paint-equation inputs."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self._live_player: Dict[int, object] = {}
        self._pre_ids: Dict[int, List[int]] = {}
        self._pre_slots: Dict[int, List[Dict]] = {}
        self._last_event: Dict[int, Dict] = {}

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._live_player.clear()
        self._pre_ids.clear()
        self._pre_slots.clear()
        self._last_event.clear()

    def begin_seat_recruit(self, seat: int, turn: int, player) -> None:
        super().begin_seat_recruit(seat, turn, player)
        self._live_player[int(seat)] = player

    def before_action(
        self, seat: int, turn: int, shop_generation: int, obs: Dict, mask,
    ) -> None:
        super().before_action(seat, turn, shop_generation, obs, mask)
        p = self._live_player.get(int(seat))
        if p is None:
            return
        board = list(getattr(p, "board", None) or [])
        self._pre_ids[int(seat)] = [id(m) for m in board]
        self._pre_slots[int(seat)] = [
            _slot_snap(m, i) for i, m in enumerate(board)
        ]

    def after_action(
        self, seat: int, turn: int, shop_generation: int, action: int,
        ended: bool, player=None,
    ) -> None:
        super().after_action(seat, turn, shop_generation, action, ended, player)
        p = player if player is not None else self._live_player.get(int(seat))
        if p is None:
            return
        ids_before = list(self._pre_ids.get(int(seat), []))
        board = list(getattr(p, "board", None) or [])
        ids_after = [id(m) for m in board]
        kind = classify_membership_event(action, ids_before, ids_after)
        if kind is not None:
            self._last_event[int(seat)] = {
                "turn": int(turn),
                "event": kind,
                "action": decode_recruit_action(action),
                "pre_slots": list(self._pre_slots.get(int(seat), [])),
                "post_slots": [_slot_snap(m, i) for i, m in enumerate(board)],
                "abstract_pool": float(getattr(p, "abstract_pool", 0.0) or 0.0),
                "board_size": len(board),
            }

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        _stamp_paint_inputs(rec, fight, env, self._last_event)


def run_allocation_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    from ml.phase_3p_prereg import assert_seed_range_allowed
    assert_seed_range_allowed(seed, lobbies)
    fights: List[Dict] = []
    lengths: List[float] = []
    turn_rows: List[Dict] = []
    replacement_events: List[Dict] = []
    board_snapshots: List[Dict] = []
    t1t3_events: List[Dict] = []
    last_t1t3_losses: List[Dict] = []
    pairing_decisions: List[Dict] = []
    hp_rows: List[Dict] = []
    eliminations: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = AllocationInputTracer(i, seed + i, arm)
                env = BGEnv(seed=seed + i)
                tracer.attach_to_env(env)
                recs = env.play_scripted(
                    [greedy_policy] * env.n_players, recruit_tracer=tracer
                )
                game_length = max((r["turn"] for r in recs), default=0)
                if tracer.game_length is None:
                    tracer.game_length = game_length
                lengths.append(float(game_length))
                fights.extend(tracer.fights)
                turn_rows.extend(tracer.turn_rows)
                replacement_events.extend(tracer.replacement_events)
                board_snapshots.extend(tracer.board_snapshots)
                t1t3_events.extend(tracer.t1t3_events)
                last_t1t3_losses.extend(tracer.last_t1t3_losses)
                pairing_decisions.extend(tracer.pairing_decisions)
                hp_rows.extend(tracer.hp_rows)
                eliminations.extend(tracer.eliminations)
                del env
                del tracer

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "board_level_abstract_scaling": bool(board_level_abstract_scaling),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "fights": fights,
        "game_lengths": lengths,
        "turn_rows": turn_rows,
        "replacement_events": replacement_events,
        "board_snapshots": board_snapshots,
        "t1t3_events": t1t3_events,
        "last_t1t3_losses": last_t1t3_losses,
        "pairing_decisions": pairing_decisions,
        "hp_rows": hp_rows,
        "eliminations": eliminations,
    }


def run_greedy_control_allocation(lobbies: int, seed: int) -> Dict:
    return run_allocation_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_allocation(lobbies: int, seed: int) -> Dict:
    return run_allocation_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _ensure_paint_rows(rows: Sequence[Dict], fights: Sequence[Dict]) -> List[Dict]:
    """Guarantee paint-equation fields even if a fight was stamped by 3O only."""
    by_id = {}
    for f in fights:
        for r in list(f.get("start_minions") or []):
            by_id[id(r)] = f
    out = []
    for r in rows:
        if r.get("painted_pool") is not None and r.get("weight_share") is not None:
            out.append(r)
            continue
        fight = by_id.get(id(r))
        pool = None
        board_rows = [r]
        if fight is not None:
            pool = fight.get("winner_abstract_pool_field")
            board_rows = list(fight.get("start_minions") or [r])
        paint = reconstruct_board_paint(board_rows, pool)
        match = next(
            (
                p for p in paint["rows"]
                if p.get("body_id") == r.get("body_id")
                and p.get("board_slot") == r.get("board_slot")
            ),
            paint["rows"][0] if paint["rows"] else r,
        )
        merged = dict(r)
        merged.update({k: match[k] for k in match if k not in (
            "survived", "died", "n_attacks",
        )})
        extra = _match_last_event_slot(
            merged, (fight or {}).get("last_membership_event_record")
        )
        merged.update(extra)
        out.append(merged)
    return out


def _pair_bodies(
    events: Sequence[Dict],
    control_fights: Dict,
    treatment_fights: Dict,
) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
    """Pair T5/T6 class-(3) starting bodies by (seed, seat, turn, slot)."""
    pairs: List[Dict] = []
    unpaired_c: List[Dict] = []
    unpaired_t: List[Dict] = []
    n_fights = 0
    for ev in events:
        if not _primary_turn(ev):
            continue
        c_fight = _fight_for_event(control_fights, ev)
        t_fight = _fight_for_event(treatment_fights, ev)
        if c_fight is None and t_fight is None:
            continue
        n_fights += 1
        if c_fight is not None:
            _stamp_paint_inputs(c_fight, c_fight)
        if t_fight is not None:
            _stamp_paint_inputs(t_fight, t_fight)
        c_rows = list((c_fight or {}).get("start_minions") or [])
        t_rows = list((t_fight or {}).get("start_minions") or [])
        c_by_slot = {_safe_int(r.get("board_slot"), i): r for i, r in enumerate(c_rows)}
        t_by_slot = {_safe_int(r.get("board_slot"), i): r for i, r in enumerate(t_rows)}
        used_t = set()
        for slot, c_row in c_by_slot.items():
            t_row = t_by_slot.get(slot)
            if t_row is None:
                unpaired_c.append(c_row)
                continue
            used_t.add(slot)
            parts = decompose_synth_pair(c_row, t_row)
            parts.update({
                "seed": ev.get("seed"),
                "causal_seat": ev.get("causal_seat"),
                "first_divergence_turn": ev.get("first_divergence_turn"),
                "tier": clamp_tier(c_row.get("tier") or t_row.get("tier")),
                "board_slot": slot,
                "control": c_row,
                "treatment": t_row,
            })
            pairs.append(parts)
        for slot, t_row in t_by_slot.items():
            if slot not in used_t:
                unpaired_t.append(t_row)
    return pairs, unpaired_c, unpaired_t, n_fights


def _by_tier_synth(rows: Sequence[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r.get("tier") or 1) == tier]
        out[str(tier)] = {
            "n_start": len(cell),
            "mean_synthetic_share": _mean(
                [float(r.get("synthetic_share") or 0) for r in cell]
            ),
            "mean_recruit_raw": _mean(
                [float(r.get("recruit_raw") or 0) for r in cell]
            ),
            "mean_painted_pool": _mean(
                [float(r.get("painted_pool") or 0) for r in cell]
            ),
            "mean_weight_share": _mean(
                [float(r["weight_share"]) for r in cell
                 if r.get("weight_share") is not None]
            ),
            "mean_board_recruit_denom": _mean(
                [float(r.get("board_recruit_denom") or 0) for r in cell]
            ),
            "mean_board_size": _mean(
                [float(r.get("board_size") or 0) for r in cell]
            ),
            "mean_rounding_contribution": _mean(
                [float(r.get("rounding_contribution") or 0) for r in cell]
            ),
            "mean_exact_proportional_share": _mean(
                [float(r.get("exact_proportional_share") or 0) for r in cell]
            ),
            "p_last_event_sell": _safe_div(
                float(sum(1 for r in cell if r.get("last_membership_event") == "sell")),
                float(len(cell)),
            ),
            "p_last_event_play": _safe_div(
                float(sum(1 for r in cell if r.get("last_membership_event") == "play")),
                float(len(cell)),
            ),
            "p_last_event_triple": _safe_div(
                float(sum(1 for r in cell if r.get("last_membership_event") == "triple")),
                float(len(cell)),
            ),
        }
    return out


def _sum_parts(pairs: Sequence[Dict]) -> Dict[str, float]:
    totals = {name: 0.0 for name in ALLOCATION_COMPONENTS}
    totals["delta_synth"] = 0.0
    for p in pairs:
        for name in ALLOCATION_COMPONENTS:
            totals[name] += float(p.get(name) or 0.0)
        totals["delta_synth"] += float(p.get("delta_synth") or 0.0)
    return totals


def tier_mass_primary(per_tier: Dict[str, Dict]) -> Dict:
    """Decision shares from within-tier |parts| so T1↓ / T3↑ cannot cancel.

    Signed pooled ΔS ≈ 0 because 2S moves the same pool off T1 onto T3.
    The 3O leftover is that cross-tier move, so primary shares weight each
    printed-tier component by n_pairs · |component|.
    """
    mass = {name: 0.0 for name in ALLOCATION_COMPONENTS}
    n_used = 0
    abs_delta = 0.0
    for cell in (per_tier or {}).values():
        n = int(cell.get("n_pairs") or 0)
        if n <= 0:
            continue
        n_used += n
        abs_delta += n * abs(float(cell.get("delta_synth") or 0.0))
        for name in ALLOCATION_COMPONENTS:
            mass[name] += n * abs(float(cell.get(name) or 0.0))
    total_mass = sum(mass.values())

    def _share(part: float) -> Optional[float]:
        if total_mass < 1e-12:
            return None
        return float(part) / total_mass

    return {
        "method": "within_tier_abs_mass_paint_identity",
        "n_pairs": n_used,
        "abs_delta_synth": (
            None if n_used <= 0 else float(abs_delta) / float(n_used)
        ),
        "component_abs_mass": mass,
        "total_abs_mass": total_mass,
        **{name: (
            None if n_used <= 0 else float(mass[name]) / float(n_used)
        ) for name in ALLOCATION_COMPONENTS},
        **{f"share_of_delta_{name}": _share(mass[name])
           for name in ALLOCATION_COMPONENTS},
        "allocation_components": list(ALLOCATION_COMPONENTS),
    }


def attribute_allocation_inputs(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    matched: Optional[Dict] = None,
    mechanics: Optional[Dict] = None,
) -> Dict:
    """3O locks plus T5/T6 paint-equation / allocation-input split."""
    if mechanics is None:
        mechanics = compare_survivor_mechanics(
            control_raw, treatment_raw, matched=matched,
        )
    events = list(iter_class3_events(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
    ))
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    rows_c, rows_t, n_c, n_t = collect_class3_minions(
        events, c_fights, t_fights, primary_only=True,
    )
    rows_c = _ensure_paint_rows(rows_c, [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(c_fights, ev),) if f
    ])
    rows_t = _ensure_paint_rows(rows_t, [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(t_fights, ev),) if f
    ])
    pairs, unpaired_c, unpaired_t, n_fights = _pair_bodies(
        events, c_fights, t_fights,
    )
    # Unpaired bodies are a membership-selection remainder: the slot set differs.
    for r in unpaired_t:
        s = float(r.get("synthetic_share") or 0)
        pairs.append({
            "delta_synth": s,
            "pool_magnitude": 0.0,
            "weight_composition": 0.0,
            "timing_membership": s,
            "integer_rounding": 0.0,
            "residual": 0.0,
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "treatment",
        })
    for r in unpaired_c:
        s = float(r.get("synthetic_share") or 0)
        pairs.append({
            "delta_synth": -s,
            "pool_magnitude": 0.0,
            "weight_composition": 0.0,
            "timing_membership": -s,
            "integer_rounding": 0.0,
            "residual": 0.0,
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "control",
        })

    totals = _sum_parts(pairs)
    n_pairs = max(1, len(pairs))
    means = {k: float(v) / float(n_pairs) for k, v in totals.items()}
    obs_delta = means["delta_synth"]

    def _share(part: float) -> Optional[float]:
        if abs(obs_delta) < 1e-12:
            return None
        return float(part) / obs_delta

    pooled_signed = {
        "method": "exact_paint_equation_paired_slot_identity",
        "n_pairs": len(pairs),
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "delta_synth": means["delta_synth"],
        **{name: means[name] for name in ALLOCATION_COMPONENTS},
        "explained_all_parts": sum(means[n] for n in ALLOCATION_COMPONENTS),
        "residual_vs_delta": (
            means["delta_synth"] - sum(means[n] for n in ALLOCATION_COMPONENTS)
        ),
        **{f"share_of_delta_{name}": _share(means[name])
           for name in ALLOCATION_COMPONENTS},
        "allocation_components": list(ALLOCATION_COMPONENTS),
    }
    per_tier = {}
    for tier in TIERS:
        cell = [p for p in pairs if int(p.get("tier") or 1) == tier]
        if not cell:
            per_tier[str(tier)] = {"n_pairs": 0}
            continue
        ct = _sum_parts(cell)
        n_cell = max(1, len(cell))
        cm = {k: float(v) / float(n_cell) for k, v in ct.items()}
        d = cm["delta_synth"]

        def _cshare(part: float, den: float = d) -> Optional[float]:
            if abs(den) < 1e-12:
                return None
            return float(part) / den

        per_tier[str(tier)] = {
            "n_pairs": len(cell),
            "delta_synth": cm["delta_synth"],
            **{name: cm[name] for name in ALLOCATION_COMPONENTS},
            **{f"share_of_delta_{name}": _cshare(cm[name])
               for name in ALLOCATION_COMPONENTS},
            "event_kind_mismatch_rate": _safe_div(
                float(sum(1 for p in cell if p.get("event_kind_mismatch"))),
                float(len(cell)),
            ),
        }

    primary = tier_mass_primary(per_tier)
    primary.update({
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "pooled_signed_delta_synth": pooled_signed["delta_synth"],
        "pooled_signed": pooled_signed,
        "residual_vs_delta": pooled_signed["residual_vs_delta"],
        "last_membership_event_play_rate_control": None,
        "last_membership_event_play_rate_treatment": None,
        "event_kind_mismatch_rate": _safe_div(
            float(sum(1 for p in pairs if p.get("event_kind_mismatch"))),
            float(len(pairs)),
        ),
    })

    c_by = _by_tier_synth(rows_c)
    t_by = _by_tier_synth(rows_t)
    play_c = [
        float((c_by[str(t)] or {}).get("p_last_event_play") or 0.0)
        * int((c_by[str(t)] or {}).get("n_start") or 0)
        for t in TIERS if (c_by.get(str(t)) or {}).get("n_start")
    ]
    play_t = [
        float((t_by[str(t)] or {}).get("p_last_event_play") or 0.0)
        * int((t_by[str(t)] or {}).get("n_start") or 0)
        for t in TIERS if (t_by.get(str(t)) or {}).get("n_start")
    ]
    n_c_ev = sum(int((c_by[str(t)] or {}).get("n_start") or 0) for t in TIERS)
    n_t_ev = sum(int((t_by[str(t)] or {}).get("n_start") or 0) for t in TIERS)
    primary["last_membership_event_play_rate_control"] = (
        None if n_c_ev <= 0 else float(sum(play_c)) / float(n_c_ev)
    )
    primary["last_membership_event_play_rate_treatment"] = (
        None if n_t_ev <= 0 else float(sum(play_t)) / float(n_t_ev)
    )
    t1_c = (c_by.get("1") or {}).get("mean_synthetic_share")
    t1_t = (t_by.get("1") or {}).get("mean_synthetic_share")
    t3_c = (c_by.get("3") or {}).get("mean_synthetic_share")
    t3_t = (t_by.get("3") or {}).get("mean_synthetic_share")

    def _close(got, published) -> bool:
        if got is None or published is None:
            return False
        return abs(float(got) - float(published)) <= _SYNTH_REPRO_TOL

    c_primary_fights = [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(c_fights, ev),) if f
    ]
    t_primary_fights = [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(t_fights, ev),) if f
    ]
    n_share_c = sum(1 for f in c_primary_fights if f.get("shares_sum_to_painted_pool") is False)
    n_share_t = sum(1 for f in t_primary_fights if f.get("shares_sum_to_painted_pool") is False)
    examples = []
    for p in pairs:
        if len(examples) >= _N_EXAMPLES:
            break
        if p.get("unpaired"):
            continue
        examples.append({
            "seed": p.get("seed"),
            "causal_seat": p.get("causal_seat"),
            "first_divergence_turn": p.get("first_divergence_turn"),
            "tier": p.get("tier"),
            "delta_synth": p.get("delta_synth"),
            "pool_magnitude": p.get("pool_magnitude"),
            "weight_composition": p.get("weight_composition"),
            "timing_membership": p.get("timing_membership"),
            "integer_rounding": p.get("integer_rounding"),
            "control_last_event": p.get("control_last_event"),
            "treatment_last_event": p.get("treatment_last_event"),
            "control_pool": (p.get("control") or {}).get("painted_pool"),
            "treatment_pool": (p.get("treatment") or {}).get("painted_pool"),
            "control_weight_share": (p.get("control") or {}).get("weight_share"),
            "treatment_weight_share": (p.get("treatment") or {}).get("weight_share"),
        })
    mech_attr = (mechanics or {}).get("attribution") or {}
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_same_outcome_damage": len(events),
        "n_primary_class3": n_c,
        "n_primary_class3_treatment": n_t,
        "published_same_outcome_damage": PHASE_3N_CLASS3,
        "published_class3_t5": PHASE_3N_CLASS3_T5,
        "published_class3_t6": PHASE_3N_CLASS3_T6,
        "published_primary_n": PHASE_3O_PRIMARY_N,
        "same_outcome_damage_reproduced": len(events) == PHASE_3N_CLASS3,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_within_tier_B_hat": mech_attr.get("phase_3n_within_tier_B_hat"),
        "phase_3n_B_reproduced": mech_attr.get("phase_3n_B_reproduced"),
        "phase_3o_t5t6_B": PHASE_3O_T5T6_B,
        "phase_3o_t5t6_B_hat": (mech_attr.get("primary") or {}).get("within_tier_B"),
        "phase_3o_B_reproduced": abs(float(
            (mech_attr.get("primary") or {}).get("within_tier_B") or 0.0
        ) - PHASE_3O_T5T6_B) <= 1e-9 if mech_attr.get("primary") else False,
        "phase_3o_share_start_stats": PHASE_3O_SHARE_START_STATS,
        "phase_3o_share_synth": PHASE_3O_SHARE_SYNTH,
        "t1_synth_control": t1_c,
        "t1_synth_treatment": t1_t,
        "t3_synth_control": t3_c,
        "t3_synth_treatment": t3_t,
        "t1_synth_reproduced": (
            _close(t1_c, PHASE_3O_T1_SYNTH_CONTROL)
            and _close(t1_t, PHASE_3O_T1_SYNTH_TREATMENT)
        ),
        "t3_synth_reproduced": (
            _close(t3_c, PHASE_3O_T3_SYNTH_CONTROL)
            and _close(t3_t, PHASE_3O_T3_SYNTH_TREATMENT)
        ),
        "primary": primary,
        "per_tier": per_tier,
        "control_by_tier": c_by,
        "treatment_by_tier": t_by,
        "paint_reconciliation": {
            "control_n_share_mismatch": n_share_c,
            "treatment_n_share_mismatch": n_share_t,
            "paint_ok": n_share_c == 0 and n_share_t == 0,
            "nested_ok": abs(float(primary.get("residual_vs_delta") or 0.0)) <= 1e-6,
            "identity": PAINT_RECONCILE_IDENTITY,
            "paint_equation_identity": PAINT_EQUATION_IDENTITY,
            "nested_identity": NESTED_ALLOCATION_IDENTITY,
        },
        "examples": examples,
        "mechanics_3o": {
            "n_primary_class3": mech_attr.get("n_primary_class3"),
            "phase_3n_B_reproduced": mech_attr.get("phase_3n_B_reproduced"),
            "primary_share_start_stats": (
                (mech_attr.get("primary") or {}).get("share_of_B_start_stats")
            ),
            "primary_share_synth": (
                (mech_attr.get("primary") or {}).get("share_of_B_synthetic_allocation")
            ),
        },
    }


def compare_allocation_inputs(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    selection: Optional[Dict] = None,
    pairing: Optional[Dict] = None,
    timing: Optional[Dict] = None,
    chain: Optional[Dict] = None,
    first: Optional[Dict] = None,
    matched: Optional[Dict] = None,
    mechanics: Optional[Dict] = None,
) -> Dict:
    """3O lock + T5/T6 allocation-input split."""
    if mechanics is None:
        mechanics = compare_survivor_mechanics(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain, first=first, matched=matched,
        )
    if matched is None:
        matched = {
            "attribution": mechanics.get("matched_state"),
            "reconciliation": mechanics.get("reconciliation"),
        }
    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch,
        still_fields_t1t3=False,
    )
    attr = attribute_allocation_inputs(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch,
        matched=matched, mechanics=mechanics,
    )
    rec = dict(mechanics.get("reconciliation") or {})
    rec.update({
        "paint_equation_identity": PAINT_EQUATION_IDENTITY,
        "paint_reconcile_identity": PAINT_RECONCILE_IDENTITY,
        "nested_allocation_identity": NESTED_ALLOCATION_IDENTITY,
        "phase_3n_B_reproduced": attr.get("phase_3n_B_reproduced"),
        "phase_3o_B_reproduced": attr.get("phase_3o_B_reproduced"),
        "t1_synth_reproduced": attr.get("t1_synth_reproduced"),
        "t3_synth_reproduced": attr.get("t3_synth_reproduced"),
        "paint_ok": (attr.get("paint_reconciliation") or {}).get("paint_ok"),
        "primary_nested_ok": (attr.get("paint_reconciliation") or {}).get("nested_ok"),
    })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": attr,
        "primary": attr.get("primary"),
        "per_tier": attr.get("per_tier"),
        "source": mechanics.get("source"),
        "matched_state": mechanics.get("matched_state"),
        "mechanics_3o": mechanics.get("primary"),
        "first_divergence_3m": mechanics.get("first_divergence_3m"),
        "reconciliation": rec,
        "decomposition_3g": mechanics.get("decomposition_3g"),
        "published_3o_locks": {
            "class3": PHASE_3N_CLASS3,
            "within_tier_B": PHASE_3N_WITHIN_TIER_B,
            "t5t6_B": PHASE_3O_T5T6_B,
            "share_start_stats": PHASE_3O_SHARE_START_STATS,
            "share_synth": PHASE_3O_SHARE_SYNTH,
            "primary_n": PHASE_3O_PRIMARY_N,
        },
        "published_3n_locks": mechanics.get("published_3n_locks"),
    }


__all__ = [
    "AllocationInputTracer",
    "attribute_allocation_inputs",
    "classify_membership_event",
    "compare_allocation_inputs",
    "decompose_synth_pair",
    "diagnose_phase_3p",
    "exact_proportional_shares",
    "paint_weight",
    "painted_pool_from_abstract",
    "reconstruct_board_paint",
    "run_greedy_2s_treatment_allocation",
    "run_greedy_control_allocation",
    "tier_mass_primary",
]
