"""Phase 3R — observational post-play scale-sync input/timing attribution.

Reuses the 3Q T5/T6 class-(3) walk on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

Snapshots both arms immediately after the last open-slot play and at
every subsequent residual/ratio scale-sync through combat start, then
splits the published 3Q subsequent-scaling term:

  Δ_scale = (S_t_start − S_t_paint) − (S_c_start − S_c_sticky)
          = input_state + sync_timing + membership + rounding + residual

On common turns:
  input_state = share_c · (R_t − R_c)
  membership  = R_t · (share_t − share_c)
  rounding    = (actual_t − exact_t) − (actual_c − exact_c)
Extra / missing sync turns are timing/count.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.allocation_input_diagnostic import (
    _ensure_paint_rows,
    _pair_bodies,
)
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.matched_state_damage_diagnostic import iter_class3_events
from ml.pairing_who_wins_diagnostic import _index_seat_fights
from ml.phase_3r_prereg import (
    INPUT_FIELDS,
    NESTED_SCALE_SYNC_IDENTITY,
    PHASE_3N_CLASS3,
    PHASE_3N_WITHIN_TIER_B,
    PHASE_3O_PRIMARY_N,
    PHASE_3O_SHARE_START_STATS,
    PHASE_3O_SHARE_SYNTH,
    PHASE_3O_T5T6_B,
    PHASE_3P_PRIMARY_N_FIGHTS,
    PHASE_3P_PRIMARY_N_PAIRS,
    PHASE_3P_SHARE_POOL,
    PHASE_3P_SHARE_ROUNDING,
    PHASE_3P_SHARE_TIMING,
    PHASE_3P_SHARE_WEIGHT,
    PHASE_3Q_PRIMARY_N_FIGHTS,
    PHASE_3Q_PRIMARY_N_PAIRS,
    PHASE_3Q_SHARE_LIFECYCLE,
    PHASE_3Q_SHARE_RESIDUAL,
    PHASE_3Q_SHARE_SAME_STATE,
    PHASE_3Q_SHARE_SCALING,
    PHASE_3Q_T1_SYNTH_CONTROL,
    PHASE_3Q_T1_SYNTH_TREATMENT,
    PHASE_3Q_T3_SYNTH_CONTROL,
    PHASE_3Q_T3_SYNTH_TREATMENT,
    SCALE_FLOW_RECONCILE_IDENTITY,
    SCALE_SYNC_COMPONENTS,
    SAME_TURN_SYNC_IDENTITY,
    diagnose_phase_3r,
)
from ml.play_lifecycle_diagnostic import (
    PlayLifecycleTracer,
    _index_plays,
    _last_play,
    _match_slot,
    _minion_snap,
    _safe_int,
    compare_play_lifecycle,
    decompose_play_pair,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.survivor_composition_diagnostic import TIERS, clamp_tier
from ml.survivor_mechanic_diagnostic import (
    _fight_for_event,
    _primary_turn,
    collect_class3_minions,
)
from ml.synthetic_allocation_diagnostic import _safe_div

METHODOLOGY_VERSION = "3r_v1"
_N_EXAMPLES = 8
_SYNTH_REPRO_TOL = 0.15
_SHARE_REPRO_TOL = 1e-9
_FLOW_ABS_TOL = 1e-6


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _as_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def exact_scale_increment(combat_raw, board_combat, residual_add) -> float:
    """Unrounded residual share: R · (body combat / board combat)."""
    board = _as_float(board_combat)
    if board <= 0:
        return 0.0
    return _as_float(residual_add) * _as_float(combat_raw) / board


def rounded_scale_increment(
    attack, health, residual_add, board_combat,
) -> int:
    """Replay ``_end_of_turn_scaling_residual`` for one body (no mutation)."""
    current = _as_float(board_combat)
    add_budget = _as_float(residual_add)
    atk = int(attack or 0)
    hp = int(health or 0)
    if add_budget <= 0 or current <= 0:
        return 0
    total = atk + hp
    if total <= 0:
        return 0
    share = total / current
    add = add_budget * share
    new_atk = max(1, round(atk + add * atk / total))
    new_hp = max(1, round(hp + add * hp / total))
    return int(new_atk + new_hp) - total


def first_input_diverge_field(control_sync: Dict, treatment_sync: Dict) -> Optional[str]:
    """Earliest budget/board input that differs on a common-turn sync."""
    for name in INPUT_FIELDS:
        c = control_sync.get(name)
        t = treatment_sync.get(name)
        if c is None and t is None:
            continue
        try:
            if abs(_as_float(c) - _as_float(t)) > 1e-9:
                return name
        except (TypeError, ValueError):
            if c != t:
                return name
    return None


def _tier_hist(slots: Sequence[Dict]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for s in slots or []:
        key = str(int(s.get("tier") or 1))
        hist[key] = hist.get(key, 0) + 1
    return hist


def _sum_field(slots: Sequence[Dict], name: str) -> float:
    return float(sum(_as_float(s.get(name)) for s in slots or []))


def body_sync_increment(
    sync: Dict, slot: int, start_row: Optional[Dict] = None,
) -> Dict:
    """Actual / exact body increment at one scale-sync."""
    pre = _match_slot(sync.get("pre_slots") or [], slot, start_row)
    post = _match_slot(sync.get("post_slots") or [], slot, start_row)
    residual_add = _as_float(sync.get("residual_add"))
    board = _as_float(sync.get("board_combat_raw"))
    combat_pre = 0.0 if pre is None else _as_float(pre.get("combat_raw"))
    share = (combat_pre / board) if board > 0 and pre is not None else 0.0
    exact = residual_add * share
    if pre is None and post is None:
        actual = 0.0
    elif pre is None:
        actual = _as_float(post.get("synthetic_share"))
    elif post is None:
        actual = -_as_float(pre.get("synthetic_share"))
    else:
        actual = (
            _as_float(post.get("synthetic_share"))
            - _as_float(pre.get("synthetic_share"))
        )
    return {
        "present_pre": pre is not None,
        "present_post": post is not None,
        "actual": float(actual),
        "exact": float(exact),
        "share": float(share),
        "residual_add": float(residual_add),
        "combat_pre": float(combat_pre),
        "board_combat": float(board),
    }


def decompose_scale_pair(
    control_start: Dict,
    treatment_start: Dict,
    control_play: Optional[Dict],
    treatment_play: Optional[Dict],
    control_syncs: Sequence[Dict],
    treatment_syncs: Sequence[Dict],
) -> Dict:
    """Exact five-way split of 3Q subsequent_scaling.

    Identity (residual ~ 0 when last-play snapshots and syncs exist):
      subsequent_scaling = input_state + timing + membership
                         + rounding + residual
      input_state = Σ_common share_c · (R_t − R_c)
      membership  = Σ_common R_t · (share_t − share_c)
      rounding    = Σ_common (actual_t − exact_t) − (actual_c − exact_c)
      timing      = Σ extra treatment actual − Σ extra control actual
    """
    q = decompose_play_pair(
        control_start, treatment_start, control_play, treatment_play,
    )
    delta = float(q.get("subsequent_scaling") or 0.0)
    slot = _safe_int(
        control_start.get("board_slot"),
        _safe_int(treatment_start.get("board_slot"), 0),
    )
    base = {
        "subsequent_scaling": delta,
        "pre_sync_input_state": 0.0,
        "sync_timing_count": 0.0,
        "membership_allocation": 0.0,
        "rounding_residue": 0.0,
        "residual": 0.0,
        "snapshots_complete": bool(q.get("snapshots_complete")),
        "control_subtype": q.get("control_subtype"),
        "treatment_subtype": q.get("treatment_subtype"),
        "subtype_mismatch": q.get("subtype_mismatch"),
        "n_syncs_control": len(control_syncs or []),
        "n_syncs_treatment": len(treatment_syncs or []),
        "sync_turns_control": [int(s.get("turn") or 0) for s in control_syncs or []],
        "sync_turns_treatment": [
            int(s.get("turn") or 0) for s in treatment_syncs or []
        ],
        "same_sync_count": len(control_syncs or []) == len(treatment_syncs or []),
        "same_sync_turns": (
            [int(s.get("turn") or 0) for s in control_syncs or []]
            == [int(s.get("turn") or 0) for s in treatment_syncs or []]
        ),
        "first_input_diverge_field": None,
        "i_c_obs": None,
        "i_t_obs": None,
        "i_c_recon": 0.0,
        "i_t_recon": 0.0,
        "flow_gap_control": 0.0,
        "flow_gap_treatment": 0.0,
        "s_c_sticky": q.get("s_c_sticky"),
        "s_t_paint": q.get("s_t_paint"),
        "s_c_start": q.get("s_c_start"),
        "s_t_start": q.get("s_t_start"),
    }
    if not q.get("snapshots_complete"):
        base["residual"] = delta
        return base

    c_by_t = {int(s.get("turn") or 0): s for s in control_syncs or []}
    t_by_t = {int(s.get("turn") or 0): s for s in treatment_syncs or []}
    only_c = set(c_by_t) - set(t_by_t)
    only_t = set(t_by_t) - set(c_by_t)
    common = sorted(set(c_by_t) & set(t_by_t))

    timing = 0.0
    i_c_recon = 0.0
    i_t_recon = 0.0
    for turn in only_t:
        inc = body_sync_increment(t_by_t[turn], slot, treatment_start)
        timing += inc["actual"]
        i_t_recon += inc["actual"]
    for turn in only_c:
        inc = body_sync_increment(c_by_t[turn], slot, control_start)
        timing -= inc["actual"]
        i_c_recon += inc["actual"]

    input_state = 0.0
    membership = 0.0
    rounding = 0.0
    first_field = None
    for turn in common:
        ci = body_sync_increment(c_by_t[turn], slot, control_start)
        ti = body_sync_increment(t_by_t[turn], slot, treatment_start)
        i_c_recon += ci["actual"]
        i_t_recon += ti["actual"]
        input_state += ci["share"] * (ti["residual_add"] - ci["residual_add"])
        membership += ti["residual_add"] * (ti["share"] - ci["share"])
        rounding += (ti["actual"] - ti["exact"]) - (ci["actual"] - ci["exact"])
        if first_field is None:
            first_field = first_input_diverge_field(c_by_t[turn], t_by_t[turn])
    if not common and (c_by_t or t_by_t):
        first_field = "sync_turn_set"

    i_c_obs = None
    i_t_obs = None
    if q.get("s_c_start") is not None and q.get("s_c_sticky") is not None:
        i_c_obs = float(q["s_c_start"]) - float(q["s_c_sticky"])
    if q.get("s_t_start") is not None and q.get("s_t_paint") is not None:
        i_t_obs = float(q["s_t_start"]) - float(q["s_t_paint"])
    residual = delta - (input_state + timing + membership + rounding)
    base.update({
        "pre_sync_input_state": float(input_state),
        "sync_timing_count": float(timing),
        "membership_allocation": float(membership),
        "rounding_residue": float(rounding),
        "residual": float(residual),
        "first_input_diverge_field": first_field,
        "i_c_obs": i_c_obs,
        "i_t_obs": i_t_obs,
        "i_c_recon": float(i_c_recon),
        "i_t_recon": float(i_t_recon),
        "flow_gap_control": (
            0.0 if i_c_obs is None else float(i_c_obs) - float(i_c_recon)
        ),
        "flow_gap_treatment": (
            0.0 if i_t_obs is None else float(i_t_obs) - float(i_t_recon)
        ),
    })
    return base


class ScaleSyncTracer(PlayLifecycleTracer):
    """3Q play-lifecycle rows plus every residual/ratio scale-sync snapshot."""

    def __init__(self, lobby_id: int, seed: int, arm: str):
        super().__init__(lobby_id, seed, arm)
        self.scale_syncs: List[Dict] = []
        self._pending_sync: Dict[int, Dict] = {}
        self._sync_seq: Dict[int, int] = {}

    def attach_to_env(self, env: BGEnv) -> None:
        super().attach_to_env(env)
        prev = env.scaling_audit_hook

        def _hook(e, player, seat, budget):
            if prev is not None:
                prev(e, player, seat, budget)
            self._on_scale_sync_audit(e, player, seat, budget)

        env.scaling_audit_hook = _hook

    def begin_lobby(self, lobby_id: int, rng_seed: int, lobby_tribes: List[str]) -> None:
        super().begin_lobby(lobby_id, rng_seed, lobby_tribes)
        self._pending_sync.clear()
        self._sync_seq.clear()

    def _on_scale_sync_audit(self, env, player, seat: int, budget: Dict) -> None:
        board = list(getattr(player, "board", None) or [])
        slots = [_minion_snap(m, i) for i, m in enumerate(board)]
        seq = int(self._sync_seq.get(int(seat), 0))
        self._sync_seq[int(seat)] = seq + 1
        rec = {
            "seed": int(self.seed),
            "lobby": int(self.lobby_id),
            "arm": self.arm,
            "seat": int(seat),
            "turn": int(getattr(env, "turn", 0) or 0),
            "sync_index": seq,
            "sync_order": seq,
            "kind": "scale_sync",
            "board_recruit_raw": _sum_field(slots, "recruit_raw"),
            "abstract_pool_entering": float(
                getattr(player, "abstract_pool", 0.0) or 0.0
            ),
            "synth_pool_entering": _sum_field(slots, "synthetic_share"),
            "board_combat_raw": _sum_field(slots, "combat_raw"),
            "board_size": len(slots),
            "board_tier_hist": _tier_hist(slots),
            "pre_slots": slots,
            "firestone_target": budget.get("firestone_target"),
            "firestone_prev": budget.get("firestone_prev"),
            "curve_ratio": budget.get("curve_ratio"),
            "growth_factor": budget.get("growth_factor"),
            "ratio_g": budget.get("ratio_g"),
            "ratio_add": budget.get("ratio_add"),
            "pace_target": budget.get("pace_target"),
            "over": budget.get("over"),
            "residual_add": budget.get("residual_add"),
            "residual_clamp_active": budget.get("residual_clamp_active"),
            "just_leveled": budget.get("just_leveled"),
            "tavern_tier": budget.get("tavern_tier"),
            "turns_since_level": budget.get("turns_since_level"),
            "end_of_recruit_pre_scaling_stats": budget.get(
                "end_of_recruit_pre_scaling_stats"
            ),
            "computed_scale_increment": budget.get("residual_add"),
        }
        self._pending_sync[int(seat)] = rec

    def after_scale_all(self, env: BGEnv) -> None:
        super().after_scale_all(env)
        for p in list(getattr(env, "players", None) or []):
            seat = int(getattr(p, "idx", -1))
            pending = self._pending_sync.pop(seat, None)
            if pending is None:
                continue
            board = list(getattr(p, "board", None) or [])
            post = [_minion_snap(m, i) for i, m in enumerate(board)]
            pending["post_slots"] = post
            pending["abstract_pool_after"] = float(
                getattr(p, "abstract_pool", 0.0) or 0.0
            )
            pending["synth_pool_after"] = _sum_field(post, "synthetic_share")
            pending["board_combat_after"] = _sum_field(post, "combat_raw")
            pending["applied_board_increment"] = (
                pending["board_combat_after"] - pending["board_combat_raw"]
            )
            bodies = []
            residual_add = _as_float(pending.get("residual_add"))
            board_combat = _as_float(pending.get("board_combat_raw"))
            for pre in pending.get("pre_slots") or []:
                slot = _safe_int(pre.get("slot"), 0)
                post_row = _match_slot(post, slot, pre)
                exact = exact_scale_increment(
                    pre.get("combat_raw"), board_combat, residual_add,
                )
                actual = 0.0
                if post_row is not None:
                    actual = (
                        _as_float(post_row.get("synthetic_share"))
                        - _as_float(pre.get("synthetic_share"))
                    )
                bodies.append({
                    "slot": slot,
                    "name": pre.get("name"),
                    "card_id": pre.get("card_id"),
                    "tier": pre.get("tier"),
                    "recruit_raw": pre.get("recruit_raw"),
                    "combat_pre": pre.get("combat_raw"),
                    "synth_pre": pre.get("synthetic_share"),
                    "synth_post": (
                        None if post_row is None
                        else post_row.get("synthetic_share")
                    ),
                    "exact_increment": exact,
                    "actual_increment": actual,
                    "rounding_residue": actual - exact,
                })
            pending["body_alloc"] = bodies
            pending["body_increment_sum"] = float(
                sum(_as_float(b.get("actual_increment")) for b in bodies)
            )
            pending["board_flow_gap"] = (
                pending["body_increment_sum"] - pending["applied_board_increment"]
            )
            self.scale_syncs.append(pending)
        self._pending_sync.clear()


def run_scale_sync_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    from ml.phase_3r_prereg import assert_seed_range_allowed
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
    play_events: List[Dict] = []
    scale_syncs: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = ScaleSyncTracer(i, seed + i, arm)
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
                play_events.extend(tracer.play_events)
                scale_syncs.extend(tracer.scale_syncs)
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
        "play_events": play_events,
        "scale_syncs": scale_syncs,
    }


def run_greedy_control_scale_sync(lobbies: int, seed: int) -> Dict:
    return run_scale_sync_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_scale_sync(lobbies: int, seed: int) -> Dict:
    return run_scale_sync_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _index_syncs(scale_syncs: Sequence[Dict]) -> Dict[Tuple[int, int], List[Dict]]:
    out: Dict[Tuple[int, int], List[Dict]] = {}
    for ev in scale_syncs or []:
        seed = _safe_int(ev.get("seed"))
        seat = _safe_int(ev.get("seat"))
        if seed is None or seat is None:
            continue
        out.setdefault((seed, seat), []).append(ev)
    for key, rows in out.items():
        rows.sort(key=lambda r: (int(r.get("turn") or 0), int(r.get("sync_index") or 0)))
        out[key] = rows
    return out


def _post_play_syncs(
    index: Dict[Tuple[int, int], List[Dict]],
    seed, seat, last_play_turn, fight_turn,
) -> List[Dict]:
    seed_i = _safe_int(seed)
    seat_i = _safe_int(seat)
    lo = _safe_int(last_play_turn)
    hi = _safe_int(fight_turn)
    if seed_i is None or seat_i is None or lo is None or hi is None:
        return []
    rows = []
    for ev in index.get((seed_i, seat_i), []) or []:
        turn = int(ev.get("turn") or 0)
        if lo <= turn <= hi:
            rows.append(ev)
    return rows


def _sum_parts(pairs: Sequence[Dict]) -> Dict[str, float]:
    totals = {name: 0.0 for name in SCALE_SYNC_COMPONENTS}
    totals["subsequent_scaling"] = 0.0
    for p in pairs:
        for name in SCALE_SYNC_COMPONENTS:
            totals[name] += float(p.get(name) or 0.0)
        totals["subsequent_scaling"] += float(p.get("subsequent_scaling") or 0.0)
    return totals


def tier_mass_scale_sync(per_tier: Dict[str, Dict]) -> Dict:
    """Decision shares from within-tier |parts| so T1↓ / T3↑ cannot cancel."""
    mass = {name: 0.0 for name in SCALE_SYNC_COMPONENTS}
    n_used = 0
    abs_delta = 0.0
    for cell in (per_tier or {}).values():
        n = int(cell.get("n_pairs") or 0)
        if n <= 0:
            continue
        n_used += n
        abs_delta += n * abs(float(cell.get("subsequent_scaling") or 0.0))
        for name in SCALE_SYNC_COMPONENTS:
            mass[name] += n * abs(float(cell.get(name) or 0.0))
    total_mass = sum(mass.values())

    def _share(part: float) -> Optional[float]:
        if total_mass < 1e-12:
            return None
        return float(part) / total_mass

    return {
        "method": "within_tier_abs_mass_scale_sync_identity",
        "n_pairs": n_used,
        "abs_subsequent_scaling": (
            None if n_used <= 0 else float(abs_delta) / float(n_used)
        ),
        "component_abs_mass": mass,
        "total_abs_mass": total_mass,
        **{name: (
            None if n_used <= 0 else float(mass[name]) / float(n_used)
        ) for name in SCALE_SYNC_COMPONENTS},
        **{f"share_of_delta_{name}": _share(mass[name])
           for name in SCALE_SYNC_COMPONENTS},
        "scale_sync_components": list(SCALE_SYNC_COMPONENTS),
    }


def _sync_arm_summary(syncs: Sequence[Dict]) -> Dict:
    n = len(syncs)
    increments = [_as_float(s.get("applied_board_increment")) for s in syncs]
    residuals = [
        _as_float(s.get("residual_add"))
        for s in syncs if s.get("residual_add") is not None
    ]
    flow_gaps = [_as_float(s.get("board_flow_gap")) for s in syncs]
    return {
        "n_syncs": n,
        "mean_board_recruit_raw": _mean([
            _as_float(s.get("board_recruit_raw")) for s in syncs
        ]),
        "mean_abstract_pool_entering": _mean([
            _as_float(s.get("abstract_pool_entering")) for s in syncs
        ]),
        "mean_synth_pool_entering": _mean([
            _as_float(s.get("synth_pool_entering")) for s in syncs
        ]),
        "mean_firestone_target": _mean([
            _as_float(s.get("firestone_target"))
            for s in syncs if s.get("firestone_target") is not None
        ]),
        "mean_residual_add": _mean(residuals),
        "mean_applied_board_increment": _mean(increments),
        "mean_board_size": _mean([_as_float(s.get("board_size")) for s in syncs]),
        "n_board_flow_mismatch": sum(
            1 for g in flow_gaps if abs(g) > _FLOW_ABS_TOL
        ),
        "max_abs_board_flow_gap": (
            None if not flow_gaps else max(abs(g) for g in flow_gaps)
        ),
    }


def attribute_scale_sync(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    matched: Optional[Dict] = None,
    mechanics: Optional[Dict] = None,
    allocation: Optional[Dict] = None,
    lifecycle: Optional[Dict] = None,
) -> Dict:
    """3Q locks plus last-play → combat-start scale-sync split."""
    if lifecycle is None:
        lifecycle = compare_play_lifecycle(
            control_raw, treatment_raw,
            matched=matched, mechanics=mechanics, allocation=allocation,
        )
    if mechanics is None:
        mechanics = {
            "attribution": (lifecycle or {}).get("matched_state"),
            "primary": (lifecycle or {}).get("mechanics_3o"),
        }
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
    alloc_pairs, unpaired_c, unpaired_t, n_fights = _pair_bodies(
        events, c_fights, t_fights,
    )
    c_plays = _index_plays(control_raw.get("play_events") or [])
    t_plays = _index_plays(treatment_raw.get("play_events") or [])
    c_syncs = _index_syncs(control_raw.get("scale_syncs") or [])
    t_syncs = _index_syncs(treatment_raw.get("scale_syncs") or [])

    pairs: List[Dict] = []
    n_complete = 0
    n_missing_play = 0
    for p in alloc_pairs:
        ev_seed = p.get("seed")
        ev_seat = p.get("causal_seat")
        ev_turn = p.get("first_divergence_turn")
        c_row = p.get("control") or {}
        t_row = p.get("treatment") or {}
        c_fight = _fight_for_event(c_fights, {
            "seed": ev_seed, "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
        })
        t_fight = _fight_for_event(t_fights, {
            "seed": ev_seed, "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
        })
        c_winner = _safe_int((c_fight or {}).get("winner_seat"), ev_seat)
        t_winner = _safe_int((t_fight or {}).get("winner_seat"), ev_seat)
        c_play = _last_play(c_plays, ev_seed, c_winner, ev_turn)
        t_play = _last_play(t_plays, ev_seed, t_winner, ev_turn)
        c_post = _post_play_syncs(
            c_syncs, ev_seed, c_winner,
            None if c_play is None else c_play.get("turn"),
            ev_turn,
        )
        t_post = _post_play_syncs(
            t_syncs, ev_seed, t_winner,
            None if t_play is None else t_play.get("turn"),
            ev_turn,
        )
        parts = decompose_scale_pair(
            c_row, t_row, c_play, t_play, c_post, t_post,
        )
        parts.update({
            "seed": ev_seed,
            "causal_seat": ev_seat,
            "first_divergence_turn": ev_turn,
            "tier": clamp_tier(p.get("tier") or c_row.get("tier") or t_row.get("tier")),
            "board_slot": p.get("board_slot"),
        })
        if parts.get("snapshots_complete"):
            n_complete += 1
        else:
            n_missing_play += 1
        pairs.append(parts)

    for r in unpaired_t:
        pairs.append({
            "subsequent_scaling": 0.0,
            "pre_sync_input_state": 0.0,
            "sync_timing_count": 0.0,
            "membership_allocation": 0.0,
            "rounding_residue": 0.0,
            "residual": 0.0,
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "treatment",
            "snapshots_complete": False,
        })
    for r in unpaired_c:
        pairs.append({
            "subsequent_scaling": 0.0,
            "pre_sync_input_state": 0.0,
            "sync_timing_count": 0.0,
            "membership_allocation": 0.0,
            "rounding_residue": 0.0,
            "residual": 0.0,
            "tier": clamp_tier(r.get("tier")),
            "unpaired": "control",
            "snapshots_complete": False,
        })

    totals = _sum_parts(pairs)
    n_pairs = max(1, len(pairs))
    means = {k: float(v) / float(n_pairs) for k, v in totals.items()}
    obs_delta = means["subsequent_scaling"]

    def _share(part: float) -> Optional[float]:
        if abs(obs_delta) < 1e-12:
            return None
        return float(part) / obs_delta

    pooled_signed = {
        "method": "exact_scale_sync_paired_slot_identity",
        "n_pairs": len(pairs),
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "n_snapshots_complete": n_complete,
        "n_missing_play": n_missing_play,
        "subsequent_scaling": means["subsequent_scaling"],
        **{name: means[name] for name in SCALE_SYNC_COMPONENTS},
        "explained_all_parts": sum(means[n] for n in SCALE_SYNC_COMPONENTS),
        "residual_vs_delta": (
            means["subsequent_scaling"]
            - sum(means[n] for n in SCALE_SYNC_COMPONENTS)
        ),
        **{f"share_of_delta_{name}": _share(means[name])
           for name in SCALE_SYNC_COMPONENTS},
        "scale_sync_components": list(SCALE_SYNC_COMPONENTS),
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
        d = cm["subsequent_scaling"]

        def _cshare(part: float, den: float = d) -> Optional[float]:
            if abs(den) < 1e-12:
                return None
            return float(part) / den

        per_tier[str(tier)] = {
            "n_pairs": len(cell),
            "subsequent_scaling": cm["subsequent_scaling"],
            **{name: cm[name] for name in SCALE_SYNC_COMPONENTS},
            **{f"share_of_delta_{name}": _cshare(cm[name])
               for name in SCALE_SYNC_COMPONENTS},
            "same_sync_turn_rate": _safe_div(
                float(sum(1 for p in cell if p.get("same_sync_turns"))),
                float(len(cell)),
            ),
            "same_sync_count_rate": _safe_div(
                float(sum(1 for p in cell if p.get("same_sync_count"))),
                float(len(cell)),
            ),
        }

    primary = tier_mass_scale_sync(per_tier)
    field_counts = Counter(
        p.get("first_input_diverge_field")
        for p in pairs
        if p.get("first_input_diverge_field")
    )
    modal_field = None
    if field_counts:
        modal_field = field_counts.most_common(1)[0][0]
    complete = [p for p in pairs if p.get("snapshots_complete")]
    primary.update({
        "n_unpaired_control": len(unpaired_c),
        "n_unpaired_treatment": len(unpaired_t),
        "n_primary_fights": n_fights,
        "n_snapshots_complete": n_complete,
        "n_missing_play": n_missing_play,
        "pooled_signed_subsequent_scaling": pooled_signed["subsequent_scaling"],
        "pooled_signed": pooled_signed,
        "residual_vs_delta": pooled_signed["residual_vs_delta"],
        "modal_first_input_diverge_field": modal_field,
        "first_input_diverge_counts": dict(field_counts),
        "same_sync_count_rate": _safe_div(
            float(sum(1 for p in complete if p.get("same_sync_count"))),
            float(len(complete)),
        ),
        "same_sync_turn_rate": _safe_div(
            float(sum(1 for p in complete if p.get("same_sync_turns"))),
            float(len(complete)),
        ),
        "mean_n_syncs_control": _mean([
            float(p.get("n_syncs_control") or 0) for p in complete
        ]),
        "mean_n_syncs_treatment": _mean([
            float(p.get("n_syncs_treatment") or 0) for p in complete
        ]),
        "mean_abs_flow_gap_control": _mean([
            abs(float(p.get("flow_gap_control") or 0.0)) for p in complete
        ]),
        "mean_abs_flow_gap_treatment": _mean([
            abs(float(p.get("flow_gap_treatment") or 0.0)) for p in complete
        ]),
        "n_body_flow_mismatch": sum(
            1 for p in complete
            if abs(float(p.get("flow_gap_control") or 0.0)) > _FLOW_ABS_TOL
            or abs(float(p.get("flow_gap_treatment") or 0.0)) > _FLOW_ABS_TOL
        ),
    })

    life_attr = (lifecycle or {}).get("attribution") or lifecycle or {}
    life_primary = (lifecycle or {}).get("primary") or life_attr.get("primary") or {}

    def _close(got, published, tol=_SYNTH_REPRO_TOL) -> bool:
        if got is None or published is None:
            return False
        return abs(float(got) - float(published)) <= tol

    t1_c = life_attr.get("t1_synth_control")
    t1_t = life_attr.get("t1_synth_treatment")
    t3_c = life_attr.get("t3_synth_control")
    t3_t = life_attr.get("t3_synth_treatment")
    share_scaling = life_primary.get("share_of_delta_subsequent_scaling")
    sample_syncs_c = []
    sample_syncs_t = []
    keys = set()
    for p in pairs:
        seed = _safe_int(p.get("seed"))
        if seed is None:
            continue
        keys.add((seed, _safe_int(p.get("causal_seat"))))
    for ev in control_raw.get("scale_syncs") or []:
        if (_safe_int(ev.get("seed")), _safe_int(ev.get("seat"))) in keys:
            sample_syncs_c.append(ev)
    for ev in treatment_raw.get("scale_syncs") or []:
        if (_safe_int(ev.get("seed")), _safe_int(ev.get("seat"))) in keys:
            sample_syncs_t.append(ev)

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
            "subsequent_scaling": p.get("subsequent_scaling"),
            "pre_sync_input_state": p.get("pre_sync_input_state"),
            "sync_timing_count": p.get("sync_timing_count"),
            "membership_allocation": p.get("membership_allocation"),
            "rounding_residue": p.get("rounding_residue"),
            "residual": p.get("residual"),
            "n_syncs_control": p.get("n_syncs_control"),
            "n_syncs_treatment": p.get("n_syncs_treatment"),
            "sync_turns_control": p.get("sync_turns_control"),
            "sync_turns_treatment": p.get("sync_turns_treatment"),
            "first_input_diverge_field": p.get("first_input_diverge_field"),
        })
    n_board_mismatch = sum(
        1 for s in (control_raw.get("scale_syncs") or [])
        + (treatment_raw.get("scale_syncs") or [])
        if abs(_as_float(s.get("board_flow_gap"))) > _FLOW_ABS_TOL
    )
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_same_outcome_damage": len(events),
        "n_primary_class3": n_c,
        "n_primary_class3_treatment": n_t,
        "published_same_outcome_damage": PHASE_3N_CLASS3,
        "published_primary_n": PHASE_3O_PRIMARY_N,
        "published_3p_n_pairs": PHASE_3P_PRIMARY_N_PAIRS,
        "published_3p_n_fights": PHASE_3P_PRIMARY_N_FIGHTS,
        "published_3q_n_pairs": PHASE_3Q_PRIMARY_N_PAIRS,
        "published_3q_n_fights": PHASE_3Q_PRIMARY_N_FIGHTS,
        "published_3q_share_scaling": PHASE_3Q_SHARE_SCALING,
        "same_outcome_damage_reproduced": len(events) == PHASE_3N_CLASS3,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_B_reproduced": life_attr.get("phase_3n_B_reproduced"),
        "phase_3o_t5t6_B": PHASE_3O_T5T6_B,
        "phase_3o_B_reproduced": life_attr.get("phase_3o_B_reproduced"),
        "phase_3o_share_start_stats": PHASE_3O_SHARE_START_STATS,
        "phase_3o_share_synth": PHASE_3O_SHARE_SYNTH,
        "phase_3p_share_timing": PHASE_3P_SHARE_TIMING,
        "phase_3p_share_pool": PHASE_3P_SHARE_POOL,
        "phase_3p_share_weight": PHASE_3P_SHARE_WEIGHT,
        "phase_3p_share_rounding": PHASE_3P_SHARE_ROUNDING,
        "phase_3q_share_same_state": life_primary.get(
            "share_of_delta_same_state_repaint"
        ),
        "phase_3q_share_lifecycle": life_primary.get(
            "share_of_delta_replacement_lifecycle"
        ),
        "phase_3q_share_scaling": share_scaling,
        "phase_3q_share_residual": life_primary.get("share_of_delta_residual"),
        "phase_3q_scaling_reproduced": _close(
            share_scaling, PHASE_3Q_SHARE_SCALING, _SHARE_REPRO_TOL,
        ),
        "t1_synth_control": t1_c,
        "t1_synth_treatment": t1_t,
        "t3_synth_control": t3_c,
        "t3_synth_treatment": t3_t,
        "t1_synth_reproduced": (
            _close(t1_c, PHASE_3Q_T1_SYNTH_CONTROL)
            and _close(t1_t, PHASE_3Q_T1_SYNTH_TREATMENT)
        ),
        "t3_synth_reproduced": (
            _close(t3_c, PHASE_3Q_T3_SYNTH_CONTROL)
            and _close(t3_t, PHASE_3Q_T3_SYNTH_TREATMENT)
        ),
        "primary": primary,
        "per_tier": per_tier,
        "syncs_control": _sync_arm_summary(sample_syncs_c),
        "syncs_treatment": _sync_arm_summary(sample_syncs_t),
        "lifecycle_3q": {
            "n_pairs": life_primary.get("n_pairs"),
            "share_same_state": life_primary.get(
                "share_of_delta_same_state_repaint"
            ),
            "share_lifecycle": life_primary.get(
                "share_of_delta_replacement_lifecycle"
            ),
            "share_scaling": share_scaling,
            "share_residual": life_primary.get("share_of_delta_residual"),
            "n_snapshots_complete": life_primary.get("n_snapshots_complete"),
        },
        "scale_reconciliation": {
            "nested_ok": abs(float(primary.get("residual_vs_delta") or 0.0)) <= 1e-6,
            "board_flow_ok": n_board_mismatch == 0,
            "n_board_flow_mismatch": n_board_mismatch,
            "n_body_flow_mismatch": primary.get("n_body_flow_mismatch"),
            "identity": NESTED_SCALE_SYNC_IDENTITY,
            "flow_identity": SCALE_FLOW_RECONCILE_IDENTITY,
            "same_turn_identity": SAME_TURN_SYNC_IDENTITY,
        },
        "examples": examples,
        "modal_first_input_diverge_field": modal_field,
        "first_input_diverge_counts": dict(field_counts),
    }


def compare_scale_sync(
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
    allocation: Optional[Dict] = None,
    lifecycle: Optional[Dict] = None,
) -> Dict:
    """3Q lock + T5/T6 post-play scale-sync split."""
    if lifecycle is None:
        lifecycle = compare_play_lifecycle(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain, first=first, matched=matched,
            mechanics=mechanics, allocation=allocation,
        )
    if mechanics is None:
        mechanics = {
            "attribution": lifecycle.get("matched_state"),
            "primary": lifecycle.get("mechanics_3o"),
            "source": lifecycle.get("source"),
            "matched_state": lifecycle.get("matched_state"),
            "first_divergence_3m": lifecycle.get("first_divergence_3m"),
            "decomposition_3g": lifecycle.get("decomposition_3g"),
            "published_3n_locks": lifecycle.get("published_3n_locks"),
        }
    if matched is None:
        matched = {
            "attribution": lifecycle.get("matched_state"),
            "reconciliation": lifecycle.get("reconciliation"),
        }
    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch,
        still_fields_t1t3=False,
    )
    attr = attribute_scale_sync(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch,
        matched=matched, mechanics=mechanics, allocation=allocation,
        lifecycle=lifecycle,
    )
    rec = dict((lifecycle or {}).get("reconciliation") or {})
    rec.update({
        "nested_scale_sync_identity": NESTED_SCALE_SYNC_IDENTITY,
        "scale_flow_reconcile_identity": SCALE_FLOW_RECONCILE_IDENTITY,
        "same_turn_sync_identity": SAME_TURN_SYNC_IDENTITY,
        "phase_3q_scaling_reproduced": attr.get("phase_3q_scaling_reproduced"),
        "t1_synth_reproduced": attr.get("t1_synth_reproduced"),
        "t3_synth_reproduced": attr.get("t3_synth_reproduced"),
        "scale_nested_ok": (attr.get("scale_reconciliation") or {}).get("nested_ok"),
        "board_flow_ok": (attr.get("scale_reconciliation") or {}).get("board_flow_ok"),
    })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": attr,
        "primary": attr.get("primary"),
        "per_tier": attr.get("per_tier"),
        "syncs_control": attr.get("syncs_control"),
        "syncs_treatment": attr.get("syncs_treatment"),
        "lifecycle_3q": attr.get("lifecycle_3q"),
        "source": lifecycle.get("source") if lifecycle else None,
        "matched_state": lifecycle.get("matched_state") if lifecycle else None,
        "mechanics_3o": lifecycle.get("mechanics_3o") if lifecycle else None,
        "allocation_3p": lifecycle.get("allocation_3p") if lifecycle else None,
        "lifecycle_primary_3q": lifecycle.get("primary") if lifecycle else None,
        "first_divergence_3m": (
            lifecycle.get("first_divergence_3m") if lifecycle else None
        ),
        "reconciliation": rec,
        "decomposition_3g": lifecycle.get("decomposition_3g") if lifecycle else None,
        "published_3q_locks": {
            "n_pairs": PHASE_3Q_PRIMARY_N_PAIRS,
            "n_fights": PHASE_3Q_PRIMARY_N_FIGHTS,
            "share_same_state": PHASE_3Q_SHARE_SAME_STATE,
            "share_lifecycle": PHASE_3Q_SHARE_LIFECYCLE,
            "share_scaling": PHASE_3Q_SHARE_SCALING,
            "share_residual": PHASE_3Q_SHARE_RESIDUAL,
            "t1_control": PHASE_3Q_T1_SYNTH_CONTROL,
            "t1_treatment": PHASE_3Q_T1_SYNTH_TREATMENT,
            "t3_control": PHASE_3Q_T3_SYNTH_CONTROL,
            "t3_treatment": PHASE_3Q_T3_SYNTH_TREATMENT,
        },
        "published_3p_locks": lifecycle.get("published_3p_locks") if lifecycle else None,
        "published_3o_locks": lifecycle.get("published_3o_locks") if lifecycle else None,
        "published_3n_locks": lifecycle.get("published_3n_locks") if lifecycle else None,
    }


__all__ = [
    "ScaleSyncTracer",
    "attribute_scale_sync",
    "body_sync_increment",
    "compare_scale_sync",
    "decompose_scale_pair",
    "diagnose_phase_3r",
    "exact_scale_increment",
    "first_input_diverge_field",
    "rounded_scale_increment",
    "run_greedy_2s_treatment_scale_sync",
    "run_greedy_control_scale_sync",
    "tier_mass_scale_sync",
]
