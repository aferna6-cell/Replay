"""Phase 3I — observational T1–T3 pairing / who-wins attribution.

Reuses the 3H BoardRetentionTracer on consumed DEV 14200–14699. Does not
change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

Restricts to late T10–T14 control T1–T3 punch rows whose paired
treatment seat is alive and still fields ≥1 T1–T3 body (the 3H 7155-row
leftover). For each leftover row the tracer matches the control fight
to the treatment fight at the same (seed, winner_seat, turn) and
classifies the missing treatment low-tier winner-start punch.
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.board_retention_diagnostic import (
    BoardRetentionTracer,
    _combat_raw_of,
    _is_low,
    _late_t1t3_rows,
    _recruit_raw_of,
    _t1t3_count,
    _t1t3_from_row,
    _t1t3_share,
    _tier_of,
    attribute_late_t1t3_collapse,
    collect_3h_leftover_rows,
    compare_retention,
)
from ml.carry_divergence_diagnostic import (
    compare_divergence,
    reconcile_history_links,
)
from ml.phase_3i_prereg import (
    FLOW_ABS_TOL,
    HISTORY_LINK_IDENTITY,
    LATE_TURNS,
    LEFTOVER_RECONCILE_IDENTITY,
    LINEAGE_IDENTITY,
    LOW_TIERS,
    PAIRED_SEAT_IDENTITY,
    PAIRING_COMPONENTS,
    PAIRING_IDENTITY,
    PHASE_3E_PUNCH_DELTA_CARRY,
    PHASE_3G_MIXTURE,
    PHASE_3G_MIXTURE_SHARE,
    PHASE_3G_N_CONTROL,
    PHASE_3G_N_TREATMENT,
    PHASE_3G_WITHIN_SHARE,
    PHASE_3H_COLLAPSE,
    PHASE_3H_LATE_CONTROL,
    PHASE_3H_LATE_TREATMENT,
    PHASE_3H_LEFTOVER,
    PHASE_3H_SHARE_LEFTOVER,
    POOL_FLOW_IDENTITY,
    REWEIGHT_ABS_TOL,
    VERY_LATE_TURNS,
    WEIGHT_RECONCILIATION_IDENTITY,
    assert_seed_range_allowed,
    classify_pairing_gap,
    share_of_leftover,
)
from ml.pool_lifecycle_diagnostic import compare_lifecycle, summarize_lifecycle_arm
from ml.punch_selection_diagnostic import collect_punch_sample_rows, compare_selection

METHODOLOGY_VERSION = "3i_v1"

_LATE = set(LATE_TURNS)
_LOW = set(LOW_TIERS)
_N_EXAMPLES = 8


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _opponent_of(fight: Optional[Dict], seat) -> Optional[int]:
    if not fight or seat in (None, ""):
        return None
    try:
        seat_i = int(seat)
    except (TypeError, ValueError):
        return None
    sa = fight.get("seat_a")
    sb = fight.get("seat_b")
    try:
        if sa not in (None, "") and int(sa) == seat_i:
            return None if sb in (None, "") else int(sb)
        if sb not in (None, "") and int(sb) == seat_i:
            return None if sa in (None, "") else int(sa)
    except (TypeError, ValueError):
        return None
    return None


def _kind_of(fight: Optional[Dict]) -> Optional[str]:
    if not fight:
        return None
    kind = fight.get("kind")
    if kind:
        return str(kind)
    if fight.get("ghost"):
        return "ghost"
    if fight.get("seat_b") in (None, ""):
        return "bye"
    return "live"


def same_pairing(control_fight: Optional[Dict], treatment_fight: Optional[Dict],
                 seat) -> bool:
    """Live vs live, same opponent seat. Ghost/bye/missing is a schedule change."""
    if not control_fight or not treatment_fight:
        return False
    if _kind_of(control_fight) != "live" or _kind_of(treatment_fight) != "live":
        return False
    c_opp = _opponent_of(control_fight, seat)
    t_opp = _opponent_of(treatment_fight, seat)
    if c_opp is None or t_opp is None:
        return False
    return int(c_opp) == int(t_opp)


def treatment_won(fight: Optional[Dict], seat) -> bool:
    if not fight or seat in (None, ""):
        return False
    winner = fight.get("winner_seat")
    if winner in (None, ""):
        return False
    try:
        return int(winner) == int(seat)
    except (TypeError, ValueError):
        return False


def _low_flags(bodies: Sequence[Dict]) -> Dict:
    low = [b for b in (bodies or []) if _tier_of(b) in _LOW]
    attacked = sum(1 for b in low if b.get("attacked"))
    survived = sum(1 for b in low if b.get("survived"))
    return {
        "n_low_tier_start": len(low),
        "n_low_tier_attacked": attacked,
        "n_low_tier_survived": survived,
        "low_tier_attacked": attacked > 0,
        "low_tier_survived": survived > 0,
    }


def _seat_board_fields(player, pre_hp) -> Dict:
    board = list(getattr(player, "board", None) or [])
    return {
        "t1t3_count": _t1t3_count(board),
        "t1t3_share": _t1t3_share(board),
        "tavern_tier": int(getattr(player, "tier", 1) or 1),
        "recruit_raw": float(sum(_recruit_raw_of(m) for m in board)),
        "abstract_pool_raw": float(getattr(player, "abstract_pool", 0.0) or 0.0),
        "combat_raw": float(sum(_combat_raw_of(m) for m in board)),
        "pre_fight_hp": None if pre_hp is None else _safe_int(pre_hp),
        "board_size": len(board),
        "alive": bool(getattr(player, "alive", True)),
    }


def _player_by_idx(env: BGEnv, seat) -> Optional[object]:
    if seat in (None, ""):
        return None
    try:
        seat_i = int(seat)
    except (TypeError, ValueError):
        return None
    for p in env.players:
        if int(getattr(p, "idx", -1)) == seat_i:
            return p
    if 0 <= seat_i < len(env.players):
        return env.players[seat_i]
    return None


class PairingWhoWinsTracer(BoardRetentionTracer):
    """3H retention plus both-board pairing / who-wins fight stamps."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        sa = fight.get("seat_a")
        sb = fight.get("seat_b")
        pa = _player_by_idx(env, sa)
        pb = _player_by_idx(env, sb)
        fields_a = _seat_board_fields(pa, fight.get("pre_hp_a")) if pa else {}
        fields_b = _seat_board_fields(pb, fight.get("pre_hp_b")) if pb else {}
        winner_low = _low_flags(rec.get("start_minions") or [])
        loser_low = _low_flags(rec.get("starting_loser") or fight.get("starting_loser") or [])
        rec["pairing"] = {
            "seat_a": None if sa in (None, "") else int(sa),
            "seat_b": None if sb in (None, "") else int(sb),
            "opponent_of_a": _opponent_of(rec, sa),
            "opponent_of_b": _opponent_of(rec, sb),
            "kind": _kind_of(rec),
            "winner_seat": rec.get("winner_seat"),
            "loser_seat": rec.get("loser_seat"),
            "fight_outcome": rec.get("fight_outcome") or rec.get("outcome"),
            "combat_margin_raw": rec.get("combat_margin_raw", rec.get("raw")),
            "applied_hp_loss": rec.get("applied_hp_loss"),
            "survivor_count": rec.get("actual_survivor_count", rec.get("survivor_count_actual")),
            "survivor_tier_sum": rec.get("actual_survivor_tier_sum", rec.get("survivor_tier_sum")),
            "a": fields_a,
            "b": fields_b,
            **winner_low,
            "loser_n_low_tier_start": loser_low["n_low_tier_start"],
            "loser_low_tier_attacked": loser_low["low_tier_attacked"],
            "loser_low_tier_survived": loser_low["low_tier_survived"],
            "alive_next": {},
        }

    def after_combat(self, env: BGEnv) -> None:
        super().after_combat(env)
        turn = int(getattr(env, "turn", 0) or 0)
        alive_map = {
            int(getattr(p, "idx", i)): bool(getattr(p, "alive", False))
            for i, p in enumerate(env.players)
        }
        for rec in reversed(self.fights):
            if int(rec.get("turn") or -1) != turn:
                break
            pairing = rec.get("pairing")
            if not isinstance(pairing, dict):
                continue
            nxt = {}
            for seat in (pairing.get("seat_a"), pairing.get("seat_b")):
                if seat in (None, ""):
                    continue
                nxt[str(int(seat))] = bool(alive_map.get(int(seat), False))
            pairing["alive_next"] = nxt


def run_pairing_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    fights: List[Dict] = []
    lengths: List[float] = []
    turn_rows: List[Dict] = []
    replacement_events: List[Dict] = []
    board_snapshots: List[Dict] = []
    t1t3_events: List[Dict] = []
    last_t1t3_losses: List[Dict] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = PairingWhoWinsTracer(i, seed + i, arm)
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
    }


def run_greedy_control_pairing(lobbies: int, seed: int) -> Dict:
    return run_pairing_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_pairing(lobbies: int, seed: int) -> Dict:
    return run_pairing_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _index_seat_fights(fights: Sequence[Dict]) -> Dict[Tuple[int, int, int], Dict]:
    out: Dict[Tuple[int, int, int], Dict] = {}
    for fight in fights or []:
        try:
            seed = int(fight["seed"])
            turn = int(fight["turn"])
        except (KeyError, TypeError, ValueError):
            continue
        for seat in (fight.get("seat_a"), fight.get("seat_b")):
            if seat in (None, ""):
                continue
            try:
                out[(seed, int(seat), turn)] = fight
            except (TypeError, ValueError):
                continue
    return out


def _seat_side_fields(fight: Optional[Dict], seat) -> Dict:
    empty = {
        "t1t3_count": None, "t1t3_share": None, "tavern_tier": None,
        "recruit_raw": None, "abstract_pool_raw": None, "combat_raw": None,
        "pre_fight_hp": None, "board_size": None, "alive": None,
    }
    if not fight or seat in (None, ""):
        return dict(empty)
    pairing = fight.get("pairing") or {}
    try:
        seat_i = int(seat)
        sa = pairing.get("seat_a")
        if sa not in (None, "") and int(sa) == seat_i:
            return dict(pairing.get("a") or empty)
        sb = pairing.get("seat_b")
        if sb not in (None, "") and int(sb) == seat_i:
            return dict(pairing.get("b") or empty)
    except (TypeError, ValueError):
        return dict(empty)
    return dict(empty)


def _alive_next(fight: Optional[Dict], seat) -> Optional[bool]:
    if not fight or seat in (None, ""):
        return None
    pairing = fight.get("pairing") or {}
    nxt = pairing.get("alive_next") or {}
    try:
        return nxt.get(str(int(seat)))
    except (TypeError, ValueError):
        return None


def _punch_key(row: Dict) -> Optional[Tuple[int, int, int]]:
    try:
        return (int(row["seed"]), int(row["winner_seat"]), int(row["turn"]))
    except (KeyError, TypeError, ValueError):
        return None


def _matched_fight_record(
    leftover_row: Dict,
    control_fight: Optional[Dict],
    treatment_fight: Optional[Dict],
    *,
    cls: str,
    same: bool,
    t_wins: bool,
) -> Dict:
    seat = leftover_row.get("winner_seat")
    c_side = _seat_side_fields(control_fight, seat)
    t_side = _seat_side_fields(treatment_fight, seat)
    c_opp = _opponent_of(control_fight, seat)
    t_opp = _opponent_of(treatment_fight, seat)
    c_opp_side = _seat_side_fields(control_fight, c_opp)
    t_opp_side = _seat_side_fields(treatment_fight, t_opp)
    c_pair = (control_fight or {}).get("pairing") or {}
    t_pair = (treatment_fight or {}).get("pairing") or {}
    return {
        "seed": leftover_row.get("seed"),
        "turn": leftover_row.get("turn"),
        "control_winner_seat": seat,
        "class": cls,
        "same_pairing": same,
        "treatment_wins": t_wins,
        "control_opponent_seat": c_opp,
        "treatment_opponent_seat": t_opp,
        "control_kind": _kind_of(control_fight),
        "treatment_kind": _kind_of(treatment_fight),
        "control_outcome": (control_fight or {}).get("fight_outcome"),
        "treatment_outcome": (treatment_fight or {}).get("fight_outcome"),
        "control_margin_raw": (control_fight or {}).get("combat_margin_raw"),
        "treatment_margin_raw": (treatment_fight or {}).get("combat_margin_raw"),
        "control_pre_hp": c_side.get("pre_fight_hp"),
        "treatment_pre_hp": t_side.get("pre_fight_hp"),
        "control_t1t3_count": c_side.get("t1t3_count"),
        "treatment_t1t3_count": t_side.get("t1t3_count"),
        "control_t1t3_share": c_side.get("t1t3_share"),
        "treatment_t1t3_share": t_side.get("t1t3_share"),
        "control_tavern_tier": c_side.get("tavern_tier"),
        "treatment_tavern_tier": t_side.get("tavern_tier"),
        "control_recruit_raw": c_side.get("recruit_raw"),
        "treatment_recruit_raw": t_side.get("recruit_raw"),
        "control_abstract_pool_raw": c_side.get("abstract_pool_raw"),
        "treatment_abstract_pool_raw": t_side.get("abstract_pool_raw"),
        "control_combat_raw": c_side.get("combat_raw"),
        "treatment_combat_raw": t_side.get("combat_raw"),
        "control_opp_t1t3_count": c_opp_side.get("t1t3_count"),
        "treatment_opp_t1t3_count": t_opp_side.get("t1t3_count"),
        "control_survivor_count": c_pair.get("survivor_count"),
        "treatment_survivor_count": t_pair.get("survivor_count"),
        "control_survivor_tier_sum": c_pair.get("survivor_tier_sum"),
        "treatment_survivor_tier_sum": t_pair.get("survivor_tier_sum"),
        "control_low_tier_attacked": c_pair.get("low_tier_attacked"),
        "treatment_low_tier_attacked": t_pair.get("low_tier_attacked"),
        "control_low_tier_survived": c_pair.get("low_tier_survived"),
        "treatment_low_tier_survived": t_pair.get("low_tier_survived"),
        "control_alive_next": _alive_next(control_fight, seat),
        "treatment_alive_next": _alive_next(treatment_fight, seat),
        "winner_start_tier": leftover_row.get("winner_start_tier"),
    }


def attribute_leftover_pairing(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    turns=None,
) -> Dict:
    """Decompose 3H leftover rows into pairing / outcome / survivor / residual."""
    window = tuple(turns or LATE_TURNS)
    t_late = _late_t1t3_rows(treatment_punch, window)
    t_punch_by_key: Dict[Tuple[int, int, int], int] = Counter()
    for row in t_late:
        key = _punch_key(row)
        if key is not None:
            t_punch_by_key[key] += 1

    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])

    grouped: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    unmatched = 0
    for row in leftover_rows:
        key = _punch_key(row)
        if key is None:
            unmatched += 1
            grouped[(-1, -1, -1)].append(row)
            continue
        grouped[key].append(row)

    counts = Counter()
    n_same_pairing = 0
    n_ctrl_fight = 0
    n_treat_fight = 0
    examples: Dict[str, List[Dict]] = {name: [] for name in PAIRING_COMPONENTS}
    fight_summaries: List[Dict] = []

    for key, rows in grouped.items():
        if key == (-1, -1, -1):
            for row in rows:
                counts["residual"] += 1
                if len(examples["residual"]) < _N_EXAMPLES:
                    examples["residual"].append({
                        "seed": row.get("seed"), "turn": row.get("turn"),
                        "class": "residual", "reason": "unparseable_key",
                    })
            continue
        seed_i, seat_i, turn_i = key
        c_fight = c_fights.get(key)
        t_fight = t_fights.get(key)
        if c_fight is not None:
            n_ctrl_fight += 1
        if t_fight is not None:
            n_treat_fight += 1
        same = same_pairing(c_fight, t_fight, seat_i)
        t_wins = treatment_won(t_fight, seat_i)
        t_n = int(t_punch_by_key.get(key, 0))
        if same:
            n_same_pairing += 1
        # Exclusive per leftover punch: pairing → outcome → uncovered sub → residual.
        n_cover = t_n if (same and t_wins) else 0
        for i, row in enumerate(rows):
            uncovered = i >= n_cover
            cls = classify_pairing_gap(
                same_pairing=same,
                treatment_wins=t_wins,
                treatment_tie_or_loss=same and not t_wins,
                treatment_t1t3_punches=t_n,
                uncovered=uncovered,
            )
            counts[cls] += 1
            rec = _matched_fight_record(
                row, c_fight, t_fight, cls=cls, same=same, t_wins=t_wins,
            )
            rec["treatment_t1t3_punches"] = t_n
            rec["uncovered"] = uncovered
            if len(examples[cls]) < _N_EXAMPLES:
                examples[cls].append(rec)
            if i == 0:
                fight_summaries.append(rec)

    leftover_n = float(len(leftover_rows))
    attributed = {name: float(counts.get(name, 0)) for name in PAIRING_COMPONENTS}
    reconstructed = sum(attributed.values())
    shares = {
        name: share_of_leftover(attributed[name], denom=leftover_n)
        for name in PAIRING_COMPONENTS
    }
    return {
        "turns": list(window),
        "n_leftover": int(leftover_n),
        "n_unparseable": unmatched,
        "n_control_fights_matched": n_ctrl_fight,
        "n_treatment_fights_matched": n_treat_fight,
        "n_same_pairing_keys": n_same_pairing,
        "n_keys": len(grouped),
        "counts": dict(counts),
        "attributed": attributed,
        "reconstructed_leftover_rows": reconstructed,
        "reconciliation_gap": leftover_n - reconstructed,
        "reconciliation_ok": abs(leftover_n - reconstructed) <= max(1.0, 1e-9 * (1 + leftover_n)),
        **{f"share_{k}": v for k, v in shares.items()},
        "examples": examples,
        "n_fight_summaries": len(fight_summaries),
    }


def _lock_3h(late: Dict) -> Dict:
    return {
        "n_control_late_t1t3_punch": late.get("n_control_late_t1t3_punch"),
        "n_treatment_late_t1t3_punch": late.get("n_treatment_late_t1t3_punch"),
        "collapse": late.get("collapse"),
        "leftover": late.get("leftover"),
        "share_leftover": late.get("share_leftover"),
        "published_leftover": PHASE_3H_LEFTOVER,
        "published_late_control": PHASE_3H_LATE_CONTROL,
        "published_late_treatment": PHASE_3H_LATE_TREATMENT,
        "published_collapse": PHASE_3H_COLLAPSE,
        "published_share_leftover": PHASE_3H_SHARE_LEFTOVER,
        "leftover_reproduced": (
            late.get("leftover") is not None
            and abs(float(late["leftover"]) - float(PHASE_3H_LEFTOVER)) < 1e-9
        ),
        "late_n_reproduced": (
            late.get("n_control_late_t1t3_punch") == PHASE_3H_LATE_CONTROL
            and late.get("n_treatment_late_t1t3_punch") == PHASE_3H_LATE_TREATMENT
        ),
    }


def compare_pairing(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    selection: Optional[Dict] = None,
    retention: Optional[Dict] = None,
) -> Dict:
    """3H leftover lock + pairing / who-wins decomposition."""
    if lifecycle_cmp is None and control_raw.get("arm") is not None:
        greedy_c = summarize_lifecycle_arm(control_raw)
        greedy_t = summarize_lifecycle_arm(treatment_raw)
        lifecycle_cmp = compare_lifecycle(greedy_c, greedy_t)
    if divergence is None:
        divergence = compare_divergence(
            control_raw, treatment_raw, lifecycle_cmp=lifecycle_cmp,
        )
    if selection is None:
        selection = compare_selection(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
        )
    if retention is None:
        retention = compare_retention(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence, selection=selection,
        )

    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch, turns=LATE_TURNS,
    )
    late = attribute_late_t1t3_collapse(
        control_raw, treatment_raw,
        control_punch=c_punch, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch, turns=VERY_LATE_TURNS,
    )
    late_attr = attribute_leftover_pairing(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch, turns=LATE_TURNS,
    )
    very_late_attr = attribute_leftover_pairing(
        control_raw, treatment_raw,
        leftover_rows=very_late_rows, treatment_punch=t_punch, turns=VERY_LATE_TURNS,
    )
    hist_c = reconcile_history_links(
        control_raw.get("fights") or [], control_raw.get("turn_rows") or [],
    )
    hist_t = reconcile_history_links(
        treatment_raw.get("fights") or [], treatment_raw.get("turn_rows") or [],
    )
    decomp = selection.get("decomposition") or {}
    rec_3h = retention.get("reconciliation") or {}
    rec = {
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "lineage_identity": LINEAGE_IDENTITY,
        "paired_seat_identity": PAIRED_SEAT_IDENTITY,
        "pairing_identity": PAIRING_IDENTITY,
        "leftover_reconcile_identity": LEFTOVER_RECONCILE_IDENTITY,
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "leftover_reconciliation_ok": late_attr.get("reconciliation_ok"),
        "leftover_n": late_attr.get("n_leftover"),
        "leftover_matches_3h_attr": (
            late.get("leftover") is not None
            and abs(float(late["leftover"]) - float(late_attr.get("n_leftover") or 0)) < 1e-9
        ),
        "phase_3g_mixture_reproduced": decomp.get("mixture_turn_winner_tier"),
        "phase_3g_mixture_share_reproduced": decomp.get("share_mixture_turn_winner_tier"),
        "phase_3g_within_share_reproduced": decomp.get("share_within_cell_opponent_carry"),
        "phase_3g_n_control": decomp.get("n_control"),
        "phase_3g_n_treatment": decomp.get("n_treatment"),
        "late_collapse_reconciliation_ok": late.get("reconciliation_ok"),
        "flow_abs_tol": FLOW_ABS_TOL,
        "reweight_abs_tol": REWEIGHT_ABS_TOL,
        "lineage_control": rec_3h.get("lineage_control"),
        "lineage_treatment": rec_3h.get("lineage_treatment"),
        "paired": rec_3h.get("paired"),
    }
    if lifecycle_cmp:
        rec["reproduced_3d_board_pool_magnitude"] = (
            (lifecycle_cmp.get("reconciliation") or {}).get(
                "reproduced_3d_board_pool_magnitude"
            )
        )
        rec["reproduced_3e_carry_share"] = (
            (lifecycle_cmp.get("reweighting") or {}).get("share_of_a1_inherited_carry_pool")
        )
        rec["flow_mismatches_control"] = (
            (lifecycle_cmp.get("reconciliation") or {}).get("flow_mismatches_control")
        )
        rec["flow_mismatches_treatment"] = (
            (lifecycle_cmp.get("reconciliation") or {}).get("flow_mismatches_treatment")
        )

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": late_attr,
        "very_late_attribution": very_late_attr,
        "leftover_3h": _lock_3h(late),
        "attribution_3h": late,
        "reconciliation": rec,
        "decomposition_3g": decomp,
        "paired_seats": retention.get("paired_seats"),
        "selection": {
            "decomposition": decomp,
            "reconciliation": selection.get("reconciliation"),
        },
        "timing_3f": None if divergence is None else divergence.get("timing"),
        "lifecycle": {
            "reweighting": None if lifecycle_cmp is None else lifecycle_cmp.get("reweighting"),
            "additive_flow": None if lifecycle_cmp is None else lifecycle_cmp.get("additive_flow"),
        },
        "published_3g_locks": {
            "mixture": PHASE_3G_MIXTURE,
            "mixture_share": PHASE_3G_MIXTURE_SHARE,
            "within_share": PHASE_3G_WITHIN_SHARE,
            "n_control": PHASE_3G_N_CONTROL,
            "n_treatment": PHASE_3G_N_TREATMENT,
            "punch_delta": PHASE_3E_PUNCH_DELTA_CARRY,
        },
        "published_3h_locks": {
            "leftover": PHASE_3H_LEFTOVER,
            "late_control": PHASE_3H_LATE_CONTROL,
            "late_treatment": PHASE_3H_LATE_TREATMENT,
            "collapse": PHASE_3H_COLLAPSE,
            "share_leftover": PHASE_3H_SHARE_LEFTOVER,
        },
    }
