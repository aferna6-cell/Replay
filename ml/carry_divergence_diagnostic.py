"""Phase 3F — observational paired carry-trajectory / selection audit.

Reuses the 3E PoolLifecycleTracer on consumed DEV 14200–14699. Does not
change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

For each (seed, seat) present in both arms, walks T7 through that seat's
eventual T7–T14 punch-row appearance (as the opposing / loser seat whose
carry is stamped on 3D punch rows) and records carry, current-turn add,
tier/board mix, alive status, fight outcome, punch-sample membership, and
the first turn the treatment–control carry path materially separates.

Unconditional paired Δcarry is then compared to the same Δ after
progressive conditioning on later punch inclusion, low winner-start tier,
and eventual fight outcome.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from ml.phase_3e_prereg import FLOW_ABS_TOL as E_FLOW_ABS_TOL
from ml.phase_3e_prereg import PHASE_3D_BOARD_POOL_MAGNITUDE
from ml.phase_3f_prereg import (
    FLOW_ABS_TOL,
    HISTORY_LINK_IDENTITY,
    INSTRUMENT_TURNS,
    LOW_WINNER_START_TIERS,
    MATERIAL_ABS,
    MATERIAL_REL,
    PHASE_3E_CARRY_DELTA,
    PHASE_3E_CARRY_SHARE_OF_A1,
    PHASE_3E_PUNCH_DELTA_CARRY,
    POOL_FLOW_IDENTITY,
    assert_seed_range_allowed,
    carry_value,
    first_separation_turn,
    materially_separated,
    scale_add_value,
    share_of_carry_term,
)
from ml.pool_lifecycle_diagnostic import (
    collect_lifecycle_minions,
    compare_lifecycle,
    run_greedy_2s_treatment_lifecycle,
    run_greedy_control_lifecycle,
    summarize_lifecycle_arm,
)
from ml.synthetic_allocation_diagnostic import _hits, _safe_div

METHODOLOGY_VERSION = "3f_v1"

_TURN_WINDOW = set(INSTRUMENT_TURNS)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _index_turn_rows(turn_rows: Sequence[Dict]) -> Dict[Tuple[int, int, int], Dict]:
    out: Dict[Tuple[int, int, int], Dict] = {}
    for row in turn_rows or []:
        try:
            seed = int(row["seed"])
            seat = int(row["seat"])
            turn = int(row["turn"])
        except (KeyError, TypeError, ValueError):
            continue
        if turn not in _TURN_WINDOW:
            continue
        out[(seed, seat, turn)] = row
    return out


def _alive(row: Optional[Dict]) -> bool:
    if not row:
        return False
    for key in ("alive_at_combat", "alive_at_recruit", "alive_at_post_scale"):
        val = row.get(key)
        if val is not None:
            return bool(val)
    return True


def _tier(row: Optional[Dict]) -> Optional[int]:
    if not row:
        return None
    for key in ("tier_at_recruit", "tavern_tier", "tier_post_scale", "tier_pre_scale"):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _board_size(row: Optional[Dict]) -> Optional[float]:
    if not row:
        return None
    for key in (
        "board_size_combat_start",
        "board_size_post_scale",
        "board_size_recruit_start",
    ):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _mean_tier(row: Optional[Dict]) -> Optional[float]:
    if not row:
        return None
    for key in (
        "mean_tier_combat_start",
        "mean_tier_post_scale",
        "mean_tier_recruit_start",
    ):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _seat_outcome(fight: Dict, seat: int) -> str:
    winner = fight.get("winner_seat")
    loser = fight.get("loser_seat")
    try:
        if winner is not None and int(winner) == int(seat):
            return "win"
        if loser is not None and int(loser) == int(seat):
            return "loss"
    except (TypeError, ValueError):
        pass
    kind = fight.get("kind") or fight.get("fight_outcome") or ""
    if kind == "bye":
        return "bye"
    if fight.get("ghost"):
        return "ghost"
    return "tie"


def _winner_start_tiers(fight: Dict) -> List[int]:
    tiers: List[int] = []
    for row in fight.get("start_minions") or []:
        raw = row.get("tier")
        if raw in (None, ""):
            continue
        try:
            tiers.append(int(raw))
        except (TypeError, ValueError):
            continue
    return tiers


def _is_punch_fight(fight: Dict) -> bool:
    if int(fight.get("turn") or 0) not in _TURN_WINDOW:
        return False
    if int(fight.get("applied_hp_loss") or 0) <= 0:
        return False
    rows = fight.get("start_minions") or []
    return any(int(r.get("n_damaging_hits") or 0) > 0 for r in rows)


def _collect_seat_fights(fights: Sequence[Dict]) -> Dict[Tuple[int, int, int], Dict]:
    """(seed, seat, turn) → fight role for live instrumented fights."""
    out: Dict[Tuple[int, int, int], Dict] = {}
    for fight in fights or []:
        turn = int(fight.get("turn") or 0)
        if turn not in _TURN_WINDOW:
            continue
        try:
            seed = int(fight["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        seats = []
        for key in ("seat_a", "seat_b", "winner_seat", "loser_seat"):
            raw = fight.get(key)
            if raw in (None, ""):
                continue
            try:
                seats.append(int(raw))
            except (TypeError, ValueError):
                continue
        punch = _is_punch_fight(fight)
        w_tiers = _winner_start_tiers(fight) if punch else []
        low_ws = bool(w_tiers) and any(t in LOW_WINNER_START_TIERS for t in w_tiers)
        min_ws = min(w_tiers) if w_tiers else None
        for seat in set(seats):
            rec = {
                "seed": seed,
                "seat": seat,
                "turn": turn,
                "outcome": _seat_outcome(fight, seat),
                "kind": fight.get("kind"),
                "ghost": bool(fight.get("ghost")),
                "applied_hp_loss": int(fight.get("applied_hp_loss") or 0),
                "winner_seat": fight.get("winner_seat"),
                "loser_seat": fight.get("loser_seat"),
                "winner_tavern_tier": fight.get("winner_tavern_tier"),
                "in_punch_sample": punch and _seat_outcome(fight, seat) == "loss",
                "punch_as_winner": punch and _seat_outcome(fight, seat) == "win",
                "winner_start_tiers": list(w_tiers),
                "min_winner_start_tier": min_ws,
                "low_winner_start": low_ws and _seat_outcome(fight, seat) == "loss",
            }
            prev = out.get((seed, seat, turn))
            if prev is None or (rec["in_punch_sample"] and not prev.get("in_punch_sample")):
                out[(seed, seat, turn)] = rec
    return out


def _punch_links(fights: Sequence[Dict]) -> List[Dict]:
    """One record per 3E punch row, joined to the opposing (loser) seat."""
    links: List[Dict] = []
    hits = _hits(fights)
    rows = collect_lifecycle_minions(hits)
    # collect_lifecycle_minions flattens start_minions; re-attach fight keys
    # by walking hits in the same order.
    fight_meta: List[Dict] = []
    for fight in hits:
        n = len(fight.get("start_minions") or [])
        meta = {
            "seed": fight.get("seed"),
            "lobby": fight.get("lobby"),
            "turn": fight.get("turn"),
            "kind": fight.get("kind"),
            "ghost": bool(fight.get("ghost")),
            "winner_seat": fight.get("winner_seat"),
            "loser_seat": fight.get("loser_seat"),
            "fight_outcome": fight.get("fight_outcome") or fight.get("outcome"),
            "winner_tavern_tier": fight.get("winner_tavern_tier"),
            "applied_hp_loss": fight.get("applied_hp_loss"),
        }
        fight_meta.extend([meta] * n)
    for row, meta in zip(rows, fight_meta):
        if int(row.get("n_damaging_hits") or 0) <= 0:
            continue
        link = dict(row)
        link.update(meta)
        try:
            link["seed"] = int(link["seed"])
            link["turn"] = int(link["turn"])
            link["loser_seat"] = (
                None if link.get("loser_seat") in (None, "") else int(link["loser_seat"])
            )
            link["winner_seat"] = (
                None if link.get("winner_seat") in (None, "") else int(link["winner_seat"])
            )
            link["winner_start_tier"] = int(row.get("tier") or 0)
        except (TypeError, ValueError):
            continue
        links.append(link)
    return links


def _first_punch_turn(events: Dict[int, Dict], *, as_opp: bool = True) -> Optional[int]:
    for turn in INSTRUMENT_TURNS:
        ev = events.get(turn)
        if not ev:
            continue
        if as_opp and ev.get("in_punch_sample"):
            return int(turn)
        if (not as_opp) and ev.get("punch_as_winner"):
            return int(turn)
    return None


def build_seat_trajectories(
    turn_rows: Sequence[Dict],
    fights: Sequence[Dict],
) -> Dict[Tuple[int, int], Dict]:
    """(seed, seat) → per-turn carry path plus punch / outcome tags."""
    turns = _index_turn_rows(turn_rows)
    events = _collect_seat_fights(fights)
    seats: Dict[Tuple[int, int], Dict] = {}
    keys = {(s, p) for (s, p, _t) in turns} | {(s, p) for (s, p, _t) in events}
    for seed, seat in keys:
        by_turn: Dict[int, Dict] = {}
        fight_by_turn: Dict[int, Dict] = {}
        for turn in INSTRUMENT_TURNS:
            row = turns.get((seed, seat, turn))
            ev = events.get((seed, seat, turn))
            if ev:
                fight_by_turn[turn] = ev
            if row is None and ev is None:
                continue
            by_turn[turn] = {
                "turn": turn,
                "carry": carry_value(row),
                "scale_add": scale_add_value(row),
                "alive": _alive(row) if row is not None else False,
                "tier": _tier(row),
                "board_size": _board_size(row),
                "mean_tier": _mean_tier(row),
                "n_alive": None if row is None else row.get("n_alive"),
                "flow_ok": None if row is None else row.get("flow_ok"),
                "outcome": None if ev is None else ev.get("outcome"),
                "in_punch_sample": bool(ev and ev.get("in_punch_sample")),
                "punch_as_winner": bool(ev and ev.get("punch_as_winner")),
                "low_winner_start": bool(ev and ev.get("low_winner_start")),
                "min_winner_start_tier": None if ev is None else ev.get("min_winner_start_tier"),
                "applied_hp_loss": None if ev is None else ev.get("applied_hp_loss"),
                "has_turn_row": row is not None,
            }
        punch_turn = _first_punch_turn(fight_by_turn, as_opp=True)
        winner_punch_turn = _first_punch_turn(fight_by_turn, as_opp=False)
        low = False
        min_ws = None
        eventual = None
        if punch_turn is not None:
            ev = fight_by_turn.get(punch_turn) or {}
            low = bool(ev.get("low_winner_start"))
            min_ws = ev.get("min_winner_start_tier")
            eventual = ev.get("outcome")
        seats[(seed, seat)] = {
            "seed": seed,
            "seat": seat,
            "by_turn": by_turn,
            "first_punch_turn": punch_turn,
            "first_winner_punch_turn": winner_punch_turn,
            "later_punch_included": punch_turn is not None,
            "low_winner_start": low,
            "min_winner_start_tier": min_ws,
            "eventual_outcome": eventual,
        }
    return seats


def pair_trajectories(
    control: Dict[Tuple[int, int], Dict],
    treatment: Dict[Tuple[int, int], Dict],
) -> List[Dict]:
    """Join control/treatment seats that share (seed, seat)."""
    keys = sorted(set(control) & set(treatment))
    pairs: List[Dict] = []
    for key in keys:
        c = control[key]
        t = treatment[key]
        by_turn: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        turn_detail: Dict[str, Dict] = {}
        for turn in INSTRUMENT_TURNS:
            cr = (c.get("by_turn") or {}).get(turn)
            tr = (t.get("by_turn") or {}).get(turn)
            cc = None if cr is None else cr.get("carry")
            tc = None if tr is None else tr.get("carry")
            if cr is None and tr is None:
                continue
            by_turn[turn] = (cc, tc)
            turn_detail[str(turn)] = {
                "control_carry": cc,
                "treatment_carry": tc,
                "delta": None if cc is None or tc is None else float(tc) - float(cc),
                "control_alive": False if cr is None else bool(cr.get("alive")),
                "treatment_alive": False if tr is None else bool(tr.get("alive")),
                "control_add": None if cr is None else cr.get("scale_add"),
                "treatment_add": None if tr is None else tr.get("scale_add"),
                "control_tier": None if cr is None else cr.get("tier"),
                "treatment_tier": None if tr is None else tr.get("tier"),
                "control_board_size": None if cr is None else cr.get("board_size"),
                "treatment_board_size": None if tr is None else tr.get("board_size"),
                "control_outcome": None if cr is None else cr.get("outcome"),
                "treatment_outcome": None if tr is None else tr.get("outcome"),
                "control_in_punch": False if cr is None else bool(cr.get("in_punch_sample")),
                "treatment_in_punch": False if tr is None else bool(tr.get("in_punch_sample")),
                "both_alive": bool(
                    cr and tr and cr.get("alive") and tr.get("alive")
                    and cc is not None and tc is not None
                ),
                "materially_separated": materially_separated(cc, tc),
            }
        sep = first_separation_turn(by_turn)
        later_punch = bool(c.get("later_punch_included") or t.get("later_punch_included"))
        low = bool(c.get("low_winner_start") or t.get("low_winner_start"))
        # Eventual outcome: prefer the first punch-turn loss on either arm.
        eventual = c.get("eventual_outcome") or t.get("eventual_outcome")
        punch_turn = c.get("first_punch_turn")
        if punch_turn is None:
            punch_turn = t.get("first_punch_turn")
        elif t.get("first_punch_turn") is not None:
            punch_turn = min(int(punch_turn), int(t["first_punch_turn"]))
        pairs.append({
            "seed": key[0],
            "seat": key[1],
            "by_turn": turn_detail,
            "first_separation_turn": sep,
            "first_punch_turn": punch_turn,
            "later_punch_included": later_punch,
            "low_winner_start": low,
            "eventual_outcome": eventual,
            "control_punch_turn": c.get("first_punch_turn"),
            "treatment_punch_turn": t.get("first_punch_turn"),
            "control_eventual_outcome": c.get("eventual_outcome"),
            "treatment_eventual_outcome": t.get("eventual_outcome"),
        })
    return pairs


def _pair_passes(pair: Dict, *, stage: str) -> bool:
    if stage == "unconditional":
        return True
    if stage == "punch_included":
        return bool(pair.get("later_punch_included"))
    if stage == "low_winner_start":
        return bool(pair.get("later_punch_included") and pair.get("low_winner_start"))
    if stage == "outcome_conditioned":
        return bool(
            pair.get("later_punch_included")
            and pair.get("low_winner_start")
            and pair.get("eventual_outcome") == "loss"
        )
    raise ValueError(f"unknown stage {stage}")


def _delta_by_turn(pairs: Sequence[Dict], *, stage: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for turn in INSTRUMENT_TURNS:
        c_vals: List[float] = []
        t_vals: List[float] = []
        n_sep = 0
        n_pairs = 0
        for pair in pairs:
            if not _pair_passes(pair, stage=stage):
                continue
            cell = (pair.get("by_turn") or {}).get(str(turn)) or {}
            if not cell.get("both_alive"):
                continue
            cc = cell.get("control_carry")
            tc = cell.get("treatment_carry")
            if cc is None or tc is None:
                continue
            n_pairs += 1
            c_vals.append(float(cc))
            t_vals.append(float(tc))
            if cell.get("materially_separated"):
                n_sep += 1
        delta = None if not c_vals else float(st.mean(t_vals) - st.mean(c_vals))
        out[str(turn)] = {
            "n_pairs": n_pairs,
            "mean_control_carry": _mean(c_vals),
            "mean_treatment_carry": _mean(t_vals),
            "delta_treatment_minus_control": delta,
            "n_materially_separated": n_sep,
            "p_materially_separated": _safe_div(float(n_sep), float(n_pairs)),
            "share_of_3e_carry": share_of_carry_term(delta),
        }
    return out


def _pooled_delta(by_turn: Dict[str, Dict]) -> Optional[float]:
    """Pair-count-weighted mean Δcarry across T7–T14."""
    num = 0.0
    den = 0.0
    for cell in by_turn.values():
        n = float(cell.get("n_pairs") or 0)
        d = cell.get("delta_treatment_minus_control")
        if n <= 0 or d is None:
            continue
        num += n * float(d)
        den += n
    if den <= 0:
        return None
    return num / den


def _delta_at_punch_appearance(pairs: Sequence[Dict], *, stage: str) -> Optional[float]:
    """Mean paired Δcarry at each pair's first punch-row appearance turn."""
    xs: List[float] = []
    for pair in pairs:
        if not _pair_passes(pair, stage=stage):
            continue
        turn = pair.get("first_punch_turn")
        if turn is None:
            continue
        cell = (pair.get("by_turn") or {}).get(str(int(turn))) or {}
        if not cell.get("both_alive"):
            continue
        d = cell.get("delta")
        if d is None:
            continue
        xs.append(float(d))
    return _mean(xs)


def _separation_hist(pairs: Sequence[Dict], *, stage: str) -> Dict:
    counts = {str(t): 0 for t in INSTRUMENT_TURNS}
    n = 0
    n_sep = 0
    before_punch = 0
    at_or_after_punch = 0
    never = 0
    for pair in pairs:
        if not _pair_passes(pair, stage=stage):
            continue
        n += 1
        sep = pair.get("first_separation_turn")
        punch = pair.get("first_punch_turn")
        if sep is None:
            never += 1
            continue
        n_sep += 1
        counts[str(int(sep))] += 1
        if punch is None or int(sep) < int(punch):
            before_punch += 1
        else:
            at_or_after_punch += 1
    return {
        "n_pairs": n,
        "n_separated": n_sep,
        "n_never_separated": never,
        "n_separated_before_punch": before_punch,
        "n_separated_at_or_after_punch": at_or_after_punch,
        "p_separated": _safe_div(float(n_sep), float(n)),
        "p_separated_before_punch": _safe_div(float(before_punch), float(n)),
        "by_turn": counts,
    }


def reconcile_history_links(
    fights: Sequence[Dict],
    turn_rows: Sequence[Dict],
) -> Dict:
    """Every punch row must join its opposing seat's exact prior-turn path."""
    turns = _index_turn_rows(turn_rows)
    links = _punch_links(fights)
    n = 0
    n_ok = 0
    n_skipped_ghost = 0
    n_missing_turn = 0
    n_carry_mismatch = 0
    n_add_mismatch = 0
    n_history_gap = 0
    n_flow_bad = 0
    examples: List[Dict] = []
    for link in links:
        if link.get("ghost") or (link.get("kind") not in (None, "live")):
            n_skipped_ghost += 1
            continue
        loser = link.get("loser_seat")
        seed = link.get("seed")
        turn = link.get("turn")
        if loser is None or seed is None or turn is None:
            n_skipped_ghost += 1
            continue
        n += 1
        row = turns.get((int(seed), int(loser), int(turn)))
        if row is None:
            n_missing_turn += 1
            if len(examples) < 6:
                examples.append({
                    "seed": seed, "seat": loser, "turn": turn,
                    "reason": "missing_turn_row",
                })
            continue
        punch_carry = carry_value(link)
        seat_carry = carry_value(row)
        punch_add = scale_add_value(link)
        seat_add = scale_add_value(row)
        carry_ok = (
            punch_carry is not None
            and seat_carry is not None
            and abs(float(punch_carry) - float(seat_carry)) <= FLOW_ABS_TOL
        )
        add_ok = True
        if punch_add is not None and seat_add is not None:
            add_ok = abs(float(punch_add) - float(seat_add)) <= FLOW_ABS_TOL
        if not carry_ok:
            n_carry_mismatch += 1
        if not add_ok:
            n_add_mismatch += 1
        if row.get("flow_ok") is False:
            n_flow_bad += 1
        gap = False
        for prior in range(7, int(turn) + 1):
            if (int(seed), int(loser), prior) not in turns:
                gap = True
                break
        if gap:
            n_history_gap += 1
        if carry_ok and add_ok and not gap and row.get("flow_ok") is not False:
            n_ok += 1
        elif len(examples) < 6:
            examples.append({
                "seed": seed, "seat": loser, "turn": turn,
                "punch_carry": punch_carry,
                "seat_carry": seat_carry,
                "carry_ok": carry_ok,
                "add_ok": add_ok,
                "history_gap": gap,
            })
    return {
        "identity": HISTORY_LINK_IDENTITY,
        "n_punch_rows": n,
        "n_skipped_ghost_or_no_loser": n_skipped_ghost,
        "n_ok": n_ok,
        "n_missing_turn_row": n_missing_turn,
        "n_carry_mismatch": n_carry_mismatch,
        "n_scale_add_mismatch": n_add_mismatch,
        "n_history_gap": n_history_gap,
        "n_flow_bad": n_flow_bad,
        "p_ok": _safe_div(float(n_ok), float(n)),
        "examples": examples,
    }


def _slim_pair(pair: Dict) -> Dict:
    keep = {
        "seed", "seat", "first_separation_turn", "first_punch_turn",
        "later_punch_included", "low_winner_start", "eventual_outcome",
        "control_punch_turn", "treatment_punch_turn",
    }
    out = {k: pair.get(k) for k in keep}
    by = pair.get("by_turn") or {}
    out["turns"] = {
        t: {
            "delta": (by.get(t) or {}).get("delta"),
            "both_alive": (by.get(t) or {}).get("both_alive"),
            "separated": (by.get(t) or {}).get("materially_separated"),
        }
        for t in (str(x) for x in INSTRUMENT_TURNS)
        if t in by
    }
    return out


def compare_divergence(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
) -> Dict:
    """Paired Δcarry by turn, then the same under progressive filters."""
    c_traj = build_seat_trajectories(
        control_raw.get("turn_rows") or [],
        control_raw.get("fights") or [],
    )
    t_traj = build_seat_trajectories(
        treatment_raw.get("turn_rows") or [],
        treatment_raw.get("fights") or [],
    )
    pairs = pair_trajectories(c_traj, t_traj)

    stages = (
        "unconditional",
        "punch_included",
        "low_winner_start",
        "outcome_conditioned",
    )
    by_stage: Dict[str, Dict] = {}
    for stage in stages:
        by_turn = _delta_by_turn(pairs, stage=stage)
        pooled = _pooled_delta(by_turn)
        at_punch = _delta_at_punch_appearance(pairs, stage=stage)
        by_stage[stage] = {
            "by_turn": by_turn,
            "pooled_delta": pooled,
            "delta_at_first_punch_appearance": at_punch,
            "share_of_3e_carry_pooled": share_of_carry_term(pooled),
            "share_of_3e_carry_at_punch": share_of_carry_term(at_punch),
            "separation": _separation_hist(pairs, stage=stage),
        }

    # Decision uses pooled unconditional paired Δ vs the extra gap that
    # appears only after punch / winner-start / outcome filters. Prefer
    # punch-appearance Δ for filtered stages (that is the #51 sampling
    # frame). Also keep the unpaired 3E punch-row Δcarry as the denom.
    uncond_delta = by_stage["unconditional"]["pooled_delta"]
    punch_delta = by_stage["punch_included"]["delta_at_first_punch_appearance"]
    if punch_delta is None:
        punch_delta = by_stage["punch_included"]["pooled_delta"]
    low_delta = by_stage["low_winner_start"]["delta_at_first_punch_appearance"]
    if low_delta is None:
        low_delta = by_stage["low_winner_start"]["pooled_delta"]
    out_delta = by_stage["outcome_conditioned"]["delta_at_first_punch_appearance"]
    if out_delta is None:
        out_delta = by_stage["outcome_conditioned"]["pooled_delta"]

    unpaired_punch = None
    if lifecycle_cmp:
        flow = (lifecycle_cmp.get("additive_flow") or {}).get(
            "delta_treatment_minus_control"
        ) or {}
        unpaired_punch = flow.get("mean_carry")
    if unpaired_punch is None:
        unpaired_punch = PHASE_3E_PUNCH_DELTA_CARRY

    share_uncond = share_of_carry_term(uncond_delta, denom=unpaired_punch)
    share_punch = share_of_carry_term(punch_delta, denom=unpaired_punch)
    share_low = share_of_carry_term(low_delta, denom=unpaired_punch)
    share_out = share_of_carry_term(out_delta, denom=unpaired_punch)
    after = [s for s in (share_punch, share_low, share_out) if s is not None]
    share_sel = None
    if share_uncond is not None:
        # Selection = punch-row crater not already present in paired uncond Δ.
        # If filters enlarge the paired gap, take that increment; else the
        # residual vs the unpaired 3E punch Δcarry is the selection piece.
        incr = 0.0 if not after else max(0.0, max(after) - float(share_uncond))
        residual_vs_unpaired = max(0.0, 1.0 - float(share_uncond))
        share_sel = max(incr, residual_vs_unpaired)

    unpaired_by_tier: Dict[str, Dict] = {}
    c_links = _punch_links(control_raw.get("fights") or [])
    t_links = _punch_links(treatment_raw.get("fights") or [])
    for tier in range(1, 7):
        cc = [carry_value(r) for r in c_links if int(r.get("winner_start_tier") or 0) == tier]
        tt = [carry_value(r) for r in t_links if int(r.get("winner_start_tier") or 0) == tier]
        cc_f = [float(x) for x in cc if x is not None]
        tt_f = [float(x) for x in tt if x is not None]
        dc = None if not cc_f or not tt_f else float(st.mean(tt_f) - st.mean(cc_f))
        unpaired_by_tier[str(tier)] = {
            "n_control": len(cc_f),
            "n_treatment": len(tt_f),
            "mean_control_carry": _mean(cc_f),
            "mean_treatment_carry": _mean(tt_f),
            "delta_treatment_minus_control": dc,
            "share_of_3e_carry": share_of_carry_term(dc, denom=unpaired_punch),
        }

    hist_c = reconcile_history_links(
        control_raw.get("fights") or [],
        control_raw.get("turn_rows") or [],
    )
    hist_t = reconcile_history_links(
        treatment_raw.get("fights") or [],
        treatment_raw.get("turn_rows") or [],
    )

    n_pairs = len(pairs)
    n_punch = sum(1 for p in pairs if p.get("later_punch_included"))
    n_low = sum(1 for p in pairs if p.get("later_punch_included") and p.get("low_winner_start"))

    timing = {
        "n_paired_seats": n_pairs,
        "n_later_punch_included": n_punch,
        "n_low_winner_start": n_low,
        "material_abs": MATERIAL_ABS,
        "material_rel": MATERIAL_REL,
        "phase_3e_punch_delta_carry": PHASE_3E_PUNCH_DELTA_CARRY,
        "phase_3e_carry_delta": PHASE_3E_CARRY_DELTA,
        "phase_3e_carry_share_of_a1": PHASE_3E_CARRY_SHARE_OF_A1,
        "unpaired_punch_delta_carry": unpaired_punch,
        "unconditional_pooled_delta": uncond_delta,
        "punch_included_delta_at_appearance": punch_delta,
        "low_winner_start_delta_at_appearance": low_delta,
        "outcome_conditioned_delta_at_appearance": out_delta,
        "share_of_3e_carry_unconditional": share_uncond,
        "share_of_3e_carry_punch_included": share_punch,
        "share_of_3e_carry_low_winner_start": share_low,
        "share_of_3e_carry_outcome_conditioned": share_out,
        "share_of_3e_carry_before_conditioning": share_uncond,
        "share_of_3e_carry_from_selection": share_sel,
        "unpaired_punch_by_winner_start_tier": unpaired_by_tier,
        "by_stage": by_stage,
    }

    rec = {
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "flow_abs_tol": FLOW_ABS_TOL,
        "phase_3d_board_pool_magnitude": PHASE_3D_BOARD_POOL_MAGNITUDE,
        "phase_3e_carry_share_of_a1": PHASE_3E_CARRY_SHARE_OF_A1,
        "e_flow_abs_tol": E_FLOW_ABS_TOL,
    }
    if lifecycle_cmp:
        lrec = lifecycle_cmp.get("reconciliation") or {}
        rec["reproduced_3d_board_pool_magnitude"] = lrec.get(
            "reproduced_3d_board_pool_magnitude"
        )
        rec["reproduced_3e_carry_share"] = (
            (lifecycle_cmp.get("reweighting") or {}).get("share_of_a1_inherited_carry_pool")
        )
        rec["flow_mismatches_control"] = lrec.get("flow_mismatches_control")
        rec["flow_mismatches_treatment"] = lrec.get("flow_mismatches_treatment")

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_paired_seats": n_pairs,
        "timing": timing,
        "reconciliation": rec,
        "example_pairs": [_slim_pair(p) for p in pairs[:12]],
        "lifecycle": {
            "reweighting": None if lifecycle_cmp is None else lifecycle_cmp.get("reweighting"),
            "additive_flow": None if lifecycle_cmp is None else lifecycle_cmp.get("additive_flow"),
            "reconciliation": None if lifecycle_cmp is None else lifecycle_cmp.get("reconciliation"),
        },
    }


def run_divergence_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    """Thin wrapper so 3F reuses the 3E lifecycle runner without new seeds."""
    assert_seed_range_allowed(seed, lobbies)
    if recruit_value_stats or board_level_abstract_scaling:
        return run_greedy_2s_treatment_lifecycle(lobbies, seed)
    return run_greedy_control_lifecycle(lobbies, seed)


def run_paired_divergence(lobbies: int, seed: int) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    control = run_greedy_control_lifecycle(lobbies, seed)
    treatment = run_greedy_2s_treatment_lifecycle(lobbies, seed)
    greedy_c = summarize_lifecycle_arm(control)
    greedy_t = summarize_lifecycle_arm(treatment)
    life = compare_lifecycle(greedy_c, greedy_t)
    cmp = compare_divergence(control, treatment, lifecycle_cmp=life)
    cmp["greedy_control_lifecycle"] = {
        "n_hits": greedy_c.get("n_hits") or greedy_c.get("_n_hits"),
        "punch_flow": greedy_c.get("punch_flow"),
        "turn_summary": greedy_c.get("turn_summary"),
        "flow_mismatches_turns": greedy_c.get("flow_mismatches_turns"),
    }
    cmp["greedy_treatment_lifecycle"] = {
        "n_hits": greedy_t.get("n_hits") or greedy_t.get("_n_hits"),
        "punch_flow": greedy_t.get("punch_flow"),
        "turn_summary": greedy_t.get("turn_summary"),
        "flow_mismatches_turns": greedy_t.get("flow_mismatches_turns"),
    }
    return cmp
