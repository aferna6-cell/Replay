"""Phase 3G — observational punch-sample selection decomposition.

Reuses the 3E / 3F PoolLifecycleTracer on consumed DEV 14200–14699. Does
not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

Reproduces the unpaired punch-row Δcarry (−196) then splits
treatment − control with a symmetric Kitagawa / Oaxaca reweight on
common support:

    (1) winner-start tier × turn mixture (who produces punch rows)
    (2) opponent carry | matched turn × winner-start tier
    (3) winner/loser role + alive / elimination selection (nested)
    (4) leftover (exclusive inner support + residual)

Exclusive outer (turn, winner-start-tier) cells count as mixture — they
are the extreme of "who produces punch rows". Role / alive is nested
inside common-support outer cells and is a refinement of (2), not an
extra slice of Δ.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from ml.carry_divergence_diagnostic import (
    compare_divergence,
    reconcile_history_links,
)
from ml.phase_3e_prereg import PHASE_3D_BOARD_POOL_MAGNITUDE
from ml.phase_3f_prereg import carry_value
from ml.phase_3g_prereg import (
    EARLY_TURNS,
    FLOW_ABS_TOL,
    HISTORY_LINK_IDENTITY,
    INSTRUMENT_TURNS,
    LOW_WINNER_START_TIERS,
    PHASE_3E_CARRY_SHARE_OF_A1,
    PHASE_3E_PUNCH_DELTA_CARRY,
    POOL_FLOW_IDENTITY,
    REWEIGHT_ABS_TOL,
    WEIGHT_RECONCILIATION_IDENTITY,
    _alive_bin,
    _role_bin,
    assert_seed_range_allowed,
    share_of_crater,
)
from ml.pool_lifecycle_diagnostic import (
    collect_lifecycle_minions,
    compare_lifecycle,
    run_greedy_2s_treatment_lifecycle,
    run_greedy_control_lifecycle,
    summarize_lifecycle_arm,
)
from ml.synthetic_allocation_diagnostic import _hits, _kitagawa_two, _safe_div

METHODOLOGY_VERSION = "3g_v1"

_TURN_WINDOW = set(INSTRUMENT_TURNS)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _pct(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(float(x) for x in xs)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * float(p)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo]) * (1.0 - frac) + float(ordered[hi]) * frac


def collect_punch_sample_rows(fights: Sequence[Dict]) -> List[Dict]:
    """3E punch-row sample (all start minions on T7–T14 hit fights) + fight meta.

    This is the sample whose mean carry reproduces the #51 −196 crater.
    `winner_start_tier` is the winner-body start tier on the row.
    """
    hits = _hits(fights)
    rows = collect_lifecycle_minions(hits)
    meta: List[Dict] = []
    for fight in hits:
        n = len(fight.get("start_minions") or [])
        rec = {
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
        meta.extend([rec] * n)
    out: List[Dict] = []
    for row, m in zip(rows, meta):
        link = dict(row)
        link.update(m)
        try:
            turn = int(link.get("turn") or 0)
            if turn not in _TURN_WINDOW:
                continue
            link["turn"] = turn
            link["winner_start_tier"] = int(row.get("tier") or 0)
            if link["winner_start_tier"] < 1:
                continue
            if link.get("seed") not in (None, ""):
                link["seed"] = int(link["seed"])
            if link.get("loser_seat") not in (None, ""):
                link["loser_seat"] = int(link["loser_seat"])
            else:
                link["loser_seat"] = None
            if link.get("winner_seat") not in (None, ""):
                link["winner_seat"] = int(link["winner_seat"])
            else:
                link["winner_seat"] = None
        except (TypeError, ValueError):
            continue
        carry = carry_value(link)
        if carry is None:
            continue
        link["carry"] = float(carry)
        link["damaging"] = int(row.get("n_damaging_hits") or 0) > 0
        link["n_alive"] = int(link.get("opp_n_alive") or 0)
        link["opp_alive"] = bool(link.get("opp_alive", True))
        try:
            link["winner_tavern_tier"] = int(link.get("winner_tavern_tier") or 0)
        except (TypeError, ValueError):
            link["winner_tavern_tier"] = 0
        link["alive_bin"] = _alive_bin(link["n_alive"])
        link["role_bin"] = _role_bin(link["winner_tavern_tier"])
        out.append(link)
    return out


def _cell_stats(rows: Sequence[Dict]) -> Dict:
    carries = [float(r["carry"]) for r in rows]
    n = len(carries)
    return {
        "n": n,
        "mean_carry": _mean(carries),
        "p25_carry": _pct(carries, 0.25),
        "p50_carry": _pct(carries, 0.50),
        "p75_carry": _pct(carries, 0.75),
    }


def _group(
    rows: Sequence[Dict],
    key_fn,
) -> Dict:
    buckets: Dict = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    return buckets


def _outer_key(row: Dict) -> Tuple[int, int]:
    return (int(row["turn"]), int(row["winner_start_tier"]))


def _inner_key(row: Dict) -> Tuple[str, str]:
    return (str(row.get("alive_bin") or _alive_bin(row.get("n_alive"))),
            str(row.get("role_bin") or _role_bin(row.get("winner_tavern_tier"))))


def kitagawa_mean_delta(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    key_fn,
    *,
    exclusive_to: str = "mixture",
) -> Dict:
    """Symmetric Kitagawa / Oaxaca split of ȳ_T − ȳ_C on cells of `key_fn`.

    Weights are n/N per arm so the cell gaps sum to the mean delta.
    Exclusive cells (one arm empty) go to `exclusive_to` (`mixture` or
    `leftover`).
    """
    c_groups = _group(control_rows, key_fn)
    t_groups = _group(treatment_rows, key_fn)
    n_c = float(len(control_rows))
    n_t = float(len(treatment_rows))
    keys = sorted(set(c_groups) | set(t_groups), key=lambda k: (str(k),))
    mixture = 0.0
    rate = 0.0
    exclusive = 0.0
    leftover = 0.0
    cells: Dict[str, Dict] = {}
    sum_n_c = 0
    sum_n_t = 0
    for key in keys:
        c_cell = c_groups.get(key) or []
        t_cell = t_groups.get(key) or []
        nc = float(len(c_cell))
        nt = float(len(t_cell))
        sum_n_c += int(nc)
        sum_n_t += int(nt)
        w_c = 0.0 if n_c <= 0 else nc / n_c
        w_t = 0.0 if n_t <= 0 else nt / n_t
        mu_c = _mean([float(r["carry"]) for r in c_cell])
        mu_t = _mean([float(r["carry"]) for r in t_cell])
        mix_k, rate_k, gap_k, excl = _kitagawa_two(w_c, w_t, mu_c, mu_t, 1.0)
        if excl:
            exclusive += gap_k
            if exclusive_to == "leftover":
                leftover += gap_k
            else:
                mixture += gap_k
        else:
            mixture += mix_k
            rate += rate_k
        label = key if isinstance(key, str) else "x".join(str(x) for x in key)
        cells[label] = {
            "key": list(key) if isinstance(key, tuple) else key,
            "n_control": int(nc),
            "n_treatment": int(nt),
            "weight_control": w_c,
            "weight_treatment": w_t,
            "mean_control_carry": mu_c,
            "mean_treatment_carry": mu_t,
            "mixture": None if excl and exclusive_to == "leftover" else (
                gap_k if excl else mix_k
            ),
            "rate": 0.0 if excl else rate_k,
            "gap": gap_k,
            "exclusive_support": excl,
            "p25_control": _pct([float(r["carry"]) for r in c_cell], 0.25),
            "p50_control": _pct([float(r["carry"]) for r in c_cell], 0.50),
            "p75_control": _pct([float(r["carry"]) for r in c_cell], 0.75),
            "p25_treatment": _pct([float(r["carry"]) for r in t_cell], 0.25),
            "p50_treatment": _pct([float(r["carry"]) for r in t_cell], 0.50),
            "p75_treatment": _pct([float(r["carry"]) for r in t_cell], 0.75),
        }
    y_c = _mean([float(r["carry"]) for r in control_rows])
    y_t = _mean([float(r["carry"]) for r in treatment_rows])
    observed = None if y_c is None or y_t is None else float(y_t) - float(y_c)
    reconstructed = mixture + rate + leftover
    return {
        "n_control": int(n_c),
        "n_treatment": int(n_t),
        "sum_n_control_cells": sum_n_c,
        "sum_n_treatment_cells": sum_n_t,
        "mean_control_carry": y_c,
        "mean_treatment_carry": y_t,
        "observed_delta": observed,
        "mixture": mixture,
        "rate": rate,
        "exclusive": exclusive,
        "leftover": leftover,
        "reconstructed": reconstructed,
        "reconciliation_gap": (
            None if observed is None else float(observed) - float(reconstructed)
        ),
        "cells": cells,
    }


def _nested_role_alive(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    outer: Dict,
) -> Dict:
    """Split each common-support (turn, tier) rate into role/alive mix + leftover."""
    c_outer = _group(control_rows, _outer_key)
    t_outer = _group(treatment_rows, _outer_key)
    n_c = float(len(control_rows))
    n_t = float(len(treatment_rows))
    role = 0.0
    within_after = 0.0
    inner_excl = 0.0
    per_cell: Dict[str, Dict] = {}
    for label, cell in (outer.get("cells") or {}).items():
        if cell.get("exclusive_support"):
            continue
        key = tuple(cell["key"])
        c_cell = c_outer.get(key) or []
        t_cell = t_outer.get(key) or []
        inner = kitagawa_mean_delta(
            c_cell, t_cell, _inner_key, exclusive_to="leftover",
        )
        w_c = 0.0 if n_c <= 0 else float(len(c_cell)) / n_c
        w_t = 0.0 if n_t <= 0 else float(len(t_cell)) / n_t
        scale = 0.5 * (w_c + w_t)
        # inner.observed_delta is Δμ_k; outer rate is n_bar * Δμ_k with
        # weights (already scale * Δμ). Scale inner parts the same way.
        dmu = inner.get("observed_delta")
        if dmu is None:
            continue
        role_k = scale * float(inner["mixture"])
        after_k = scale * float(inner["rate"])
        excl_k = scale * float(inner["leftover"])
        role += role_k
        within_after += after_k
        inner_excl += excl_k
        per_cell[label] = {
            "scale": scale,
            "delta_mu": dmu,
            "role_alive": role_k,
            "within_after_role": after_k,
            "inner_exclusive": excl_k,
            "inner": {
                "mixture": inner["mixture"],
                "rate": inner["rate"],
                "leftover": inner["leftover"],
                "n_control": inner["n_control"],
                "n_treatment": inner["n_treatment"],
                "cells": inner["cells"],
            },
        }
    return {
        "role_alive_selection": role,
        "within_after_role": within_after,
        "inner_exclusive": inner_excl,
        "reconstructed_outer_rate": role + within_after + inner_excl,
        "per_outer_cell": per_cell,
    }


def _turn_tier_table(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
) -> Dict[str, Dict]:
    c_groups = _group(control_rows, _outer_key)
    t_groups = _group(treatment_rows, _outer_key)
    n_c = float(len(control_rows)) or 1.0
    n_t = float(len(treatment_rows)) or 1.0
    out: Dict[str, Dict] = {}
    for turn in INSTRUMENT_TURNS:
        for tier in range(1, 7):
            key = (turn, tier)
            c_cell = c_groups.get(key) or []
            t_cell = t_groups.get(key) or []
            cs = _cell_stats(c_cell)
            ts = _cell_stats(t_cell)
            dc = None
            if cs["mean_carry"] is not None and ts["mean_carry"] is not None:
                dc = float(ts["mean_carry"]) - float(cs["mean_carry"])
            label = f"T{turn}_W{tier}"
            out[label] = {
                "turn": turn,
                "winner_start_tier": tier,
                "n_control": cs["n"],
                "n_treatment": ts["n"],
                "weight_control": cs["n"] / n_c,
                "weight_treatment": ts["n"] / n_t,
                "mean_control_carry": cs["mean_carry"],
                "mean_treatment_carry": ts["mean_carry"],
                "delta_treatment_minus_control": dc,
                "p25_control": cs["p25_carry"],
                "p50_control": cs["p50_carry"],
                "p75_control": cs["p75_carry"],
                "p25_treatment": ts["p25_carry"],
                "p50_treatment": ts["p50_carry"],
                "p75_treatment": ts["p75_carry"],
            }
    return out


def _low_tier_early_diagnostic(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
) -> Dict:
    """Are treatment T1–T3 punch rows disproportionately early / low-carry?"""
    def _low(rows: Sequence[Dict]) -> List[Dict]:
        return [
            r for r in rows
            if int(r.get("winner_start_tier") or 0) in LOW_WINNER_START_TIERS
        ]

    c_low = _low(control_rows)
    t_low = _low(treatment_rows)
    split = kitagawa_mean_delta(
        c_low, t_low, lambda r: (int(r["turn"]),), exclusive_to="mixture",
    )

    def _arm(rows: Sequence[Dict]) -> Dict:
        if not rows:
            return {
                "n": 0, "p_early": None, "mean_turn": None,
                "mean_carry": None, "mean_carry_early": None,
                "mean_carry_late": None,
            }
        early = [r for r in rows if int(r["turn"]) in EARLY_TURNS]
        late = [r for r in rows if int(r["turn"]) not in EARLY_TURNS]
        return {
            "n": len(rows),
            "p_early": _safe_div(float(len(early)), float(len(rows))),
            "mean_turn": _mean([float(r["turn"]) for r in rows]),
            "mean_carry": _mean([float(r["carry"]) for r in rows]),
            "mean_carry_early": _mean([float(r["carry"]) for r in early]),
            "mean_carry_late": _mean([float(r["carry"]) for r in late]),
        }

    c_arm = _arm(c_low)
    t_arm = _arm(t_low)
    raw_delta = None
    if c_arm["mean_carry"] is not None and t_arm["mean_carry"] is not None:
        raw_delta = float(t_arm["mean_carry"]) - float(c_arm["mean_carry"])
    mix_share = share_of_crater(split["mixture"], denom=raw_delta) if raw_delta else None
    rate_share = share_of_crater(split["rate"], denom=raw_delta) if raw_delta else None
    early_shift = None
    if c_arm["p_early"] is not None and t_arm["p_early"] is not None:
        early_shift = float(t_arm["p_early"]) - float(c_arm["p_early"])

    if mix_share is not None and mix_share > 0.50:
        verdict = "disproportionately_early_low_carry"
    elif rate_share is not None and rate_share > 0.50:
        verdict = "true_within_cell_pool_deficit"
    elif mix_share is not None and rate_share is not None:
        verdict = "mixed_early_and_within_cell"
    else:
        verdict = "insufficient_low_tier_rows"

    return {
        "verdict": verdict,
        "control": c_arm,
        "treatment": t_arm,
        "early_weight_shift_treatment_minus_control": early_shift,
        "raw_delta": raw_delta,
        "turn_mixture": split["mixture"],
        "within_turn_rate": split["rate"],
        "exclusive": split["exclusive"],
        "share_of_t1t3_delta_from_turn_mixture": mix_share,
        "share_of_t1t3_delta_from_within_turn": rate_share,
        "by_turn": split["cells"],
    }


def decompose_punch_selection(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    unpaired_punch: Optional[float] = None,
) -> Dict:
    """Full 3G decomposition + tables + weight reconciliation."""
    if unpaired_punch is None:
        unpaired_punch = PHASE_3E_PUNCH_DELTA_CARRY
    y_c = _mean([float(r["carry"]) for r in control_rows])
    y_t = _mean([float(r["carry"]) for r in treatment_rows])
    observed = None if y_c is None or y_t is None else float(y_t) - float(y_c)

    outer = kitagawa_mean_delta(
        control_rows, treatment_rows, _outer_key, exclusive_to="mixture",
    )
    nested = _nested_role_alive(control_rows, treatment_rows, outer)
    leftover = float(outer["leftover"]) + (
        0.0 if outer.get("reconciliation_gap") is None
        else float(outer["reconciliation_gap"])
    )
    mixture = float(outer["mixture"])
    within = float(outer["rate"])
    role = float(nested["role_alive_selection"])
    reconstructed = mixture + within + leftover

    denom = float(unpaired_punch if unpaired_punch is not None else (observed or 0.0))
    share_mix = share_of_crater(mixture, denom=denom)
    share_cell = share_of_crater(within, denom=denom)
    share_role = share_of_crater(role, denom=denom)
    share_left = share_of_crater(leftover, denom=denom)
    share_mix_role = None
    if share_mix is not None or share_role is not None:
        share_mix_role = (
            (0.0 if share_mix is None else float(share_mix))
            + (0.0 if share_role is None else float(share_role))
        )

    n_c = len(control_rows)
    n_t = len(treatment_rows)
    w_c_sum = sum(
        float(c["weight_control"]) for c in (outer.get("cells") or {}).values()
    )
    w_t_sum = sum(
        float(c["weight_treatment"]) for c in (outer.get("cells") or {}).values()
    )
    rec = {
        "identity": WEIGHT_RECONCILIATION_IDENTITY,
        "n_control": n_c,
        "n_treatment": n_t,
        "sum_n_control_cells": outer["sum_n_control_cells"],
        "sum_n_treatment_cells": outer["sum_n_treatment_cells"],
        "counts_match_control": outer["sum_n_control_cells"] == n_c,
        "counts_match_treatment": outer["sum_n_treatment_cells"] == n_t,
        "weight_sum_control": w_c_sum,
        "weight_sum_treatment": w_t_sum,
        "weights_sum_to_one_control": abs(w_c_sum - 1.0) <= 1e-9 if n_c else True,
        "weights_sum_to_one_treatment": abs(w_t_sum - 1.0) <= 1e-9 if n_t else True,
        "observed_delta": observed,
        "unpaired_punch_delta_carry": unpaired_punch,
        "mixture": mixture,
        "within_cell": within,
        "role_alive": role,
        "leftover": leftover,
        "reconstructed": reconstructed,
        "reconciliation_gap": (
            None if observed is None else float(observed) - float(reconstructed)
        ),
        "reconciliation_ok": (
            observed is not None
            and abs(float(observed) - float(reconstructed)) <= max(
                REWEIGHT_ABS_TOL, 1e-9 * (1.0 + abs(float(observed)))
            )
        ),
        "nested_rate_gap": (
            within - float(nested["reconstructed_outer_rate"])
        ),
    }

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_control": n_c,
        "n_treatment": n_t,
        "mean_control_carry": y_c,
        "mean_treatment_carry": y_t,
        "observed_delta": observed,
        "unpaired_punch_delta_carry": unpaired_punch,
        "mixture_turn_winner_tier": mixture,
        "within_cell_opponent_carry": within,
        "role_alive_selection": role,
        "within_after_role": nested["within_after_role"],
        "inner_exclusive": nested["inner_exclusive"],
        "leftover": leftover,
        "share_mixture_turn_winner_tier": share_mix,
        "share_within_cell_opponent_carry": share_cell,
        "share_role_alive_selection": share_role,
        "share_leftover": share_left,
        "share_mixture_plus_role": share_mix_role,
        "outer_kitagawa": {
            "mixture": outer["mixture"],
            "rate": outer["rate"],
            "exclusive": outer["exclusive"],
            "leftover": outer["leftover"],
            "observed_delta": outer["observed_delta"],
            "reconciliation_gap": outer["reconciliation_gap"],
        },
        "nested_role_alive": {
            "role_alive_selection": nested["role_alive_selection"],
            "within_after_role": nested["within_after_role"],
            "inner_exclusive": nested["inner_exclusive"],
            "reconstructed_outer_rate": nested["reconstructed_outer_rate"],
        },
        "by_turn_winner_tier": _turn_tier_table(control_rows, treatment_rows),
        "low_tier_early": _low_tier_early_diagnostic(control_rows, treatment_rows),
        "reconciliation": rec,
        "outer_cells": outer["cells"],
        "nested_cells": nested["per_outer_cell"],
    }


def compare_selection(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    lifecycle_cmp: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
) -> Dict:
    """Run 3G decomposition on a paired 3E/3F raw-arm pair."""
    c_rows = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_rows = collect_punch_sample_rows(treatment_raw.get("fights") or [])

    unpaired = None
    if lifecycle_cmp:
        flow = (lifecycle_cmp.get("additive_flow") or {}).get(
            "delta_treatment_minus_control"
        ) or {}
        unpaired = flow.get("mean_carry")
    if unpaired is None:
        unpaired = PHASE_3E_PUNCH_DELTA_CARRY

    decomp = decompose_punch_selection(c_rows, t_rows, unpaired_punch=unpaired)

    hist_c = reconcile_history_links(
        control_raw.get("fights") or [],
        control_raw.get("turn_rows") or [],
    )
    hist_t = reconcile_history_links(
        treatment_raw.get("fights") or [],
        treatment_raw.get("turn_rows") or [],
    )

    rec = {
        "history_link_identity": HISTORY_LINK_IDENTITY,
        "weight_reconciliation_identity": WEIGHT_RECONCILIATION_IDENTITY,
        "pool_flow_identity": POOL_FLOW_IDENTITY,
        "history_link_control": hist_c,
        "history_link_treatment": hist_t,
        "history_link_mismatches_control": int(hist_c["n_punch_rows"] - hist_c["n_ok"]),
        "history_link_mismatches_treatment": int(hist_t["n_punch_rows"] - hist_t["n_ok"]),
        "flow_abs_tol": FLOW_ABS_TOL,
        "phase_3d_board_pool_magnitude": PHASE_3D_BOARD_POOL_MAGNITUDE,
        "phase_3e_carry_share_of_a1": PHASE_3E_CARRY_SHARE_OF_A1,
        "weight_reconciliation": decomp.get("reconciliation"),
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
        rec["reproduced_punch_delta_carry"] = unpaired
    if divergence:
        rec["reproduced_3f_uncond_share"] = (
            (divergence.get("timing") or {}).get("share_of_3e_carry_unconditional")
        )
        rec["reproduced_3f_selection_share"] = (
            (divergence.get("timing") or {}).get("share_of_3e_carry_from_selection")
        )

    damaging_c = [r for r in c_rows if r.get("damaging")]
    damaging_t = [r for r in t_rows if r.get("damaging")]
    damaging = None
    if damaging_c or damaging_t:
        damaging = decompose_punch_selection(
            damaging_c, damaging_t, unpaired_punch=unpaired,
        )

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "decomposition": decomp,
        "damaging_only": damaging,
        "reconciliation": rec,
        "lifecycle": {
            "reweighting": None if lifecycle_cmp is None else lifecycle_cmp.get("reweighting"),
            "additive_flow": None if lifecycle_cmp is None else lifecycle_cmp.get("additive_flow"),
            "reconciliation": None if lifecycle_cmp is None else lifecycle_cmp.get("reconciliation"),
        },
        "timing": None if divergence is None else divergence.get("timing"),
    }


def run_paired_selection(lobbies: int, seed: int) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    control = run_greedy_control_lifecycle(lobbies, seed)
    treatment = run_greedy_2s_treatment_lifecycle(lobbies, seed)
    greedy_c = summarize_lifecycle_arm(control)
    greedy_t = summarize_lifecycle_arm(treatment)
    life = compare_lifecycle(greedy_c, greedy_t)
    diverg = compare_divergence(control, treatment, lifecycle_cmp=life)
    cmp = compare_selection(
        control, treatment, lifecycle_cmp=life, divergence=diverg,
    )
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
