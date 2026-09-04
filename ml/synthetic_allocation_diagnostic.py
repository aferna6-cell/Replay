"""Phase 2X — observational synthetic-allocation vs within-tier survival.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. For every decisive T7–T14 hit, records each winner
starting minion's printed tier, recruit/base raw, synthetic abstract-pool
share at combat start, combat raw, board slot, golden, survived/died, and
whether it attacked before death.

Then standardizes on tier + recruit/raw and splits the leftover 2V
within-tier term (B ≈ +1.678 / hit) into extra synthetic allocation vs
residual combat-order / position.

Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.phase_2x_prereg import (
    INSTRUMENT_TURNS,
    N_DECILES,
    PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
    PHASE_2V_WITHIN_TIER_B,
    assert_seed_range_allowed,
    diagnose_phase_2x,
)
from ml.survivor_composition_diagnostic import (
    TIERS,
    SurvivorCompositionTracer,
    clamp_tier,
    compare_composition,
    decompose_gap,
    summarize_composition_arm,
    traced_body_ids,
)

METHODOLOGY_VERSION = "2x_v1"


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None:
        return None
    if abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def _quantile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(float(x) for x in xs)
    if len(s) == 1:
        return s[0]
    i = (len(s) - 1) * float(q)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    frac = i - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def largest_remainder_shares(
    weights: Sequence[int],
    pool_int: int,
) -> List[int]:
    """Observational copy of Phase 2S largest-remainder integer split."""
    if not weights:
        return []
    if int(pool_int) <= 0:
        return [0] * len(weights)
    w = [int(x) if int(x) > 0 else 1 for x in weights]
    total_w = sum(w)
    raw = [int(pool_int) * wi / total_w for wi in w]
    adds = [int(x) for x in raw]
    leftover = int(pool_int) - sum(adds)
    order = sorted(range(len(w)), key=lambda i: (-(raw[i] - adds[i]), i))
    step = 1 if leftover > 0 else -1
    for j in range(abs(leftover)):
        adds[order[j % len(order)]] += step
    return adds


def decile_edges(values: Sequence[float], n: int = N_DECILES) -> List[float]:
    """Interior quantile cuts; duplicates collapsed so bins stay ordered."""
    if not values:
        return []
    cuts = []
    for i in range(1, n):
        q = _quantile(values, i / n)
        if q is None:
            continue
        if not cuts or q > cuts[-1] + 1e-12:
            cuts.append(float(q))
    return cuts


def bin_value(value: float, edges: Sequence[float]) -> int:
    """0-based bin; last bin is open on the right."""
    v = float(value)
    for i, e in enumerate(edges):
        if v <= float(e) + 1e-12:
            return i
    return len(edges)


def classify_start_minion(body: Dict, slot: int, survived: bool) -> Dict:
    """Winner starting-body row at combat start (recruit + painted synthetic)."""
    recruit = int(body.get("recruit_raw") or 0)
    combat = int(body.get("combat_raw") or 0)
    synth = combat - recruit
    combat_share = _safe_div(float(synth), float(combat)) if combat else None
    slot_i = body.get("board_slot")
    if slot_i is None:
        slot_i = slot
    return {
        "name": str(body.get("name") or ""),
        "card_id": str(body.get("card_id") or ""),
        "body_id": str(body.get("body_id") or ""),
        "tier": clamp_tier(body.get("tier")),
        "golden": bool(body.get("golden")),
        "board_slot": int(slot_i),
        "recruit_raw": recruit,
        "synthetic_share": int(synth),
        "combat_raw": combat,
        "synthetic_share_of_combat": combat_share,
        "survived": bool(survived),
        "died": not bool(survived),
        "attacked": bool(body.get("attacked")),
        "n_attacks": int(body.get("n_attacks") or 0),
        "attacked_before_death": bool(body.get("attacked")) and not bool(survived),
    }


class SyntheticAllocationTracer(SurvivorCompositionTracer):
    """2V composition rows plus per-minion synthetic-share / attack flags."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        start = list(rec.get("start_combat_bodies") or [])
        survivors = list(rec.get("actual_survivors") or [])
        surv_ids = traced_body_ids(survivors)
        rows = [
            classify_start_minion(
                b, i, str(b.get("body_id") or "") in surv_ids
            )
            for i, b in enumerate(start)
        ]
        share_sum = int(sum(int(r["synthetic_share"]) for r in rows))
        winner_seat = fight.get("winner_seat")
        pool_field = None
        if winner_seat is not None:
            for p in env.players:
                if int(p.idx) == int(winner_seat):
                    pool_field = float(getattr(p, "abstract_pool", 0.0) or 0.0)
                    break
        # 2S ON: player.abstract_pool is the conserved pool.
        # 2S OFF: abstract_pool is unused (0); implicit pool is on-body synthetic.
        if pool_field is not None and abs(float(pool_field)) > 1e-9:
            player_pool = int(round(float(pool_field)))
        else:
            player_pool = share_sum
        expected = largest_remainder_shares(
            [int(r["recruit_raw"]) for r in rows], player_pool
        )
        rec.update({
            "start_minions": rows,
            "winner_abstract_pool_field": pool_field,
            "winner_player_pool": player_pool,
            "synthetic_shares_sum": share_sum,
            "shares_sum_to_pool": share_sum == player_pool,
            "expected_synthetic_shares": expected,
            "painted_matches_expected": (
                [int(r["synthetic_share"]) for r in rows] == expected
                if rows else True
            ),
        })


def run_allocation_arm(
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

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = SyntheticAllocationTracer(i, seed + i, arm)
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
    }


def run_greedy_control(lobbies: int, seed: int) -> Dict:
    return run_allocation_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_allocation_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _in_window(turn: int) -> bool:
    return int(turn) in INSTRUMENT_TURNS


def _hits(fights: Sequence[Dict]) -> List[Dict]:
    return [
        f for f in fights
        if _in_window(f["turn"]) and int(f.get("applied_hp_loss") or 0) > 0
    ]


def collect_start_minions(hits: Sequence[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for f in hits:
        n_hits_scale = 1.0
        for r in f.get("start_minions") or []:
            row = dict(r)
            row["winner_tavern_tier"] = int(f.get("winner_tavern_tier") or 1)
            row["_hit_weight"] = n_hits_scale
            rows.append(row)
    return rows


def _by_tier_stats(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r["tier"]) == tier]
        n = len(cell)
        surv = [r for r in cell if r.get("survived")]
        died = [r for r in cell if r.get("died")]
        synth = [float(r["synthetic_share"]) for r in cell]
        combat = [float(r["combat_raw"]) for r in cell]
        recruit = [float(r["recruit_raw"]) for r in cell]
        shares = [
            float(r["synthetic_share_of_combat"])
            for r in cell
            if r.get("synthetic_share_of_combat") is not None
        ]
        surv_tier_sum = float(tier) * len(surv)
        out[str(tier)] = {
            "n_start": n,
            "n_survived": len(surv),
            "n_died": len(died),
            "mean_per_hit": _safe_div(float(n), float(n_hits)) if n_hits else None,
            "p_survive": _safe_div(float(len(surv)), float(n)),
            "mean_synthetic_share": _mean(synth),
            "mean_recruit_raw": _mean(recruit),
            "mean_combat_raw": _mean(combat),
            "mean_synthetic_share_of_combat": _mean(shares),
            "survivor_tier_sum_contrib_per_hit": _safe_div(
                surv_tier_sum, float(n_hits)
            ) if n_hits else None,
            "p_attacked": _safe_div(
                float(sum(1 for r in cell if r.get("attacked"))), float(n)
            ),
            "p_attacked_given_died": _safe_div(
                float(sum(1 for r in died if r.get("attacked"))),
                float(len(died)),
            ),
            "p_attacked_given_survived": _safe_div(
                float(sum(1 for r in surv if r.get("attacked"))),
                float(len(surv)),
            ),
            "mean_board_slot": _mean(
                [float(r["board_slot"]) for r in cell]
            ),
            "golden_share": _safe_div(
                float(sum(1 for r in cell if r.get("golden"))), float(n)
            ),
        }
    return out


def _cross_survival(
    rows: Sequence[Dict],
    *,
    edges_by_tier: Dict[int, List[float]],
    key: str,
) -> Dict[str, Dict[str, Optional[float]]]:
    """P(survive | tier, decile) with per-tier edges of ``key``."""
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for tier in TIERS:
        edges = edges_by_tier.get(tier) or []
        n_bins = len(edges) + 1 if edges or any(
            int(r["tier"]) == tier for r in rows
        ) else 0
        row = {}
        for b in range(max(n_bins, N_DECILES)):
            cell = [
                r for r in rows
                if int(r["tier"]) == tier
                and bin_value(float(r[key]), edges) == b
            ]
            row[str(b)] = {
                "n": len(cell),
                "p_survive": _safe_div(
                    float(sum(1 for r in cell if r.get("survived"))),
                    float(len(cell)),
                ),
                "mean_key": _mean([float(r[key]) for r in cell]),
            } if cell else {"n": 0, "p_survive": None, "mean_key": None}
        table[str(tier)] = row
    return table


def _kitagawa_two(
    n_c: float,
    n_t: float,
    p_c: Optional[float],
    p_t: Optional[float],
    scale: float,
) -> Tuple[float, float, float, bool]:
    """Return (fielded/mix, rate, starting_gap, exclusive)."""
    pc = float(p_c) if p_c is not None else 0.0
    pt = float(p_t) if p_t is not None else 0.0
    gap = scale * (n_t * pt - n_c * pc)
    exclusive = n_c < 1e-12 or n_t < 1e-12
    if exclusive:
        return gap, 0.0, gap, True
    dn = n_t - n_c
    dp = pt - pc
    p_bar = 0.5 * (pc + pt)
    n_bar = 0.5 * (n_c + n_t)
    return scale * dn * p_bar, scale * n_bar * dp, gap, False


def _cond_p(surv: float, n: float) -> Optional[float]:
    return None if n < 1e-15 else float(surv) / float(n)


def _kitagawa_prob_delta(
    p_x_c: float,
    p_x_t: float,
    p_surv_c: Optional[float],
    p_surv_t: Optional[float],
) -> Tuple[float, float, bool]:
    """Split ΔP into mix(x) + rate | x. Exclusive x → all mix."""
    pc = float(p_surv_c) if p_surv_c is not None else 0.0
    pt = float(p_surv_t) if p_surv_t is not None else 0.0
    exclusive = p_x_c < 1e-15 or p_x_t < 1e-15
    if exclusive:
        return (p_x_t * pt - p_x_c * pc), 0.0, True
    mix = (p_x_t - p_x_c) * 0.5 * (pc + pt)
    rate = 0.5 * (p_x_c + p_x_t) * (pt - pc)
    return mix, rate, False


def reweight_within_tier(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    n_hits_c: int,
    n_hits_t: int,
    observed_B: Optional[float] = None,
) -> Dict:
    """Hold tier + recruit/raw fixed; split leftover B into synth vs residual.

    2V B is Σ_t t · n̄_t · ΔP(survive|t). This splits ΔP(survive|t) with
    nested Kitagawa on P(recruit-decile|t) then P(synth-decile|t,r).
    Exclusive-tier support stays in 2V A (not B). Exclusive inner cells
    assign to the outer mix (recruit-mix, then synthetic).
    """
    pooled_by_tier: Dict[int, List[float]] = {t: [] for t in TIERS}
    for r in list(control_rows) + list(treatment_rows):
        pooled_by_tier[int(r["tier"])].append(float(r["recruit_raw"]))
    recruit_edges = {t: decile_edges(vs) for t, vs in pooled_by_tier.items()}

    synth_edges: Dict[Tuple[int, int], List[float]] = {}
    for t in TIERS:
        n_r = len(recruit_edges[t]) + 1
        for rb in range(n_r):
            vs = [
                float(r["synthetic_share"])
                for r in list(control_rows) + list(treatment_rows)
                if int(r["tier"]) == t
                and bin_value(float(r["recruit_raw"]), recruit_edges[t]) == rb
            ]
            synth_edges[(t, rb)] = decile_edges(vs)

    def _arm_cells(rows: Sequence[Dict], n_hits: int):
        n_t = {t: 0.0 for t in TIERS}
        s_t = {t: 0.0 for t in TIERS}
        n_tr: Dict[Tuple[int, int], float] = defaultdict(float)
        s_tr: Dict[Tuple[int, int], float] = defaultdict(float)
        n_trs: Dict[Tuple[int, int, int], float] = defaultdict(float)
        s_trs: Dict[Tuple[int, int, int], float] = defaultdict(float)
        scale = 1.0 / float(n_hits) if n_hits else 0.0
        for r in rows:
            t = int(r["tier"])
            rb = bin_value(float(r["recruit_raw"]), recruit_edges[t])
            sb = bin_value(float(r["synthetic_share"]), synth_edges[(t, rb)])
            w = scale
            n_t[t] += w
            n_tr[(t, rb)] += w
            n_trs[(t, rb, sb)] += w
            if r.get("survived"):
                s_t[t] += w
                s_tr[(t, rb)] += w
                s_trs[(t, rb, sb)] += w
        return n_t, s_t, n_tr, s_tr, n_trs, s_trs

    c = _arm_cells(control_rows, n_hits_c)
    t = _arm_cells(treatment_rows, n_hits_t)
    n_c, s_c, n_cr, s_cr, n_crs, s_crs = c
    n_t, s_t, n_tr, s_tr, n_trs, s_trs = t

    b_direct = 0.0
    b_recruit = 0.0
    b_synth = 0.0
    b_resid = 0.0
    per_tier = {}
    for tier in TIERS:
        nc, nt = n_c[tier], n_t[tier]
        pc = _cond_p(s_c[tier], nc)
        pt = _cond_p(s_t[tier], nt)
        _mix_t, rate_t, _gap_t, excl_t = _kitagawa_two(
            nc, nt, pc, pt, float(tier)
        )
        # 2V assigns exclusive-tier support (T6) to fielded A, not B.
        if excl_t:
            per_tier[str(tier)] = {
                "exclusive_support": True,
                "n_start_control": nc,
                "n_start_treatment": nt,
                "p_survive_control": pc,
                "p_survive_treatment": pt,
                "within_tier_B": 0.0,
                "recruit_mix": 0.0,
                "synthetic_allocation": 0.0,
                "residual_position": 0.0,
            }
            continue
        n_bar = 0.5 * (nc + nt)
        b_direct += rate_t

        n_r_bins = len(recruit_edges[tier]) + 1
        rec_mix_dp = 0.0
        synth_dp = 0.0
        resid_dp = 0.0
        per_r = {}
        for rb in range(n_r_bins):
            ncr = n_cr[(tier, rb)]
            ntr = n_tr[(tier, rb)]
            p_r_c = (ncr / nc) if nc > 1e-15 else 0.0
            p_r_t = (ntr / nt) if nt > 1e-15 else 0.0
            pcr = _cond_p(s_cr[(tier, rb)], ncr)
            ptr = _cond_p(s_tr[(tier, rb)], ntr)
            mix_r, rate_r, excl_r = _kitagawa_prob_delta(
                p_r_c, p_r_t, pcr, ptr
            )
            rec_mix_dp += mix_r
            if excl_r:
                per_r[str(rb)] = {
                    "exclusive_support": True,
                    "delta_p_mix": mix_r,
                    "delta_p_rate": 0.0,
                    "synthetic_allocation_dp": 0.0,
                    "residual_position_dp": 0.0,
                }
                continue
            n_s_bins = len(synth_edges[(tier, rb)]) + 1
            s_mix_dp = 0.0
            s_rate_dp = 0.0
            for sb in range(n_s_bins):
                ncs = n_crs[(tier, rb, sb)]
                nts = n_trs[(tier, rb, sb)]
                p_s_c = (ncs / ncr) if ncr > 1e-15 else 0.0
                p_s_t = (nts / ntr) if ntr > 1e-15 else 0.0
                pcs = _cond_p(s_crs[(tier, rb, sb)], ncs)
                pts = _cond_p(s_trs[(tier, rb, sb)], nts)
                mix_s, rate_s, _excl_s = _kitagawa_prob_delta(
                    p_s_c, p_s_t, pcs, pts
                )
                s_mix_dp += mix_s
                s_rate_dp += rate_s
            p_r_bar = 0.5 * (p_r_c + p_r_t)
            synth_dp += p_r_bar * s_mix_dp
            resid_dp += p_r_bar * s_rate_dp
            per_r[str(rb)] = {
                "exclusive_support": False,
                "delta_p_mix": mix_r,
                "delta_p_rate": rate_r,
                "synthetic_allocation_dp": s_mix_dp,
                "residual_position_dp": s_rate_dp,
            }

        rec_part = float(tier) * n_bar * rec_mix_dp
        synth_part = float(tier) * n_bar * synth_dp
        resid_part = float(tier) * n_bar * resid_dp
        b_recruit += rec_part
        b_synth += synth_part
        b_resid += resid_part
        per_tier[str(tier)] = {
            "exclusive_support": False,
            "n_start_control": nc,
            "n_start_treatment": nt,
            "p_survive_control": pc,
            "p_survive_treatment": pt,
            "within_tier_B": rate_t,
            "recruit_mix": rec_part,
            "synthetic_allocation": synth_part,
            "residual_position": resid_part,
            "nested_residual": (
                rate_t - rec_part - synth_part - resid_part
            ),
            "recruit_deciles": per_r,
        }

    explained = b_recruit + b_synth + b_resid
    obs_b = float(observed_B) if observed_B is not None else b_direct

    def _share(part: float) -> Optional[float]:
        if abs(obs_b) < 1e-12:
            return None
        return float(part) / obs_b

    return {
        "method": (
            "nested_kitagawa_tier_then_recruit_decile_then_synthetic_decile"
        ),
        "n_deciles": N_DECILES,
        "within_tier_B": b_direct,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "observed_B_used_for_shares": obs_b,
        "recruit_mix": b_recruit,
        "synthetic_allocation": b_synth,
        "residual_position": b_resid,
        "explained_mix_plus_synth_plus_resid": explained,
        "residual_vs_direct_B": b_direct - explained,
        "share_of_B_recruit_mix": _share(b_recruit),
        "share_of_B_synthetic": _share(b_synth),
        "share_of_B_residual_position": _share(b_resid),
        "share_synthetic_of_standardized": _safe_div(
            b_synth, b_synth + b_resid
        ),
        "per_tier": per_tier,
    }


def summarize_allocation_arm(raw: Dict) -> Dict:
    summary = summarize_composition_arm(raw)
    hits = _hits(raw["fights"])
    rows = collect_start_minions(hits)
    n_hits = len(hits)
    by_tier = _by_tier_stats(rows, n_hits)
    summary.update({
        "n_start_minions": len(rows),
        "by_tier": by_tier,
        "mean_synthetic_share": _mean(
            [float(r["synthetic_share"]) for r in rows]
        ),
        "mean_synthetic_share_of_combat": _mean(
            [
                float(r["synthetic_share_of_combat"])
                for r in rows
                if r.get("synthetic_share_of_combat") is not None
            ]
        ),
        "p_survive": _safe_div(
            float(sum(1 for r in rows if r.get("survived"))),
            float(len(rows)),
        ),
        "p_attacked": _safe_div(
            float(sum(1 for r in rows if r.get("attacked"))),
            float(len(rows)),
        ),
        "p_attacked_given_died": _safe_div(
            float(sum(1 for r in rows if r.get("died") and r.get("attacked"))),
            float(sum(1 for r in rows if r.get("died"))),
        ),
        "n_share_pool_mismatch": sum(
            1 for f in hits if not f.get("shares_sum_to_pool", True)
        ),
        "n_expected_paint_mismatch": sum(
            1 for f in hits if not f.get("painted_matches_expected", True)
        ),
        "example_minions": rows[:8],
        "_rows": rows,
        "_n_hits": n_hits,
    })
    return summary


def compare_allocation(control: Dict, treatment: Dict) -> Dict:
    base = compare_composition(control, treatment)
    rows_c = list(control.get("_rows") or [])
    rows_t = list(treatment.get("_rows") or [])
    n_c = int(control.get("_n_hits") or control.get("n_hits") or 0)
    n_t = int(treatment.get("_n_hits") or treatment.get("n_hits") or 0)

    combat_edges = {}
    recruit_edges = {}
    for tier in TIERS:
        combat_edges[tier] = decile_edges(
            [float(r["combat_raw"]) for r in rows_c + rows_t
             if int(r["tier"]) == tier]
        )
        recruit_edges[tier] = decile_edges(
            [float(r["recruit_raw"]) for r in rows_c + rows_t
             if int(r["tier"]) == tier]
        )

    surv_combat_c = _cross_survival(rows_c, edges_by_tier=combat_edges, key="combat_raw")
    surv_combat_t = _cross_survival(rows_t, edges_by_tier=combat_edges, key="combat_raw")

    decomp = base.get("decomposition") or decompose_gap(control, treatment)
    b_obs = decomp.get("within_tier_survival_B")
    if b_obs is None:
        b_obs = PHASE_2V_WITHIN_TIER_B
    reweight = reweight_within_tier(
        rows_c, rows_t, n_hits_c=n_c, n_hits_t=n_t, observed_B=b_obs
    )

    def _tier_delta(key: str) -> Dict[str, Optional[float]]:
        out = {}
        for tier in TIERS:
            k = str(tier)
            c = (control.get("by_tier") or {}).get(k) or {}
            t = (treatment.get("by_tier") or {}).get(k) or {}
            a, b = c.get(key), t.get(key)
            out[k] = None if a is None or b is None else float(b) - float(a)
        return out

    base["by_tier"] = {
        "control": control.get("by_tier"),
        "treatment": treatment.get("by_tier"),
        "delta_mean_synthetic_share": _tier_delta("mean_synthetic_share"),
        "delta_mean_synthetic_share_of_combat": _tier_delta(
            "mean_synthetic_share_of_combat"
        ),
        "delta_p_survive": _tier_delta("p_survive"),
        "delta_survivor_tier_sum_contrib": _tier_delta(
            "survivor_tier_sum_contrib_per_hit"
        ),
    }
    base["survival_by_tier_combat_decile"] = {
        "edges": {str(t): combat_edges[t] for t in TIERS},
        "control": surv_combat_c,
        "treatment": surv_combat_t,
    }
    base["survival_by_tier_recruit_decile"] = {
        "edges": {str(t): recruit_edges[t] for t in TIERS},
        "control": _cross_survival(
            rows_c, edges_by_tier=recruit_edges, key="recruit_raw"
        ),
        "treatment": _cross_survival(
            rows_t, edges_by_tier=recruit_edges, key="recruit_raw"
        ),
    }
    base["reweighting"] = reweight
    rec = dict(base.get("reconciliation") or {})
    rec.update({
        "share_pool_mismatch_control": control.get("n_share_pool_mismatch"),
        "share_pool_mismatch_treatment": treatment.get("n_share_pool_mismatch"),
        "expected_paint_mismatch_control": control.get(
            "n_expected_paint_mismatch"
        ),
        "expected_paint_mismatch_treatment": treatment.get(
            "n_expected_paint_mismatch"
        ),
        "phase_2v_survivor_tier_sum_delta": PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "reproduced_within_tier_B": decomp.get("within_tier_survival_B"),
        "reweight_direct_B": reweight.get("within_tier_B"),
    })
    base["reconciliation"] = rec
    base["example_minions"] = {
        "control": control.get("example_minions") or [],
        "treatment": treatment.get("example_minions") or [],
    }
    return base


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_2x(comparison, non_evaluative=non_evaluative)
