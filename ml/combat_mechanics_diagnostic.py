"""Phase 2Z — observational targeting / cursor / represented-DR split.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. Holds the 2Y leftover cells (tier + recruit/raw +
synth share + slot bin + teammate-raw), then splits leftover survival into
taunt-forced vs open targeting, attack-cursor / initiative, faithfully
represented generated-body / deathrattle effects, marked unsupported-effect
coverage, and still unexplained.

Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.
Unsupported mechanics are tagged, never approximated.
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
from ml.phase_2z_prereg import (
    N_CURSOR_BINS,
    N_DECILES,
    N_GEN_BINS,
    N_TARGET_BINS,
    N_TEAM_BINS,
    N_UNSUP_BINS,
    PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    PHASE_2Y_UNEXPLAINED,
    SLOT_BIN_CAP,
    assert_seed_range_allowed,
    cursor_bin,
    diagnose_phase_2z,
    gen_bin,
    slot_bin,
    target_bin,
    unsupported_bin,
)
from ml.position_order_diagnostic import PositionOrderTracer
from ml.survivor_composition_diagnostic import TIERS
from ml.synthetic_allocation_diagnostic import (
    _cond_p,
    _cross_survival,
    _hits,
    _kitagawa_prob_delta,
    _kitagawa_two,
    _safe_div,
    bin_value,
    compare_allocation,
    decile_edges,
    reweight_within_tier,
    summarize_allocation_arm,
)

METHODOLOGY_VERSION = "2z_v1"

_EXTRA_KEYS = (
    "n_targeted_forced",
    "n_targeted_open",
    "taunt_forced_target",
    "open_target",
    "death_cause",
    "killed_by_body_id",
    "last_attacker_id",
    "start_health",
    "end_health",
    "spawned_represented",
    "cursor_wrapped_before_first",
    "side_first",
    "effect_status",
    "has_unsupported_effect",
    "has_represented_generated_effect",
)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


class CombatMechanicsTracer(PositionOrderTracer):
    """2Y rows plus targeting / cursor / generated-body / unsupported tags."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        start_combat = list(rec.get("start_combat_bodies") or [])
        by_id = {str(b.get("body_id") or ""): b for b in start_combat}
        created = list(rec.get("created_winner") or [])
        n_gen_rep = int(fight.get("n_board_generated_represented") or 0)
        if n_gen_rep <= 0:
            n_gen_rep = sum(1 for c in created if c.get("represented_generated"))
        counts = dict(fight.get("event_counts") or {})
        if not counts:
            counts = dict(rec.get("event_counts") or {})
        rows = list(rec.get("start_minions") or [])
        for r in rows:
            src = by_id.get(str(r.get("body_id") or ""), {})
            for k in _EXTRA_KEYS:
                if k in src:
                    r[k] = src[k]
                elif k not in r:
                    r[k] = src.get(k)
            r["n_board_generated_represented"] = n_gen_rep
            r["target_bin"] = target_bin(r)
            r["cursor_bin"] = cursor_bin(r)
            r["gen_bin"] = gen_bin(r)
            r["unsupported_bin"] = unsupported_bin(r)
            r["side_first"] = bool(src.get("side_first", r.get("side_first")))
        rec["n_board_generated_represented"] = n_gen_rep
        rec["event_counts"] = counts
        rec["side_first"] = fight.get("side_first")
        rec["event_counts_ok"] = _event_counts_ok(counts)


def _event_counts_ok(counts: Dict) -> bool:
    if not counts:
        return True
    flags = (
        "attacks_reconcile",
        "targets_reconcile",
        "forced_open_reconcile",
        "created_reconcile",
        "deaths_reconcile",
    )
    return all(bool(counts.get(k, True)) for k in flags)


def run_mechanics_arm(
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
                tracer = CombatMechanicsTracer(i, seed + i, arm)
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
    return run_mechanics_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_mechanics_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def collect_mechanics_minions(hits: Sequence[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for f in hits:
        start = list(f.get("start_minions") or [])
        n_gen = int(f.get("n_board_generated_represented") or 0)
        for r in start:
            row = dict(r)
            row["n_board_generated_represented"] = int(
                row.get("n_board_generated_represented") or n_gen
            )
            row["target_bin"] = target_bin(row)
            row["cursor_bin"] = cursor_bin(row)
            row["gen_bin"] = gen_bin(row)
            row["unsupported_bin"] = unsupported_bin(row)
            row["slot_bin"] = slot_bin(row.get("board_slot"))
            row["winner_tavern_tier"] = int(f.get("winner_tavern_tier") or 1)
            rows.append(row)
    return rows


def _by_tier_mechanics(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r["tier"]) == tier]
        n = len(cell)
        out[str(tier)] = {
            "n_start": n,
            "p_survive": _safe_div(
                float(sum(1 for r in cell if r.get("survived"))), float(n)
            ),
            "p_taunt": _safe_div(
                float(sum(1 for r in cell if r.get("taunt"))), float(n)
            ),
            "p_taunt_forced_target": _safe_div(
                float(sum(1 for r in cell if r.get("taunt_forced_target")
                          or int(r.get("n_targeted_forced") or 0) > 0)),
                float(n),
            ),
            "p_open_target": _safe_div(
                float(sum(1 for r in cell if int(r.get("n_targeted_open") or 0) > 0)),
                float(n),
            ),
            "p_side_first": _safe_div(
                float(sum(1 for r in cell if r.get("side_first"))), float(n)
            ),
            "p_cursor_wrap_before_first": _safe_div(
                float(sum(1 for r in cell if r.get("cursor_wrapped_before_first"))),
                float(n),
            ),
            "p_represented_generated": _safe_div(
                float(sum(1 for r in cell if gen_bin(r) == 1)), float(n)
            ),
            "p_unsupported_effect": _safe_div(
                float(sum(1 for r in cell if unsupported_bin(r) == 1)), float(n)
            ),
            "mean_n_targeted_forced": _mean(
                [float(r.get("n_targeted_forced") or 0) for r in cell]
            ),
            "mean_n_targeted_open": _mean(
                [float(r.get("n_targeted_open") or 0) for r in cell]
            ),
            "mean_first_attack_index": _mean(
                [
                    float(r["first_attack_index"])
                    for r in cell
                    if r.get("first_attack_index") is not None
                ]
            ),
            "mean_per_hit": _safe_div(float(n), float(n_hits)) if n_hits else None,
        }
    return out


def reweight_combat_mechanics(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    n_hits_c: int,
    n_hits_t: int,
    observed_B: Optional[float] = None,
    observed_residual: Optional[float] = None,
    observed_leftover: Optional[float] = None,
) -> Dict:
    """Hold 2Y leftover cells, then split C into targeting / cursor / DR / gap.

    Nested Kitagawa:

        hold P(recruit-raw | tier) then P(synth | …) then P(slot) then P(teammate)
            ↓
        hold P(target_bin | 2Y cells)     →  (A) targeting / taunt
            ↓
        hold P(cursor_bin | …)            →  (B) attack-cursor / initiative
            ↓
        hold P(gen_bin | …)               →  (C) represented generated / DR
            ↓
        hold P(unsupported_bin | …)       →  (D) unsupported coverage (marked)
            ↓
        leftover P(survive | all)         →  (E) still unexplained
    """
    pooled = list(control_rows) + list(treatment_rows)
    recruit_edges: Dict[int, List[float]] = {}
    for t in TIERS:
        recruit_edges[t] = decile_edges(
            [float(r["recruit_raw"]) for r in pooled if int(r["tier"]) == t]
        )

    synth_edges: Dict[Tuple[int, int], List[float]] = {}
    for t in TIERS:
        n_r = len(recruit_edges[t]) + 1
        for rb in range(n_r):
            vs = [
                float(r["synthetic_share"])
                for r in pooled
                if int(r["tier"]) == t
                and bin_value(float(r["recruit_raw"]), recruit_edges[t]) == rb
            ]
            synth_edges[(t, rb)] = decile_edges(vs)

    team_edges: Dict[int, List[float]] = {}
    for t in TIERS:
        team_edges[t] = decile_edges(
            [float(r.get("teammate_combat_raw") or 0)
             for r in pooled if int(r["tier"]) == t],
            n=N_TEAM_BINS,
        )

    def _arm_cells(rows: Sequence[Dict], n_hits: int):
        n_t = {t: 0.0 for t in TIERS}
        s_t = {t: 0.0 for t in TIERS}
        n_tr: Dict[Tuple[int, int], float] = defaultdict(float)
        s_tr: Dict[Tuple[int, int], float] = defaultdict(float)
        n_trs: Dict[Tuple[int, int, int], float] = defaultdict(float)
        s_trs: Dict[Tuple[int, int, int], float] = defaultdict(float)
        n_trsk: Dict[Tuple[int, int, int, int], float] = defaultdict(float)
        s_trsk: Dict[Tuple[int, int, int, int], float] = defaultdict(float)
        n_trskm: Dict[Tuple[int, int, int, int, int], float] = defaultdict(float)
        s_trskm: Dict[Tuple[int, int, int, int, int], float] = defaultdict(float)
        n_tgt: Dict[Tuple, float] = defaultdict(float)
        s_tgt: Dict[Tuple, float] = defaultdict(float)
        n_cur: Dict[Tuple, float] = defaultdict(float)
        s_cur: Dict[Tuple, float] = defaultdict(float)
        n_gen: Dict[Tuple, float] = defaultdict(float)
        s_gen: Dict[Tuple, float] = defaultdict(float)
        n_uns: Dict[Tuple, float] = defaultdict(float)
        s_uns: Dict[Tuple, float] = defaultdict(float)
        scale = 1.0 / float(n_hits) if n_hits else 0.0
        for r in rows:
            t = int(r["tier"])
            rb = bin_value(float(r["recruit_raw"]), recruit_edges[t])
            sb = bin_value(float(r["synthetic_share"]), synth_edges[(t, rb)])
            kb = slot_bin(r.get("board_slot"))
            mb = bin_value(
                float(r.get("teammate_combat_raw") or 0), team_edges[t]
            )
            tb = target_bin(r)
            cb = cursor_bin(r)
            gb = gen_bin(r)
            ub = unsupported_bin(r)
            w = scale
            n_t[t] += w
            n_tr[(t, rb)] += w
            n_trs[(t, rb, sb)] += w
            n_trsk[(t, rb, sb, kb)] += w
            n_trskm[(t, rb, sb, kb, mb)] += w
            n_tgt[(t, rb, sb, kb, mb, tb)] += w
            n_cur[(t, rb, sb, kb, mb, tb, cb)] += w
            n_gen[(t, rb, sb, kb, mb, tb, cb, gb)] += w
            n_uns[(t, rb, sb, kb, mb, tb, cb, gb, ub)] += w
            if r.get("survived"):
                s_t[t] += w
                s_tr[(t, rb)] += w
                s_trs[(t, rb, sb)] += w
                s_trsk[(t, rb, sb, kb)] += w
                s_trskm[(t, rb, sb, kb, mb)] += w
                s_tgt[(t, rb, sb, kb, mb, tb)] += w
                s_cur[(t, rb, sb, kb, mb, tb, cb)] += w
                s_gen[(t, rb, sb, kb, mb, tb, cb, gb)] += w
                s_uns[(t, rb, sb, kb, mb, tb, cb, gb, ub)] += w
        return (
            n_t, s_t, n_tr, s_tr, n_trs, s_trs, n_trsk, s_trsk, n_trskm, s_trskm,
            n_tgt, s_tgt, n_cur, s_cur, n_gen, s_gen, n_uns, s_uns,
        )

    c = _arm_cells(control_rows, n_hits_c)
    t = _arm_cells(treatment_rows, n_hits_t)
    (n_c, s_c, n_cr, s_cr, n_crs, s_crs,
     n_crsk, s_crsk, n_crskm, s_crskm,
     n_ct, s_ct, n_cc, s_cc, n_cg, s_cg, n_cu, s_cu) = c
    (n_t, s_t, n_tr, s_tr, n_trs, s_trs,
     n_trsk, s_trsk, n_trskm, s_trskm,
     n_tt, s_tt, n_tc, s_tc, n_tg, s_tg, n_tu, s_tu) = t

    b_direct = 0.0
    b_recruit = 0.0
    b_synth = 0.0
    b_slot = 0.0
    b_team = 0.0
    b_target = 0.0
    b_cursor = 0.0
    b_gen = 0.0
    b_unsup = 0.0
    b_unexpl = 0.0
    per_tier = {}
    for tier in TIERS:
        nc, nt = n_c[tier], n_t[tier]
        pc = _cond_p(s_c[tier], nc)
        pt = _cond_p(s_t[tier], nt)
        _mix_t, rate_t, _gap_t, excl_t = _kitagawa_two(
            nc, nt, pc, pt, float(tier)
        )
        zero = {
            "exclusive_support": True,
            "n_start_control": nc,
            "n_start_treatment": nt,
            "p_survive_control": pc,
            "p_survive_treatment": pt,
            "within_tier_B": 0.0,
            "recruit_mix": 0.0,
            "synthetic_allocation": 0.0,
            "slot_opportunity": 0.0,
            "teammate_protection": 0.0,
            "targeting_taunt": 0.0,
            "attack_cursor": 0.0,
            "represented_generated": 0.0,
            "unsupported_coverage": 0.0,
            "still_unexplained": 0.0,
            "phase_2y_unexplained_hat": 0.0,
        }
        if excl_t:
            per_tier[str(tier)] = zero
            continue
        n_bar = 0.5 * (nc + nt)
        b_direct += rate_t

        n_r_bins = len(recruit_edges[tier]) + 1
        rec_mix_dp = 0.0
        synth_dp = 0.0
        slot_dp = 0.0
        team_dp = 0.0
        tgt_dp = 0.0
        cur_dp = 0.0
        gen_dp = 0.0
        uns_dp = 0.0
        unexpl_dp = 0.0
        for rb in range(n_r_bins):
            ncr = n_cr[(tier, rb)]
            ntr = n_tr[(tier, rb)]
            p_r_c = (ncr / nc) if nc > 1e-15 else 0.0
            p_r_t = (ntr / nt) if nt > 1e-15 else 0.0
            pcr = _cond_p(s_cr[(tier, rb)], ncr)
            ptr = _cond_p(s_tr[(tier, rb)], ntr)
            mix_r, _rate_r, excl_r = _kitagawa_prob_delta(
                p_r_c, p_r_t, pcr, ptr
            )
            rec_mix_dp += mix_r
            if excl_r:
                continue
            n_s_bins = len(synth_edges[(tier, rb)]) + 1
            s_mix_dp = 0.0
            k_mix_dp = 0.0
            m_mix_dp = 0.0
            tgt_mix_dp = 0.0
            cur_mix_dp = 0.0
            gen_mix_dp = 0.0
            uns_mix_dp = 0.0
            leftover_dp = 0.0
            for sb in range(n_s_bins):
                ncs = n_crs[(tier, rb, sb)]
                nts = n_trs[(tier, rb, sb)]
                p_s_c = (ncs / ncr) if ncr > 1e-15 else 0.0
                p_s_t = (nts / ntr) if ntr > 1e-15 else 0.0
                pcs = _cond_p(s_crs[(tier, rb, sb)], ncs)
                pts = _cond_p(s_trs[(tier, rb, sb)], nts)
                mix_s, _rate_s, excl_s = _kitagawa_prob_delta(
                    p_s_c, p_s_t, pcs, pts
                )
                s_mix_dp += mix_s
                if excl_s:
                    continue
                k_inner_mix = 0.0
                m_inner_mix = 0.0
                tgt_inner = 0.0
                cur_inner = 0.0
                gen_inner = 0.0
                uns_inner = 0.0
                left_inner = 0.0
                for kb in range(SLOT_BIN_CAP + 1):
                    nck = n_crsk[(tier, rb, sb, kb)]
                    ntk = n_trsk[(tier, rb, sb, kb)]
                    p_k_c = (nck / ncs) if ncs > 1e-15 else 0.0
                    p_k_t = (ntk / nts) if nts > 1e-15 else 0.0
                    pck = _cond_p(s_crsk[(tier, rb, sb, kb)], nck)
                    ptk = _cond_p(s_trsk[(tier, rb, sb, kb)], ntk)
                    mix_k, _rate_k, excl_k = _kitagawa_prob_delta(
                        p_k_c, p_k_t, pck, ptk
                    )
                    k_inner_mix += mix_k
                    if excl_k:
                        continue
                    n_m_bins = len(team_edges[tier]) + 1
                    tm_mix = 0.0
                    tb_mix = 0.0
                    cb_mix = 0.0
                    gb_mix = 0.0
                    ub_mix = 0.0
                    ub_rate = 0.0
                    for mb in range(n_m_bins):
                        ncm = n_crskm[(tier, rb, sb, kb, mb)]
                        ntm = n_trskm[(tier, rb, sb, kb, mb)]
                        p_m_c = (ncm / nck) if nck > 1e-15 else 0.0
                        p_m_t = (ntm / ntk) if ntk > 1e-15 else 0.0
                        pcm = _cond_p(s_crskm[(tier, rb, sb, kb, mb)], ncm)
                        ptm = _cond_p(s_trskm[(tier, rb, sb, kb, mb)], ntm)
                        mix_m, _rate_m, excl_m = _kitagawa_prob_delta(
                            p_m_c, p_m_t, pcm, ptm
                        )
                        tm_mix += mix_m
                        if excl_m:
                            continue
                        t_mix = 0.0
                        c_mix_i = 0.0
                        g_mix_i = 0.0
                        u_mix_i = 0.0
                        u_rate_i = 0.0
                        for tb in range(N_TARGET_BINS):
                            nctb = n_ct[(tier, rb, sb, kb, mb, tb)]
                            nttb = n_tt[(tier, rb, sb, kb, mb, tb)]
                            p_tb_c = (nctb / ncm) if ncm > 1e-15 else 0.0
                            p_tb_t = (nttb / ntm) if ntm > 1e-15 else 0.0
                            pctb = _cond_p(s_ct[(tier, rb, sb, kb, mb, tb)], nctb)
                            pttb = _cond_p(s_tt[(tier, rb, sb, kb, mb, tb)], nttb)
                            mix_tb, _rtb, excl_tb = _kitagawa_prob_delta(
                                p_tb_c, p_tb_t, pctb, pttb
                            )
                            t_mix += mix_tb
                            if excl_tb:
                                continue
                            c_mix = 0.0
                            g_mix = 0.0
                            u_mix = 0.0
                            u_rate = 0.0
                            for cb in range(N_CURSOR_BINS):
                                ncc = n_cc[(tier, rb, sb, kb, mb, tb, cb)]
                                ntc = n_tc[(tier, rb, sb, kb, mb, tb, cb)]
                                p_cb_c = (ncc / nctb) if nctb > 1e-15 else 0.0
                                p_cb_t = (ntc / nttb) if nttb > 1e-15 else 0.0
                                pcc = _cond_p(
                                    s_cc[(tier, rb, sb, kb, mb, tb, cb)], ncc
                                )
                                ptc_s = _cond_p(
                                    s_tc[(tier, rb, sb, kb, mb, tb, cb)], ntc
                                )
                                mix_cb, _rcb, excl_cb = _kitagawa_prob_delta(
                                    p_cb_c, p_cb_t, pcc, ptc_s
                                )
                                c_mix += mix_cb
                                if excl_cb:
                                    continue
                                g_mix_c = 0.0
                                u_mix_c = 0.0
                                u_rate_c = 0.0
                                for gb in range(N_GEN_BINS):
                                    ncg = n_cg[(tier, rb, sb, kb, mb, tb, cb, gb)]
                                    ntg = n_tg[(tier, rb, sb, kb, mb, tb, cb, gb)]
                                    p_gb_c = (ncg / ncc) if ncc > 1e-15 else 0.0
                                    p_gb_t = (ntg / ntc) if ntc > 1e-15 else 0.0
                                    pcg = _cond_p(
                                        s_cg[(tier, rb, sb, kb, mb, tb, cb, gb)],
                                        ncg,
                                    )
                                    ptg = _cond_p(
                                        s_tg[(tier, rb, sb, kb, mb, tb, cb, gb)],
                                        ntg,
                                    )
                                    mix_gb, _rgb, excl_gb = _kitagawa_prob_delta(
                                        p_gb_c, p_gb_t, pcg, ptg
                                    )
                                    g_mix_c += mix_gb
                                    if excl_gb:
                                        continue
                                    u_mix_g = 0.0
                                    u_rate_g = 0.0
                                    for ub in range(N_UNSUP_BINS):
                                        ncu = n_cu[(
                                            tier, rb, sb, kb, mb, tb, cb, gb, ub
                                        )]
                                        ntu = n_tu[(
                                            tier, rb, sb, kb, mb, tb, cb, gb, ub
                                        )]
                                        p_ub_c = (ncu / ncg) if ncg > 1e-15 else 0.0
                                        p_ub_t = (ntu / ntg) if ntg > 1e-15 else 0.0
                                        pcu = _cond_p(
                                            s_cu[(
                                                tier, rb, sb, kb, mb, tb, cb, gb, ub
                                            )],
                                            ncu,
                                        )
                                        ptu = _cond_p(
                                            s_tu[(
                                                tier, rb, sb, kb, mb, tb, cb, gb, ub
                                            )],
                                            ntu,
                                        )
                                        mix_ub, rate_ub, _exu = _kitagawa_prob_delta(
                                            p_ub_c, p_ub_t, pcu, ptu
                                        )
                                        u_mix_g += mix_ub
                                        u_rate_g += rate_ub
                                    p_gb_bar = 0.5 * (p_gb_c + p_gb_t)
                                    u_mix_c += p_gb_bar * u_mix_g
                                    u_rate_c += p_gb_bar * u_rate_g
                                p_cb_bar = 0.5 * (p_cb_c + p_cb_t)
                                g_mix += p_cb_bar * g_mix_c
                                u_mix += p_cb_bar * u_mix_c
                                u_rate += p_cb_bar * u_rate_c
                            p_tb_bar = 0.5 * (p_tb_c + p_tb_t)
                            c_mix_i += p_tb_bar * c_mix
                            g_mix_i += p_tb_bar * g_mix
                            u_mix_i += p_tb_bar * u_mix
                            u_rate_i += p_tb_bar * u_rate
                        p_m_bar = 0.5 * (p_m_c + p_m_t)
                        tb_mix += p_m_bar * t_mix
                        cb_mix += p_m_bar * c_mix_i
                        gb_mix += p_m_bar * g_mix_i
                        ub_mix += p_m_bar * u_mix_i
                        ub_rate += p_m_bar * u_rate_i
                    p_k_bar = 0.5 * (p_k_c + p_k_t)
                    m_inner_mix += p_k_bar * tm_mix
                    tgt_inner += p_k_bar * tb_mix
                    cur_inner += p_k_bar * cb_mix
                    gen_inner += p_k_bar * gb_mix
                    uns_inner += p_k_bar * ub_mix
                    left_inner += p_k_bar * ub_rate
                p_s_bar = 0.5 * (p_s_c + p_s_t)
                k_mix_dp += p_s_bar * k_inner_mix
                m_mix_dp += p_s_bar * m_inner_mix
                tgt_mix_dp += p_s_bar * tgt_inner
                cur_mix_dp += p_s_bar * cur_inner
                gen_mix_dp += p_s_bar * gen_inner
                uns_mix_dp += p_s_bar * uns_inner
                leftover_dp += p_s_bar * left_inner
            p_r_bar = 0.5 * (p_r_c + p_r_t)
            synth_dp += p_r_bar * s_mix_dp
            slot_dp += p_r_bar * k_mix_dp
            team_dp += p_r_bar * m_mix_dp
            tgt_dp += p_r_bar * tgt_mix_dp
            cur_dp += p_r_bar * cur_mix_dp
            gen_dp += p_r_bar * gen_mix_dp
            uns_dp += p_r_bar * uns_mix_dp
            unexpl_dp += p_r_bar * leftover_dp

        rec_part = float(tier) * n_bar * rec_mix_dp
        synth_part = float(tier) * n_bar * synth_dp
        slot_part = float(tier) * n_bar * slot_dp
        team_part = float(tier) * n_bar * team_dp
        tgt_part = float(tier) * n_bar * tgt_dp
        cur_part = float(tier) * n_bar * cur_dp
        gen_part = float(tier) * n_bar * gen_dp
        uns_part = float(tier) * n_bar * uns_dp
        unexpl_part = float(tier) * n_bar * unexpl_dp
        b_recruit += rec_part
        b_synth += synth_part
        b_slot += slot_part
        b_team += team_part
        b_target += tgt_part
        b_cursor += cur_part
        b_gen += gen_part
        b_unsup += uns_part
        b_unexpl += unexpl_part
        leftover_hat = tgt_part + cur_part + gen_part + uns_part + unexpl_part
        per_tier[str(tier)] = {
            "exclusive_support": False,
            "n_start_control": nc,
            "n_start_treatment": nt,
            "p_survive_control": pc,
            "p_survive_treatment": pt,
            "within_tier_B": rate_t,
            "recruit_mix": rec_part,
            "synthetic_allocation": synth_part,
            "slot_opportunity": slot_part,
            "teammate_protection": team_part,
            "targeting_taunt": tgt_part,
            "attack_cursor": cur_part,
            "represented_generated": gen_part,
            "unsupported_coverage": uns_part,
            "still_unexplained": unexpl_part,
            "phase_2y_unexplained_hat": leftover_hat,
            "nested_residual": (
                rate_t - rec_part - synth_part - slot_part - team_part
                - leftover_hat
            ),
        }

    explained = (
        b_recruit + b_synth + b_slot + b_team
        + b_target + b_cursor + b_gen + b_unsup + b_unexpl
    )
    leftover_hat = b_target + b_cursor + b_gen + b_unsup + b_unexpl
    resid_hat = b_slot + b_team + leftover_hat
    obs_b = float(observed_B) if observed_B is not None else b_direct
    obs_r = (
        float(observed_residual)
        if observed_residual is not None
        else resid_hat
    )
    obs_c = (
        float(observed_leftover)
        if observed_leftover is not None
        else leftover_hat
    )

    def _share_b(part: float) -> Optional[float]:
        if abs(obs_b) < 1e-12:
            return None
        return float(part) / obs_b

    def _share_r(part: float) -> Optional[float]:
        if abs(obs_r) < 1e-12:
            return None
        return float(part) / obs_r

    def _share_c(part: float) -> Optional[float]:
        if abs(obs_c) < 1e-12:
            return None
        return float(part) / obs_c

    return {
        "method": (
            "nested_kitagawa_2y_cells_then_target_cursor_gen_unsupported"
        ),
        "n_deciles": N_DECILES,
        "n_team_bins": N_TEAM_BINS,
        "slot_bin_cap": SLOT_BIN_CAP,
        "n_target_bins": N_TARGET_BINS,
        "n_cursor_bins": N_CURSOR_BINS,
        "n_gen_bins": N_GEN_BINS,
        "n_unsup_bins": N_UNSUP_BINS,
        "within_tier_B": b_direct,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "observed_B_used_for_shares": obs_b,
        "observed_residual_used_for_shares": obs_r,
        "observed_leftover_used_for_shares": obs_c,
        "recruit_mix": b_recruit,
        "synthetic_allocation": b_synth,
        "slot_opportunity": b_slot,
        "teammate_protection": b_team,
        "targeting_taunt": b_target,
        "attack_cursor": b_cursor,
        "represented_generated": b_gen,
        "unsupported_coverage": b_unsup,
        "still_unexplained": b_unexpl,
        "phase_2y_unexplained_hat": leftover_hat,
        "residual_position_hat": resid_hat,
        "explained_all_parts": explained,
        "residual_vs_direct_B": b_direct - explained,
        "share_of_B_recruit_mix": _share_b(b_recruit),
        "share_of_B_synthetic": _share_b(b_synth),
        "share_of_leftover_targeting_taunt": _share_c(b_target),
        "share_of_leftover_attack_cursor": _share_c(b_cursor),
        "share_of_leftover_represented_generated": _share_c(b_gen),
        "share_of_leftover_unsupported_coverage": _share_c(b_unsup),
        "share_of_leftover_still_unexplained": _share_c(b_unexpl),
        "share_of_residual_targeting_taunt": _share_r(b_target),
        "share_of_residual_attack_cursor": _share_r(b_cursor),
        "share_of_residual_represented_generated": _share_r(b_gen),
        "share_of_residual_unsupported_coverage": _share_r(b_unsup),
        "share_of_residual_still_unexplained": _share_r(b_unexpl),
        "per_tier": per_tier,
    }


def _slim_example(row: Dict) -> Dict:
    keep = {
        "name", "tier", "survived", "died", "taunt", "board_slot", "slot_bin",
        "n_attacks", "first_attack_index", "n_targeted", "n_targeted_forced",
        "n_targeted_open", "death_cause", "start_health", "end_health",
        "effect_status", "has_unsupported_effect",
        "has_represented_generated_effect", "spawned_represented",
        "cursor_wrapped_before_first", "side_first", "target_bin",
        "cursor_bin", "gen_bin", "unsupported_bin", "teammate_combat_raw",
        "recruit_raw", "synthetic_share",
    }
    return {k: row.get(k) for k in keep}


def summarize_mechanics_arm(raw: Dict) -> Dict:
    summary = summarize_allocation_arm(raw)
    hits = _hits(raw["fights"])
    rows = collect_mechanics_minions(hits)
    n_hits = len(hits)
    by_tier = _by_tier_mechanics(rows, n_hits)
    n_mismatch = 0
    n_counted = 0
    death_causes: Dict[str, int] = defaultdict(int)
    for f in hits:
        counts = f.get("event_counts") or {}
        n_counted += 1
        if not _event_counts_ok(counts):
            n_mismatch += 1
        for r in f.get("start_minions") or []:
            cause = r.get("death_cause")
            if cause:
                death_causes[str(cause)] += 1
            elif r.get("died"):
                death_causes["unrecorded"] += 1
    summary.update({
        "n_start_minions": len(rows),
        "by_tier_mechanics": by_tier,
        "p_taunt": _safe_div(
            float(sum(1 for r in rows if r.get("taunt"))), float(len(rows))
        ),
        "p_taunt_forced_target": _safe_div(
            float(sum(1 for r in rows if int(r.get("n_targeted_forced") or 0) > 0)),
            float(len(rows)),
        ),
        "p_open_target": _safe_div(
            float(sum(1 for r in rows if int(r.get("n_targeted_open") or 0) > 0)),
            float(len(rows)),
        ),
        "p_side_first": _safe_div(
            float(sum(1 for r in rows if r.get("side_first"))), float(len(rows))
        ),
        "p_represented_generated": _safe_div(
            float(sum(1 for r in rows if gen_bin(r) == 1)), float(len(rows))
        ),
        "p_unsupported_effect": _safe_div(
            float(sum(1 for r in rows if unsupported_bin(r) == 1)),
            float(len(rows)),
        ),
        "death_causes": dict(death_causes),
        "event_count_mismatches": n_mismatch,
        "n_hits_counted": n_counted,
        "example_minions": [_slim_example(r) for r in rows[:8]],
        "_rows": rows,
        "_n_hits": n_hits,
    })
    return summary


def compare_mechanics(control: Dict, treatment: Dict) -> Dict:
    base = compare_allocation(control, treatment)
    rows_c = list(control.get("_rows") or [])
    rows_t = list(treatment.get("_rows") or [])
    n_c = int(control.get("_n_hits") or control.get("n_hits") or 0)
    n_t = int(treatment.get("_n_hits") or treatment.get("n_hits") or 0)

    decomp = base.get("decomposition") or {}
    b_obs = decomp.get("within_tier_survival_B")
    if b_obs is None:
        b_obs = PHASE_2V_WITHIN_TIER_B

    two_x = reweight_within_tier(
        rows_c, rows_t, n_hits_c=n_c, n_hits_t=n_t, observed_B=b_obs
    )
    resid_obs = two_x.get("residual_position")
    if resid_obs is None:
        resid_obs = PHASE_2X_RESIDUAL_POSITION

    reweight = reweight_combat_mechanics(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover=PHASE_2Y_UNEXPLAINED,
    )

    def _tier_delta(table_c: Dict, table_t: Dict, key: str) -> Dict[str, Optional[float]]:
        out = {}
        for tier in TIERS:
            k = str(tier)
            a = (table_c or {}).get(k, {}).get(key)
            b = (table_t or {}).get(k, {}).get(key)
            out[k] = None if a is None or b is None else float(b) - float(a)
        return out

    mech_c = control.get("by_tier_mechanics") or {}
    mech_t = treatment.get("by_tier_mechanics") or {}
    base["by_tier_mechanics"] = {
        "control": mech_c,
        "treatment": mech_t,
        "delta_p_taunt_forced_target": _tier_delta(
            mech_c, mech_t, "p_taunt_forced_target"
        ),
        "delta_p_open_target": _tier_delta(mech_c, mech_t, "p_open_target"),
        "delta_p_side_first": _tier_delta(mech_c, mech_t, "p_side_first"),
        "delta_p_represented_generated": _tier_delta(
            mech_c, mech_t, "p_represented_generated"
        ),
        "delta_p_unsupported_effect": _tier_delta(
            mech_c, mech_t, "p_unsupported_effect"
        ),
        "delta_p_survive": _tier_delta(mech_c, mech_t, "p_survive"),
    }

    tgt_edges = {t: [0.5, 1.5] for t in TIERS}
    cur_edges = {t: [0.5, 1.5] for t in TIERS}
    gen_edges = {t: [0.5] for t in TIERS}
    uns_edges = {t: [0.5] for t in TIERS}
    base["survival_by_tier_target_bin"] = {
        "edges": {str(t): tgt_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=tgt_edges, key="target_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=tgt_edges, key="target_bin"),
    }
    base["survival_by_tier_cursor_bin"] = {
        "edges": {str(t): cur_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=cur_edges, key="cursor_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=cur_edges, key="cursor_bin"),
    }
    base["survival_by_tier_gen_bin"] = {
        "edges": {str(t): gen_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=gen_edges, key="gen_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=gen_edges, key="gen_bin"),
    }
    base["survival_by_tier_unsupported_bin"] = {
        "edges": {str(t): uns_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=uns_edges, key="unsupported_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=uns_edges, key="unsupported_bin"),
    }
    base["reweighting"] = reweight
    base["reweighting_2x"] = {
        "residual_position": two_x.get("residual_position"),
        "share_of_B_residual_position": two_x.get("share_of_B_residual_position"),
        "synthetic_allocation": two_x.get("synthetic_allocation"),
        "recruit_mix": two_x.get("recruit_mix"),
        "within_tier_B": two_x.get("within_tier_B"),
    }
    rec = dict(base.get("reconciliation") or {})
    leftover_sum = (
        float(reweight.get("targeting_taunt") or 0.0)
        + float(reweight.get("attack_cursor") or 0.0)
        + float(reweight.get("represented_generated") or 0.0)
        + float(reweight.get("unsupported_coverage") or 0.0)
        + float(reweight.get("still_unexplained") or 0.0)
    )
    rec.update({
        "phase_2v_survivor_tier_sum_delta": PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "reproduced_within_tier_B": decomp.get("within_tier_survival_B"),
        "reproduced_2x_residual": two_x.get("residual_position"),
        "reweight_direct_B": reweight.get("within_tier_B"),
        "reweight_leftover_hat": reweight.get("phase_2y_unexplained_hat"),
        "mechanics_sum": leftover_sum,
        "nested_residual_vs_2y": leftover_sum - PHASE_2Y_UNEXPLAINED,
        "event_count_mismatches_control": int(
            control.get("event_count_mismatches") or 0
        ),
        "event_count_mismatches_treatment": int(
            treatment.get("event_count_mismatches") or 0
        ),
        "death_causes_control": control.get("death_causes") or {},
        "death_causes_treatment": treatment.get("death_causes") or {},
    })
    base["reconciliation"] = rec
    base["example_minions"] = {
        "control": control.get("example_minions") or [],
        "treatment": treatment.get("example_minions") or [],
    }
    return base


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_2z(comparison, non_evaluative=non_evaluative)
