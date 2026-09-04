"""Phase 2Y — observational slot/attack-order vs teammate-protection split.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. For every decisive T7–T14 hit, each winner starting
body is tagged with board slot, first-attack index, n_attacks,
death-before-first-attack, taunt / defender-target counts (if traced),
teammate combat-raw excluding self, board size, and survival.

Standardizes first on tier + recruit/raw + synth share (2X leftover), then
adds slot / attack-opportunity, then teammate-strength / board-size.

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
from ml.phase_2y_prereg import (
    INSTRUMENT_TURNS,
    N_DECILES,
    N_TEAM_BINS,
    PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    SLOT_BIN_CAP,
    assert_seed_range_allowed,
    diagnose_phase_2y,
    slot_bin,
)
from ml.survivor_composition_diagnostic import TIERS
from ml.synthetic_allocation_diagnostic import (
    SyntheticAllocationTracer,
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

METHODOLOGY_VERSION = "2y_v1"


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


class PositionOrderTracer(SyntheticAllocationTracer):
    """2X synthetic-share rows plus slot/attack/teammate protection fields."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        start_combat = list(rec.get("start_combat_bodies") or [])
        by_id = {str(b.get("body_id") or ""): b for b in start_combat}
        rows = list(rec.get("start_minions") or [])
        board_size = len(rows)
        total_raw = int(sum(int(r.get("combat_raw") or 0) for r in rows))
        for r in rows:
            src = by_id.get(str(r.get("body_id") or ""), {})
            n_att = int(r.get("n_attacks") or src.get("n_attacks") or 0)
            died = bool(r.get("died"))
            first_idx = src.get("first_attack_index")
            if first_idx is None:
                first_idx = r.get("first_attack_index")
            n_targeted = int(src.get("n_targeted") or r.get("n_targeted") or 0)
            combat = int(r.get("combat_raw") or 0)
            teammate = total_raw - combat
            r["first_attack_index"] = (
                None if first_idx is None else int(first_idx)
            )
            r["n_targeted"] = n_targeted
            r["was_targeted"] = n_targeted > 0
            r["taunt"] = bool(src.get("taunt") or r.get("taunt"))
            r["death_before_first_attack"] = died and n_att == 0
            r["board_size"] = board_size
            r["teammate_combat_raw"] = teammate
            r["mean_teammate_combat_raw"] = (
                float(teammate) / float(board_size - 1) if board_size > 1 else 0.0
            )
            r["slot_bin"] = slot_bin(r.get("board_slot"))
        rec["board_size"] = board_size
        rec["winner_board_combat_raw"] = total_raw


def run_position_arm(
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
                tracer = PositionOrderTracer(i, seed + i, arm)
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
    return run_position_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_position_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def collect_position_minions(hits: Sequence[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for f in hits:
        start = list(f.get("start_minions") or [])
        board_size = len(start)
        total_raw = int(sum(int(r.get("combat_raw") or 0) for r in start))
        for r in start:
            row = dict(r)
            combat = int(row.get("combat_raw") or 0)
            teammate = int(row.get("teammate_combat_raw")
                           if row.get("teammate_combat_raw") is not None
                           else total_raw - combat)
            n_att = int(row.get("n_attacks") or 0)
            row["board_size"] = int(row.get("board_size") or board_size)
            row["teammate_combat_raw"] = teammate
            row["mean_teammate_combat_raw"] = (
                float(teammate) / float(row["board_size"] - 1)
                if row["board_size"] > 1 else 0.0
            )
            row["slot_bin"] = slot_bin(row.get("board_slot"))
            row["death_before_first_attack"] = bool(
                row.get("death_before_first_attack")
                if "death_before_first_attack" in row
                else (bool(row.get("died")) and n_att == 0)
            )
            row["winner_tavern_tier"] = int(f.get("winner_tavern_tier") or 1)
            rows.append(row)
    return rows


def _by_tier_position(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r["tier"]) == tier]
        n = len(cell)
        surv = [r for r in cell if r.get("survived")]
        died = [r for r in cell if r.get("died")]
        out[str(tier)] = {
            "n_start": n,
            "n_survived": len(surv),
            "p_survive": _safe_div(float(len(surv)), float(n)),
            "mean_board_slot": _mean(
                [float(r.get("board_slot") or 0) for r in cell]
            ),
            "mean_slot_bin": _mean(
                [float(r.get("slot_bin") or 0) for r in cell]
            ),
            "mean_n_attacks": _mean(
                [float(r.get("n_attacks") or 0) for r in cell]
            ),
            "mean_first_attack_index": _mean(
                [
                    float(r["first_attack_index"])
                    for r in cell
                    if r.get("first_attack_index") is not None
                ]
            ),
            "p_attacked": _safe_div(
                float(sum(1 for r in cell if r.get("attacked"))), float(n)
            ),
            "p_death_before_first_attack": _safe_div(
                float(sum(1 for r in cell if r.get("death_before_first_attack"))),
                float(n),
            ),
            "p_death_before_first_given_died": _safe_div(
                float(sum(1 for r in died if r.get("death_before_first_attack"))),
                float(len(died)),
            ),
            "mean_teammate_combat_raw": _mean(
                [float(r.get("teammate_combat_raw") or 0) for r in cell]
            ),
            "mean_board_size": _mean(
                [float(r.get("board_size") or 0) for r in cell]
            ),
            "p_taunt": _safe_div(
                float(sum(1 for r in cell if r.get("taunt"))), float(n)
            ),
            "p_was_targeted": _safe_div(
                float(sum(1 for r in cell if r.get("was_targeted"))), float(n)
            ),
            "mean_n_targeted": _mean(
                [float(r.get("n_targeted") or 0) for r in cell]
            ),
            "mean_per_hit": _safe_div(float(n), float(n_hits)) if n_hits else None,
        }
    return out


def reweight_position_protection(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    n_hits_c: int,
    n_hits_t: int,
    observed_B: Optional[float] = None,
    observed_residual: Optional[float] = None,
) -> Dict:
    """Hold tier + recruit + synth, then split leftover into slot vs teammates.

    Nested Kitagawa:

        hold P(recruit-raw decile | tier)
            ↓
        hold P(synth share | tier, recruit)
            ↓
        hold P(slot_bin | tier, recruit, synth)     →  (A) slot / opportunity
            ↓
        hold P(teammate-raw quintile | …, slot)     →  (B) teammate protection
            ↓
        leftover P(survive | …)                     →  (C) unexplained
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
        scale = 1.0 / float(n_hits) if n_hits else 0.0
        for r in rows:
            t = int(r["tier"])
            rb = bin_value(float(r["recruit_raw"]), recruit_edges[t])
            sb = bin_value(float(r["synthetic_share"]), synth_edges[(t, rb)])
            kb = slot_bin(r.get("board_slot"))
            mb = bin_value(
                float(r.get("teammate_combat_raw") or 0), team_edges[t]
            )
            w = scale
            n_t[t] += w
            n_tr[(t, rb)] += w
            n_trs[(t, rb, sb)] += w
            n_trsk[(t, rb, sb, kb)] += w
            n_trskm[(t, rb, sb, kb, mb)] += w
            if r.get("survived"):
                s_t[t] += w
                s_tr[(t, rb)] += w
                s_trs[(t, rb, sb)] += w
                s_trsk[(t, rb, sb, kb)] += w
                s_trskm[(t, rb, sb, kb, mb)] += w
        return (n_t, s_t, n_tr, s_tr, n_trs, s_trs,
                n_trsk, s_trsk, n_trskm, s_trskm)

    c = _arm_cells(control_rows, n_hits_c)
    t = _arm_cells(treatment_rows, n_hits_t)
    (n_c, s_c, n_cr, s_cr, n_crs, s_crs,
     n_crsk, s_crsk, n_crskm, s_crskm) = c
    (n_t, s_t, n_tr, s_tr, n_trs, s_trs,
     n_trsk, s_trsk, n_trskm, s_trskm) = t

    b_direct = 0.0
    b_recruit = 0.0
    b_synth = 0.0
    b_slot = 0.0
    b_team = 0.0
    b_unexpl = 0.0
    per_tier = {}
    for tier in TIERS:
        nc, nt = n_c[tier], n_t[tier]
        pc = _cond_p(s_c[tier], nc)
        pt = _cond_p(s_t[tier], nt)
        _mix_t, rate_t, _gap_t, excl_t = _kitagawa_two(
            nc, nt, pc, pt, float(tier)
        )
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
                "slot_opportunity": 0.0,
                "teammate_protection": 0.0,
                "unexplained": 0.0,
            }
            continue
        n_bar = 0.5 * (nc + nt)
        b_direct += rate_t

        n_r_bins = len(recruit_edges[tier]) + 1
        rec_mix_dp = 0.0
        synth_dp = 0.0
        slot_dp = 0.0
        team_dp = 0.0
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
            m_rate_dp = 0.0
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
                m_inner_rate = 0.0
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
                    tm_rate = 0.0
                    for mb in range(n_m_bins):
                        ncm = n_crskm[(tier, rb, sb, kb, mb)]
                        ntm = n_trskm[(tier, rb, sb, kb, mb)]
                        p_m_c = (ncm / nck) if nck > 1e-15 else 0.0
                        p_m_t = (ntm / ntk) if ntk > 1e-15 else 0.0
                        pcm = _cond_p(s_crskm[(tier, rb, sb, kb, mb)], ncm)
                        ptm = _cond_p(s_trskm[(tier, rb, sb, kb, mb)], ntm)
                        mix_m, rate_m, _excl_m = _kitagawa_prob_delta(
                            p_m_c, p_m_t, pcm, ptm
                        )
                        tm_mix += mix_m
                        tm_rate += rate_m
                    p_k_bar = 0.5 * (p_k_c + p_k_t)
                    m_inner_mix += p_k_bar * tm_mix
                    m_inner_rate += p_k_bar * tm_rate
                p_s_bar = 0.5 * (p_s_c + p_s_t)
                k_mix_dp += p_s_bar * k_inner_mix
                m_mix_dp += p_s_bar * m_inner_mix
                m_rate_dp += p_s_bar * m_inner_rate
            p_r_bar = 0.5 * (p_r_c + p_r_t)
            synth_dp += p_r_bar * s_mix_dp
            slot_dp += p_r_bar * k_mix_dp
            team_dp += p_r_bar * m_mix_dp
            unexpl_dp += p_r_bar * m_rate_dp

        rec_part = float(tier) * n_bar * rec_mix_dp
        synth_part = float(tier) * n_bar * synth_dp
        slot_part = float(tier) * n_bar * slot_dp
        team_part = float(tier) * n_bar * team_dp
        unexpl_part = float(tier) * n_bar * unexpl_dp
        b_recruit += rec_part
        b_synth += synth_part
        b_slot += slot_part
        b_team += team_part
        b_unexpl += unexpl_part
        resid_hat = slot_part + team_part + unexpl_part
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
            "unexplained": unexpl_part,
            "residual_position_hat": resid_hat,
            "nested_residual": (
                rate_t - rec_part - synth_part - resid_hat
            ),
        }

    explained = b_recruit + b_synth + b_slot + b_team + b_unexpl
    resid_hat = b_slot + b_team + b_unexpl
    obs_b = float(observed_B) if observed_B is not None else b_direct
    obs_r = (
        float(observed_residual)
        if observed_residual is not None
        else resid_hat
    )

    def _share_b(part: float) -> Optional[float]:
        if abs(obs_b) < 1e-12:
            return None
        return float(part) / obs_b

    def _share_r(part: float) -> Optional[float]:
        if abs(obs_r) < 1e-12:
            return None
        return float(part) / obs_r

    return {
        "method": (
            "nested_kitagawa_tier_recruit_synth_then_slot_bin_then_teammate_quintile"
        ),
        "n_deciles": N_DECILES,
        "n_team_bins": N_TEAM_BINS,
        "slot_bin_cap": SLOT_BIN_CAP,
        "within_tier_B": b_direct,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "observed_B_used_for_shares": obs_b,
        "observed_residual_used_for_shares": obs_r,
        "recruit_mix": b_recruit,
        "synthetic_allocation": b_synth,
        "slot_opportunity": b_slot,
        "teammate_protection": b_team,
        "unexplained": b_unexpl,
        "residual_position_hat": resid_hat,
        "explained_all_parts": explained,
        "residual_vs_direct_B": b_direct - explained,
        "share_of_B_recruit_mix": _share_b(b_recruit),
        "share_of_B_synthetic": _share_b(b_synth),
        "share_of_B_slot_opportunity": _share_b(b_slot),
        "share_of_B_teammate_protection": _share_b(b_team),
        "share_of_B_unexplained": _share_b(b_unexpl),
        "share_of_residual_slot_opportunity": _share_r(b_slot),
        "share_of_residual_teammate_protection": _share_r(b_team),
        "share_of_residual_unexplained": _share_r(b_unexpl),
        "phase_2x_residual_position_hat": resid_hat,
        "per_tier": per_tier,
    }


def summarize_position_arm(raw: Dict) -> Dict:
    summary = summarize_allocation_arm(raw)
    hits = _hits(raw["fights"])
    rows = collect_position_minions(hits)
    n_hits = len(hits)
    by_tier = _by_tier_position(rows, n_hits)
    summary.update({
        "n_start_minions": len(rows),
        "by_tier_position": by_tier,
        "mean_board_slot": _mean(
            [float(r.get("board_slot") or 0) for r in rows]
        ),
        "mean_n_attacks": _mean(
            [float(r.get("n_attacks") or 0) for r in rows]
        ),
        "p_death_before_first_attack": _safe_div(
            float(sum(1 for r in rows if r.get("death_before_first_attack"))),
            float(len(rows)),
        ),
        "mean_teammate_combat_raw": _mean(
            [float(r.get("teammate_combat_raw") or 0) for r in rows]
        ),
        "mean_board_size": _mean(
            [float(r.get("board_size") or 0) for r in rows]
        ),
        "p_taunt": _safe_div(
            float(sum(1 for r in rows if r.get("taunt"))), float(len(rows))
        ),
        "p_was_targeted": _safe_div(
            float(sum(1 for r in rows if r.get("was_targeted"))),
            float(len(rows)),
        ),
        "example_minions": rows[:8],
        "_rows": rows,
        "_n_hits": n_hits,
    })
    return summary


def compare_position(control: Dict, treatment: Dict) -> Dict:
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

    reweight = reweight_position_protection(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
    )

    def _tier_delta(table_c: Dict, table_t: Dict, key: str) -> Dict[str, Optional[float]]:
        out = {}
        for tier in TIERS:
            k = str(tier)
            a = (table_c or {}).get(k, {}).get(key)
            b = (table_t or {}).get(k, {}).get(key)
            out[k] = None if a is None or b is None else float(b) - float(a)
        return out

    pos_c = control.get("by_tier_position") or {}
    pos_t = treatment.get("by_tier_position") or {}
    base["by_tier_position"] = {
        "control": pos_c,
        "treatment": pos_t,
        "delta_mean_board_slot": _tier_delta(pos_c, pos_t, "mean_board_slot"),
        "delta_mean_n_attacks": _tier_delta(pos_c, pos_t, "mean_n_attacks"),
        "delta_p_death_before_first_attack": _tier_delta(
            pos_c, pos_t, "p_death_before_first_attack"
        ),
        "delta_mean_teammate_combat_raw": _tier_delta(
            pos_c, pos_t, "mean_teammate_combat_raw"
        ),
        "delta_mean_board_size": _tier_delta(pos_c, pos_t, "mean_board_size"),
        "delta_p_was_targeted": _tier_delta(pos_c, pos_t, "p_was_targeted"),
        "delta_p_survive": _tier_delta(pos_c, pos_t, "p_survive"),
    }

    slot_edges = {}
    team_edges = {}
    for tier in TIERS:
        slot_edges[tier] = [float(i) + 0.5 for i in range(SLOT_BIN_CAP)]
        team_edges[tier] = decile_edges(
            [float(r.get("teammate_combat_raw") or 0)
             for r in rows_c + rows_t if int(r["tier"]) == tier],
            n=N_TEAM_BINS,
        )
    # _cross_survival bins on a numeric key; expose slot_bin / teammate_raw.
    for r in rows_c + rows_t:
        r.setdefault("slot_bin", slot_bin(r.get("board_slot")))
        r.setdefault("teammate_combat_raw", 0)

    base["survival_by_tier_slot_bin"] = {
        "edges": {str(t): slot_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=slot_edges, key="slot_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=slot_edges, key="slot_bin"),
    }
    base["survival_by_tier_teammate_decile"] = {
        "edges": {str(t): team_edges[t] for t in TIERS},
        "control": _cross_survival(
            rows_c, edges_by_tier=team_edges, key="teammate_combat_raw"
        ),
        "treatment": _cross_survival(
            rows_t, edges_by_tier=team_edges, key="teammate_combat_raw"
        ),
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
    rec.update({
        "phase_2v_survivor_tier_sum_delta": PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "reproduced_within_tier_B": decomp.get("within_tier_survival_B"),
        "reproduced_2x_residual": two_x.get("residual_position"),
        "reweight_direct_B": reweight.get("within_tier_B"),
        "reweight_residual_hat": reweight.get("residual_position_hat"),
        "slot_plus_team_plus_unexpl": (
            float(reweight.get("slot_opportunity") or 0.0)
            + float(reweight.get("teammate_protection") or 0.0)
            + float(reweight.get("unexplained") or 0.0)
        ),
        "nested_residual_vs_2x": (
            None if two_x.get("residual_position") is None
            else (
                float(reweight.get("residual_position_hat") or 0.0)
                - float(two_x.get("residual_position") or 0.0)
            )
        ),
    })
    base["reconciliation"] = rec
    base["example_minions"] = {
        "control": control.get("example_minions") or [],
        "treatment": treatment.get("example_minions") or [],
    }
    return base


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_2y(comparison, non_evaluative=non_evaluative)
