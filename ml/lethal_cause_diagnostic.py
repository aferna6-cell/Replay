"""Phase 3A — observational lethal-cause / keyword split of the 2Z leftover.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. Holds the 2Z leftover cells (tier + recruit/raw +
synth share + slot + teammate-raw + target + cursor + gen-DR + unsupported),
then splits leftover survival into divine-shield, poisonous/venomous, cleave
primary/secondary, represented start-of-combat, ordinary attack/counterattack,
and still unexplained.

Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.combat_mechanics_diagnostic import (
    CombatMechanicsTracer,
    reweight_combat_mechanics,
)
from ml.phase_2z_prereg import (
    cursor_bin,
    gen_bin,
    target_bin,
    unsupported_bin,
)
from ml.phase_3a_prereg import (
    N_CLEAVE_BINS,
    N_CURSOR_BINS,
    N_DECILES,
    N_DS_BINS,
    N_GEN_BINS,
    N_ORDINARY_BINS,
    N_POISON_BINS,
    N_SOC_BINS,
    N_TARGET_BINS,
    N_TEAM_BINS,
    N_UNSUP_BINS,
    PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    PHASE_2Y_UNEXPLAINED,
    PHASE_2Z_UNEXPLAINED,
    SLOT_BIN_CAP,
    assert_seed_range_allowed,
    cleave_bin,
    diagnose_phase_3a,
    ds_bin,
    ordinary_bin,
    poison_bin,
    slot_bin,
    soc_bin,
)
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

METHODOLOGY_VERSION = "3a_v1"

_LETHAL_KEYS = (
    "start_divine_shield",
    "end_divine_shield",
    "ds_before_last_hit",
    "ds_after_last_hit",
    "n_shield_pops",
    "shield_pop_cause",
    "n_hits",
    "n_hits_poison",
    "poison_lethal",
    "n_cleave_primary",
    "n_cleave_secondary",
    "cleave_lethal",
    "n_soc_hits",
    "soc_lethal",
    "n_ordinary_attack_hits",
    "n_ordinary_counter_hits",
    "ordinary_lethal",
    "divine_shield",
    "poisonous",
    "cleave",
)

_LETHAL_RECONCILE = (
    "hits_reconcile",
    "shield_pops_reconcile",
    "poison_hits_reconcile",
    "cleave_primary_reconcile",
    "cleave_secondary_reconcile",
    "soc_hits_reconcile",
    "ordinary_attack_reconcile",
    "ordinary_counter_reconcile",
    "death_causes_reconcile",
)

# Nested parts after within-tier B, in walk order.
_PART_NAMES = (
    "recruit_mix",
    "synthetic_allocation",
    "slot_opportunity",
    "teammate_protection",
    "targeting_taunt",
    "attack_cursor",
    "represented_generated",
    "unsupported_coverage",
    "divine_shield",
    "poison_venomous",
    "cleave",
    "start_of_combat",
    "ordinary_combat",
    "still_unexplained",
)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


class LethalCauseTracer(CombatMechanicsTracer):
    """2Z rows plus per-body DS / poison / cleave / SOC / ordinary tags."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        start_combat = list(rec.get("start_combat_bodies") or [])
        by_id = {str(b.get("body_id") or ""): b for b in start_combat}
        rows = list(rec.get("start_minions") or [])
        for r in rows:
            src = by_id.get(str(r.get("body_id") or ""), {})
            for k in _LETHAL_KEYS:
                if k in src:
                    r[k] = src[k]
                elif k not in r:
                    r[k] = src.get(k)
            r["ds_bin"] = ds_bin(r)
            r["poison_bin"] = poison_bin(r)
            r["cleave_bin"] = cleave_bin(r)
            r["soc_bin"] = soc_bin(r)
            r["ordinary_bin"] = ordinary_bin(r)
        counts = dict(rec.get("event_counts") or fight.get("event_counts") or {})
        rec["event_counts"] = counts
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
    ) + _LETHAL_RECONCILE
    return all(bool(counts.get(k, True)) for k in flags)


def run_lethal_arm(
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
                tracer = LethalCauseTracer(i, seed + i, arm)
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
    return run_lethal_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_lethal_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def collect_lethal_minions(hits: Sequence[Dict]) -> List[Dict]:
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
            row["ds_bin"] = ds_bin(row)
            row["poison_bin"] = poison_bin(row)
            row["cleave_bin"] = cleave_bin(row)
            row["soc_bin"] = soc_bin(row)
            row["ordinary_bin"] = ordinary_bin(row)
            row["slot_bin"] = slot_bin(row.get("board_slot"))
            row["winner_tavern_tier"] = int(f.get("winner_tavern_tier") or 1)
            rows.append(row)
    return rows


def _by_tier_lethal(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r["tier"]) == tier]
        n = len(cell)
        out[str(tier)] = {
            "n_start": n,
            "p_survive": _safe_div(
                float(sum(1 for r in cell if r.get("survived"))), float(n)
            ),
            "p_start_divine_shield": _safe_div(
                float(sum(1 for r in cell if r.get("start_divine_shield"))), float(n)
            ),
            "p_shield_popped": _safe_div(
                float(sum(1 for r in cell if int(r.get("n_shield_pops") or 0) > 0)),
                float(n),
            ),
            "p_poison_hit": _safe_div(
                float(sum(1 for r in cell if poison_bin(r) == 1)), float(n)
            ),
            "p_poison_lethal": _safe_div(
                float(sum(1 for r in cell if r.get("poison_lethal"))), float(n)
            ),
            "p_cleave_primary": _safe_div(
                float(sum(1 for r in cell if int(r.get("n_cleave_primary") or 0) > 0)),
                float(n),
            ),
            "p_cleave_secondary": _safe_div(
                float(sum(1 for r in cell if int(r.get("n_cleave_secondary") or 0) > 0)),
                float(n),
            ),
            "p_soc_hit": _safe_div(
                float(sum(1 for r in cell if soc_bin(r) == 1)), float(n)
            ),
            "p_ordinary_attack_hit": _safe_div(
                float(sum(
                    1 for r in cell if int(r.get("n_ordinary_attack_hits") or 0) > 0
                )),
                float(n),
            ),
            "p_ordinary_counter_hit": _safe_div(
                float(sum(
                    1 for r in cell
                    if int(r.get("n_ordinary_counter_hits") or 0) > 0
                )),
                float(n),
            ),
            "mean_per_hit": _safe_div(float(n), float(n_hits)) if n_hits else None,
        }
    return out


def _row_key(
    r: Dict,
    recruit_edges: Dict[int, List[float]],
    synth_edges: Dict[Tuple[int, int], List[float]],
    team_edges: Dict[int, List[float]],
) -> Tuple:
    t = int(r["tier"])
    rb = bin_value(float(r["recruit_raw"]), recruit_edges[t])
    sb = bin_value(float(r["synthetic_share"]), synth_edges[(t, rb)])
    kb = slot_bin(r.get("board_slot"))
    mb = bin_value(float(r.get("teammate_combat_raw") or 0), team_edges[t])
    return (
        t, rb, sb, kb, mb,
        target_bin(r), cursor_bin(r), gen_bin(r), unsupported_bin(r),
        ds_bin(r), poison_bin(r), cleave_bin(r), soc_bin(r), ordinary_bin(r),
    )


def _arm_prefixes(rows: Sequence[Dict], n_hits: int, key_fn: Callable) -> Tuple:
    n: Dict[Tuple, float] = defaultdict(float)
    s: Dict[Tuple, float] = defaultdict(float)
    n_t = {t: 0.0 for t in TIERS}
    s_t = {t: 0.0 for t in TIERS}
    scale = 1.0 / float(n_hits) if n_hits else 0.0
    for r in rows:
        key = key_fn(r)
        t = int(key[0])
        w = scale
        n_t[t] += w
        n[key[:1]] += w
        if r.get("survived"):
            s_t[t] += w
            s[key[:1]] += w
        for d in range(2, len(key) + 1):
            pref = key[:d]
            n[pref] += w
            if r.get("survived"):
                s[pref] += w
    return n_t, s_t, n, s


def _walk(
    prefix: Tuple,
    depth: int,
    n_bins_at: Callable,
    n_c: Dict,
    s_c: Dict,
    n_t: Dict,
    s_t: Dict,
    parent_nc: float,
    parent_nt: float,
    n_depths: int,
) -> List[float]:
    """Nested Kitagawa from ``depth``..end. Returns [mix_depth, ..., leftover]."""
    n_left = n_depths - depth
    acc = [0.0] * (n_left + 1)
    n_bins = n_bins_at(depth, prefix)
    last = depth == n_depths - 1
    for b in range(n_bins):
        key = prefix + (b,)
        nc = float(n_c.get(key, 0.0))
        nt = float(n_t.get(key, 0.0))
        if nc <= 0.0 and nt <= 0.0:
            continue
        p_c = (nc / parent_nc) if parent_nc > 1e-15 else 0.0
        p_t = (nt / parent_nt) if parent_nt > 1e-15 else 0.0
        pc = _cond_p(float(s_c.get(key, 0.0)), nc)
        pt = _cond_p(float(s_t.get(key, 0.0)), nt)
        mix_b, rate_b, excl = _kitagawa_prob_delta(p_c, p_t, pc, pt)
        acc[0] += mix_b
        if last:
            acc[-1] += rate_b
            continue
        if excl:
            continue
        inner = _walk(
            key, depth + 1, n_bins_at, n_c, s_c, n_t, s_t, nc, nt, n_depths,
        )
        p_bar = 0.5 * (p_c + p_t)
        for i, v in enumerate(inner):
            acc[i + 1] += p_bar * v
    return acc


def reweight_lethal_cause(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    n_hits_c: int,
    n_hits_t: int,
    observed_B: Optional[float] = None,
    observed_residual: Optional[float] = None,
    observed_leftover_2y: Optional[float] = None,
    observed_leftover_2z: Optional[float] = None,
) -> Dict:
    """Hold 2Z leftover cells, then split E into DS / poison / cleave / SOC / ordinary / gap.

    Nested Kitagawa:

        hold P(recruit-raw | tier) … P(unsupported | 2Z cells)
            ↓
        hold P(ds_bin | 2Z leftover cells)          →  (A) divine shield
            ↓
        hold P(poison_bin | …)                      →  (B) poisonous / venomous
            ↓
        hold P(cleave_bin | …)                      →  (C) cleave
            ↓
        hold P(soc_bin | …)                         →  (D) start-of-combat
            ↓
        hold P(ordinary_bin | …)                    →  (E) ordinary combat
            ↓
        leftover P(survive | all)                   →  (F) still unexplained
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

    def _key(r: Dict) -> Tuple:
        return _row_key(r, recruit_edges, synth_edges, team_edges)

    n_c_t, s_c_t, n_c, s_c = _arm_prefixes(control_rows, n_hits_c, _key)
    n_t_t, s_t_t, n_t, s_t = _arm_prefixes(treatment_rows, n_hits_t, _key)

    const_bins = {
        5: N_TARGET_BINS,
        6: N_CURSOR_BINS,
        7: N_GEN_BINS,
        8: N_UNSUP_BINS,
        9: N_DS_BINS,
        10: N_POISON_BINS,
        11: N_CLEAVE_BINS,
        12: N_SOC_BINS,
        13: N_ORDINARY_BINS,
    }
    n_depths = 14  # tier + 13 nested (recruit … ordinary)

    def n_bins_at(depth: int, prefix: Tuple) -> int:
        if depth == 1:
            return len(recruit_edges[prefix[0]]) + 1
        if depth == 2:
            return len(synth_edges[(prefix[0], prefix[1])]) + 1
        if depth == 3:
            return SLOT_BIN_CAP + 1
        if depth == 4:
            return len(team_edges[prefix[0]]) + 1
        return int(const_bins[depth])

    totals = {name: 0.0 for name in _PART_NAMES}
    b_direct = 0.0
    per_tier = {}
    for tier in TIERS:
        nc, nt = n_c_t[tier], n_t_t[tier]
        pc = _cond_p(s_c_t[tier], nc)
        pt = _cond_p(s_t_t[tier], nt)
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
            **{name: 0.0 for name in _PART_NAMES},
            "phase_2y_unexplained_hat": 0.0,
            "phase_2z_unexplained_hat": 0.0,
        }
        if excl_t:
            per_tier[str(tier)] = zero
            continue
        n_bar = 0.5 * (nc + nt)
        b_direct += rate_t
        parts = _walk(
            (tier,), 1, n_bins_at, n_c, s_c, n_t, s_t, nc, nt, n_depths,
        )
        scale = float(tier) * n_bar
        scaled = [scale * v for v in parts]
        cell = {
            "exclusive_support": False,
            "n_start_control": nc,
            "n_start_treatment": nt,
            "p_survive_control": pc,
            "p_survive_treatment": pt,
            "within_tier_B": rate_t,
        }
        for name, val in zip(_PART_NAMES, scaled):
            cell[name] = val
            totals[name] += val
        leftover_2y = (
            cell["targeting_taunt"] + cell["attack_cursor"]
            + cell["represented_generated"] + cell["unsupported_coverage"]
            + cell["divine_shield"] + cell["poison_venomous"] + cell["cleave"]
            + cell["start_of_combat"] + cell["ordinary_combat"]
            + cell["still_unexplained"]
        )
        leftover_2z = (
            cell["divine_shield"] + cell["poison_venomous"] + cell["cleave"]
            + cell["start_of_combat"] + cell["ordinary_combat"]
            + cell["still_unexplained"]
        )
        cell["phase_2y_unexplained_hat"] = leftover_2y
        cell["phase_2z_unexplained_hat"] = leftover_2z
        cell["nested_residual"] = (
            rate_t
            - cell["recruit_mix"] - cell["synthetic_allocation"]
            - cell["slot_opportunity"] - cell["teammate_protection"]
            - leftover_2y
        )
        per_tier[str(tier)] = cell

    leftover_2y = (
        totals["targeting_taunt"] + totals["attack_cursor"]
        + totals["represented_generated"] + totals["unsupported_coverage"]
        + totals["divine_shield"] + totals["poison_venomous"] + totals["cleave"]
        + totals["start_of_combat"] + totals["ordinary_combat"]
        + totals["still_unexplained"]
    )
    leftover_2z = (
        totals["divine_shield"] + totals["poison_venomous"] + totals["cleave"]
        + totals["start_of_combat"] + totals["ordinary_combat"]
        + totals["still_unexplained"]
    )
    resid_hat = (
        totals["slot_opportunity"] + totals["teammate_protection"] + leftover_2y
    )
    explained = (
        totals["recruit_mix"] + totals["synthetic_allocation"]
        + totals["slot_opportunity"] + totals["teammate_protection"]
        + leftover_2y
    )
    obs_b = float(observed_B) if observed_B is not None else b_direct
    obs_r = (
        float(observed_residual)
        if observed_residual is not None
        else resid_hat
    )
    obs_c = (
        float(observed_leftover_2y)
        if observed_leftover_2y is not None
        else leftover_2y
    )
    obs_e = (
        float(observed_leftover_2z)
        if observed_leftover_2z is not None
        else leftover_2z
    )

    def _share(part: float, denom: float) -> Optional[float]:
        if abs(denom) < 1e-12:
            return None
        return float(part) / denom

    return {
        "method": (
            "nested_kitagawa_2z_cells_then_ds_poison_cleave_soc_ordinary"
        ),
        "n_deciles": N_DECILES,
        "n_team_bins": N_TEAM_BINS,
        "slot_bin_cap": SLOT_BIN_CAP,
        "n_target_bins": N_TARGET_BINS,
        "n_cursor_bins": N_CURSOR_BINS,
        "n_gen_bins": N_GEN_BINS,
        "n_unsup_bins": N_UNSUP_BINS,
        "n_ds_bins": N_DS_BINS,
        "n_poison_bins": N_POISON_BINS,
        "n_cleave_bins": N_CLEAVE_BINS,
        "n_soc_bins": N_SOC_BINS,
        "n_ordinary_bins": N_ORDINARY_BINS,
        "within_tier_B": b_direct,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "phase_2z_unexplained": PHASE_2Z_UNEXPLAINED,
        "observed_B_used_for_shares": obs_b,
        "observed_residual_used_for_shares": obs_r,
        "observed_leftover_2y_used_for_shares": obs_c,
        "observed_leftover_used_for_shares": obs_e,
        **totals,
        "phase_2y_unexplained_hat": leftover_2y,
        "phase_2z_unexplained_hat": leftover_2z,
        "residual_position_hat": resid_hat,
        "explained_all_parts": explained,
        "residual_vs_direct_B": b_direct - explained,
        "share_of_leftover_divine_shield": _share(totals["divine_shield"], obs_e),
        "share_of_leftover_poison_venomous": _share(
            totals["poison_venomous"], obs_e
        ),
        "share_of_leftover_cleave": _share(totals["cleave"], obs_e),
        "share_of_leftover_start_of_combat": _share(
            totals["start_of_combat"], obs_e
        ),
        "share_of_leftover_ordinary_combat": _share(
            totals["ordinary_combat"], obs_e
        ),
        "share_of_leftover_still_unexplained": _share(
            totals["still_unexplained"], obs_e
        ),
        "share_of_2y_divine_shield": _share(totals["divine_shield"], obs_c),
        "share_of_2y_poison_venomous": _share(totals["poison_venomous"], obs_c),
        "share_of_2y_cleave": _share(totals["cleave"], obs_c),
        "share_of_2y_start_of_combat": _share(totals["start_of_combat"], obs_c),
        "share_of_2y_ordinary_combat": _share(totals["ordinary_combat"], obs_c),
        "share_of_2y_still_unexplained": _share(
            totals["still_unexplained"], obs_c
        ),
        "per_tier": per_tier,
    }


def _slim_example(row: Dict) -> Dict:
    keep = {
        "name", "tier", "survived", "died", "taunt", "board_slot", "slot_bin",
        "n_attacks", "first_attack_index", "n_targeted", "n_targeted_forced",
        "n_targeted_open", "death_cause", "start_health", "end_health",
        "start_divine_shield", "end_divine_shield", "n_shield_pops",
        "shield_pop_cause", "ds_before_last_hit", "ds_after_last_hit",
        "n_hits_poison", "poison_lethal", "n_cleave_primary",
        "n_cleave_secondary", "cleave_lethal", "n_soc_hits", "soc_lethal",
        "n_ordinary_attack_hits", "n_ordinary_counter_hits", "ordinary_lethal",
        "effect_status", "target_bin", "cursor_bin", "gen_bin",
        "unsupported_bin", "ds_bin", "poison_bin", "cleave_bin", "soc_bin",
        "ordinary_bin", "teammate_combat_raw", "recruit_raw", "synthetic_share",
    }
    return {k: row.get(k) for k in keep}


def summarize_lethal_arm(raw: Dict) -> Dict:
    summary = summarize_allocation_arm(raw)
    hits = _hits(raw["fights"])
    rows = collect_lethal_minions(hits)
    n_hits = len(hits)
    by_tier = _by_tier_lethal(rows, n_hits)
    n_mismatch = 0
    n_counted = 0
    death_causes: Dict[str, int] = defaultdict(int)
    shield_pop_causes: Dict[str, int] = defaultdict(int)
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
            pop = r.get("shield_pop_cause")
            if pop:
                shield_pop_causes[str(pop)] += 1
    summary.update({
        "n_start_minions": len(rows),
        "by_tier_lethal": by_tier,
        "p_start_divine_shield": _safe_div(
            float(sum(1 for r in rows if r.get("start_divine_shield"))),
            float(len(rows)),
        ),
        "p_shield_popped": _safe_div(
            float(sum(1 for r in rows if int(r.get("n_shield_pops") or 0) > 0)),
            float(len(rows)),
        ),
        "p_poison_hit": _safe_div(
            float(sum(1 for r in rows if poison_bin(r) == 1)), float(len(rows))
        ),
        "p_poison_lethal": _safe_div(
            float(sum(1 for r in rows if r.get("poison_lethal"))),
            float(len(rows)),
        ),
        "p_cleave_secondary": _safe_div(
            float(sum(1 for r in rows if int(r.get("n_cleave_secondary") or 0) > 0)),
            float(len(rows)),
        ),
        "p_soc_hit": _safe_div(
            float(sum(1 for r in rows if soc_bin(r) == 1)), float(len(rows))
        ),
        "p_ordinary_hit": _safe_div(
            float(sum(1 for r in rows if ordinary_bin(r) > 0)), float(len(rows))
        ),
        "death_causes": dict(death_causes),
        "shield_pop_causes": dict(shield_pop_causes),
        "event_count_mismatches": n_mismatch,
        "n_hits_counted": n_counted,
        "example_minions": [_slim_example(r) for r in rows[:8]],
        "_rows": rows,
        "_n_hits": n_hits,
    })
    return summary


def compare_lethal(control: Dict, treatment: Dict) -> Dict:
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

    two_z = reweight_combat_mechanics(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover=PHASE_2Y_UNEXPLAINED,
    )

    reweight = reweight_lethal_cause(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover_2y=PHASE_2Y_UNEXPLAINED,
        observed_leftover_2z=PHASE_2Z_UNEXPLAINED,
    )

    def _tier_delta(table_c: Dict, table_t: Dict, key: str) -> Dict[str, Optional[float]]:
        out = {}
        for tier in TIERS:
            k = str(tier)
            a = (table_c or {}).get(k, {}).get(key)
            b = (table_t or {}).get(k, {}).get(key)
            out[k] = None if a is None or b is None else float(b) - float(a)
        return out

    leth_c = control.get("by_tier_lethal") or {}
    leth_t = treatment.get("by_tier_lethal") or {}
    base["by_tier_lethal"] = {
        "control": leth_c,
        "treatment": leth_t,
        "delta_p_start_divine_shield": _tier_delta(
            leth_c, leth_t, "p_start_divine_shield"
        ),
        "delta_p_shield_popped": _tier_delta(leth_c, leth_t, "p_shield_popped"),
        "delta_p_poison_hit": _tier_delta(leth_c, leth_t, "p_poison_hit"),
        "delta_p_poison_lethal": _tier_delta(leth_c, leth_t, "p_poison_lethal"),
        "delta_p_cleave_secondary": _tier_delta(
            leth_c, leth_t, "p_cleave_secondary"
        ),
        "delta_p_soc_hit": _tier_delta(leth_c, leth_t, "p_soc_hit"),
        "delta_p_survive": _tier_delta(leth_c, leth_t, "p_survive"),
    }

    ds_edges = {t: [0.5, 1.5] for t in TIERS}
    poi_edges = {t: [0.5] for t in TIERS}
    cl_edges = {t: [0.5, 1.5] for t in TIERS}
    soc_edges = {t: [0.5] for t in TIERS}
    ord_edges = {t: [0.5, 1.5] for t in TIERS}
    base["survival_by_tier_ds_bin"] = {
        "edges": {str(t): ds_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=ds_edges, key="ds_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=ds_edges, key="ds_bin"),
    }
    base["survival_by_tier_poison_bin"] = {
        "edges": {str(t): poi_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=poi_edges, key="poison_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=poi_edges, key="poison_bin"),
    }
    base["survival_by_tier_cleave_bin"] = {
        "edges": {str(t): cl_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=cl_edges, key="cleave_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=cl_edges, key="cleave_bin"),
    }
    base["survival_by_tier_soc_bin"] = {
        "edges": {str(t): soc_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=soc_edges, key="soc_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=soc_edges, key="soc_bin"),
    }
    base["survival_by_tier_ordinary_bin"] = {
        "edges": {str(t): ord_edges[t] for t in TIERS},
        "control": _cross_survival(rows_c, edges_by_tier=ord_edges, key="ordinary_bin"),
        "treatment": _cross_survival(rows_t, edges_by_tier=ord_edges, key="ordinary_bin"),
    }
    base["reweighting"] = reweight
    base["reweighting_2z"] = {
        "still_unexplained": two_z.get("still_unexplained"),
        "share_of_leftover_still_unexplained": two_z.get(
            "share_of_leftover_still_unexplained"
        ),
        "phase_2y_unexplained_hat": two_z.get("phase_2y_unexplained_hat"),
        "targeting_taunt": two_z.get("targeting_taunt"),
        "attack_cursor": two_z.get("attack_cursor"),
        "represented_generated": two_z.get("represented_generated"),
        "unsupported_coverage": two_z.get("unsupported_coverage"),
        "within_tier_B": two_z.get("within_tier_B"),
    }
    base["reweighting_2x"] = {
        "residual_position": two_x.get("residual_position"),
        "share_of_B_residual_position": two_x.get("share_of_B_residual_position"),
        "synthetic_allocation": two_x.get("synthetic_allocation"),
        "recruit_mix": two_x.get("recruit_mix"),
        "within_tier_B": two_x.get("within_tier_B"),
    }
    rec = dict(base.get("reconciliation") or {})
    lethal_sum = (
        float(reweight.get("divine_shield") or 0.0)
        + float(reweight.get("poison_venomous") or 0.0)
        + float(reweight.get("cleave") or 0.0)
        + float(reweight.get("start_of_combat") or 0.0)
        + float(reweight.get("ordinary_combat") or 0.0)
        + float(reweight.get("still_unexplained") or 0.0)
    )
    rec.update({
        "phase_2v_survivor_tier_sum_delta": PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "phase_2z_unexplained": PHASE_2Z_UNEXPLAINED,
        "reproduced_within_tier_B": decomp.get("within_tier_survival_B"),
        "reproduced_2x_residual": two_x.get("residual_position"),
        "reproduced_2z_leftover": two_z.get("still_unexplained"),
        "reweight_direct_B": reweight.get("within_tier_B"),
        "reweight_2y_hat": reweight.get("phase_2y_unexplained_hat"),
        "reweight_2z_hat": reweight.get("phase_2z_unexplained_hat"),
        "lethal_sum": lethal_sum,
        "nested_residual_vs_2z": lethal_sum - PHASE_2Z_UNEXPLAINED,
        "event_count_mismatches_control": int(
            control.get("event_count_mismatches") or 0
        ),
        "event_count_mismatches_treatment": int(
            treatment.get("event_count_mismatches") or 0
        ),
        "death_causes_control": control.get("death_causes") or {},
        "death_causes_treatment": treatment.get("death_causes") or {},
        "shield_pop_causes_control": control.get("shield_pop_causes") or {},
        "shield_pop_causes_treatment": treatment.get("shield_pop_causes") or {},
    })
    base["reconciliation"] = rec
    base["example_minions"] = {
        "control": control.get("example_minions") or [],
        "treatment": treatment.get("example_minions") or [],
    }
    return base


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_3a(comparison, non_evaluative=non_evaluative)
