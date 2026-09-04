"""Phase 3D — observational split of the 3C attacker-attack-strength term.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. Holds the 3B cells through damaging-hit count,
then splits 3C A (+0.512 attack-strength mix) into opposing board-pool
magnitude, allocation concentration onto attacking bodies, in-combat
attack mutation, and leftover residual attack-strength mix.

Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.
"""

from __future__ import annotations

import statistics as st
from typing import Dict, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.attacker_punch_diagnostic import (
    AttackPunchTracer,
    _arm_prefixes,
    _walk,
    collect_punch_minions,
    reweight_attacker_punch,
    summarize_punch_arm,
)
from ml.combat_mechanics_diagnostic import reweight_combat_mechanics
from ml.hp_depletion_diagnostic import reweight_hp_depletion
from ml.lethal_cause_diagnostic import reweight_lethal_cause
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
    SLOT_BIN_CAP,
    cleave_bin,
    ds_bin,
    ordinary_bin,
    poison_bin,
    slot_bin,
    soc_bin,
)
from ml.phase_3b_prereg import (
    N_HIT_BINS,
    hit_count_bin,
)
from ml.phase_3c_prereg import (
    N_ATK_BINS,
    N_PAIR_BINS,
    N_SYNTH_ATK_BINS,
    attacker_attack_value,
    attacker_synth_share_value,
    pairing_order_value,
)
from ml.phase_3d_prereg import (
    N_CONC_BINS,
    N_DELTA_BINS,
    N_POOL_BINS,
    PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
    PHASE_2V_WITHIN_TIER_B,
    PHASE_2X_RESIDUAL_POSITION,
    PHASE_2Y_UNEXPLAINED,
    PHASE_2Z_UNEXPLAINED,
    PHASE_3A_UNEXPLAINED,
    PHASE_3B_DAMAGE_PER_HIT,
    PHASE_3B_DAMAGING_HITS,
    PHASE_3B_OVERKILL,
    PHASE_3B_UNEXPLAINED,
    PHASE_3C_ATTACKER_ATTACK_STRENGTH,
    PHASE_3C_PAIRING_ORDER,
    PHASE_3C_SHARE_ATTACKER_ATTACK,
    PHASE_3C_SYNTH_COMPOSITION,
    PHASE_3C_UNEXPLAINED,
    allocation_concentration_value,
    assert_seed_range_allowed,
    board_pool_value,
    combat_delta_value,
    diagnose_phase_3d,
)
from ml.survivor_composition_diagnostic import TIERS
from ml.synthetic_allocation_diagnostic import (
    _cond_p,
    _hits,
    _kitagawa_two,
    _safe_div,
    bin_value,
    compare_allocation,
    decile_edges,
    reweight_within_tier,
)

METHODOLOGY_VERSION = "3d_v1"

_SOURCE_KEYS = (
    "mean_attacker_start_attack",
    "mean_attacker_start_recruit",
    "mean_attacker_start_pool",
    "mean_attacker_combat_delta",
    "mean_attacker_pool_share_of_board",
    "mean_attacker_pool_rank",
    "mean_attacker_board_pool",
    "mean_attacker_board_recruit",
    "mean_attacker_board_size",
    "mean_attacker_board_mean_tier",
    "n_attack_identity",
    "n_attack_identity_ok",
    "n_attacker_start_represented",
    "attack_identity_ok",
    "opp_board_pool_attack",
    "opp_board_recruit_attack",
    "opp_board_size",
    "opp_board_mean_tier",
    "opp_board_tier_hist",
    "opp_attacking_pool_attack",
    "opp_n_attacked",
    "opp_pool_on_attackers_share",
)

_SOURCE_RECONCILE = (
    "attack_identity_reconcile",
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
    "damaging_hits",
    "board_pool_magnitude",
    "allocation_concentration",
    "combat_mutation",
    "attacker_attack_strength",
    "attacker_synth_composition",
    "pairing_order",
    "still_unexplained",
)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


class AttackSourceTracer(AttackPunchTracer):
    """3C punch rows plus combat-start recruit / pool / combat-delta tags."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        start_combat = list(rec.get("start_combat_bodies") or [])
        by_id = {str(b.get("body_id") or ""): b for b in start_combat}
        rows = list(rec.get("start_minions") or [])
        for r in rows:
            src = by_id.get(str(r.get("body_id") or ""), {})
            for k in _SOURCE_KEYS:
                if k in src:
                    r[k] = src[k]
                elif k not in r:
                    r[k] = src.get(k)
        counts = dict(rec.get("event_counts") or fight.get("event_counts") or {})
        rec["event_counts"] = counts
        rec["event_counts_ok"] = _event_counts_ok(counts)
        rec["starting_loser"] = list(fight.get("starting_loser") or [])


def _event_counts_ok(counts: Dict) -> bool:
    if not counts:
        return True
    flags = (
        "attacks_reconcile",
        "targets_reconcile",
        "forced_open_reconcile",
        "created_reconcile",
        "deaths_reconcile",
        "hits_reconcile",
        "hp_flow_reconcile",
        "incoming_reconcile",
        "damaging_hits_reconcile",
        "ordinary_hp_loss_reconcile",
        "ordinary_kind_reconcile",
        "ordinary_ok_reconcile",
    ) + _SOURCE_RECONCILE
    return all(bool(counts.get(k, True)) for k in flags)


def run_source_arm(
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
                tracer = AttackSourceTracer(i, seed + i, arm)
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


def run_greedy_control_source(lobbies: int, seed: int) -> Dict:
    return run_source_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_source(lobbies: int, seed: int) -> Dict:
    return run_source_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def collect_source_minions(hits: Sequence[Dict]) -> List[Dict]:
    rows = collect_punch_minions(hits)
    for row in rows:
        if row.get("opp_board_pool_attack") in (None,):
            row["opp_board_pool_attack"] = float(
                row.get("mean_attacker_board_pool") or 0
            )
        if row.get("opp_pool_on_attackers_share") in (None,):
            row["opp_pool_on_attackers_share"] = float(
                row.get("mean_attacker_pool_share_of_board") or 0
            )
        if row.get("mean_attacker_combat_delta") in (None,):
            row["mean_attacker_combat_delta"] = 0.0
        if row.get("attack_identity_ok") in (None,):
            row["attack_identity_ok"] = True
    return rows


def _by_tier_source(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r["tier"]) == tier]
        n = len(cell)
        punched = [r for r in cell if int(r.get("n_damaging_hits") or 0) > 0]
        out[str(tier)] = {
            "n_start": n,
            "p_survive": _safe_div(
                float(sum(1 for r in cell if r.get("survived"))), float(n)
            ),
            "mean_attacker_attack": _mean([
                attacker_attack_value(r) for r in punched
            ]),
            "mean_attacker_start_recruit": _mean([
                float(r.get("mean_attacker_start_recruit") or 0) for r in punched
            ]),
            "mean_attacker_start_pool": _mean([
                float(r.get("mean_attacker_start_pool") or 0) for r in punched
            ]),
            "mean_attacker_combat_delta": _mean([
                combat_delta_value(r) for r in punched
            ]),
            "mean_opp_board_pool": _mean([
                board_pool_value(r) for r in cell
            ]),
            "mean_opp_board_recruit": _mean([
                float(r.get("opp_board_recruit_attack") or 0) for r in cell
            ]),
            "mean_opp_board_size": _mean([
                float(r.get("opp_board_size") or 0) for r in cell
            ]),
            "mean_opp_board_mean_tier": _mean([
                float(r.get("opp_board_mean_tier") or 0) for r in cell
            ]),
            "mean_pool_on_attackers_share": _mean([
                allocation_concentration_value(r) for r in cell
            ]),
            "mean_attacker_pool_share_of_board": _mean([
                float(r.get("mean_attacker_pool_share_of_board") or 0)
                for r in punched
            ]),
            "mean_attacker_pool_rank": _mean([
                float(r.get("mean_attacker_pool_rank") or 0) for r in punched
            ]),
            "p_attack_identity_ok": _safe_div(
                float(sum(1 for r in cell if r.get("attack_identity_ok") is True)),
                float(n),
            ),
            "mean_per_hit": _safe_div(float(n), float(n_hits)) if n_hits else None,
        }
    return out


def _source_row_key(
    r: Dict,
    recruit_edges: Dict[int, List[float]],
    synth_edges: Dict[Tuple[int, int], List[float]],
    team_edges: Dict[int, List[float]],
    pool_edges: Dict[int, List[float]],
    conc_edges: Dict[int, List[float]],
    delta_edges: Dict[int, List[float]],
    atk_edges: Dict[int, List[float]],
    synth_atk_edges: Dict[int, List[float]],
    pair_edges: Dict[int, List[float]],
) -> Tuple:
    t = int(r["tier"])
    rb = bin_value(float(r["recruit_raw"]), recruit_edges[t])
    sb = bin_value(float(r["synthetic_share"]), synth_edges[(t, rb)])
    kb = slot_bin(r.get("board_slot"))
    mb = bin_value(float(r.get("teammate_combat_raw") or 0), team_edges[t])
    hb = hit_count_bin(r)
    pb_pool = bin_value(board_pool_value(r), pool_edges[t])
    cb = bin_value(allocation_concentration_value(r), conc_edges[t])
    db = bin_value(combat_delta_value(r), delta_edges[t])
    ab = bin_value(attacker_attack_value(r), atk_edges[t])
    yb = bin_value(attacker_synth_share_value(r), synth_atk_edges[t])
    pb = bin_value(pairing_order_value(r), pair_edges[t])
    return (
        t, rb, sb, kb, mb,
        target_bin(r), cursor_bin(r), gen_bin(r), unsupported_bin(r),
        ds_bin(r), poison_bin(r), cleave_bin(r), soc_bin(r), ordinary_bin(r),
        hb, pb_pool, cb, db, ab, yb, pb,
    )


def reweight_attack_source(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    n_hits_c: int,
    n_hits_t: int,
    observed_B: Optional[float] = None,
    observed_residual: Optional[float] = None,
    observed_leftover_2y: Optional[float] = None,
    observed_leftover_2z: Optional[float] = None,
    observed_leftover_3a: Optional[float] = None,
    observed_damage_per_hit: Optional[float] = None,
    observed_attack_strength: Optional[float] = None,
) -> Dict:
    """Hold 3B cells through hit-count, then split +0.512 into A1/A2/A3/A4.

    Nested Kitagawa:

        hold P(recruit-raw | tier) … P(hit_count_bin | 3A cells)
            ↓
        hold P(opp board-pool quintile | …)           →  (A1) pool magnitude
            ↓
        hold P(pool-on-attackers quintile | …)        →  (A2) concentration
            ↓
        hold P(combat-delta quintile | …)             →  (A3) combat mutation
            ↓
        hold P(attacker_attack quintile | …)          →  leftover of A
            ↓
        hold P(synth_attack_share / pairing | …)      →  3C B/C
            ↓
        leftover P(survive | all)                     →  3C D
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
    pool_edges: Dict[int, List[float]] = {}
    conc_edges: Dict[int, List[float]] = {}
    delta_edges: Dict[int, List[float]] = {}
    atk_edges: Dict[int, List[float]] = {}
    synth_atk_edges: Dict[int, List[float]] = {}
    pair_edges: Dict[int, List[float]] = {}
    for t in TIERS:
        cell = [r for r in pooled if int(r["tier"]) == t]
        team_edges[t] = decile_edges(
            [float(r.get("teammate_combat_raw") or 0) for r in cell],
            n=N_TEAM_BINS,
        )
        pool_edges[t] = decile_edges(
            [board_pool_value(r) for r in cell], n=N_POOL_BINS,
        )
        conc_edges[t] = decile_edges(
            [allocation_concentration_value(r) for r in cell], n=N_CONC_BINS,
        )
        delta_edges[t] = decile_edges(
            [combat_delta_value(r) for r in cell], n=N_DELTA_BINS,
        )
        atk_edges[t] = decile_edges(
            [attacker_attack_value(r) for r in cell], n=N_ATK_BINS,
        )
        synth_atk_edges[t] = decile_edges(
            [attacker_synth_share_value(r) for r in cell], n=N_SYNTH_ATK_BINS,
        )
        pair_edges[t] = decile_edges(
            [pairing_order_value(r) for r in cell], n=N_PAIR_BINS,
        )

    def _key(r: Dict) -> Tuple:
        return _source_row_key(
            r, recruit_edges, synth_edges, team_edges,
            pool_edges, conc_edges, delta_edges,
            atk_edges, synth_atk_edges, pair_edges,
        )

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
        14: N_HIT_BINS,
    }
    n_depths = 21  # tier + 20 nested (recruit … pairing)

    def n_bins_at(depth: int, prefix: Tuple) -> int:
        if depth == 1:
            return len(recruit_edges[prefix[0]]) + 1
        if depth == 2:
            return len(synth_edges[(prefix[0], prefix[1])]) + 1
        if depth == 3:
            return SLOT_BIN_CAP + 1
        if depth == 4:
            return len(team_edges[prefix[0]]) + 1
        if depth == 15:
            return len(pool_edges[prefix[0]]) + 1
        if depth == 16:
            return len(conc_edges[prefix[0]]) + 1
        if depth == 17:
            return len(delta_edges[prefix[0]]) + 1
        if depth == 18:
            return len(atk_edges[prefix[0]]) + 1
        if depth == 19:
            return len(synth_atk_edges[prefix[0]]) + 1
        if depth == 20:
            return len(pair_edges[prefix[0]]) + 1
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
            "phase_3a_unexplained_hat": 0.0,
            "phase_3b_damage_per_hit_hat": 0.0,
            "phase_3c_attacker_attack_strength_hat": 0.0,
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
            + cell["damaging_hits"] + cell["board_pool_magnitude"]
            + cell["allocation_concentration"] + cell["combat_mutation"]
            + cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"] + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        leftover_2z = (
            cell["divine_shield"] + cell["poison_venomous"] + cell["cleave"]
            + cell["start_of_combat"] + cell["ordinary_combat"]
            + cell["damaging_hits"] + cell["board_pool_magnitude"]
            + cell["allocation_concentration"] + cell["combat_mutation"]
            + cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"] + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        leftover_3a = (
            cell["damaging_hits"] + cell["board_pool_magnitude"]
            + cell["allocation_concentration"] + cell["combat_mutation"]
            + cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"] + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        punch_sum = (
            cell["board_pool_magnitude"] + cell["allocation_concentration"]
            + cell["combat_mutation"] + cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"] + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        a_hat = (
            cell["board_pool_magnitude"] + cell["allocation_concentration"]
            + cell["combat_mutation"] + cell["attacker_attack_strength"]
        )
        cell["phase_2y_unexplained_hat"] = leftover_2y
        cell["phase_2z_unexplained_hat"] = leftover_2z
        cell["phase_3a_unexplained_hat"] = leftover_3a
        cell["phase_3b_damage_per_hit_hat"] = punch_sum
        cell["phase_3c_attacker_attack_strength_hat"] = a_hat
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
        + totals["damaging_hits"] + totals["board_pool_magnitude"]
        + totals["allocation_concentration"] + totals["combat_mutation"]
        + totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"] + totals["pairing_order"]
        + totals["still_unexplained"]
    )
    leftover_2z = (
        totals["divine_shield"] + totals["poison_venomous"] + totals["cleave"]
        + totals["start_of_combat"] + totals["ordinary_combat"]
        + totals["damaging_hits"] + totals["board_pool_magnitude"]
        + totals["allocation_concentration"] + totals["combat_mutation"]
        + totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"] + totals["pairing_order"]
        + totals["still_unexplained"]
    )
    leftover_3a = (
        totals["damaging_hits"] + totals["board_pool_magnitude"]
        + totals["allocation_concentration"] + totals["combat_mutation"]
        + totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"] + totals["pairing_order"]
        + totals["still_unexplained"]
    )
    punch_sum = (
        totals["board_pool_magnitude"] + totals["allocation_concentration"]
        + totals["combat_mutation"] + totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"] + totals["pairing_order"]
        + totals["still_unexplained"]
    )
    a_hat = (
        totals["board_pool_magnitude"] + totals["allocation_concentration"]
        + totals["combat_mutation"] + totals["attacker_attack_strength"]
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
    obs_f = (
        float(observed_leftover_3a)
        if observed_leftover_3a is not None
        else leftover_3a
    )
    obs_dmg = (
        float(observed_damage_per_hit)
        if observed_damage_per_hit is not None
        else PHASE_3B_DAMAGE_PER_HIT
    )
    obs_a = (
        float(observed_attack_strength)
        if observed_attack_strength is not None
        else PHASE_3C_ATTACKER_ATTACK_STRENGTH
    )

    def _share(part: float, denom: float) -> Optional[float]:
        if abs(denom) < 1e-12:
            return None
        return float(part) / denom

    return {
        "method": (
            "nested_kitagawa_3b_hit_cells_then_pool_alloc_combat_attack"
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
        "n_hit_bins": N_HIT_BINS,
        "n_pool_bins": N_POOL_BINS,
        "n_conc_bins": N_CONC_BINS,
        "n_delta_bins": N_DELTA_BINS,
        "n_atk_bins": N_ATK_BINS,
        "n_synth_atk_bins": N_SYNTH_ATK_BINS,
        "n_pair_bins": N_PAIR_BINS,
        "within_tier_B": b_direct,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "phase_2z_unexplained": PHASE_2Z_UNEXPLAINED,
        "phase_3a_unexplained": PHASE_3A_UNEXPLAINED,
        "phase_3b_damage_per_hit": PHASE_3B_DAMAGE_PER_HIT,
        "phase_3c_attacker_attack_strength": PHASE_3C_ATTACKER_ATTACK_STRENGTH,
        "observed_B_used_for_shares": obs_b,
        "observed_residual_used_for_shares": obs_r,
        "observed_leftover_2y_used_for_shares": obs_c,
        "observed_leftover_2z_used_for_shares": obs_e,
        "observed_leftover_used_for_shares": obs_f,
        "observed_damage_per_hit_used_for_shares": obs_dmg,
        "observed_attack_strength_used_for_shares": obs_a,
        **totals,
        "phase_2y_unexplained_hat": leftover_2y,
        "phase_2z_unexplained_hat": leftover_2z,
        "phase_3a_unexplained_hat": leftover_3a,
        "phase_3b_damage_per_hit_hat": punch_sum,
        "phase_3c_attacker_attack_strength_hat": a_hat,
        "residual_position_hat": resid_hat,
        "explained_all_parts": explained,
        "residual_vs_direct_B": b_direct - explained,
        "share_of_a_board_pool_magnitude": _share(
            totals["board_pool_magnitude"], obs_a
        ),
        "share_of_a_allocation_concentration": _share(
            totals["allocation_concentration"], obs_a
        ),
        "share_of_a_combat_mutation": _share(
            totals["combat_mutation"], obs_a
        ),
        "share_of_a_still_unexplained": _share(
            totals["attacker_attack_strength"], obs_a
        ),
        "share_of_b_board_pool_magnitude": _share(
            totals["board_pool_magnitude"], obs_dmg
        ),
        "share_of_b_allocation_concentration": _share(
            totals["allocation_concentration"], obs_dmg
        ),
        "share_of_b_combat_mutation": _share(
            totals["combat_mutation"], obs_dmg
        ),
        "per_tier": per_tier,
    }


def _slim_example(row: Dict) -> Dict:
    keep = {
        "name", "tier", "survived", "died", "taunt", "board_slot", "slot_bin",
        "n_attacks", "first_attack_index", "n_targeted", "death_cause",
        "start_health", "end_health", "cumulative_incoming",
        "n_damaging_hits", "n_hits", "overkill_on_death",
        "mean_incoming_dmg", "hp_depletion_margin", "hp_flow_ok",
        "ordinary_hp_loss_ok", "attack_identity_ok",
        "mean_attacker_attack", "mean_attacker_recruit_attack",
        "mean_attacker_start_recruit", "mean_attacker_start_pool",
        "mean_attacker_combat_delta", "mean_attacker_synth_share",
        "mean_attacker_pool_share_of_board", "mean_attacker_pool_rank",
        "opp_board_pool_attack", "opp_board_recruit_attack", "opp_board_size",
        "opp_board_mean_tier", "opp_pool_on_attackers_share",
        "opp_attacking_pool_attack", "opp_n_attacked",
        "first_attacker_id", "first_attacker_name", "pairing_order_value",
        "effect_status", "target_bin", "cursor_bin", "gen_bin",
        "unsupported_bin", "ds_bin", "poison_bin", "cleave_bin", "soc_bin",
        "ordinary_bin", "hit_count_bin", "teammate_combat_raw",
        "recruit_raw", "synthetic_share",
    }
    return {k: row.get(k) for k in keep}


def summarize_source_arm(raw: Dict) -> Dict:
    summary = summarize_punch_arm(raw)
    hits = _hits(raw["fights"])
    rows = collect_source_minions(hits)
    n_hits = len(hits)
    by_tier = _by_tier_source(rows, n_hits)
    n_mismatch = 0
    n_counted = 0
    n_identity_bad = 0
    for f in hits:
        counts = f.get("event_counts") or {}
        n_counted += 1
        if not _event_counts_ok(counts):
            n_mismatch += 1
        if counts.get("attack_identity_reconcile") is False:
            n_identity_bad += 1
    punched = [r for r in rows if int(r.get("n_damaging_hits") or 0) > 0]
    summary.update({
        "n_start_minions": len(rows),
        "by_tier_source": by_tier,
        "mean_attacker_attack": _mean([
            attacker_attack_value(r) for r in punched
        ]),
        "mean_attacker_start_recruit": _mean([
            float(r.get("mean_attacker_start_recruit") or 0) for r in punched
        ]),
        "mean_attacker_start_pool": _mean([
            float(r.get("mean_attacker_start_pool") or 0) for r in punched
        ]),
        "mean_attacker_combat_delta": _mean([
            combat_delta_value(r) for r in punched
        ]),
        "mean_opp_board_pool": _mean([board_pool_value(r) for r in rows]),
        "mean_opp_board_recruit": _mean([
            float(r.get("opp_board_recruit_attack") or 0) for r in rows
        ]),
        "mean_pool_on_attackers_share": _mean([
            allocation_concentration_value(r) for r in rows
        ]),
        "p_attack_identity_ok": _safe_div(
            float(sum(1 for r in rows if r.get("attack_identity_ok") is True)),
            float(len(rows)),
        ),
        "event_count_mismatches": n_mismatch,
        "attack_identity_mismatches": n_identity_bad,
        "n_hits_counted": n_counted,
        "example_minions": [_slim_example(r) for r in rows[:8]],
        "_rows": rows,
        "_n_hits": n_hits,
    })
    return summary


def compare_source(control: Dict, treatment: Dict) -> Dict:
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
    three_a = reweight_lethal_cause(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover_2y=PHASE_2Y_UNEXPLAINED,
        observed_leftover_2z=PHASE_2Z_UNEXPLAINED,
    )
    three_b = reweight_hp_depletion(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover_2y=PHASE_2Y_UNEXPLAINED,
        observed_leftover_2z=PHASE_2Z_UNEXPLAINED,
        observed_leftover_3a=PHASE_3A_UNEXPLAINED,
    )
    three_c = reweight_attacker_punch(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover_2y=PHASE_2Y_UNEXPLAINED,
        observed_leftover_2z=PHASE_2Z_UNEXPLAINED,
        observed_leftover_3a=PHASE_3A_UNEXPLAINED,
        observed_damage_per_hit=PHASE_3B_DAMAGE_PER_HIT,
    )
    reweight = reweight_attack_source(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover_2y=PHASE_2Y_UNEXPLAINED,
        observed_leftover_2z=PHASE_2Z_UNEXPLAINED,
        observed_leftover_3a=PHASE_3A_UNEXPLAINED,
        observed_damage_per_hit=PHASE_3B_DAMAGE_PER_HIT,
        observed_attack_strength=PHASE_3C_ATTACKER_ATTACK_STRENGTH,
    )

    def _tier_delta(table_c: Dict, table_t: Dict, key: str) -> Dict[str, Optional[float]]:
        out = {}
        for tier in TIERS:
            k = str(tier)
            a = (table_c or {}).get(k, {}).get(key)
            b = (table_t or {}).get(k, {}).get(key)
            out[k] = None if a is None or b is None else float(b) - float(a)
        return out

    src_c = control.get("by_tier_source") or {}
    src_t = treatment.get("by_tier_source") or {}
    base["by_tier_source"] = {
        "control": src_c,
        "treatment": src_t,
        "delta_mean_opp_board_pool": _tier_delta(
            src_c, src_t, "mean_opp_board_pool"
        ),
        "delta_mean_pool_on_attackers_share": _tier_delta(
            src_c, src_t, "mean_pool_on_attackers_share"
        ),
        "delta_mean_attacker_combat_delta": _tier_delta(
            src_c, src_t, "mean_attacker_combat_delta"
        ),
        "delta_mean_attacker_attack": _tier_delta(
            src_c, src_t, "mean_attacker_attack"
        ),
        "delta_p_survive": _tier_delta(src_c, src_t, "p_survive"),
    }
    base["reweighting"] = reweight
    base["reweighting_3c"] = {
        "attacker_attack_strength": three_c.get("attacker_attack_strength"),
        "share_of_b_attacker_attack_strength": three_c.get(
            "share_of_b_attacker_attack_strength"
        ),
        "attacker_synth_composition": three_c.get("attacker_synth_composition"),
        "pairing_order": three_c.get("pairing_order"),
        "still_unexplained": three_c.get("still_unexplained"),
        "phase_3b_damage_per_hit_hat": three_c.get("phase_3b_damage_per_hit_hat"),
        "within_tier_B": three_c.get("within_tier_B"),
    }
    base["reweighting_3b"] = {
        "damage_per_hit": three_b.get("damage_per_hit"),
        "share_of_leftover_damage_per_hit": three_b.get(
            "share_of_leftover_damage_per_hit"
        ),
        "damaging_hits": three_b.get("damaging_hits"),
        "overkill_threshold": three_b.get("overkill_threshold"),
        "still_unexplained": three_b.get("still_unexplained"),
        "phase_3a_unexplained_hat": three_b.get("phase_3a_unexplained_hat"),
        "within_tier_B": three_b.get("within_tier_B"),
    }
    rec = dict(base.get("reconciliation") or {})
    a_parts = (
        float(reweight.get("board_pool_magnitude") or 0.0)
        + float(reweight.get("allocation_concentration") or 0.0)
        + float(reweight.get("combat_mutation") or 0.0)
        + float(reweight.get("attacker_attack_strength") or 0.0)
    )
    punch_parts = (
        a_parts
        + float(reweight.get("attacker_synth_composition") or 0.0)
        + float(reweight.get("pairing_order") or 0.0)
        + float(reweight.get("still_unexplained") or 0.0)
    )
    remainder_3b = (
        PHASE_3B_DAMAGE_PER_HIT + PHASE_3B_OVERKILL + PHASE_3B_UNEXPLAINED
    )
    rec.update({
        "phase_2v_survivor_tier_sum_delta": PHASE_2V_SURVIVOR_TIER_SUM_DELTA,
        "phase_2v_within_tier_B": PHASE_2V_WITHIN_TIER_B,
        "phase_2x_residual_position": PHASE_2X_RESIDUAL_POSITION,
        "phase_2y_unexplained": PHASE_2Y_UNEXPLAINED,
        "phase_2z_unexplained": PHASE_2Z_UNEXPLAINED,
        "phase_3a_unexplained": PHASE_3A_UNEXPLAINED,
        "phase_3b_damage_per_hit": PHASE_3B_DAMAGE_PER_HIT,
        "phase_3b_damaging_hits": PHASE_3B_DAMAGING_HITS,
        "phase_3c_attacker_attack_strength": PHASE_3C_ATTACKER_ATTACK_STRENGTH,
        "phase_3c_share_attacker_attack": PHASE_3C_SHARE_ATTACKER_ATTACK,
        "phase_3c_synth_composition": PHASE_3C_SYNTH_COMPOSITION,
        "phase_3c_pairing_order": PHASE_3C_PAIRING_ORDER,
        "phase_3c_unexplained": PHASE_3C_UNEXPLAINED,
        "reproduced_within_tier_B": decomp.get("within_tier_survival_B"),
        "reproduced_2x_residual": two_x.get("residual_position"),
        "reproduced_2z_leftover": two_z.get("still_unexplained"),
        "reproduced_3a_leftover": three_a.get("still_unexplained"),
        "reproduced_3b_damage_per_hit": three_b.get("damage_per_hit"),
        "reproduced_3c_attacker_attack_strength": three_c.get(
            "attacker_attack_strength"
        ),
        "reweight_direct_B": reweight.get("within_tier_B"),
        "reweight_2y_hat": reweight.get("phase_2y_unexplained_hat"),
        "reweight_2z_hat": reweight.get("phase_2z_unexplained_hat"),
        "reweight_3a_hat": reweight.get("phase_3a_unexplained_hat"),
        "reweight_3b_hat": reweight.get("phase_3b_damage_per_hit_hat"),
        "reweight_3c_a_hat": reweight.get("phase_3c_attacker_attack_strength_hat"),
        "source_sum": a_parts,
        "punch_sum": punch_parts,
        "phase_3b_remainder_after_hits": remainder_3b,
        "nested_residual_vs_3b_remainder": punch_parts - remainder_3b,
        "nested_residual_vs_3c_a": a_parts - PHASE_3C_ATTACKER_ATTACK_STRENGTH,
        "event_count_mismatches_control": int(
            control.get("event_count_mismatches") or 0
        ),
        "event_count_mismatches_treatment": int(
            treatment.get("event_count_mismatches") or 0
        ),
        "attack_identity_mismatches_control": int(
            control.get("attack_identity_mismatches") or 0
        ),
        "attack_identity_mismatches_treatment": int(
            treatment.get("attack_identity_mismatches") or 0
        ),
        "ordinary_hp_loss_mismatches_control": int(
            control.get("ordinary_hp_loss_mismatches") or 0
        ),
        "ordinary_hp_loss_mismatches_treatment": int(
            treatment.get("ordinary_hp_loss_mismatches") or 0
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
    return diagnose_phase_3d(comparison, non_evaluative=non_evaluative)
