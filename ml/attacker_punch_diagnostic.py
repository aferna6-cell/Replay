"""Phase 3C — observational attacker-punch split of the 3B damage-per-hit term.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. Holds the 3B cells through damaging-hit count
(tier + recruit/raw + synth share + slot + teammate-raw + target + cursor +
gen-DR + unsupported + DS + poison + cleave + SOC + ordinary + hit-count),
then splits the 3B damage-per-hit / HP-margin term into attacker
attack-strength mix, synthetic-vs-recruit attack composition, same-strength
pairing / attack-order, and still unexplained.

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
from ml.combat_mechanics_diagnostic import reweight_combat_mechanics
from ml.hp_depletion_diagnostic import (
    HpDepletionTracer,
    reweight_hp_depletion,
)
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
    hp_margin_value,
    overkill_bin,
)
from ml.phase_3c_prereg import (
    N_ATK_BINS,
    N_PAIR_BINS,
    N_SYNTH_ATK_BINS,
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
    assert_seed_range_allowed,
    attacker_attack_value,
    attacker_synth_share_value,
    diagnose_phase_3c,
    pairing_order_value,
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

METHODOLOGY_VERSION = "3c_v1"

_PUNCH_KEYS = (
    "mean_attacker_attack",
    "mean_attacker_recruit_attack",
    "mean_attacker_synth_attack",
    "mean_attacker_synth_share",
    "mean_attacker_tier",
    "mean_attacker_slot",
    "mean_relative_slot",
    "mean_attacker_first_attack_index",
    "mean_defender_hp_before",
    "mean_hp_loss",
    "p_attacker_survived_swing",
    "p_hit_was_counterattack",
    "p_attacker_golden",
    "p_attacker_generated",
    "first_attacker_id",
    "first_attacker_name",
    "last_attacker_id",
    "last_attacker_attack",
    "pairing_order_value",
    "n_punch_hits",
    "n_ordinary_kind",
    "n_ordinary_ok",
    "n_shield_kind",
    "n_poison_kind",
    "n_cleave_kind",
    "n_soc_kind",
    "n_other_kind",
    "ordinary_hp_loss_ok",
)

_PUNCH_RECONCILE = (
    "ordinary_kind_reconcile",
    "ordinary_ok_reconcile",
    "shield_kind_reconcile",
    "poison_kind_reconcile",
    "ordinary_hp_loss_reconcile",
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
    "attacker_attack_strength",
    "attacker_synth_composition",
    "pairing_order",
    "still_unexplained",
)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


class AttackPunchTracer(HpDepletionTracer):
    """3B rows plus per-body attacker-punch / pairing tags."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        start_combat = list(rec.get("start_combat_bodies") or [])
        by_id = {str(b.get("body_id") or ""): b for b in start_combat}
        rows = list(rec.get("start_minions") or [])
        for r in rows:
            src = by_id.get(str(r.get("body_id") or ""), {})
            for k in _PUNCH_KEYS:
                if k in src:
                    r[k] = src[k]
                elif k not in r:
                    r[k] = src.get(k)
            if r.get("pairing_order_value") in (None,):
                r["pairing_order_value"] = pairing_order_value(r)
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
        "hits_reconcile",
        "hp_flow_reconcile",
        "incoming_reconcile",
        "damaging_hits_reconcile",
    ) + _PUNCH_RECONCILE
    return all(bool(counts.get(k, True)) for k in flags)


def run_punch_arm(
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
                tracer = AttackPunchTracer(i, seed + i, arm)
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
    return run_punch_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_punch_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def collect_punch_minions(hits: Sequence[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for f in hits:
        start = list(f.get("start_minions") or [])
        n_gen = int(f.get("n_board_generated_represented") or 0)
        for r in start:
            row = dict(r)
            row.pop("hit_events", None)
            row["n_board_generated_represented"] = int(
                row.get("n_board_generated_represented") or n_gen
            )
            n = int(row.get("n_hits") or 0)
            tot = float(row.get("cumulative_incoming") or 0)
            if row.get("mean_incoming_dmg") in (None,):
                row["mean_incoming_dmg"] = (tot / n) if n else 0.0
            if row.get("hp_depletion_margin") in (None,):
                row["hp_depletion_margin"] = hp_margin_value(row)
            if row.get("pairing_order_value") in (None,):
                row["pairing_order_value"] = pairing_order_value(row)
            row["target_bin"] = target_bin(row)
            row["cursor_bin"] = cursor_bin(row)
            row["gen_bin"] = gen_bin(row)
            row["unsupported_bin"] = unsupported_bin(row)
            row["ds_bin"] = ds_bin(row)
            row["poison_bin"] = poison_bin(row)
            row["cleave_bin"] = cleave_bin(row)
            row["soc_bin"] = soc_bin(row)
            row["ordinary_bin"] = ordinary_bin(row)
            row["hit_count_bin"] = hit_count_bin(row)
            row["overkill_bin"] = overkill_bin(row)
            row["slot_bin"] = slot_bin(row.get("board_slot"))
            row["winner_tavern_tier"] = int(f.get("winner_tavern_tier") or 1)
            rows.append(row)
    return rows


def _by_tier_punch(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
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
            "mean_attacker_recruit_attack": _mean([
                float(r.get("mean_attacker_recruit_attack") or 0) for r in punched
            ]),
            "mean_attacker_synth_share": _mean([
                attacker_synth_share_value(r) for r in punched
            ]),
            "mean_pairing_order": _mean([
                pairing_order_value(r) for r in punched
            ]),
            "mean_hp_loss": _mean([
                float(r.get("mean_hp_loss") or 0) for r in punched
            ]),
            "p_attacker_survived_swing": _mean([
                float(r.get("p_attacker_survived_swing") or 0) for r in punched
            ]),
            "p_hit_was_counterattack": _mean([
                float(r.get("p_hit_was_counterattack") or 0) for r in punched
            ]),
            "p_ordinary_hp_loss_ok": _safe_div(
                float(sum(1 for r in cell if r.get("ordinary_hp_loss_ok") is True)),
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
    ab = bin_value(attacker_attack_value(r), atk_edges[t])
    yb = bin_value(attacker_synth_share_value(r), synth_atk_edges[t])
    pb = bin_value(pairing_order_value(r), pair_edges[t])
    return (
        t, rb, sb, kb, mb,
        target_bin(r), cursor_bin(r), gen_bin(r), unsupported_bin(r),
        ds_bin(r), poison_bin(r), cleave_bin(r), soc_bin(r), ordinary_bin(r),
        hb, ab, yb, pb,
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


def reweight_attacker_punch(
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
) -> Dict:
    """Hold 3B cells through hit-count, then split +0.939 into A/B/C/D.

    Nested Kitagawa:

        hold P(recruit-raw | tier) … P(hit_count_bin | 3A cells)
            ↓
        hold P(attacker_attack quintile | …)          →  (A) attack-strength
            ↓
        hold P(synth_attack_share quintile | …)       →  (B) synth vs recruit
            ↓
        hold P(pairing_order quintile | …)            →  (C) pairing / order
            ↓
        leftover P(survive | all)                     →  (D) still unexplained
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

    atk_edges: Dict[int, List[float]] = {}
    synth_atk_edges: Dict[int, List[float]] = {}
    pair_edges: Dict[int, List[float]] = {}
    for t in TIERS:
        cell = [r for r in pooled if int(r["tier"]) == t]
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
        return _row_key(
            r, recruit_edges, synth_edges, team_edges,
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
    n_depths = 18  # tier + 17 nested (recruit … pairing)

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
            return len(atk_edges[prefix[0]]) + 1
        if depth == 16:
            return len(synth_atk_edges[prefix[0]]) + 1
        if depth == 17:
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
            + cell["damaging_hits"] + cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"] + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        leftover_2z = (
            cell["divine_shield"] + cell["poison_venomous"] + cell["cleave"]
            + cell["start_of_combat"] + cell["ordinary_combat"]
            + cell["damaging_hits"] + cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"] + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        leftover_3a = (
            cell["damaging_hits"] + cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"] + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        punch_sum = (
            cell["attacker_attack_strength"]
            + cell["attacker_synth_composition"]
            + cell["pairing_order"]
            + cell["still_unexplained"]
        )
        cell["phase_2y_unexplained_hat"] = leftover_2y
        cell["phase_2z_unexplained_hat"] = leftover_2z
        cell["phase_3a_unexplained_hat"] = leftover_3a
        cell["phase_3b_damage_per_hit_hat"] = punch_sum
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
        + totals["damaging_hits"] + totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"] + totals["pairing_order"]
        + totals["still_unexplained"]
    )
    leftover_2z = (
        totals["divine_shield"] + totals["poison_venomous"] + totals["cleave"]
        + totals["start_of_combat"] + totals["ordinary_combat"]
        + totals["damaging_hits"] + totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"] + totals["pairing_order"]
        + totals["still_unexplained"]
    )
    leftover_3a = (
        totals["damaging_hits"] + totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"] + totals["pairing_order"]
        + totals["still_unexplained"]
    )
    punch_sum = (
        totals["attacker_attack_strength"]
        + totals["attacker_synth_composition"]
        + totals["pairing_order"]
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

    def _share(part: float, denom: float) -> Optional[float]:
        if abs(denom) < 1e-12:
            return None
        return float(part) / denom

    return {
        "method": (
            "nested_kitagawa_3b_hit_cells_then_attack_synth_pairing"
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
        "observed_B_used_for_shares": obs_b,
        "observed_residual_used_for_shares": obs_r,
        "observed_leftover_2y_used_for_shares": obs_c,
        "observed_leftover_2z_used_for_shares": obs_e,
        "observed_leftover_used_for_shares": obs_f,
        "observed_damage_per_hit_used_for_shares": obs_dmg,
        **totals,
        "phase_2y_unexplained_hat": leftover_2y,
        "phase_2z_unexplained_hat": leftover_2z,
        "phase_3a_unexplained_hat": leftover_3a,
        "phase_3b_damage_per_hit_hat": punch_sum,
        "residual_position_hat": resid_hat,
        "explained_all_parts": explained,
        "residual_vs_direct_B": b_direct - explained,
        "share_of_b_attacker_attack_strength": _share(
            totals["attacker_attack_strength"], obs_dmg
        ),
        "share_of_b_attacker_synth_composition": _share(
            totals["attacker_synth_composition"], obs_dmg
        ),
        "share_of_b_pairing_order": _share(totals["pairing_order"], obs_dmg),
        "share_of_b_still_unexplained": _share(
            totals["still_unexplained"], obs_dmg
        ),
        "share_of_3a_damaging_hits": _share(totals["damaging_hits"], leftover_3a),
        "share_of_3a_attacker_attack_strength": _share(
            totals["attacker_attack_strength"], leftover_3a
        ),
        "share_of_3a_attacker_synth_composition": _share(
            totals["attacker_synth_composition"], leftover_3a
        ),
        "share_of_3a_pairing_order": _share(totals["pairing_order"], leftover_3a),
        "share_of_3a_still_unexplained": _share(
            totals["still_unexplained"], leftover_3a
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
        "ordinary_hp_loss_ok",
        "mean_attacker_attack", "mean_attacker_recruit_attack",
        "mean_attacker_synth_share", "mean_attacker_tier", "mean_attacker_slot",
        "mean_relative_slot", "mean_attacker_first_attack_index",
        "mean_defender_hp_before", "mean_hp_loss",
        "p_attacker_survived_swing", "p_hit_was_counterattack",
        "p_attacker_golden", "p_attacker_generated",
        "first_attacker_id", "first_attacker_name", "pairing_order_value",
        "n_ordinary_kind", "n_ordinary_ok", "n_shield_kind", "n_poison_kind",
        "effect_status", "target_bin", "cursor_bin", "gen_bin",
        "unsupported_bin", "ds_bin", "poison_bin", "cleave_bin", "soc_bin",
        "ordinary_bin", "hit_count_bin", "teammate_combat_raw",
        "recruit_raw", "synthetic_share",
    }
    return {k: row.get(k) for k in keep}


def summarize_punch_arm(raw: Dict) -> Dict:
    summary = summarize_allocation_arm(raw)
    hits = _hits(raw["fights"])
    rows = collect_punch_minions(hits)
    n_hits = len(hits)
    by_tier = _by_tier_punch(rows, n_hits)
    n_mismatch = 0
    n_counted = 0
    n_ordinary_bad = 0
    death_causes: Dict[str, int] = defaultdict(int)
    for f in hits:
        counts = f.get("event_counts") or {}
        n_counted += 1
        if not _event_counts_ok(counts):
            n_mismatch += 1
        if counts.get("ordinary_hp_loss_reconcile") is False:
            n_ordinary_bad += 1
        for r in f.get("start_minions") or []:
            cause = r.get("death_cause")
            if cause:
                death_causes[str(cause)] += 1
            elif r.get("died"):
                death_causes["unrecorded"] += 1
    punched = [r for r in rows if int(r.get("n_damaging_hits") or 0) > 0]
    summary.update({
        "n_start_minions": len(rows),
        "by_tier_punch": by_tier,
        "mean_attacker_attack": _mean([
            attacker_attack_value(r) for r in punched
        ]),
        "mean_attacker_recruit_attack": _mean([
            float(r.get("mean_attacker_recruit_attack") or 0) for r in punched
        ]),
        "mean_attacker_synth_share": _mean([
            attacker_synth_share_value(r) for r in punched
        ]),
        "mean_pairing_order": _mean([
            pairing_order_value(r) for r in punched
        ]),
        "mean_hp_loss": _mean([
            float(r.get("mean_hp_loss") or 0) for r in punched
        ]),
        "p_attacker_survived_swing": _mean([
            float(r.get("p_attacker_survived_swing") or 0) for r in punched
        ]),
        "p_hit_was_counterattack": _mean([
            float(r.get("p_hit_was_counterattack") or 0) for r in punched
        ]),
        "p_ordinary_hp_loss_ok": _safe_div(
            float(sum(1 for r in rows if r.get("ordinary_hp_loss_ok") is True)),
            float(len(rows)),
        ),
        "death_causes": dict(death_causes),
        "event_count_mismatches": n_mismatch,
        "ordinary_hp_loss_mismatches": n_ordinary_bad,
        "n_hits_counted": n_counted,
        "example_minions": [_slim_example(r) for r in rows[:8]],
        "_rows": rows,
        "_n_hits": n_hits,
    })
    return summary


def compare_punch(control: Dict, treatment: Dict) -> Dict:
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

    reweight = reweight_attacker_punch(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=b_obs,
        observed_residual=resid_obs,
        observed_leftover_2y=PHASE_2Y_UNEXPLAINED,
        observed_leftover_2z=PHASE_2Z_UNEXPLAINED,
        observed_leftover_3a=PHASE_3A_UNEXPLAINED,
        observed_damage_per_hit=PHASE_3B_DAMAGE_PER_HIT,
    )

    def _tier_delta(table_c: Dict, table_t: Dict, key: str) -> Dict[str, Optional[float]]:
        out = {}
        for tier in TIERS:
            k = str(tier)
            a = (table_c or {}).get(k, {}).get(key)
            b = (table_t or {}).get(k, {}).get(key)
            out[k] = None if a is None or b is None else float(b) - float(a)
        return out

    punch_c = control.get("by_tier_punch") or {}
    punch_t = treatment.get("by_tier_punch") or {}
    base["by_tier_punch"] = {
        "control": punch_c,
        "treatment": punch_t,
        "delta_mean_attacker_attack": _tier_delta(
            punch_c, punch_t, "mean_attacker_attack"
        ),
        "delta_mean_attacker_synth_share": _tier_delta(
            punch_c, punch_t, "mean_attacker_synth_share"
        ),
        "delta_mean_pairing_order": _tier_delta(
            punch_c, punch_t, "mean_pairing_order"
        ),
        "delta_p_survive": _tier_delta(punch_c, punch_t, "p_survive"),
    }
    base["reweighting"] = reweight
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
    base["reweighting_3a"] = {
        "still_unexplained": three_a.get("still_unexplained"),
        "share_of_leftover_still_unexplained": three_a.get(
            "share_of_leftover_still_unexplained"
        ),
        "phase_2z_unexplained_hat": three_a.get("phase_2z_unexplained_hat"),
        "within_tier_B": three_a.get("within_tier_B"),
    }
    base["reweighting_2z"] = {
        "still_unexplained": two_z.get("still_unexplained"),
        "share_of_leftover_still_unexplained": two_z.get(
            "share_of_leftover_still_unexplained"
        ),
        "phase_2y_unexplained_hat": two_z.get("phase_2y_unexplained_hat"),
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
    punch_parts = (
        float(reweight.get("attacker_attack_strength") or 0.0)
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
        "reproduced_within_tier_B": decomp.get("within_tier_survival_B"),
        "reproduced_2x_residual": two_x.get("residual_position"),
        "reproduced_2z_leftover": two_z.get("still_unexplained"),
        "reproduced_3a_leftover": three_a.get("still_unexplained"),
        "reproduced_3b_damage_per_hit": three_b.get("damage_per_hit"),
        "reweight_direct_B": reweight.get("within_tier_B"),
        "reweight_2y_hat": reweight.get("phase_2y_unexplained_hat"),
        "reweight_2z_hat": reweight.get("phase_2z_unexplained_hat"),
        "reweight_3a_hat": reweight.get("phase_3a_unexplained_hat"),
        "reweight_3b_hat": reweight.get("phase_3b_damage_per_hit_hat"),
        "punch_sum": punch_parts,
        "phase_3b_remainder_after_hits": remainder_3b,
        "nested_residual_vs_3b_remainder": punch_parts - remainder_3b,
        "nested_residual_vs_3b_b": punch_parts - PHASE_3B_DAMAGE_PER_HIT,
        "event_count_mismatches_control": int(
            control.get("event_count_mismatches") or 0
        ),
        "event_count_mismatches_treatment": int(
            treatment.get("event_count_mismatches") or 0
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
    return diagnose_phase_3c(comparison, non_evaluative=non_evaluative)
