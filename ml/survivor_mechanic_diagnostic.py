"""Phase 3O — observational matched-board survivor-mechanic attribution.

Reuses the 3N first-split walk on consumed DEV 14200–14699.
Does not change α, scaling math, `_hero_damage`, gates, defaults, or 2Q.

Restricts primary analysis to T5/T6 class-(3) fights. For every winner
starting body records printed tier, recruit atk/hp, synthetic share,
combat stats, slot / attack order, attacks made, death-before-first-attack,
incoming target count, taunt-forced vs open targeting, DS / poison /
cleave / SOC / generated flags, killer attack/tier, and survival.

Reproduces 3N within-tier B (+0.688), then sequentially standardizes
that gap into start-stats, attack opportunity, target exposure,
represented keywords, teammate protection, and residual.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.board_retention_diagnostic import collect_3h_leftover_rows
from ml.elimination_chain_diagnostic import compare_chain
from ml.elimination_timing_diagnostic import compare_elimination
from ml.hp_divergence_diagnostic import compare_first_divergence
from ml.lethal_cause_diagnostic import _arm_prefixes, _walk
from ml.matched_state_damage_diagnostic import (
    MatchedStateDamageTracer,
    compare_matched_state_damage,
    iter_class3_events,
)
from ml.pairing_who_wins_diagnostic import _index_seat_fights
from ml.phase_2z_prereg import gen_bin, target_bin
from ml.phase_3a_prereg import (
    cleave_bin,
    ds_bin,
    poison_bin,
    slot_bin,
    soc_bin,
)
from ml.phase_3o_prereg import (
    BODY_EVENT_RECONCILE_IDENTITY,
    KEYWORD_PARTS,
    MECHANIC_COMPONENTS,
    N_CLEAVE_BINS,
    N_DECILES,
    N_DS_BINS,
    N_GEN_BINS,
    N_POISON_BINS,
    N_SOC_BINS,
    N_TARGET_BINS,
    N_TEAM_BINS,
    NESTED_SURVIVAL_IDENTITY,
    PHASE_3N_CLASS3,
    PHASE_3N_CLASS3_T5,
    PHASE_3N_CLASS3_T6,
    PHASE_3N_WITHIN_TIER_B,
    PRIMARY_TURNS,
    SLOT_BIN_CAP,
    START_STATS_PARTS,
    WALK_LEAF_NAMES,
    attack_opp_bin,
    diagnose_phase_3o,
    keyword_bin,
)
from ml.punch_selection_diagnostic import collect_punch_sample_rows
from ml.survivor_composition_diagnostic import (
    TIERS,
    clamp_tier,
    survivors_subset_of_traced,
    traced_body_ids,
)
from ml.synthetic_allocation_diagnostic import (
    _cond_p,
    _kitagawa_two,
    _safe_div,
    bin_value,
    classify_start_minion,
    decile_edges,
    largest_remainder_shares,
)

METHODOLOGY_VERSION = "3o_v1"
_N_EXAMPLES = 8
_EVENT_FLAGS = (
    "attacks_reconcile",
    "targets_reconcile",
    "forced_open_reconcile",
    "created_reconcile",
    "deaths_reconcile",
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


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _safe_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _event_counts_ok(counts: Dict) -> bool:
    if not counts:
        return True
    return all(bool(counts.get(k, True)) for k in _EVENT_FLAGS)


def _killer_fields(src: Dict) -> Tuple[Optional[int], Optional[int]]:
    events = list(src.get("hit_events") or [])
    last = events[-1] if events else {}
    atk = last.get("attacker_attack")
    if atk is None:
        atk = src.get("last_attacker_attack") or src.get("killer_attack")
    tier = last.get("attacker_tier")
    if tier is None:
        tier = src.get("killer_tier") or src.get("mean_attacker_tier")
    return _safe_int(atk), _safe_int(tier)


def classify_mechanic_minion(body: Dict, slot: int, survived: bool) -> Dict:
    """Winner starting-body row with 3O combat-mechanic fields."""
    row = classify_start_minion(body, slot, survived)
    n_att = int(body.get("n_attacks") or row.get("n_attacks") or 0)
    died = not bool(survived)
    first_idx = body.get("first_attack_index")
    n_targeted = int(body.get("n_targeted") or 0)
    n_forced = int(body.get("n_targeted_forced") or 0)
    n_open = int(body.get("n_targeted_open") or 0)
    start_hp = int(
        body.get("start_health")
        or body.get("health")
        or body.get("recruit_health")
        or 0
    )
    recruit_atk = int(body.get("recruit_attack") or 0)
    recruit_hp = int(body.get("recruit_health") or start_hp)
    combat_atk = int(body.get("attack") or body.get("start_attack") or 0)
    combat_hp = int(body.get("health") or start_hp)
    synth_atk = combat_atk - recruit_atk
    synth_hp = combat_hp - recruit_hp
    killer_atk, killer_tier = _killer_fields(body)
    row.update({
        "recruit_attack": recruit_atk,
        "recruit_health": recruit_hp,
        "synthetic_attack": synth_atk,
        "synthetic_health": synth_hp,
        "start_health": start_hp,
        "start_attack": int(body.get("start_attack") or combat_atk),
        "n_attacks": n_att,
        "first_attack_index": None if first_idx is None else int(first_idx),
        "death_before_first_attack": died and n_att == 0,
        "n_targeted": n_targeted,
        "was_targeted": n_targeted > 0,
        "n_targeted_forced": n_forced,
        "n_targeted_open": n_open,
        "taunt_forced_target": bool(body.get("taunt_forced_target") or n_forced > 0),
        "open_target": bool(body.get("open_target") or (n_open > 0 and n_forced == 0)),
        "taunt": bool(body.get("taunt")),
        "divine_shield": bool(body.get("divine_shield") or body.get("start_divine_shield")),
        "start_divine_shield": bool(body.get("start_divine_shield") or body.get("divine_shield")),
        "poisonous": bool(body.get("poisonous")),
        "cleave": bool(body.get("cleave")),
        "generated": bool(body.get("generated")),
        "n_shield_pops": int(body.get("n_shield_pops") or 0),
        "n_hits_poison": int(body.get("n_hits_poison") or 0),
        "poison_lethal": bool(body.get("poison_lethal")),
        "n_cleave_primary": int(body.get("n_cleave_primary") or 0),
        "n_cleave_secondary": int(body.get("n_cleave_secondary") or 0),
        "cleave_lethal": bool(body.get("cleave_lethal")),
        "n_soc_hits": int(body.get("n_soc_hits") or 0),
        "soc_lethal": bool(body.get("soc_lethal")),
        "has_represented_generated_effect": bool(
            body.get("has_represented_generated_effect")
        ),
        "spawned_represented": int(body.get("spawned_represented") or 0),
        "n_board_generated_represented": int(
            body.get("n_board_generated_represented") or 0
        ),
        "effect_status": body.get("effect_status") or "unregistered",
        "has_unsupported_effect": bool(body.get("has_unsupported_effect")),
        "death_cause": body.get("death_cause"),
        "killed_by_body_id": body.get("killed_by_body_id"),
        "killer_attack": killer_atk,
        "killer_tier": killer_tier,
        "teammate_combat_raw": int(body.get("teammate_combat_raw") or 0),
        "board_size": int(body.get("board_size") or 0),
        "hp_flow_ok": bool(body.get("hp_flow_ok", True)),
    })
    row["slot_bin"] = slot_bin(row.get("board_slot"))
    row["target_bin"] = target_bin(row)
    row["ds_bin"] = ds_bin(row)
    row["poison_bin"] = poison_bin(row)
    row["cleave_bin"] = cleave_bin(row)
    row["soc_bin"] = soc_bin(row)
    row["gen_bin"] = gen_bin(row)
    row["keyword_bin"] = keyword_bin(row)
    row["attack_opp_bin"] = attack_opp_bin(row)
    return row


def _stamp_start_minions(rec: Dict, fight: Dict, env: Optional[BGEnv] = None) -> None:
    start_combat = list(
        rec.get("start_combat_bodies")
        or fight.get("starting_winner")
        or []
    )
    if not start_combat:
        winner = fight.get("winner_seat")
        sa = fight.get("seat_a")
        if winner is not None and sa is not None and int(winner) == int(sa):
            start_combat = list(fight.get("starting_a") or [])
        else:
            start_combat = list(fight.get("starting_b") or [])
    survivors = list(
        fight.get("survivors")
        or fight.get("actual_survivors")
        or rec.get("actual_survivors")
        or []
    )
    surv_ids = traced_body_ids(survivors)
    existing = {
        str(r.get("body_id") or ""): r
        for r in list(rec.get("start_minions") or [])
    }
    board_size = len(start_combat) or len(existing)
    total_raw = int(sum(
        int((existing.get(str(b.get("body_id") or ""), b).get("combat_raw")
             or b.get("combat_raw") or 0))
        for b in start_combat
    )) if start_combat else int(sum(
        int(r.get("combat_raw") or 0) for r in existing.values()
    ))
    rows = list(rec.get("start_minions") or [])
    if not rows and start_combat:
        rows = [
            classify_start_minion(
                b, i, str(b.get("body_id") or "") in surv_ids
            )
            for i, b in enumerate(start_combat)
        ]
    by_combat = {str(b.get("body_id") or ""): b for b in start_combat}
    stamped = []
    for i, r in enumerate(rows):
        src = dict(by_combat.get(str(r.get("body_id") or ""), {}))
        src.update(r)
        combat = int(src.get("combat_raw") or r.get("combat_raw") or 0)
        src["teammate_combat_raw"] = int(
            src.get("teammate_combat_raw")
            if src.get("teammate_combat_raw") is not None
            else total_raw - combat
        )
        src["board_size"] = int(src.get("board_size") or board_size)
        src["n_board_generated_represented"] = int(
            fight.get("n_board_generated_represented")
            or rec.get("n_board_generated_represented")
            or src.get("n_board_generated_represented")
            or 0
        )
        survived = bool(
            r.get("survived")
            if "survived" in r
            else str(r.get("body_id") or "") in surv_ids
        )
        stamped.append(classify_mechanic_minion(src, i, survived))
    share_sum = int(sum(int(r["synthetic_share"]) for r in stamped))
    pool_field = rec.get("winner_abstract_pool_field")
    if pool_field is None and env is not None and fight.get("winner_seat") is not None:
        for p in env.players:
            if int(p.idx) == int(fight["winner_seat"]):
                pool_field = float(getattr(p, "abstract_pool", 0.0) or 0.0)
                break
    if pool_field is not None and abs(float(pool_field)) > 1e-9:
        player_pool = int(round(float(pool_field)))
    else:
        player_pool = int(rec.get("winner_player_pool") or share_sum)
    expected = largest_remainder_shares(
        [int(r["recruit_raw"]) for r in stamped], player_pool
    )
    created = list(fight.get("created_winner") or rec.get("created_winner") or [])
    counts = dict(fight.get("event_counts") or rec.get("event_counts") or {})
    rec.update({
        "start_minions": stamped,
        "start_combat_bodies": start_combat or rec.get("start_combat_bodies") or [],
        "winner_abstract_pool_field": pool_field,
        "winner_player_pool": player_pool,
        "synthetic_shares_sum": share_sum,
        "shares_sum_to_pool": share_sum == player_pool,
        "expected_synthetic_shares": expected,
        "painted_matches_expected": (
            [int(r["synthetic_share"]) for r in stamped] == expected
            if stamped else True
        ),
        "event_counts": counts,
        "event_counts_ok": _event_counts_ok(counts),
        "survivors_subset_of_traced": survivors_subset_of_traced(
            survivors, start_combat, created
        ),
        "n_start_minions": len(stamped),
    })


class SurvivorMechanicTracer(MatchedStateDamageTracer):
    """3N matched-state rows plus per-body combat-mechanic tags."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        _stamp_start_minions(rec, fight, env)


def run_mechanic_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    from ml.phase_3o_prereg import assert_seed_range_allowed
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

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = SurvivorMechanicTracer(i, seed + i, arm)
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
    }


def run_greedy_control_mechanic(lobbies: int, seed: int) -> Dict:
    return run_mechanic_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment_mechanic(lobbies: int, seed: int) -> Dict:
    return run_mechanic_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _primary_turn(event: Dict) -> bool:
    turn = _safe_int(event.get("first_divergence_turn"))
    return turn in PRIMARY_TURNS


def _fight_for_event(index: Dict, event: Dict) -> Optional[Dict]:
    seed = _safe_int(event.get("seed"))
    seat = _safe_int(event.get("causal_seat"))
    turn = _safe_int(event.get("first_divergence_turn"))
    if seed is None or seat is None or turn is None:
        return None
    return index.get((seed, seat, turn))


def collect_class3_minions(
    events: Sequence[Dict],
    control_fights: Dict,
    treatment_fights: Dict,
    *,
    primary_only: bool = False,
) -> Tuple[List[Dict], List[Dict], int, int]:
    """Winner starting bodies on class-(3) first-split fights."""
    rows_c: List[Dict] = []
    rows_t: List[Dict] = []
    n_c = 0
    n_t = 0
    for ev in events:
        if primary_only and not _primary_turn(ev):
            continue
        c_fight = _fight_for_event(control_fights, ev)
        t_fight = _fight_for_event(treatment_fights, ev)
        turn = _safe_int(ev.get("first_divergence_turn"))
        if c_fight is not None:
            if not c_fight.get("start_minions"):
                _stamp_start_minions(c_fight, c_fight)
            for r in c_fight.get("start_minions") or []:
                row = dict(r)
                row["first_divergence_turn"] = turn
                row["winner_tavern_tier"] = int(
                    c_fight.get("winner_tavern_tier") or 1
                )
                rows_c.append(row)
            n_c += 1
        if t_fight is not None:
            if not t_fight.get("start_minions"):
                _stamp_start_minions(t_fight, t_fight)
            for r in t_fight.get("start_minions") or []:
                row = dict(r)
                row["first_divergence_turn"] = turn
                row["winner_tavern_tier"] = int(
                    t_fight.get("winner_tavern_tier") or 1
                )
                rows_t.append(row)
            n_t += 1
    return rows_c, rows_t, n_c, n_t


def _by_tier_mechanic(rows: Sequence[Dict], n_hits: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        cell = [r for r in rows if int(r.get("tier") or 1) == tier]
        n = len(cell)
        surv = [r for r in cell if r.get("survived")]
        out[str(tier)] = {
            "n_start": n,
            "n_survived": len(surv),
            "p_survive": _safe_div(float(len(surv)), float(n)),
            "mean_per_hit": _safe_div(float(n), float(n_hits)) if n_hits else None,
            "mean_recruit_raw": _mean(
                [float(r.get("recruit_raw") or 0) for r in cell]
            ),
            "mean_synthetic_share": _mean(
                [float(r.get("synthetic_share") or 0) for r in cell]
            ),
            "mean_combat_raw": _mean(
                [float(r.get("combat_raw") or 0) for r in cell]
            ),
            "mean_start_health": _mean(
                [float(r.get("start_health") or 0) for r in cell]
            ),
            "mean_board_slot": _mean(
                [float(r.get("board_slot") or 0) for r in cell]
            ),
            "mean_n_attacks": _mean(
                [float(r.get("n_attacks") or 0) for r in cell]
            ),
            "p_death_before_first_attack": _safe_div(
                float(sum(1 for r in cell if r.get("death_before_first_attack"))),
                float(n),
            ),
            "mean_n_targeted": _mean(
                [float(r.get("n_targeted") or 0) for r in cell]
            ),
            "p_taunt_forced": _safe_div(
                float(sum(1 for r in cell if r.get("taunt_forced_target"))),
                float(n),
            ),
            "p_open_target": _safe_div(
                float(sum(1 for r in cell if r.get("open_target"))),
                float(n),
            ),
            "p_keyword": _safe_div(
                float(sum(1 for r in cell if keyword_bin(r))), float(n)
            ),
            "p_start_divine_shield": _safe_div(
                float(sum(1 for r in cell if r.get("start_divine_shield"))),
                float(n),
            ),
            "mean_teammate_combat_raw": _mean(
                [float(r.get("teammate_combat_raw") or 0) for r in cell]
            ),
            "mean_killer_attack": _mean(
                [float(r["killer_attack"]) for r in cell
                 if r.get("killer_attack") is not None]
            ),
            "mean_killer_tier": _mean(
                [float(r["killer_tier"]) for r in cell
                 if r.get("killer_tier") is not None]
            ),
        }
    return out


def reweight_survivor_mechanics(
    control_rows: Sequence[Dict],
    treatment_rows: Sequence[Dict],
    *,
    n_hits_c: int,
    n_hits_t: int,
    observed_B: Optional[float] = None,
) -> Dict:
    """Nested Kitagawa of within-tier survival in the 3O walk order."""
    pooled = list(control_rows) + list(treatment_rows)
    recruit_edges: Dict[int, List[float]] = {}
    hp_edges: Dict[Tuple[int, int, int], List[float]] = {}
    synth_edges: Dict[Tuple[int, int], List[float]] = {}
    team_edges: Dict[int, List[float]] = {}
    for t in TIERS:
        recruit_edges[t] = decile_edges(
            [float(r.get("recruit_raw") or 0) for r in pooled if int(r["tier"]) == t]
        )
        team_edges[t] = decile_edges(
            [float(r.get("teammate_combat_raw") or 0)
             for r in pooled if int(r["tier"]) == t],
            n=N_TEAM_BINS,
        )
        n_r = len(recruit_edges[t]) + 1
        for rb in range(n_r):
            vs = [
                float(r.get("synthetic_share") or 0)
                for r in pooled
                if int(r["tier"]) == t
                and bin_value(float(r.get("recruit_raw") or 0), recruit_edges[t]) == rb
            ]
            synth_edges[(t, rb)] = decile_edges(vs)
            n_s = len(synth_edges[(t, rb)]) + 1
            for sb in range(n_s):
                hvs = [
                    float(r.get("start_health") or r.get("recruit_health") or 0)
                    for r in pooled
                    if int(r["tier"]) == t
                    and bin_value(float(r.get("recruit_raw") or 0), recruit_edges[t]) == rb
                    and bin_value(
                        float(r.get("synthetic_share") or 0),
                        synth_edges[(t, rb)],
                    ) == sb
                ]
                hp_edges[(t, rb, sb)] = decile_edges(hvs)

    def _key(r: Dict) -> Tuple:
        t = int(r["tier"])
        rb = bin_value(float(r.get("recruit_raw") or 0), recruit_edges[t])
        sb = bin_value(float(r.get("synthetic_share") or 0), synth_edges[(t, rb)])
        hb = bin_value(
            float(r.get("start_health") or r.get("recruit_health") or 0),
            hp_edges[(t, rb, sb)],
        )
        return (
            t, rb, sb, hb,
            slot_bin(r.get("board_slot")),
            target_bin(r),
            ds_bin(r),
            poison_bin(r),
            cleave_bin(r),
            soc_bin(r),
            gen_bin(r),
            bin_value(float(r.get("teammate_combat_raw") or 0), team_edges[t]),
        )

    n_c_t, s_c_t, n_c, s_c = _arm_prefixes(control_rows, n_hits_c, _key)
    n_t_t, s_t_t, n_t, s_t = _arm_prefixes(treatment_rows, n_hits_t, _key)
    const_bins = {
        4: SLOT_BIN_CAP + 1,
        5: N_TARGET_BINS,
        6: N_DS_BINS,
        7: N_POISON_BINS,
        8: N_CLEAVE_BINS,
        9: N_SOC_BINS,
        10: N_GEN_BINS,
    }
    n_depths = 12

    def n_bins_at(depth: int, prefix: Tuple) -> int:
        if depth == 1:
            return len(recruit_edges[prefix[0]]) + 1
        if depth == 2:
            return len(synth_edges[(prefix[0], prefix[1])]) + 1
        if depth == 3:
            return len(hp_edges[(prefix[0], prefix[1], prefix[2])]) + 1
        if depth == 11:
            return len(team_edges[prefix[0]]) + 1
        return int(const_bins[depth])

    totals = {name: 0.0 for name in WALK_LEAF_NAMES}
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
            **{name: 0.0 for name in WALK_LEAF_NAMES},
            "start_stats": 0.0,
            "attack_opportunity": 0.0,
            "represented_keywords": 0.0,
            "residual": 0.0,
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
        for name, val in zip(WALK_LEAF_NAMES, scaled):
            cell[name] = val
            totals[name] += val
        cell["start_stats"] = sum(cell[n] for n in START_STATS_PARTS)
        cell["attack_opportunity"] = cell["slot_opportunity"]
        cell["represented_keywords"] = sum(cell[n] for n in KEYWORD_PARTS)
        cell["residual"] = cell["still_unexplained"]
        cell["nested_residual"] = (
            rate_t
            - cell["start_stats"]
            - cell["attack_opportunity"]
            - cell["target_exposure"]
            - cell["represented_keywords"]
            - cell["teammate_protection"]
            - cell["residual"]
        )
        per_tier[str(tier)] = cell

    start_stats = sum(totals[n] for n in START_STATS_PARTS)
    keywords = sum(totals[n] for n in KEYWORD_PARTS)
    attack = totals["slot_opportunity"]
    target = totals["target_exposure"]
    teammate = totals["teammate_protection"]
    residual = totals["still_unexplained"]
    explained = (
        start_stats + attack + target + keywords + teammate + residual
    )
    obs_b = float(observed_B) if observed_B is not None else b_direct

    def _share(part: float) -> Optional[float]:
        if abs(obs_b) < 1e-12:
            return None
        return float(part) / obs_b

    return {
        "method": (
            "nested_kitagawa_tier_then_start_stats_then_slot_then_target_"
            "then_keywords_then_teammate"
        ),
        "n_deciles": N_DECILES,
        "n_team_bins": N_TEAM_BINS,
        "slot_bin_cap": SLOT_BIN_CAP,
        "within_tier_B": b_direct,
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "observed_B_used_for_shares": obs_b,
        **totals,
        "start_stats": start_stats,
        "attack_opportunity": attack,
        "target_exposure": target,
        "represented_keywords": keywords,
        "teammate_protection": teammate,
        "residual": residual,
        "explained_all_parts": explained,
        "residual_vs_direct_B": b_direct - explained,
        "share_of_B_start_stats": _share(start_stats),
        "share_of_B_attack_opportunity": _share(attack),
        "share_of_B_target_exposure": _share(target),
        "share_of_B_represented_keywords": _share(keywords),
        "share_of_B_teammate_protection": _share(teammate),
        "share_of_B_residual": _share(residual),
        "share_of_B_recruit_mix": _share(totals["recruit_mix"]),
        "share_of_B_synthetic_allocation": _share(totals["synthetic_allocation"]),
        "share_of_B_start_hp": _share(totals["start_hp"]),
        "share_of_B_divine_shield": _share(totals["divine_shield"]),
        "share_of_B_poison_venomous": _share(totals["poison_venomous"]),
        "share_of_B_cleave": _share(totals["cleave"]),
        "share_of_B_start_of_combat": _share(totals["start_of_combat"]),
        "share_of_B_generated": _share(totals["generated"]),
        "per_tier": per_tier,
        "mechanic_components": list(MECHANIC_COMPONENTS),
    }


def _recon_bodies(rows: Sequence[Dict], fights: Sequence[Dict]) -> Dict:
    n_share = sum(1 for f in fights if f.get("shares_sum_to_pool") is False)
    n_event = sum(1 for f in fights if f.get("event_counts_ok") is False)
    n_subset = sum(
        1 for f in fights if f.get("survivors_subset_of_traced") is False
    )
    n_hp = sum(1 for r in rows if r.get("hp_flow_ok") is False)
    return {
        "n_share_pool_mismatch": n_share,
        "n_event_count_mismatch": n_event,
        "n_survivor_subset_mismatch": n_subset,
        "n_body_hp_flow_mismatch": n_hp,
        "body_event_ok": n_share == 0 and n_event == 0 and n_subset == 0,
    }


def attribute_survivor_mechanics(
    control_raw: Dict,
    treatment_raw: Dict,
    *,
    leftover_rows: Sequence[Dict],
    treatment_punch: Sequence[Dict],
    matched: Optional[Dict] = None,
) -> Dict:
    """3N class-(3) lock plus T5/T6 nested mechanic split."""
    if matched is None:
        matched = compare_matched_state_damage(
            control_raw, treatment_raw,
        )
    events = list(iter_class3_events(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=treatment_punch,
    ))
    c_fights = _index_seat_fights(control_raw.get("fights") or [])
    t_fights = _index_seat_fights(treatment_raw.get("fights") or [])
    turn_counts = Counter(
        str(ev.get("first_divergence_turn"))
        for ev in events
        if ev.get("first_divergence_turn") is not None
    )
    rows_c_all, rows_t_all, n_c_all, n_t_all = collect_class3_minions(
        events, c_fights, t_fights, primary_only=False,
    )
    rows_c, rows_t, n_c, n_t = collect_class3_minions(
        events, c_fights, t_fights, primary_only=True,
    )
    late = matched.get("attribution") or {}
    kit = late.get("kitagawa") or {}
    b_3n = float(kit.get("within_tier_survival_B") or 0.0)
    full = reweight_survivor_mechanics(
        rows_c_all, rows_t_all,
        n_hits_c=n_c_all, n_hits_t=n_t_all, observed_B=b_3n,
    )
    primary = reweight_survivor_mechanics(
        rows_c, rows_t,
        n_hits_c=n_c, n_hits_t=n_t,
        observed_B=full["within_tier_B"] if abs(b_3n) < 1e-12 else None,
    )
    # Primary shares are of the T5/T6 within-tier B, not the 1059 lock.
    primary["observed_B_used_for_shares"] = primary["within_tier_B"]
    if abs(primary["within_tier_B"]) > 1e-12:
        def _pshare(part: float) -> Optional[float]:
            return float(part) / float(primary["within_tier_B"])
        for name in MECHANIC_COMPONENTS:
            primary[f"share_of_B_{name}"] = _pshare(float(primary.get(name) or 0.0))

    c_primary_fights = [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(c_fights, ev),) if f
    ]
    t_primary_fights = [
        f for ev in events if _primary_turn(ev)
        for f in (_fight_for_event(t_fights, ev),) if f
    ]
    recon_c = _recon_bodies(rows_c, c_primary_fights)
    recon_t = _recon_bodies(rows_t, t_primary_fights)
    examples = []
    for ev in events:
        if not _primary_turn(ev):
            continue
        if len(examples) >= _N_EXAMPLES:
            break
        c_fight = _fight_for_event(c_fights, ev)
        t_fight = _fight_for_event(t_fights, ev)
        examples.append({
            "seed": ev.get("seed"),
            "causal_seat": ev.get("causal_seat"),
            "first_divergence_turn": ev.get("first_divergence_turn"),
            "control_n_start": len((c_fight or {}).get("start_minions") or []),
            "treatment_n_start": len((t_fight or {}).get("start_minions") or []),
            "control_bodies": ((c_fight or {}).get("start_minions") or [])[:4],
            "treatment_bodies": ((t_fight or {}).get("start_minions") or [])[:4],
        })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "n_same_outcome_damage": len(events),
        "n_primary_class3": n_c,
        "n_primary_class3_treatment": n_t,
        "published_same_outcome_damage": PHASE_3N_CLASS3,
        "published_class3_t5": PHASE_3N_CLASS3_T5,
        "published_class3_t6": PHASE_3N_CLASS3_T6,
        "same_outcome_damage_reproduced": len(events) == PHASE_3N_CLASS3,
        "first_divergence_turn_counts": dict(turn_counts),
        "n_start_minions_control": len(rows_c),
        "n_start_minions_treatment": len(rows_t),
        "n_start_minions_control_all": len(rows_c_all),
        "n_start_minions_treatment_all": len(rows_t_all),
        "phase_3n_within_tier_B": PHASE_3N_WITHIN_TIER_B,
        "phase_3n_within_tier_B_hat": b_3n,
        "phase_3n_B_reproduced": abs(b_3n - PHASE_3N_WITHIN_TIER_B) <= 1e-9,
        "full_class3": full,
        "primary": primary,
        "control_by_tier": _by_tier_mechanic(rows_c, n_c),
        "treatment_by_tier": _by_tier_mechanic(rows_t, n_t),
        "body_reconciliation": {
            "control": recon_c,
            "treatment": recon_t,
            "body_event_ok": (
                recon_c["body_event_ok"] and recon_t["body_event_ok"]
            ),
            "identity": BODY_EVENT_RECONCILE_IDENTITY,
            "nested_identity": NESTED_SURVIVAL_IDENTITY,
        },
        "examples": examples,
        "matched_state": {
            "n_same_outcome_damage": late.get("n_same_outcome_damage"),
            "reconciliation_ok": late.get("reconciliation_ok"),
            "row_damage_ok": late.get("row_damage_ok"),
            "n_matched_pre_fight_board": late.get("n_matched_pre_fight_board"),
            "kitagawa": kit,
        },
    }


def compare_survivor_mechanics(
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
) -> Dict:
    """3N lock + T5/T6 survivor-mechanic split."""
    if matched is None:
        matched = compare_matched_state_damage(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain, first=first,
        )
    if first is None:
        first = compare_first_divergence(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
            chain=chain,
        )
    if chain is None:
        chain = compare_chain(
            control_raw, treatment_raw,
            lifecycle_cmp=lifecycle_cmp, divergence=divergence,
            selection=selection, pairing=pairing, timing=timing,
        )
    c_punch = collect_punch_sample_rows(control_raw.get("fights") or [])
    t_punch = collect_punch_sample_rows(treatment_raw.get("fights") or [])
    leftover_rows = collect_3h_leftover_rows(
        control_raw, treatment_raw, control_punch=c_punch,
        still_fields_t1t3=False,
    )
    attr = attribute_survivor_mechanics(
        control_raw, treatment_raw,
        leftover_rows=leftover_rows, treatment_punch=t_punch,
        matched=matched,
    )
    rec = dict(matched.get("reconciliation") or {})
    rec.update({
        "body_event_reconcile_identity": BODY_EVENT_RECONCILE_IDENTITY,
        "nested_survival_identity": NESTED_SURVIVAL_IDENTITY,
        "phase_3n_B_reproduced": attr.get("phase_3n_B_reproduced"),
        "same_outcome_damage_reproduced": attr.get("same_outcome_damage_reproduced"),
        "body_event_ok": (attr.get("body_reconciliation") or {}).get("body_event_ok"),
        "primary_nested_ok": abs(float(
            (attr.get("primary") or {}).get("residual_vs_direct_B") or 0.0
        )) <= 1e-6,
    })
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "attribution": attr,
        "primary": attr.get("primary"),
        "full_class3": attr.get("full_class3"),
        "source": matched.get("source"),
        "matched_state": matched.get("attribution"),
        "very_late_attribution": matched.get("very_late_attribution"),
        "first_divergence_3m": matched.get("first_divergence_3m"),
        "chain_3l": matched.get("chain_3l"),
        "timing_3k": matched.get("timing_3k"),
        "matchmaking_3j": matched.get("matchmaking_3j"),
        "pairing_3i": matched.get("pairing_3i"),
        "leftover_3h": matched.get("leftover_3h"),
        "reconciliation": rec,
        "decomposition_3g": matched.get("decomposition_3g"),
        "published_3n_locks": {
            "class3": PHASE_3N_CLASS3,
            "within_tier_B": PHASE_3N_WITHIN_TIER_B,
            "class3_t5": PHASE_3N_CLASS3_T5,
            "class3_t6": PHASE_3N_CLASS3_T6,
        },
        "published_3m_locks": matched.get("published_3m_locks"),
        "published_3l_locks": matched.get("published_3l_locks"),
        "published_3k_locks": matched.get("published_3k_locks"),
        "published_3j_locks": matched.get("published_3j_locks"),
        "published_3i_locks": matched.get("published_3i_locks"),
        "published_3h_locks": matched.get("published_3h_locks"),
        "published_3g_locks": matched.get("published_3g_locks"),
    }


# Diagnose is imported for runners; keep a local alias used by tests.
__all__ = [
    "SurvivorMechanicTracer",
    "attribute_survivor_mechanics",
    "classify_mechanic_minion",
    "collect_class3_minions",
    "compare_survivor_mechanics",
    "diagnose_phase_3o",
    "reweight_survivor_mechanics",
    "run_greedy_2s_treatment_mechanic",
    "run_greedy_control_mechanic",
]
