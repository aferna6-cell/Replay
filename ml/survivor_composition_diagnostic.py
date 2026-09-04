"""Phase 2V — observational survivor-composition attribution.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. For every decisive T7–T14 hit, classifies the
winner board-at-start and actual combat survivors, then decomposes the
treatment−control survivor-tier-sum gap into:

* (A) higher-tier cards fielded / recruited
* (B) same-tier cards surviving more often
* (C) token / generated combat bodies

Does not change α, scaling math, `_hero_damage`, gates, or defaults.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from typing import Dict, List, Optional, Sequence

from hsbg_coach.bg_env import (
    BGEnv,
    EnvMinion,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.phase_2v_prereg import (
    CHAFF_TIER_MAX,
    HIGH_TIER_MIN,
    INSTRUMENT_TURNS,
    PHASE_2U_SURVIVOR_TIER_SUM_DELTA,
    SHARE_DOMINANT,
    assert_seed_range_allowed,
    diagnose_phase_2v,
)
from ml.survivor_tier_damage_diagnostic import (
    SurvivorTierTracer,
    rules_faithful_hero_damage,
)

METHODOLOGY_VERSION = "2v_v1"
TIERS = (1, 2, 3, 4, 5, 6)


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None:
        return None
    if abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def classify_env_minion(m: EnvMinion, slot: int) -> Dict:
    """Winner-board-at-start row from the env minion (recruit + combat stats)."""
    tribes = [str(t) for t in (getattr(m, "tribes", None) or [])]
    atk = int(getattr(m, "attack", 0) or 0)
    hp = int(getattr(m, "health", 0) or 0)
    ra = getattr(m, "recruit_attack", None)
    rh = getattr(m, "recruit_health", None)
    try:
        recruit_attack = int(ra) if ra not in (None, "") else atk
        recruit_health = int(rh) if rh not in (None, "") else hp
    except (TypeError, ValueError):
        recruit_attack, recruit_health = atk, hp
    return {
        "name": str(getattr(m, "name", "") or ""),
        "card_id": str(getattr(m, "card_id", "") or ""),
        "tier": int(getattr(m, "tier", 1) or 1),
        "golden": bool(getattr(m, "golden", False)),
        "tribes": tribes,
        "archetype": tribes[0] if tribes else "tribeless",
        "board_slot": int(slot),
        "recruit_attack": recruit_attack,
        "recruit_health": recruit_health,
        "combat_attack": atk,
        "combat_health": hp,
        "recruit_raw": recruit_attack + recruit_health,
        "combat_raw": atk + hp,
        "origin": "starting",
        "generated": False,
        "token": False,
    }


def clamp_tier(raw) -> int:
    try:
        t = int(raw)
    except (TypeError, ValueError):
        t = 1
    return min(6, max(1, t))


def tier_histogram(bodies: Sequence[Dict]) -> Dict[str, int]:
    hist = {str(t): 0 for t in TIERS}
    for b in bodies:
        hist[str(clamp_tier(b.get("tier")))] += 1
    return hist


def tier_sum(bodies: Sequence[Dict]) -> int:
    return int(sum(clamp_tier(b.get("tier")) for b in bodies))


def traced_body_ids(bodies: Sequence[Dict]) -> set:
    return {str(b.get("body_id") or "") for b in bodies if b.get("body_id")}


def survivors_subset_of_traced(survivors: Sequence[Dict],
                               starting: Sequence[Dict],
                               created: Sequence[Dict]) -> bool:
    """Every survivor with a body_id is in starting ∪ created combat bodies."""
    allowed = traced_body_ids(starting) | traced_body_ids(created)
    if not allowed:
        return len(survivors) == 0
    for s in survivors:
        bid = str(s.get("body_id") or "")
        if bid and bid not in allowed:
            return False
    return True


def fight_tier_buckets_reconcile(fight: Dict) -> bool:
    survivors = list(fight.get("actual_survivors") or [])
    expect = int(fight.get("actual_survivor_tier_sum") or 0)
    return tier_sum(survivors) == expect


class SurvivorCompositionTracer(SurvivorTierTracer):
    """2U fight rows plus start-board / survivor composition classification."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        winner_board = list(fight.get("winner_board") or [])
        start_env = [
            classify_env_minion(m, i) for i, m in enumerate(winner_board)
        ]
        start_combat = list(fight.get("starting_winner") or [])
        created = list(fight.get("created_winner") or [])
        # Full combatant_trace_row fields (body_id / origin / golden / stats).
        survivors = list(fight.get("survivors") or [])
        applied = int(rec.get("applied_hp_loss") or 0)
        raw = int(rec.get("raw") or 0)
        if raw == 0 or applied <= 0:
            start_env = []
            start_combat = []
            created = []
            survivors = []
        rec["actual_survivors"] = survivors

        start_ids = traced_body_ids(start_combat)
        created_ids = traced_body_ids(created)
        surv_start = [
            s for s in survivors
            if str(s.get("origin") or "starting") == "starting"
            or str(s.get("body_id") or "") in start_ids
        ]
        surv_gen = [
            s for s in survivors
            if bool(s.get("generated")) or str(s.get("origin") or "") in (
                "token", "reborn"
            ) or str(s.get("body_id") or "") in created_ids
        ]
        # A body can theoretically match both if ids collide; keep generated
        # exclusive of starting-origin for the A/B vs C split.
        surv_start = [
            s for s in surv_start
            if not (
                bool(s.get("generated"))
                or str(s.get("origin") or "") in ("token", "reborn")
            )
        ]

        rec.update({
            "start_board": start_env,
            "start_combat_bodies": start_combat,
            "created_winner": created,
            "surv_starting": surv_start,
            "surv_generated": surv_gen,
            "start_tier_hist": tier_histogram(start_env),
            "start_combat_tier_hist": tier_histogram(start_combat),
            "survivor_tier_hist": tier_histogram(survivors),
            "surv_starting_tier_hist": tier_histogram(surv_start),
            "surv_generated_tier_hist": tier_histogram(surv_gen),
            "start_tier_sum": tier_sum(start_env),
            "surv_starting_tier_sum": tier_sum(surv_start),
            "surv_generated_tier_sum": tier_sum(surv_gen),
            "n_start": len(start_env),
            "n_start_combat": len(start_combat),
            "n_created": len(created),
            "n_surv_starting": len(surv_start),
            "n_surv_generated": len(surv_gen),
            "n_surv_token": sum(1 for s in survivors if s.get("token")),
            "n_surv_golden": sum(1 for s in survivors if s.get("golden")),
            "n_surv_chaff": sum(
                1 for s in survivors if clamp_tier(s.get("tier")) <= CHAFF_TIER_MAX
            ),
            "n_surv_high_tier": sum(
                1 for s in survivors if clamp_tier(s.get("tier")) >= HIGH_TIER_MIN
            ),
            "n_start_golden": sum(1 for s in start_env if s.get("golden")),
            "n_start_high_tier": sum(
                1 for s in start_env if clamp_tier(s.get("tier")) >= HIGH_TIER_MIN
            ),
            "tier_buckets_reconcile": (
                tier_sum(survivors) == int(rec.get("actual_survivor_tier_sum") or 0)
            ),
            "survivors_subset_of_traced": survivors_subset_of_traced(
                survivors, start_combat, created
            ),
            "start_env_vs_combat_n_match": len(start_env) == len(start_combat),
        })


def run_composition_arm(
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
                tracer = SurvivorCompositionTracer(i, seed + i, arm)
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
    return run_composition_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_composition_arm(
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


def _hist_mean(hits: Sequence[Dict], key: str) -> Dict[str, Optional[float]]:
    if not hits:
        return {str(t): None for t in TIERS}
    acc = {str(t): 0.0 for t in TIERS}
    for f in hits:
        hist = f.get(key) or {}
        for t in TIERS:
            acc[str(t)] += float(hist.get(str(t), 0) or 0)
    n = float(len(hits))
    return {k: v / n for k, v in acc.items()}


def _share_from_mean_hist(mean_hist: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    total = sum(float(v) for v in mean_hist.values() if v is not None)
    if total <= 1e-12:
        return {k: None for k in mean_hist}
    return {
        k: (None if v is None else float(v) / total)
        for k, v in mean_hist.items()
    }


def _cond_survival(
    start_mean: Dict[str, Optional[float]],
    surv_start_mean: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    return {
        str(t): _safe_div(surv_start_mean.get(str(t)), start_mean.get(str(t)))
        for t in TIERS
    }


def _archetype_share(hits: Sequence[Dict], body_key: str) -> Dict[str, float]:
    c: Counter = Counter()
    n = 0
    for f in hits:
        for b in f.get(body_key) or []:
            c[str(b.get("archetype") or "tribeless")] += 1
            n += 1
    if n <= 0:
        return {}
    return {k: v / n for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))}


def _tier_damage_contrib(surv_mean: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Each tier bucket's mean contribution to Σ survivor tiers (CF − tavern)."""
    return {
        str(t): (None if surv_mean.get(str(t)) is None else float(t) * float(surv_mean[str(t)]))
        for t in TIERS
    }


def _summarize_hits(hits: List[Dict]) -> Dict:
    start_mean = _hist_mean(hits, "start_tier_hist")
    start_combat_mean = _hist_mean(hits, "start_combat_tier_hist")
    surv_mean = _hist_mean(hits, "survivor_tier_hist")
    surv_start_mean = _hist_mean(hits, "surv_starting_tier_hist")
    surv_gen_mean = _hist_mean(hits, "surv_generated_tier_hist")
    n_surv = [float(f.get("actual_survivor_count") or 0) for f in hits]
    n_start = [float(f.get("n_start") or 0) for f in hits]
    high = [float(f.get("n_surv_high_tier") or 0) for f in hits]
    chaff = [float(f.get("n_surv_chaff") or 0) for f in hits]
    token = [float(f.get("n_surv_token") or 0) for f in hits]
    gen = [float(f.get("n_surv_generated") or 0) for f in hits]
    golden = [float(f.get("n_surv_golden") or 0) for f in hits]
    start_high = [float(f.get("n_start_high_tier") or 0) for f in hits]
    start_golden = [float(f.get("n_start_golden") or 0) for f in hits]
    cf = [float(f.get("counterfactual_damage") or 0) for f in hits]
    tavern = [float(f.get("winner_tavern_tier") or 0) for f in hits]
    tier_sum_hit = [float(f.get("actual_survivor_tier_sum") or 0) for f in hits]
    start_sum = [float(f.get("start_tier_sum") or 0) for f in hits]
    surv_start_sum = [float(f.get("surv_starting_tier_sum") or 0) for f in hits]
    surv_gen_sum = [float(f.get("surv_generated_tier_sum") or 0) for f in hits]
    recruit_raw_surv = [
        float(s.get("recruit_raw") or 0)
        for f in hits for s in (f.get("actual_survivors") or [])
    ]
    combat_raw_surv = [
        float(s.get("combat_raw") or 0)
        for f in hits for s in (f.get("actual_survivors") or [])
    ]
    recruit_raw_start = [
        float(s.get("recruit_raw") or 0)
        for f in hits for s in (f.get("start_board") or [])
    ]
    combat_raw_start = [
        float(s.get("combat_raw") or 0)
        for f in hits for s in (f.get("start_board") or [])
    ]
    contrib = _tier_damage_contrib(surv_mean)
    return {
        "n_hits": len(hits),
        "mean_survivor_count": _mean(n_surv),
        "mean_start_n": _mean(n_start),
        "mean_survivor_tier_sum": _mean(tier_sum_hit),
        "mean_start_tier_sum": _mean(start_sum),
        "mean_surv_starting_tier_sum": _mean(surv_start_sum),
        "mean_surv_generated_tier_sum": _mean(surv_gen_sum),
        "mean_counterfactual": _mean(cf),
        "mean_winner_tavern_tier": _mean(tavern),
        "start_tier_histogram": start_mean,
        "start_combat_tier_histogram": start_combat_mean,
        "survivor_tier_histogram": surv_mean,
        "surv_starting_tier_histogram": surv_start_mean,
        "surv_generated_tier_histogram": surv_gen_mean,
        "start_board_share_by_tier": _share_from_mean_hist(start_mean),
        "survivor_share_by_tier": _share_from_mean_hist(surv_mean),
        "survival_prob_by_tier": _cond_survival(start_combat_mean, surv_start_mean),
        "high_tier_survivor_share": _safe_div(_mean(high), _mean(n_surv)),
        "high_tier_start_share": _safe_div(_mean(start_high), _mean(n_start)),
        "chaff_survivor_share": _safe_div(_mean(chaff), _mean(n_surv)),
        "token_survivor_share": _safe_div(_mean(token), _mean(n_surv)),
        "generated_survivor_share": _safe_div(_mean(gen), _mean(n_surv)),
        "golden_survivor_share": _safe_div(_mean(golden), _mean(n_surv)),
        "golden_start_share": _safe_div(_mean(start_golden), _mean(n_start)),
        "tier_contrib_to_survivor_tier_sum": contrib,
        "tier_share_of_survivor_tier_sum": _share_from_mean_hist(contrib),
        "mean_survivor_recruit_raw": _mean(recruit_raw_surv),
        "mean_survivor_combat_raw": _mean(combat_raw_surv),
        "mean_start_recruit_raw": _mean(recruit_raw_start),
        "mean_start_combat_raw": _mean(combat_raw_start),
        "survivor_archetype_share": _archetype_share(hits, "actual_survivors"),
        "start_archetype_share": _archetype_share(hits, "start_board"),
        "n_tier_bucket_mismatch": sum(
            1 for f in hits if not f.get("tier_buckets_reconcile", True)
        ),
        "n_subset_mismatch": sum(
            1 for f in hits if not f.get("survivors_subset_of_traced", True)
        ),
        "n_start_env_combat_n_mismatch": sum(
            1 for f in hits if not f.get("start_env_vs_combat_n_match", True)
        ),
    }


def _slim_example(f: Dict) -> Dict:
    keep_surv = (
        "body_id", "name", "card_id", "tier", "golden", "token", "generated",
        "origin", "archetype", "board_slot", "recruit_raw", "combat_raw",
    )
    def _slim_bodies(rows, n=7):
        out = []
        for b in list(rows or [])[:n]:
            out.append({k: b.get(k) for k in keep_surv})
        return out
    return {
        "lobby": f.get("lobby"),
        "turn": f.get("turn"),
        "kind": f.get("kind"),
        "winner_tavern_tier": f.get("winner_tavern_tier"),
        "actual_survivor_count": f.get("actual_survivor_count"),
        "actual_survivor_tier_sum": f.get("actual_survivor_tier_sum"),
        "counterfactual_damage": f.get("counterfactual_damage"),
        "start_board": _slim_bodies(f.get("start_board")),
        "actual_survivors": _slim_bodies(f.get("actual_survivors")),
        "created_winner": _slim_bodies(f.get("created_winner")),
        "survivors_subset_of_traced": f.get("survivors_subset_of_traced"),
        "tier_buckets_reconcile": f.get("tier_buckets_reconcile"),
    }


def summarize_composition_arm(raw: Dict) -> Dict:
    fights_all = raw["fights"]
    fights = [f for f in fights_all if _in_window(f["turn"])]
    decisive = [
        f for f in fights
        if f["kind"] != "bye" and int(f.get("raw") or 0) != 0
    ]
    hits = [f for f in fights if int(f.get("applied_hp_loss") or 0) > 0]
    lengths = list(raw["game_lengths"])
    summary = _summarize_hits(hits)
    by_turn: Dict[str, Dict] = {}
    for t in INSTRUMENT_TURNS:
        ht = [f for f in hits if int(f["turn"]) == t]
        by_turn[str(t)] = _summarize_hits(ht)
    summary.update({
        "arm": raw["arm"],
        "recruit_value_stats": raw["recruit_value_stats"],
        "board_level_abstract_scaling": raw["board_level_abstract_scaling"],
        "n_lobbies": int(raw["n_lobbies"] or 1),
        "seed_base": raw["seed_base"],
        "mean_game_length": _mean(lengths),
        "n_fights_t7_t14": len(fights),
        "n_decisive": len(decisive),
        "per_turn": by_turn,
        "example_fights": [_slim_example(f) for f in hits[:4]],
        "survivor_count_matches_raw": all(
            int(f.get("actual_survivor_count") or 0)
            == int(f.get("survivor_count") or 0)
            for f in hits
        ) if hits else True,
    })
    return summary


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _hist_delta(c: Dict, t: Dict) -> Dict[str, Optional[float]]:
    keys = set(c or {}) | set(t or {})
    return {k: _delta((c or {}).get(k), (t or {}).get(k)) for k in sorted(keys)}


def decompose_gap(
    control: Dict,
    treatment: Dict,
    *,
    observed_delta: Optional[float] = None,
) -> Dict:
    """Kitagawa + sequential split of treatment−control survivor tier sum.

    Starting-origin only for (A)/(B). (C) is generated-survivor tier sum.
    Kitagawa uses mid-point weights so A+B equals the starting-origin gap.
    """
    n_c = control.get("start_combat_tier_histogram") or {}
    n_t = treatment.get("start_combat_tier_histogram") or {}
    p_c = control.get("survival_prob_by_tier") or {}
    p_t = treatment.get("survival_prob_by_tier") or {}

    a_kit = 0.0
    b_kit = 0.0
    a_seq = 0.0
    b_seq = 0.0
    start_gap = 0.0
    exclusive_to_a = 0.0
    per_tier = {}
    for tier in TIERS:
        k = str(tier)
        nc = float(n_c.get(k) or 0.0)
        nt = float(n_t.get(k) or 0.0)
        pc = p_c.get(k)
        pt = p_t.get(k)
        pc_f = float(pc) if pc is not None else 0.0
        pt_f = float(pt) if pt is not None else 0.0
        start_c = nc * pc_f * float(tier)
        start_t = nt * pt_f * float(tier)
        gap_t = start_t - start_c
        start_gap += gap_t
        # Same-tier survival is only defined when both arms field the tier.
        exclusive = nc < 1e-12 or nt < 1e-12
        if exclusive:
            a_t = gap_t
            b_t = 0.0
            a_s = gap_t
            b_s = 0.0
            exclusive_to_a += gap_t
        else:
            dn = nt - nc
            dp = pt_f - pc_f
            p_bar = 0.5 * (pc_f + pt_f)
            n_bar = 0.5 * (nc + nt)
            a_t = float(tier) * dn * p_bar
            b_t = float(tier) * n_bar * dp
            a_s = float(tier) * dn * pc_f
            b_s = float(tier) * nt * dp
        a_kit += a_t
        b_kit += b_t
        a_seq += a_s
        b_seq += b_s
        per_tier[k] = {
            "n_start_control": nc,
            "n_start_treatment": nt,
            "p_survive_control": pc,
            "p_survive_treatment": pt,
            "kitagawa_fielded": a_t,
            "kitagawa_survival": b_t,
            "starting_origin_tier_sum_delta": gap_t,
            "exclusive_support": exclusive,
        }

    c_gap = _delta(
        control.get("mean_surv_generated_tier_sum"),
        treatment.get("mean_surv_generated_tier_sum"),
    )
    c_gap_f = float(c_gap) if c_gap is not None else 0.0
    obs = observed_delta
    if obs is None:
        obs = _delta(
            control.get("mean_survivor_tier_sum"),
            treatment.get("mean_survivor_tier_sum"),
        )
    obs_f = float(obs) if obs is not None else None
    explained = a_kit + b_kit + c_gap_f
    residual = None if obs_f is None else obs_f - explained

    def _share(part: float) -> Optional[float]:
        if obs_f is None or abs(obs_f) < 1e-12:
            return None
        return float(part) / obs_f

    return {
        "method": "kitagawa_common_support_exclusive_to_fielded",
        "exclusive_support_assigned_to_A": exclusive_to_a,
        "observed_survivor_tier_sum_delta": obs_f,
        "phase_2u_survivor_tier_sum_delta": PHASE_2U_SURVIVOR_TIER_SUM_DELTA,
        "fielded_composition_A": a_kit,
        "within_tier_survival_B": b_kit,
        "token_generated_C": c_gap_f,
        "starting_origin_gap": start_gap,
        "explained_A_plus_B_plus_C": explained,
        "residual": residual,
        "share_fielded_composition": _share(a_kit),
        "share_within_tier_survival": _share(b_kit),
        "share_token_generated": _share(c_gap_f),
        "share_dominant_threshold": SHARE_DOMINANT,
        "sequential_fielded_A": a_seq,
        "sequential_survival_B": b_seq,
        "sequential_explained": a_seq + b_seq + c_gap_f,
        "per_tier": per_tier,
    }


def compare_composition(control: Dict, treatment: Dict) -> Dict:
    scalar_keys = (
        "mean_game_length",
        "n_hits",
        "mean_survivor_count",
        "mean_start_n",
        "mean_survivor_tier_sum",
        "mean_start_tier_sum",
        "mean_surv_starting_tier_sum",
        "mean_surv_generated_tier_sum",
        "mean_counterfactual",
        "mean_winner_tavern_tier",
        "high_tier_survivor_share",
        "high_tier_start_share",
        "chaff_survivor_share",
        "token_survivor_share",
        "generated_survivor_share",
        "golden_survivor_share",
        "golden_start_share",
        "mean_survivor_recruit_raw",
        "mean_survivor_combat_raw",
        "mean_start_recruit_raw",
        "mean_start_combat_raw",
    )
    deltas = {k: _delta(control.get(k), treatment.get(k)) for k in scalar_keys}
    decomp = decompose_gap(
        control, treatment,
        observed_delta=deltas.get("mean_survivor_tier_sum"),
    )
    per_turn = {}
    for t in INSTRUMENT_TURNS:
        key = str(t)
        c = (control.get("per_turn") or {}).get(key) or {}
        t_ = (treatment.get("per_turn") or {}).get(key) or {}
        turn_delta = {
            m: _delta(c.get(m), t_.get(m))
            for m in (
                "mean_survivor_tier_sum",
                "mean_start_tier_sum",
                "mean_surv_starting_tier_sum",
                "mean_surv_generated_tier_sum",
                "high_tier_survivor_share",
                "token_survivor_share",
                "generated_survivor_share",
                "chaff_survivor_share",
                "golden_survivor_share",
                "n_hits",
            )
        }
        turn_delta["decomposition"] = decompose_gap(
            c, t_,
            observed_delta=turn_delta.get("mean_survivor_tier_sum"),
        )
        turn_delta["survivor_tier_histogram"] = _hist_delta(
            c.get("survivor_tier_histogram") or {},
            t_.get("survivor_tier_histogram") or {},
        )
        turn_delta["survival_prob_by_tier"] = _hist_delta(
            c.get("survival_prob_by_tier") or {},
            t_.get("survival_prob_by_tier") or {},
        )
        turn_delta["start_board_share_by_tier"] = _hist_delta(
            c.get("start_board_share_by_tier") or {},
            t_.get("start_board_share_by_tier") or {},
        )
        turn_delta["survivor_share_by_tier"] = _hist_delta(
            c.get("survivor_share_by_tier") or {},
            t_.get("survivor_share_by_tier") or {},
        )
        per_turn[key] = turn_delta

    return {
        "deltas": deltas,
        "decomposition": decomp,
        "histogram_delta": {
            "start_tier_histogram": _hist_delta(
                control.get("start_tier_histogram") or {},
                treatment.get("start_tier_histogram") or {},
            ),
            "survivor_tier_histogram": _hist_delta(
                control.get("survivor_tier_histogram") or {},
                treatment.get("survivor_tier_histogram") or {},
            ),
            "survival_prob_by_tier": _hist_delta(
                control.get("survival_prob_by_tier") or {},
                treatment.get("survival_prob_by_tier") or {},
            ),
            "start_board_share_by_tier": _hist_delta(
                control.get("start_board_share_by_tier") or {},
                treatment.get("start_board_share_by_tier") or {},
            ),
            "survivor_share_by_tier": _hist_delta(
                control.get("survivor_share_by_tier") or {},
                treatment.get("survivor_share_by_tier") or {},
            ),
            "tier_contrib_to_survivor_tier_sum": _hist_delta(
                control.get("tier_contrib_to_survivor_tier_sum") or {},
                treatment.get("tier_contrib_to_survivor_tier_sum") or {},
            ),
        },
        "per_turn_delta": per_turn,
        "reconciliation": {
            "tier_bucket_mismatch_control": control.get("n_tier_bucket_mismatch"),
            "tier_bucket_mismatch_treatment": treatment.get("n_tier_bucket_mismatch"),
            "subset_mismatch_control": control.get("n_subset_mismatch"),
            "subset_mismatch_treatment": treatment.get("n_subset_mismatch"),
            "start_env_combat_n_mismatch_control": control.get(
                "n_start_env_combat_n_mismatch"
            ),
            "start_env_combat_n_mismatch_treatment": treatment.get(
                "n_start_env_combat_n_mismatch"
            ),
            "decomp_residual": decomp.get("residual"),
        },
        "control": {k: control.get(k) for k in scalar_keys},
        "treatment": {k: treatment.get(k) for k in scalar_keys},
    }


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_2v(comparison, non_evaluative=non_evaluative)
