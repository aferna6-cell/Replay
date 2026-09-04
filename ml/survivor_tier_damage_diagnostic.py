"""Phase 2U — observational survivor-tier vs `_hero_damage` proxy fidelity.

Reuses paired greedy control (2Q/2S OFF) vs 2S treatment (2Q + 2S ON) on
consumed DEV 14200–14699. For every decisive T7–T14 fight computes:

* current applied proxy (`BGEnv._hero_damage`)
* count-only (`winner tavern + survivor count`, sim.py raw)
* rules-faithful counterfactual (`winner tavern + sum(actual survivor tiers)`)

Does not change α, scaling math, `_hero_damage`, gates, or defaults.
"""

from __future__ import annotations

import statistics as st
from typing import Dict, List, Optional, Sequence

from hsbg_coach.bg_env import (
    BGEnv,
    board_level_abstract_scaling_enabled,
    greedy_policy,
    recruit_value_stats_enabled,
)
from ml.game_length_damage_diagnostic import GameLengthDamageTracer
from ml.phase_2u_prereg import (
    INSTRUMENT_TURNS,
    PHASE_2T_AMP_DELTA_WHEN_HIT,
    SHARE_REMOVED_MOST,
    assert_seed_range_allowed,
    diagnose_phase_2u,
)

METHODOLOGY_VERSION = "2u_v1"


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return float(st.median(xs)) if xs else None


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


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if xs else None
    return float(st.pstdev(xs))


def rules_faithful_hero_damage(
    winner_tavern_tier: int,
    survivor_tiers: Sequence[int],
) -> int:
    """Winner tavern tier + sum of actual surviving minion tavern tiers."""
    return int(winner_tavern_tier) + int(sum(int(t) for t in survivor_tiers))


def error_summary(errors: List[float]) -> Dict:
    return {
        "n": len(errors),
        "mean": _mean(errors),
        "median": _median(errors),
        "std": _std(errors),
        "p10": _quantile(errors, 0.10),
        "p90": _quantile(errors, 0.90),
        "min": float(min(errors)) if errors else None,
        "max": float(max(errors)) if errors else None,
    }


class SurvivorTierTracer(GameLengthDamageTracer):
    """2T fight rows plus actual combat-survivor identities/tiers."""

    def on_fight(self, env: BGEnv, fight: Dict) -> None:
        super().on_fight(env, fight)
        rec = self.fights[-1]
        survivors = list(fight.get("survivors") or [])
        tiers = [int(s.get("tier") or 1) for s in survivors]
        actual_n = int(fight.get("survivor_count_actual") or len(survivors))
        actual_sum = int(fight.get("survivor_tier_sum") or sum(tiers))
        winner_tavern = int(rec.get("winner_tavern_tier") or 1)
        applied = int(rec.get("applied_hp_loss") or 0)
        count_only = int(rec.get("count_only_damage") or 0)
        raw = int(rec.get("raw") or 0)
        if raw == 0 or applied <= 0:
            cf = 0
            proxy_err = 0
            lethal_cf = False
            lethal_flip = False
            actual_n = 0
            actual_sum = 0
            survivors = []
        else:
            cf = rules_faithful_hero_damage(winner_tavern, tiers)
            proxy_err = applied - cf
            loser_pre = None
            if fight.get("loser_seat") == fight.get("seat_a"):
                loser_pre = fight.get("pre_hp_a")
            elif fight.get("loser_seat") == fight.get("seat_b"):
                loser_pre = fight.get("pre_hp_b")
            elif fight.get("ghost") and raw < 0:
                loser_pre = fight.get("pre_hp_a")
            lethal_cf = bool(
                loser_pre is not None
                and int(loser_pre) > 0
                and int(loser_pre) - cf <= 0
            )
            lethal_flip = bool(rec.get("lethal")) != lethal_cf

        rec.update({
            "actual_survivors": [
                {
                    "name": s.get("name"),
                    "card_id": s.get("card_id"),
                    "tier": int(s.get("tier") or 1),
                }
                for s in survivors
            ],
            "actual_survivor_count": actual_n,
            "actual_survivor_tier_sum": actual_sum,
            "counterfactual_damage": int(cf),
            "proxy_minus_counterfactual": int(proxy_err),
            "cf_amplification": int(cf) - int(count_only) if applied > 0 else 0,
            "lethal_counterfactual": lethal_cf,
            "lethal_flip": lethal_flip,
        })


def run_survivor_tier_arm(
    lobbies: int,
    seed: int,
    *,
    arm: str,
    recruit_value_stats: bool,
    board_level_abstract_scaling: bool,
) -> Dict:
    assert_seed_range_allowed(seed, lobbies)
    fights: List[Dict] = []
    hp_rows: List[Dict] = []
    eliminations: List[Dict] = []
    lengths: List[float] = []

    with recruit_value_stats_enabled(recruit_value_stats):
        with board_level_abstract_scaling_enabled(board_level_abstract_scaling):
            for i in range(lobbies):
                tracer = SurvivorTierTracer(i, seed + i, arm)
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
                hp_rows.extend(tracer.hp_rows)
                eliminations.extend(tracer.eliminations)
                del env

    return {
        "arm": arm,
        "recruit_value_stats": bool(recruit_value_stats),
        "board_level_abstract_scaling": bool(board_level_abstract_scaling),
        "n_lobbies": lobbies,
        "seed_base": seed,
        "fights": fights,
        "hp_rows": hp_rows,
        "eliminations": eliminations,
        "game_lengths": lengths,
    }


def run_greedy_control(lobbies: int, seed: int) -> Dict:
    return run_survivor_tier_arm(
        lobbies, seed, arm="greedy_control",
        recruit_value_stats=False, board_level_abstract_scaling=False,
    )


def run_greedy_2s_treatment(lobbies: int, seed: int) -> Dict:
    return run_survivor_tier_arm(
        lobbies, seed, arm="greedy_2s_treatment",
        recruit_value_stats=True, board_level_abstract_scaling=True,
    )


def _in_window(turn: int) -> bool:
    return int(turn) in INSTRUMENT_TURNS


def _slim_example(f: Dict) -> Dict:
    keep = (
        "lobby", "turn", "kind", "raw", "applied_hp_loss",
        "count_only_damage", "counterfactual_damage",
        "proxy_minus_counterfactual", "amplification", "cf_amplification",
        "winner_tavern_tier", "actual_survivor_count",
        "actual_survivor_tier_sum", "winner_minion_tier_mean",
        "lethal", "lethal_counterfactual", "lethal_flip",
        "actual_survivors",
    )
    return {k: f.get(k) for k in keep}


def summarize_survivor_arm(raw: Dict) -> Dict:
    fights_all = raw["fights"]
    fights = [f for f in fights_all if _in_window(f["turn"])]
    decisive = [
        f for f in fights
        if f["kind"] != "bye" and int(f.get("raw") or 0) != 0
    ]
    hits = [f for f in fights if int(f.get("applied_hp_loss") or 0) > 0]
    lengths = list(raw["game_lengths"])
    n_lobbies = int(raw["n_lobbies"] or 1)

    applied_hit = [float(f["applied_hp_loss"]) for f in hits]
    count_hit = [float(f["count_only_damage"]) for f in hits]
    cf_hit = [float(f["counterfactual_damage"]) for f in hits]
    amp_hit = [float(f["amplification"]) for f in hits]
    cf_amp_hit = [float(f["cf_amplification"]) for f in hits]
    err_hit = [float(f["proxy_minus_counterfactual"]) for f in hits]
    err_dec = [float(f["proxy_minus_counterfactual"]) for f in decisive]

    count_match = [
        int(f["actual_survivor_count"]) == int(f["survivor_count"])
        for f in hits
    ]
    lethal_applied_n = sum(1 for f in hits if f.get("lethal"))
    lethal_cf_n = sum(1 for f in hits if f.get("lethal_counterfactual"))
    flip_n = sum(1 for f in hits if f.get("lethal_flip"))
    overkill_n = sum(
        1 for f in hits
        if f.get("lethal") and not f.get("lethal_counterfactual")
    )
    underkill_n = sum(
        1 for f in hits
        if (not f.get("lethal")) and f.get("lethal_counterfactual")
    )

    by_turn: Dict[str, Dict] = {}
    for t in INSTRUMENT_TURNS:
        key = str(t)
        ft = [f for f in fights if int(f["turn"]) == t]
        ht = [f for f in ft if int(f.get("applied_hp_loss") or 0) > 0]
        dec = [
            f for f in ft
            if f["kind"] != "bye" and int(f.get("raw") or 0) != 0
        ]
        err_t = [float(f["proxy_minus_counterfactual"]) for f in ht]
        by_turn[key] = {
            "n_decisive": len(dec),
            "n_hits": len(ht),
            "mean_applied_when_hit": _mean(
                [float(f["applied_hp_loss"]) for f in ht]
            ),
            "mean_count_only_when_hit": _mean(
                [float(f["count_only_damage"]) for f in ht]
            ),
            "mean_counterfactual_when_hit": _mean(
                [float(f["counterfactual_damage"]) for f in ht]
            ),
            "mean_proxy_minus_cf_when_hit": _mean(err_t),
            "median_proxy_minus_cf_when_hit": _median(err_t),
            "p10_proxy_minus_cf_when_hit": _quantile(err_t, 0.10),
            "p90_proxy_minus_cf_when_hit": _quantile(err_t, 0.90),
            "mean_amplification_when_hit": _mean(
                [float(f["amplification"]) for f in ht]
            ),
            "mean_cf_amplification_when_hit": _mean(
                [float(f["cf_amplification"]) for f in ht]
            ),
            "mean_actual_survivor_count": _mean(
                [float(f["actual_survivor_count"]) for f in ht]
            ),
            "mean_actual_survivor_tier_sum": _mean(
                [float(f["actual_survivor_tier_sum"]) for f in ht]
            ),
            "mean_winner_board_tier_mean": _mean(
                [float(f["winner_minion_tier_mean"]) for f in ht
                 if f.get("winner_minion_tier_mean") is not None]
            ),
            "lethal_rate_applied": (
                sum(1 for f in ht if f.get("lethal")) / len(ht) if ht else None
            ),
            "lethal_rate_counterfactual": (
                sum(1 for f in ht if f.get("lethal_counterfactual")) / len(ht)
                if ht else None
            ),
            "lethal_flip_rate": (
                sum(1 for f in ht if f.get("lethal_flip")) / len(ht)
                if ht else None
            ),
            "error": error_summary(err_t),
        }

    return {
        "arm": raw["arm"],
        "recruit_value_stats": raw["recruit_value_stats"],
        "board_level_abstract_scaling": raw["board_level_abstract_scaling"],
        "n_lobbies": n_lobbies,
        "seed_base": raw["seed_base"],
        "mean_game_length": _mean(lengths),
        "n_fights_t7_t14": len(fights),
        "n_decisive": len(decisive),
        "n_hits": len(hits),
        "mean_applied_when_hit": _mean(applied_hit),
        "mean_count_only_when_hit": _mean(count_hit),
        "mean_counterfactual_when_hit": _mean(cf_hit),
        "mean_amplification_when_hit": _mean(amp_hit),
        "mean_cf_amplification_when_hit": _mean(cf_amp_hit),
        "mean_proxy_minus_cf_when_hit": _mean(err_hit),
        "mean_actual_survivor_count_when_hit": _mean(
            [float(f["actual_survivor_count"]) for f in hits]
        ),
        "mean_actual_survivor_tier_sum_when_hit": _mean(
            [float(f["actual_survivor_tier_sum"]) for f in hits]
        ),
        "mean_winner_board_tier_mean_when_hit": _mean(
            [float(f["winner_minion_tier_mean"]) for f in hits
             if f.get("winner_minion_tier_mean") is not None]
        ),
        "error_when_hit": error_summary(err_hit),
        "error_decisive": error_summary(err_dec),
        "survivor_count_matches_raw": all(count_match) if count_match else True,
        "n_survivor_count_mismatch": sum(1 for ok in count_match if not ok),
        "lethal_applied_n": lethal_applied_n,
        "lethal_counterfactual_n": lethal_cf_n,
        "lethal_flip_n": flip_n,
        "lethal_overkill_proxy_n": overkill_n,
        "lethal_underkill_proxy_n": underkill_n,
        "lethal_rate_applied": (
            lethal_applied_n / len(hits) if hits else None
        ),
        "lethal_rate_counterfactual": (
            lethal_cf_n / len(hits) if hits else None
        ),
        "lethal_flip_rate": (flip_n / len(hits) if hits else None),
        "per_turn": by_turn,
        "example_fights": [_slim_example(f) for f in hits[:8]],
    }


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(b) - float(a)


def compare_survivor_fidelity(control: Dict, treatment: Dict) -> Dict:
    keys = (
        "mean_game_length",
        "mean_applied_when_hit",
        "mean_count_only_when_hit",
        "mean_counterfactual_when_hit",
        "mean_amplification_when_hit",
        "mean_cf_amplification_when_hit",
        "mean_proxy_minus_cf_when_hit",
        "mean_actual_survivor_count_when_hit",
        "mean_actual_survivor_tier_sum_when_hit",
        "mean_winner_board_tier_mean_when_hit",
        "lethal_rate_applied",
        "lethal_rate_counterfactual",
        "lethal_flip_rate",
        "n_hits",
        "n_decisive",
    )
    deltas = {k: _delta(control.get(k), treatment.get(k)) for k in keys}

    per_turn = {}
    for t in INSTRUMENT_TURNS:
        key = str(t)
        c = (control.get("per_turn") or {}).get(key) or {}
        t_ = (treatment.get("per_turn") or {}).get(key) or {}
        per_turn[key] = {
            m: _delta(c.get(m), t_.get(m))
            for m in (
                "mean_applied_when_hit",
                "mean_count_only_when_hit",
                "mean_counterfactual_when_hit",
                "mean_proxy_minus_cf_when_hit",
                "mean_amplification_when_hit",
                "mean_cf_amplification_when_hit",
                "lethal_rate_applied",
                "lethal_rate_counterfactual",
                "lethal_flip_rate",
            )
        }

    proxy_amp_delta = deltas.get("mean_amplification_when_hit")
    cf_amp_delta = deltas.get("mean_cf_amplification_when_hit")
    share_remaining = None
    share_removed = None
    if proxy_amp_delta is not None and abs(float(proxy_amp_delta)) > 1e-9:
        if cf_amp_delta is not None:
            share_remaining = float(cf_amp_delta) / float(proxy_amp_delta)
            share_removed = 1.0 - share_remaining

    fid = {
        "proxy_amplification_delta_when_hit": proxy_amp_delta,
        "counterfactual_amplification_delta_when_hit": cf_amp_delta,
        "phase_2t_amp_delta_when_hit": PHASE_2T_AMP_DELTA_WHEN_HIT,
        "amp_delta_disappeared": (
            float(proxy_amp_delta) - float(cf_amp_delta)
            if proxy_amp_delta is not None and cf_amp_delta is not None
            else None
        ),
        "share_of_amp_delta_remaining": share_remaining,
        "share_of_amp_delta_removed": share_removed,
        "share_removed_most_threshold": SHARE_REMOVED_MOST,
        "applied_when_hit_delta": deltas.get("mean_applied_when_hit"),
        "counterfactual_when_hit_delta": deltas.get(
            "mean_counterfactual_when_hit"
        ),
        "count_only_when_hit_delta": deltas.get("mean_count_only_when_hit"),
        "lethal_flip_n_control": control.get("lethal_flip_n"),
        "lethal_flip_n_treatment": treatment.get("lethal_flip_n"),
        "lethal_overkill_proxy_n_control": control.get(
            "lethal_overkill_proxy_n"
        ),
        "lethal_overkill_proxy_n_treatment": treatment.get(
            "lethal_overkill_proxy_n"
        ),
        "survivor_count_matches_raw_control": control.get(
            "survivor_count_matches_raw"
        ),
        "survivor_count_matches_raw_treatment": treatment.get(
            "survivor_count_matches_raw"
        ),
    }

    return {
        "deltas": deltas,
        "control": {k: control.get(k) for k in (
            "mean_game_length", "n_hits", "n_decisive",
            "mean_applied_when_hit", "mean_count_only_when_hit",
            "mean_counterfactual_when_hit", "mean_amplification_when_hit",
            "mean_cf_amplification_when_hit",
            "mean_proxy_minus_cf_when_hit",
            "lethal_rate_applied", "lethal_rate_counterfactual",
            "lethal_flip_rate", "lethal_flip_n",
            "error_when_hit",
        )},
        "treatment": {k: treatment.get(k) for k in (
            "mean_game_length", "n_hits", "n_decisive",
            "mean_applied_when_hit", "mean_count_only_when_hit",
            "mean_counterfactual_when_hit", "mean_amplification_when_hit",
            "mean_cf_amplification_when_hit",
            "mean_proxy_minus_cf_when_hit",
            "lethal_rate_applied", "lethal_rate_counterfactual",
            "lethal_flip_rate", "lethal_flip_n",
            "error_when_hit",
        )},
        "per_turn_delta": per_turn,
        "fidelity": fid,
        "error_by_arm": {
            "control": control.get("error_when_hit"),
            "treatment": treatment.get("error_when_hit"),
        },
        "error_by_turn": {
            arm: {
                t: ((summary.get("per_turn") or {}).get(str(t)) or {}).get(
                    "error"
                )
                for t in INSTRUMENT_TURNS
            }
            for arm, summary in (("control", control), ("treatment", treatment))
        },
    }


def diagnose_from_comparison(comparison: Dict, *, non_evaluative: bool = False) -> Dict:
    return diagnose_phase_2u(comparison, non_evaluative=non_evaluative)
