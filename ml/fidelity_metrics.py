"""Simulator-vs-real fidelity metrics for Phase 2A."""

from __future__ import annotations

import statistics as st
from typing import Callable, Dict, List, Optional, Tuple

from hsbg_coach.bg_env import BGEnv, greedy_policy
from hsbg_coach.build_path import infer_target, load_archetypes
from hsbg_coach.pace import board_stats, load_pace, _at as pace_at
from ml.econ_env import alive_at, gold_at

COMPOSITION_TURNS = tuple(range(8, 15))
PRIMARY_TURNS = tuple(range(1, 17))


def run_fidelity_rollouts(lobbies: int, seed: int = 0,
                          policy: Callable = greedy_policy) -> List[Dict]:
    """Play greedy lobbies via ``BGEnv.play_scripted`` and collect snapshots."""
    rows: List[Dict] = []
    for i in range(lobbies):
        env = BGEnv(seed=seed + i)
        recs = env.play_scripted([policy] * env.n_players)
        game_length = max((r["turn"] for r in recs), default=0)
        for r in recs:
            s = r["state"]
            rows.append({
                "lobby": i,
                "seed": seed + i,
                "seat": r["seat"],
                "turn": r["turn"],
                "game_length": game_length,
                "tavern_tier": float(s["tavern_tier"]),
                "gold": float(s["gold"]),
                "board_size": float(len(s["board"])),
                "board_stats": float(board_stats(s)),
                "board": s["board"],
                "players_alive": float(s["players_alive"]),
            })
        del env
    return rows


def _mean(xs: List[float]) -> float:
    return st.mean(xs) if xs else 0.0


def real_composition_baseline() -> Dict[str, float]:
    """Mean build-path coverage on real winning example boards."""
    archetypes = load_archetypes()
    if not archetypes:
        return {"real_coverage_mean": 0.0, "n_boards": 0}
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "stats",
                        "firestone_final_boards.json")
    raw = json.load(open(path, encoding="utf-8"))
    coverages = []
    for comp in raw.get("boards", []):
        for ex in comp.get("examples", []):
            board = ex.get("minions") or []
            fit = infer_target(board, archetypes)
            if fit:
                coverages.append(fit.coverage)
    return {
        "real_coverage_mean": _mean(coverages),
        "real_coverage_median": st.median(coverages) if coverages else 0.0,
        "n_boards": len(coverages),
    }


def aggregate_turn_curves(rows: List[Dict], pace: Optional[Dict] = None
                          ) -> Dict[str, Dict]:
    """Per-turn sim averages vs Firestone reference curves."""
    pace = pace or load_pace()
    by_turn: Dict[int, Dict[str, List[float]]] = {}
    for r in rows:
        t = r["turn"]
        bucket = by_turn.setdefault(t, {
            "tier": [], "stats": [], "board_size": [], "gold": [],
            "players_alive": [],
        })
        bucket["tier"].append(float(r["tavern_tier"]))
        bucket["stats"].append(float(r["board_stats"]))
        bucket["board_size"].append(float(r["board_size"]))
        bucket["gold"].append(float(r["gold"]))
        bucket["players_alive"].append(float(r["players_alive"]))

    out = {}
    for t in sorted(by_turn):
        b = by_turn[t]
        sim_tier = _mean(b["tier"])
        sim_stats = _mean(b["stats"])
        sim_board = _mean(b["board_size"])
        sim_gold = _mean(b["gold"])
        sim_alive = _mean(b["players_alive"])
        real_tier = pace_at(pace.get("tavern_tier", {}), t)
        real_stats = pace_at(pace.get("scaling", {}), t)
        real_level = pace_at(pace.get("leveling", {}), t)
        ref_alive = float(alive_at(t))
        ref_gold = float(gold_at(t))

        tier_err = (sim_tier - real_tier) if real_tier is not None else None
        stats_ratio = (sim_stats / real_stats) if real_stats else None
        stats_rel_err = ((sim_stats - real_stats) / real_stats
                         if real_stats else None)

        out[str(t)] = {
            "sim_tavern_tier": sim_tier,
            "real_tavern_tier": real_tier,
            "tier_error": tier_err,
            "sim_board_stats": sim_stats,
            "real_board_stats": real_stats,
            "stats_ratio_sim_over_real": stats_ratio,
            "stats_relative_error": stats_rel_err,
            "sim_board_size": sim_board,
            "sim_gold_end_recruit": sim_gold,
            "reference_gold_income": ref_gold,
            "sim_players_alive": sim_alive,
            "reference_alive_prior": ref_alive,
            "alive_error_vs_prior": sim_alive - ref_alive,
            "real_leveling_minion_tier": real_level,
            "n_samples": len(b["tier"]),
        }
    return out


def aggregate_composition(rows: List[Dict]) -> Dict[str, float]:
    archetypes = load_archetypes()
    coverages = []
    for r in rows:
        if r["turn"] not in COMPOSITION_TURNS:
            continue
        fit = infer_target(r["board"], archetypes)
        if fit:
            coverages.append(fit.coverage)
    real = real_composition_baseline()
    sim_mean = _mean(coverages)
    return {
        "sim_coverage_mean_turns_8_14": sim_mean,
        "sim_coverage_median_turns_8_14": st.median(coverages) if coverages else 0.0,
        "sim_coverage_n": len(coverages),
        **real,
        "coverage_ratio_sim_over_real": (
            sim_mean / real["real_coverage_mean"]
            if real["real_coverage_mean"] else None),
    }


def aggregate_lobby_dynamics(rows: List[Dict]) -> Dict:
    lengths = []
    alive_curve: Dict[int, List[float]] = {}
    for r in rows:
        if "game_length" in r:
            lengths.append(float(r["game_length"]))
        t = r["turn"]
        alive_curve.setdefault(t, []).append(float(r["players_alive"]))
    avg_alive = {str(t): _mean(v) for t, v in sorted(alive_curve.items())}
    ref_alive = {str(t): float(alive_at(t)) for t in range(1, 17)}
    return {
        "avg_game_length": _mean(lengths),
        "median_game_length": st.median(lengths) if lengths else 0.0,
        "sim_alive_by_turn": avg_alive,
        "reference_alive_prior_by_turn": ref_alive,
        "n_lobbies": len(set(r["lobby"] for r in rows)),
    }


def summarize_divergence(turn_curves: Dict[str, Dict],
                         primary_turns: Tuple[int, ...] = PRIMARY_TURNS
                         ) -> Dict[str, Optional[float]]:
    """Headline ratios at key turns for regression gates."""
    out = {}
    for t in primary_turns:
        row = turn_curves.get(str(t))
        if not row:
            continue
        out[f"stats_ratio_turn_{t}"] = row.get("stats_ratio_sim_over_real")
        out[f"tier_error_turn_{t}"] = row.get("tier_error")
    return out
