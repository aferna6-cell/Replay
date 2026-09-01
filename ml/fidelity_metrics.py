"""Simulator-vs-real fidelity metrics for Phase 2A."""

from __future__ import annotations

import statistics as st
from typing import Callable, Dict, List, Optional, Tuple

from hsbg_coach.bg_env import BGEnv, greedy_policy
from hsbg_coach.build_path import infer_target, load_archetypes
from hsbg_coach.pace import board_stats, load_pace
from ml.econ_env import alive_at, gold_at
from ml.fidelity_reference import reference_at_exact

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
                "placement": r.get("placement"),
            })
        del env
    return rows


def _mean(xs: List[float]) -> float:
    return st.mean(xs) if xs else 0.0


def _coverage_stats(coverages: List[float]) -> Dict[str, float]:
    return {
        "mean": _mean(coverages),
        "median": st.median(coverages) if coverages else 0.0,
        "n": len(coverages),
    }


def real_composition_baseline() -> Dict[str, float]:
    """Mean build-path coverage on real final winning example boards."""
    archetypes = load_archetypes()
    if not archetypes:
        return {
            "real_final_winner_coverage_mean": 0.0,
            "real_final_winner_coverage_median": 0.0,
            "n_boards": 0,
        }
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
    stats = _coverage_stats(coverages)
    return {
        "real_final_winner_coverage_mean": stats["mean"],
        "real_final_winner_coverage_median": stats["median"],
        "n_boards": stats["n"],
    }


def _final_winner_boards(rows: List[Dict]) -> List[Dict]:
    """One final end-of-recruit board per lobby for the 1st-place player."""
    by_lobby: Dict[int, List[Dict]] = {}
    for r in rows:
        by_lobby.setdefault(r["lobby"], []).append(r)
    finals: List[Dict] = []
    for recs in by_lobby.values():
        winners = [r for r in recs if r.get("placement") == 1]
        if not winners:
            continue
        finals.append(max(winners, key=lambda r: r["turn"]))
    return finals


def _reference_status(curve: Dict[int, float], turn: int,
                      value: Optional[float]) -> str:
    if value is None:
        return "unmeasured"
    return "measured"


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

        real_tier = reference_at_exact(pace.get("tavern_tier", {}), t)
        real_stats = reference_at_exact(pace.get("scaling", {}), t)
        real_level = reference_at_exact(pace.get("leveling", {}), t)
        ref_alive = float(alive_at(t))
        ref_gold = float(gold_at(t))

        tier_err = (sim_tier - real_tier) if real_tier is not None else None
        stats_ratio = (sim_stats / real_stats) if real_stats else None
        stats_rel_err = ((sim_stats - real_stats) / real_stats
                         if real_stats else None)

        out[str(t)] = {
            "sim_tavern_tier": sim_tier,
            "real_tavern_tier": real_tier,
            "real_tavern_tier_status": _reference_status(
                pace.get("tavern_tier", {}), t, real_tier),
            "tier_error": tier_err,
            "sim_board_stats": sim_stats,
            "real_board_stats": real_stats,
            "real_board_stats_status": _reference_status(
                pace.get("scaling", {}), t, real_stats),
            "stats_ratio_sim_over_real": stats_ratio,
            "stats_relative_error": stats_rel_err,
            "sim_board_size": sim_board,
            "sim_gold_end_recruit": sim_gold,
            "reference_gold_income": ref_gold,
            "sim_players_alive": sim_alive,
            "reference_alive_prior": ref_alive,
            "alive_error_vs_prior": sim_alive - ref_alive,
            "real_leveling_minion_tier": real_level,
            "real_leveling_status": _reference_status(
                pace.get("leveling", {}), t, real_level),
            "n_samples": len(b["tier"]),
        }
    return out


def aggregate_composition(rows: List[Dict]) -> Dict:
    archetypes = load_archetypes()
    midgame_coverages: List[float] = []
    for r in rows:
        if r["turn"] not in COMPOSITION_TURNS:
            continue
        fit = infer_target(r["board"], archetypes)
        if fit:
            midgame_coverages.append(fit.coverage)

    sim_final_coverages: List[float] = []
    for r in _final_winner_boards(rows):
        fit = infer_target(r["board"], archetypes)
        if fit:
            sim_final_coverages.append(fit.coverage)

    real = real_composition_baseline()
    midgame = _coverage_stats(midgame_coverages)
    sim_final = _coverage_stats(sim_final_coverages)
    real_mean = real["real_final_winner_coverage_mean"]

    return {
        "sim_midgame_to_final_winner_coverage": {
            "sim_mean": midgame["mean"],
            "sim_median": midgame["median"],
            "sim_n": midgame["n"],
            "real_final_winner_mean": real_mean,
            "real_final_winner_median": real["real_final_winner_coverage_median"],
            "real_n": real["n_boards"],
            "note": (
                "Diagnostic, not calibrated fidelity: sim boards from turns 8–14 "
                "vs real final winning-board examples (different stages)."
            ),
        },
        "final_winner_coverage": {
            "real_mean": real_mean,
            "real_median": real["real_final_winner_coverage_median"],
            "real_n": real["n_boards"],
            "sim_mean": sim_final["mean"],
            "sim_median": sim_final["median"],
            "sim_n": sim_final["n"],
            "ratio_sim_over_real": (
                sim_final["mean"] / real_mean if real_mean else None),
            "sim_distribution": sim_final_coverages,
        },
    }


def aggregate_lobby_dynamics(rows: List[Dict]) -> Dict:
    lobby_lengths: Dict[int, float] = {}
    alive_curve: Dict[int, List[float]] = {}
    for r in rows:
        lobby = r["lobby"]
        lobby_lengths[lobby] = max(lobby_lengths.get(lobby, 0.0), float(r["turn"]))
        t = r["turn"]
        alive_curve.setdefault(t, []).append(float(r["players_alive"]))
    lengths = list(lobby_lengths.values())
    avg_alive = {str(t): _mean(v) for t, v in sorted(alive_curve.items())}
    ref_alive = {str(t): float(alive_at(t)) for t in range(1, 17)}
    return {
        "avg_game_length": _mean(lengths),
        "median_game_length": st.median(lengths) if lengths else 0.0,
        "sim_alive_by_turn": avg_alive,
        "reference_alive_prior_by_turn": ref_alive,
        "n_lobbies": len(lobby_lengths),
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
