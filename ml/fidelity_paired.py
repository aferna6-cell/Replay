"""Paired lobby-level fidelity statistics and pre-specified success gates."""

from __future__ import annotations

import random
import statistics as st
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hsbg_coach.pace import load_pace

from ml.fidelity_reference import reference_at_exact

MEASURED_TURNS = (8, 9, 10, 11, 12, 13, 14)
PRIMARY_TURNS = (10, 12, 14)


def per_lobby_turn_means(rows: Iterable[Dict]) -> Dict[int, Dict[int, float]]:
    """lobby -> turn -> mean board_stats across alive seats at end-recruit."""
    acc: Dict[int, Dict[int, List[float]]] = {}
    for r in rows:
        lobby = r["lobby"]
        turn = r["turn"]
        acc.setdefault(lobby, {}).setdefault(turn, []).append(float(r["board_stats"]))
    return {
        lobby: {t: st.mean(vals) for t, vals in turns.items()}
        for lobby, turns in acc.items()
    }


def aggregate_ratio_at_turn(per_lobby: Dict[int, Dict[int, float]], turn: int,
                            real_stats: float) -> Optional[float]:
    vals = [per_lobby[l][turn] for l in per_lobby if turn in per_lobby[l]]
    if not vals or not real_stats:
        return None
    return st.mean(vals) / real_stats


def per_lobby_ratios(per_lobby: Dict[int, Dict[int, float]], turn: int,
                     real_stats: float) -> List[float]:
    if not real_stats:
        return []
    return [per_lobby[l][turn] / real_stats
            for l in sorted(per_lobby) if turn in per_lobby[l]]


def bootstrap_ratio_ci(per_lobby: Dict[int, Dict[int, float]], turn: int,
                       real_stats: float, *, n_samples: int = 2000,
                       seed: int = 0) -> Dict[str, float]:
    """Bootstrap CI for the aggregate sim/real stats ratio at one turn."""
    lobby_ids = sorted(per_lobby)
    if not lobby_ids or not real_stats:
        return {}
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_samples):
        draw = [rng.choice(lobby_ids) for _ in lobby_ids]
        mean_sim = st.mean(per_lobby[i][turn] for i in draw if turn in per_lobby[i])
        samples.append(mean_sim / real_stats)
    samples.sort()
    n = len(samples)
    return {
        "turn": turn,
        "mean_ratio": st.mean(samples),
        "p2_5": samples[max(0, int(0.025 * n) - 1)],
        "p50": samples[int(0.50 * n)],
        "p97_5": samples[min(n - 1, int(0.975 * n))],
        "std": st.pstdev(samples),
        "n_lobbies": len(lobby_ids),
        "n_bootstrap": n_samples,
    }


def freeze_success_thresholds(per_lobby_v1: Dict[int, Dict[int, float]],
                            *, seed: int = 0) -> Dict:
    """Derive pre-specified gates from Simulator v1 lobby variability only."""
    scaling = load_pace().get("scaling", {})
    turn_stats = {}
    for t in PRIMARY_TURNS + (8, 9, 11, 13):
        real = reference_at_exact(scaling, t)
        if real is None:
            continue
        turn_stats[str(t)] = {
            "real_board_stats": real,
            "baseline_ratio": aggregate_ratio_at_turn(per_lobby_v1, t, real),
            "bootstrap": bootstrap_ratio_ci(per_lobby_v1, t, real, seed=seed + t),
        }

    b10 = turn_stats["10"]["bootstrap"]
    b12 = turn_stats["12"]["bootstrap"]
    b14 = turn_stats["14"]["bootstrap"]

    return {
        "derived_from": "Simulator v1 per-lobby rollouts (ratio scaling)",
        "measured_turns_only": True,
        "turns": turn_stats,
        "gates": {
            "turn_14_primary_max_ratio": round(
                min(b14["mean_ratio"] - b14["std"],
                    b14["mean_ratio"] * 0.70), 3),
            "turn_12_secondary_max_ratio": round(
                min(b12["mean_ratio"] - 0.5 * b12["std"],
                    b12["mean_ratio"]), 3),
            "turn_10_regression_band": round(
                max(0.05, 2.0 * b10["std"]), 3),
            "turn_10_center_ratio": round(b10["mean_ratio"], 3),
            "tier_alive_game_length": {
                "note": "Report only; no numeric gate frozen here.",
            },
        },
    }


def paired_turn_comparison(per_lobby_v1: Dict[int, Dict[int, float]],
                           per_lobby_v11: Dict[int, Dict[int, float]],
                           turns: Sequence[int] = PRIMARY_TURNS) -> Dict[str, Dict]:
    scaling = load_pace().get("scaling", {})
    out: Dict[str, Dict] = {}
    for t in turns:
        real = reference_at_exact(scaling, t)
        if real is None:
            continue
        shared = sorted(set(per_lobby_v1) & set(per_lobby_v11))
        v1_ratios = [per_lobby_v1[l][t] / real for l in shared if t in per_lobby_v1[l]]
        v11_ratios = [per_lobby_v11[l][t] / real for l in shared if t in per_lobby_v11[l]]
        paired_delta = [
            per_lobby_v11[l][t] - per_lobby_v1[l][t]
            for l in shared if t in per_lobby_v1[l] and t in per_lobby_v11[l]
        ]
        out[str(t)] = {
            "real_board_stats": real,
            "v1_mean_ratio": st.mean(v1_ratios) if v1_ratios else None,
            "v1_1_mean_ratio": st.mean(v11_ratios) if v11_ratios else None,
            "paired_mean_delta_sim_stats": st.mean(paired_delta) if paired_delta else None,
            "paired_median_delta_sim_stats": (
                st.median(paired_delta) if paired_delta else None),
            "n_paired_lobbies": len(paired_delta),
            "v1_lobbies_improved_ratio": sum(
                1 for a, b in zip(v1_ratios, v11_ratios) if b < a),
        }
    return out


def evaluate_gates(thresholds: Dict, paired: Dict[str, Dict],
                   turn_curves_v11: Dict[str, Dict],
                   lobby_v11: Dict) -> Dict:
    """Accept/reject Simulator v1.1 against frozen pre-specified gates."""
    gates = thresholds["gates"]
    results = {}

    t10 = paired.get("10", {})
    t12 = paired.get("12", {})
    t14 = paired.get("14", {})
    v11_r10 = t10.get("v1_1_mean_ratio")
    v11_r12 = t12.get("v1_1_mean_ratio")
    v11_r14 = t14.get("v1_1_mean_ratio")
    center = gates["turn_10_center_ratio"]
    band = gates["turn_10_regression_band"]

    results["turn_14_primary"] = {
        "value": v11_r14,
        "max_allowed": gates["turn_14_primary_max_ratio"],
        "passed": v11_r14 is not None and v11_r14 <= gates["turn_14_primary_max_ratio"],
    }
    results["turn_12_secondary"] = {
        "value": v11_r12,
        "baseline": t12.get("v1_mean_ratio"),
        "max_allowed": gates["turn_12_secondary_max_ratio"],
        "passed": (
            v11_r12 is not None and t12.get("v1_mean_ratio") is not None
            and v11_r12 <= gates["turn_12_secondary_max_ratio"]
            and v11_r12 < t12["v1_mean_ratio"]),
    }
    results["turn_10_regression"] = {
        "value": v11_r10,
        "baseline_center": center,
        "allowed_abs_delta": band,
        "passed": (
            v11_r10 is not None and abs(v11_r10 - center) <= band),
    }

    tier_ok = all(
        abs((turn_curves_v11.get(str(t)) or {}).get("tier_error") or 0.0) <= 0.75
        for t in (8, 9, 10, 11, 12, 13, 14)
        if turn_curves_v11.get(str(t)) and
        (turn_curves_v11[str(t)].get("real_tavern_tier") is not None))
    results["tavern_tier_unchanged"] = {"passed": tier_ok}

    alive_ok = all(
        abs((turn_curves_v11.get(str(t)) or {}).get("alive_error_vs_prior") or 0.0) <= 1.5
        for t in (8, 9, 10, 11, 12, 13, 14)
        if turn_curves_v11.get(str(t)))
    results["alive_curve_unchanged"] = {"passed": alive_ok}

    results["game_length"] = {
        "value": lobby_v11.get("avg_game_length"),
        "passed": lobby_v11.get("avg_game_length") is not None,
    }

    core = [
        results["turn_14_primary"]["passed"],
        results["turn_12_secondary"]["passed"],
        results["turn_10_regression"]["passed"],
        results["tavern_tier_unchanged"]["passed"],
        results["alive_curve_unchanged"]["passed"],
    ]
    results["accept_v1_1"] = all(core)
    return results
