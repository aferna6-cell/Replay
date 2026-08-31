"""Integrity-aware DEV evaluation for policies that may not terminate.

The canonical evaluator intentionally aborts on the first unfinished game.
This companion preserves that rule (unfinished games are never assigned a
placement) while scanning the complete pre-specified seed block. It records
nullable placements, exact failure seeds, complete-case descriptive metrics,
and worst/best-case bounds. It is not a Benchmark v1 TEST entry point.
"""

import argparse
import json
import time

from .benchmark import (BenchmarkIntegrityError, compute_metrics, latency_stats,
                        make_agent, run_game)
from .dev_benchmark import DEV_VERSION, dev_field_seats, field_composition
from .seeds import eval_game_seed, validate_dev_range


def run_integrity_dev(agent, field, games, base_seed):
    validate_dev_range(base_seed, games)
    seats = dev_field_seats(field)
    placements = []
    latencies = []
    failures = []
    started = time.time()
    for i in range(games):
        seed = eval_game_seed(base_seed, i)
        try:
            game = run_game(agent, seats, seed)
            placements.append(game["placement"])
            latencies.extend(game["latencies"])
        except BenchmarkIntegrityError as error:
            placements.append(None)
            failures.append({"seed": seed, "error": str(error)})
    complete = [p for p in placements if p is not None]
    missing = len(failures)
    return {
        "benchmark_version": DEV_VERSION,
        "evaluation_split": "dev",
        "evaluation_kind": "integrity-aware; unfinished games are unscored",
        "agent": agent.name,
        "checkpoint": agent.checkpoint,
        "field": field,
        "field_composition": field_composition(field),
        "games_requested": games,
        "games_completed": len(complete),
        "games_unfinished": missing,
        "seed_range": [base_seed, base_seed + games - 1],
        "placements_nullable": placements,
        "failures": failures,
        "complete_case_metrics": compute_metrics(complete) if complete else None,
        "mean_placement_sensitivity_bounds": [
            (sum(complete) + missing * 1) / games,
            (sum(complete) + missing * 8) / games,
        ],
        "latency_ms_completed_games": latency_stats(latencies),
        "elapsed_seconds": time.time() - started,
        "integrity_note": (
            "No placement is imputed for an unfinished game. Complete-case "
            "statistics exclude failures; bounds assign every unfinished game "
            "placement 1 and 8 respectively."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--field", choices=("greedy", "greedy4_random3"),
                        required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    agent = make_agent("policy", args.checkpoint, args.name)
    result = run_integrity_dev(agent, args.field, args.games, args.seed)
    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.json_out} ({result['games_unfinished']} unfinished)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
