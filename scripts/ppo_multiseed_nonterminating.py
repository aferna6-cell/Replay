"""Count the DEV lobbies a checkpoint cannot finish.

``ml.benchmark.run_game`` refuses to score a game that has not terminated
within ``MAX_DECISIONS`` — a policy that never ends its turn would otherwise
be silently recorded as an 8th place. Experiment 3 hit that guard: one
training seed's 5,120-episode policy stalls on some DEV lobbies.

This script measures how often, without changing the instrument. It replays
the identical DEV seeds one game at a time and records, per game, either the
placement or a non-termination, so the failure can be reported as the
measurement it is instead of being worked around.

    python scripts/ppo_multiseed_nonterminating.py \
        --checkpoint results/ppo_multiseed_v1/seed_1/checkpoints/iter_320.pt \
        --field greedy --games 1000 --json-out /tmp/seed1_iter320.json

DEV split only; the seed range is validated against the reserved DEV
interval exactly as the normal evaluation path does.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.benchmark import (BenchmarkIntegrityError, make_agent,  # noqa: E402
                          run_game)
from ml.dev_benchmark import dev_field_seats                    # noqa: E402
from ml.seeds import (DEV_SEED_START, eval_game_seed,           # noqa: E402
                      validate_dev_range)


def scan(checkpoint: str, field: str, games: int,
         base_seed: int = DEV_SEED_START, name: str = "scan") -> dict:
    validate_dev_range(base_seed, games)
    agent = make_agent("policy", checkpoint, name)
    seats = dev_field_seats(field)
    placements, stalled = [], []
    for i in range(games):
        seed = eval_game_seed(base_seed, i)
        try:
            placements.append({"index": i, "seed": seed,
                               "placement": run_game(agent, seats,
                                                     seed)["placement"]})
        except BenchmarkIntegrityError as e:
            stalled.append({"index": i, "seed": seed, "error": str(e)})
    n_ok = len(placements)
    return {
        "checkpoint": os.path.basename(checkpoint), "field": field,
        "games_requested": games, "base_seed": base_seed,
        "seed_range": [base_seed, base_seed + games - 1],
        "games_completed": n_ok,
        "games_non_terminating": len(stalled),
        "non_termination_rate": len(stalled) / games if games else None,
        "non_terminating_seeds": [s["seed"] for s in stalled],
        "completed_avg_placement":
            sum(p["placement"] for p in placements) / n_ok if n_ok else None,
        "completed_placements": [p["placement"] for p in placements],
        "completed_indices": [p["index"] for p in placements],
        "note": ("the completed-subset average EXCLUDES the lobbies the "
                 "policy could not finish and is therefore biased in the "
                 "policy's favor; it is not a benchmark result"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ppo_multiseed_nonterminating")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--field", default="greedy")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=DEV_SEED_START)
    p.add_argument("--name", default="scan")
    p.add_argument("--json-out")
    a = p.parse_args(argv)
    out = scan(a.checkpoint, a.field, a.games, a.seed, a.name)
    print(f"{out['checkpoint']} vs {a.field}: "
          f"{out['games_non_terminating']}/{a.games} lobbies did not "
          f"terminate ({100 * out['non_termination_rate']:.1f}%)")
    if out["completed_avg_placement"] is not None:
        print(f"  completed-subset avg placement "
              f"{out['completed_avg_placement']:.3f} "
              f"(biased — excludes the stalled lobbies)")
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
