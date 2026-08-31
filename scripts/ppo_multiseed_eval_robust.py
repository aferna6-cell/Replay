"""Robust DEV evaluation for a seed whose checkpoints contain a
non-terminating episode on one or more DEV lobbies.

Genuine finding, not a pipeline bug: Experiment 3 training seed 1's
iteration-320 checkpoint enters a deterministic infinite loop (repeatedly
selecting "freeze", an idempotent no-op once the shop-frozen observation
settles into a 1- or 2-cycle under a fully-drifted deterministic argmax
policy) on a handful of the 1000/500 DEV lobbies. ``ml.benchmark``'s
integrity check is correct to refuse to score these as a silent 8th place —
but Experiment 2's tooling assumes every checkpoint of a trajectory can
complete the full seed range, which iteration 320 of this seed cannot.

To keep every paired comparison for this training seed valid (identical
game-by-game seeds, sample-by-sample aligned, across every primary
checkpoint), this script:

  1. Scans every primary checkpoint {0, 40, 80, 160, 320} of the seed for
     non-terminating games, over the full requested seed range, for a given
     field.
  2. Takes the UNION of every problem seed found at ANY checkpoint.
  3. Re-scores ALL FIVE checkpoints over the seed range with that union
     excluded, so every checkpoint's result file has identical, aligned
     placements arrays (just fewer games than the nominal 1000/500).

The excluded seeds and the reason are recorded explicitly in every affected
result JSON (``excluded_seeds``, ``requested_games``, ``non_termination_note``)
— nothing is silently dropped or scored as an unfinished game.

    python -m scripts.ppo_multiseed_eval_robust --seed 1 --field greedy --games 1000
    python -m scripts.ppo_multiseed_eval_robust --seed 1 --field greedy4_random3 --games 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.benchmark import (BenchmarkIntegrityError, bootstrap_ci,
                          compute_metrics, latency_stats, make_agent,
                          run_game)                                    # noqa: E402
from ml.dev_benchmark import (DEV_BANNER, DEV_VERSION, dev_field_seats,
                              field_composition)                       # noqa: E402
from ml.seeds import DEV_SEED_END, DEV_SEED_START                      # noqa: E402
from hsbg_coach.bg_env import MAX_TURNS                                # noqa: E402
from ml.rl_common import MAX_DECISIONS                                 # noqa: E402
from ml.benchmark import FIELD_SIZE, BEAT_FIELD_THRESHOLD              # noqa: E402
import json

ITERS = [0, 40, 80, 160, 320]


def scan_problem_seeds(checkpoint_paths, field, games, base_seed):
    """Union of DEV seeds (base_seed + i) that fail to terminate on ANY of
    the given checkpoints, for this field."""
    seats = dev_field_seats(field)
    problems = set()
    for path in checkpoint_paths:
        agent = make_agent("policy", path, os.path.basename(path))
        for i in range(games):
            seed = base_seed + i
            try:
                run_game(agent, seats, seed)
            except BenchmarkIntegrityError:
                problems.add(seed)
    return problems


def evaluate_excluding(checkpoint_path, field, games, base_seed, exclude):
    seats = dev_field_seats(field)
    agent = make_agent("policy", checkpoint_path, os.path.basename(checkpoint_path))
    placements, latencies, used_seeds = [], [], []
    for i in range(games):
        seed = base_seed + i
        if seed in exclude:
            continue
        g = run_game(agent, seats, seed)
        placements.append(g["placement"])
        latencies.extend(g["latencies"])
        used_seeds.append(seed)
    return placements, latencies, used_seeds


def result_json(agent_name, checkpoint_path, field, requested_games,
                base_seed, excluded, placements, latencies, used_seeds):
    import hashlib
    with open(checkpoint_path, "rb") as f:
        checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()
    metrics = compute_metrics(placements)
    return {
        "benchmark_version": DEV_VERSION,
        "evaluation_split": "dev",
        "agent": agent_name,
        "agent_kind": "policy",
        "checkpoint": os.path.basename(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "field": field,
        "field_composition": field_composition(field),
        "games": len(placements),
        "requested_games": requested_games,
        "base_seed": base_seed,
        "seed_range": [used_seeds[0], used_seeds[-1]] if used_seeds else None,
        "excluded_seeds": sorted(excluded),
        "non_termination_note": (
            "these DEV seeds were excluded from EVERY primary checkpoint of "
            "this training seed (uniformly) because at least one checkpoint "
            "entered a genuine non-terminating decision loop "
            "(BenchmarkIntegrityError, MAX_DECISIONS cap) on them; excluding "
            "the same seeds from every checkpoint keeps the paired "
            "comparison sample-by-sample aligned. This is a real drift "
            "finding (a fully-drifted deterministic policy can get stuck "
            "repeatedly selecting an idempotent action), not a pipeline "
            "defect, and is reported explicitly rather than silently scored "
            "as an unfinished game or dropped without a record."
        ) if excluded else None,
        "seed_policy": (f"development-split seeds from the reserved DEV "
                        f"interval [{DEV_SEED_START}, {DEV_SEED_END}]; "
                        f"never train on them; NOT Benchmark v1 test "
                        f"results (see ml/seeds.py)"),
        "environment": {"env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
                        "field_size": FIELD_SIZE, "max_turns": MAX_TURNS,
                        "max_decisions": MAX_DECISIONS, "agent_seat": 0},
        "field_kind": ("dev diagnostic (mixed opponents)" if field != "greedy"
                       else "homogeneous"),
        "beat_field_threshold": BEAT_FIELD_THRESHOLD if field == "greedy" else None,
        "beats_field": (metrics["avg_placement"] < BEAT_FIELD_THRESHOLD
                        if field == "greedy" else None),
        "metrics": metrics,
        "placements": list(placements),
        "avg_placement_ci95": bootstrap_ci(placements, seed=base_seed),
        "decision_latency": latency_stats(latencies),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--field", required=True, choices=["greedy", "greedy4_random3"])
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--base-seed", type=int, default=DEV_SEED_START)
    a = p.parse_args(argv)

    ckpt_dir = f"results/ppo_multiseed_v1/seed_{a.seed}/checkpoints"
    dev_dir = f"results/ppo_multiseed_v1/seed_{a.seed}/dev"
    os.makedirs(dev_dir, exist_ok=True)
    paths = [os.path.join(ckpt_dir, f"iter_{it:03d}.pt") for it in ITERS]

    print(DEV_BANNER)
    print(f"Scanning {len(paths)} checkpoints for non-terminating episodes "
          f"({a.field}, {a.games} games, base {a.base_seed})…")
    problems = scan_problem_seeds(paths, a.field, a.games, a.base_seed)
    if problems:
        print(f"  Non-terminating DEV seeds found (excluded from ALL "
              f"checkpoints of seed {a.seed}, {a.field}): {sorted(problems)}")
    else:
        print("  None found.")

    for it, path in zip(ITERS, paths):
        placements, latencies, used_seeds = evaluate_excluding(
            path, a.field, a.games, a.base_seed, problems)
        blob = result_json(f"PPO_seed{a.seed}_iter{it}", path, a.field,
                           a.games, a.base_seed, problems, placements,
                           latencies, used_seeds)
        out_path = os.path.join(dev_dir, f"iter{it:03d}_vs_{a.field}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        print(f"  iter{it:03d}: {blob['metrics']['avg_placement']:.3f} avg "
              f"placement over {blob['games']}/{a.games} games "
              f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
