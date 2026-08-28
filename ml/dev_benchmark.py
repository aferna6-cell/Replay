"""Replay DEV Evaluation — development-split model evaluation.

    NOT Benchmark v1 test results.

Reuses the Benchmark v1 machinery (same agents, metrics, CI, latency,
fingerprints, JSON schema) but runs on the reserved DEV seed interval
(``ml/seeds.py``: [DEV_SEED_START, DEV_SEED_END]). This is the split for
model development — diagnosing checkpoints, comparing experimental variants,
choosing hyperparameters and training durations. ``python -m ml.benchmark``
(the TEST split) stays reserved for sparing final confirmation of a chosen
model; iterating against it would turn the test set into a dev set and
invalidate the published baselines.

    python -m ml.dev_benchmark --agent policy --checkpoint ml/policy_ppo.pt \
        --name PPO --games 200 --field greedy --json-out results/dev/ppo.json

DEV result JSON is explicitly labeled ("evaluation_split": "dev", its own
version string) so it can never masquerade as a canonical Benchmark v1
result: compare mode and the paired analysis both treat the version/split
fields as compatibility keys, keeping dev and test files from silently
mixing. DEV-vs-DEV results with equal config pair game-by-game exactly like
test results (``ml/analyze_benchmark.py``).
"""

import argparse
import sys
from typing import Dict, Optional, Sequence

from .benchmark import (BenchmarkResult, _FIELDS, _run_games, _write_json,
                        make_agent, print_summary, result_to_json)
from .seeds import DEV_SEED_END, DEV_SEED_START, validate_dev_range

DEV_VERSION = "Replay DEV Evaluation (Benchmark v1 machinery)"
DEV_BANNER = "Replay DEV Evaluation\nNOT Benchmark v1 test results"


def run_dev_benchmark(agent, field: str, games: int,
                      base_seed: int = DEV_SEED_START,
                      progress: bool = False) -> BenchmarkResult:
    """Deterministic DEV evaluation: game i uses seed base_seed + i,
    validated against the reserved DEV interval."""
    validate_dev_range(base_seed, games)
    return _run_games(agent, field, games, base_seed, progress)


def dev_result_to_json(res: BenchmarkResult) -> Dict:
    """Benchmark v1 result schema, explicitly re-labeled as the dev split."""
    blob = result_to_json(res)
    blob["benchmark_version"] = DEV_VERSION
    blob["evaluation_split"] = "dev"
    blob["seed_policy"] = (f"development-split seeds from the reserved DEV "
                           f"interval [{DEV_SEED_START}, {DEV_SEED_END}]; "
                           f"never train on them; NOT Benchmark v1 test "
                           f"results (see ml/seeds.py)")
    return blob


def save_dev_json(res: BenchmarkResult, path: str) -> None:
    _write_json(dev_result_to_json(res), path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="ml.dev_benchmark",
        description="Replay DEV Evaluation — development-split evaluation on "
                    f"reserved DEV seeds [{DEV_SEED_START}, {DEV_SEED_END}]. "
                    "NOT Benchmark v1 test results.")
    p.add_argument("--agent", choices=["random", "greedy", "policy"],
                   required=True)
    p.add_argument("--checkpoint", help="PolicyNet .pt file (with --agent policy)")
    p.add_argument("--name", help="display name for the tested agent")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=DEV_SEED_START,
                   help=f"base DEV seed (default {DEV_SEED_START}; game i "
                        "uses seed+i; the range must stay inside the DEV "
                        "interval)")
    p.add_argument("--field", choices=sorted(_FIELDS), default="greedy")
    p.add_argument("--json-out", help="write machine-readable results here")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    a = p.parse_args(argv)

    try:
        validate_dev_range(a.seed, a.games)
        agent = make_agent(a.agent, a.checkpoint, a.name)
    except ValueError as e:
        p.error(str(e))

    print(DEV_BANNER)
    print(f"Evaluation games: {a.games}")
    print(f"DEV seed range: {a.seed}-{a.seed + a.games - 1}")
    print(f"Field: 7x {a.field}")
    res = run_dev_benchmark(agent, a.field, a.games, a.seed,
                            progress=not a.quiet)
    print_summary(res)
    print("\n(dev split — not a Benchmark v1 test result)")
    if a.json_out:
        save_dev_json(res, a.json_out)
        print(f"Saved -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
