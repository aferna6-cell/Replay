"""Generate the Replay Experiment 3 reproducibility manifest."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy
import torch

from ml.model_fingerprint import checkpoint_fingerprint
from ml.ppo_multiseed import (ALL_SEEDS, CORPUS_FINGERPRINT, CORPUS_STATES,
                              DEV_BASE_SEED, EPISODES_PER_ITERATION,
                              EXPERIMENT2_DIR, EXPERIMENT_DIR, GREEDY_GAMES,
                              ITERATIONS, MIXED_GAMES, REPORT_PATH,
                              SHAPING_HORIZON, TRAINING_SEEDS,
                              WARMSTART_PARAMETER_SHA256, eval_command,
                              seed_dir, train_command, training_seed_span,
                              validate_protocol)
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY, LEAGUE_MAX,
                          PPO_EPOCHS, VALUE_COEF)


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    validate_protocol()
    warm = checkpoint_fingerprint("ml/policy_bc.pt")
    if warm["parameter_sha256"] != WARMSTART_PARAMETER_SHA256:
        raise ValueError("warm-start parameter SHA256 differs from Experiment 2")

    runs = []
    for seed in TRAINING_SEEDS:
        checkpoints = []
        for iteration in ITERATIONS:
            path = seed_dir(seed) / "checkpoints" / f"iter_{iteration:03d}.pt"
            fp = checkpoint_fingerprint(str(path))
            if iteration == 0 and fp["parameter_sha256"] != WARMSTART_PARAMETER_SHA256:
                raise ValueError(f"seed {seed} iteration-0 is not the warm start")
            checkpoints.append({
                "iteration": iteration,
                "cumulative_episodes": iteration * EPISODES_PER_ITERATION,
                "file": str(path),
                **fp,
            })
        drift = json.load(open(seed_dir(seed) / "policy_drift.json"))
        if (drift["corpus"]["states"] != CORPUS_STATES or
                drift["corpus"]["fingerprint_sha256"] != CORPUS_FINGERPRINT):
            raise ValueError(f"seed {seed} corpus fingerprint mismatch")
        runs.append({
            "training_seed": seed,
            "training_seed_span": list(training_seed_span(seed)),
            "training_command": " ".join(train_command(seed)),
            "checkpoints": checkpoints,
            "evaluation_commands": [
                " ".join(eval_command(seed, iteration, field))
                for iteration in ITERATIONS
                for field in ("greedy", "greedy4_random3")
            ],
            "diagnostic_commands": [
                "python3 -m ml.policy_drift "
                f"--reference {seed_dir(seed)}/checkpoints/iter_000.pt "
                "--checkpoints " +
                " ".join(str(seed_dir(seed) / "checkpoints" /
                             f"iter_{i:03d}.pt") for i in ITERATIONS) +
                f" --json-out {seed_dir(seed)}/policy_drift.json "
                f"--categories-out {seed_dir(seed)}/action_category_drift.json"
            ],
        })

    manifest = {
        "experiment": "Replay Experiment 3 — Multi-seed PPO Budget Replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "protocol": {
            "frozen_from": "Replay Experiment 2",
            "new_training_seeds": list(TRAINING_SEEDS),
            "seed0_policy": ("read committed results/ppo_budget_v1 artifacts only; "
                             "never retrained or re-evaluated"),
            "iterations": 320,
            "episodes_per_iteration": EPISODES_PER_ITERATION,
            "total_episodes_per_seed": 320 * EPISODES_PER_ITERATION,
            "checkpoints": list(ITERATIONS),
            "shaping_initial": 1.0,
            "shaping_horizon": SHAPING_HORIZON,
            "warm_start": {"file": "ml/policy_bc.pt", **warm},
            "hyperparameters": {
                "lr": 0.0003, "weight_decay": 0.0001, "optimizer": "AdamW",
                "gamma": GAMMA, "lambda": LAM, "clip": CLIP,
                "entropy_coefficient": ENTROPY,
                "value_coefficient": VALUE_COEF,
                "ppo_epochs": PPO_EPOCHS, "minibatch": 256,
                "gradient_clip_norm": 1.0,
                "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX,
            },
        },
        "training_runs": runs,
        "evaluation": {
            "split": "DEV",
            "base_seed": DEV_BASE_SEED,
            "greedy": {"games": GREEDY_GAMES,
                       "seed_range": [DEV_BASE_SEED,
                                      DEV_BASE_SEED + GREEDY_GAMES - 1],
                       "field": "7x greedy"},
            "greedy4_random3": {
                "games": MIXED_GAMES,
                "seed_range": [DEV_BASE_SEED,
                               DEV_BASE_SEED + MIXED_GAMES - 1],
                "field": "seats 1-4 greedy; seats 5-7 random",
                "kind": "diagnostic; 4.5 threshold does not apply",
            },
            "test_interval_NOT_USED": [10_250_000, 10_299_999],
        },
        "drift_corpus": {
            "states": CORPUS_STATES,
            "fingerprint_sha256": CORPUS_FINGERPRINT,
            "source": "exact Experiment 2 frozen corpus definition",
        },
        "analysis": {
            "all_seeds": list(ALL_SEEDS),
            "paired_bootstrap": "10000 resamples per seed and field",
            "artifacts": {
                "aggregate": str(EXPERIMENT_DIR / "aggregate.json"),
                "paired": str(EXPERIMENT_DIR / "paired_analysis.json"),
                "rl_signal": str(EXPERIMENT_DIR / "rl_signal.json"),
                "plots": str(EXPERIMENT_DIR / "plots"),
                "report": str(REPORT_PATH),
            },
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
        "checkpoint_storage": ("checkpoint binaries are gitignored by repository "
                               "convention; exact hashes are committed here"),
        "guardrails": [
            "No PPO hyperparameter tuning",
            "No seed or checkpoint selection",
            "No Experiment 4 execution",
            "No TEST seed use",
        ],
    }
    path = EXPERIMENT_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
