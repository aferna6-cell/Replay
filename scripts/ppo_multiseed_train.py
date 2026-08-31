"""Launch one Experiment 3 PPO trajectory with the frozen Experiment 2 recipe.

    python scripts/ppo_multiseed_train.py --seed 1

Does not change architecture, optimizer, learning rate, shaping schedule,
league, or any other hyperparameter. The only CLI variable is --seed.
Refuses seed 0 (already complete in Experiment 2).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_fingerprint import checkpoint_parameter_sha256  # noqa: E402
from ml.multiseed_analysis import (  # noqa: E402
    EPISODES_PER_ITER, SHAPING_HORIZON, TRAINING_ITERS,
    WARM_START_PARAMETER_SHA256, assert_training_seeds_isolated,
    assert_warmstart_hash, seed_dir,
)
from ml.train_ppo import main as ppo_main  # noqa: E402


def train_argv(seed: int, directory: str) -> list:
    os.makedirs(os.path.join(directory, "checkpoints"), exist_ok=True)
    return [
        "--iters", str(TRAINING_ITERS),
        "--episodes", str(EPISODES_PER_ITER),
        "--seed", str(seed),
        "--shaping", "1.0",
        "--shaping-horizon", str(SHAPING_HORIZON),
        "--from-bc", "ml/policy_bc.pt",
        "--out", os.path.join(directory, "final.pt"),
        "--save-iters", "0,10,20,40,80,120,160,240,320",
        "--save-dir", os.path.join(directory, "checkpoints"),
        "--diag-log", os.path.join(directory, "train_diag.jsonl"),
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Experiment 3 frozen-recipe PPO")
    p.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--dir")
    a = p.parse_args(argv)
    assert_training_seeds_isolated([a.seed])
    if not os.path.isfile("ml/policy_bc.pt"):
        p.error("ml/policy_bc.pt is missing — reproduce the historical "
                "BC+DAgger warm start before training")
    assert_warmstart_hash(checkpoint_parameter_sha256("ml/policy_bc.pt"))
    directory = a.dir or seed_dir(a.seed)
    args = train_argv(a.seed, directory)
    print("Warm-start parameter_sha256:", WARM_START_PARAMETER_SHA256)
    print("Command: python -m ml.train_ppo", " ".join(args))
    return ppo_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
