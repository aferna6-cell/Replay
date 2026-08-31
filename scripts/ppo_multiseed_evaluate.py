"""Run the frozen Experiment 3 DEV and drift evaluations for one seed."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.model_fingerprint import checkpoint_parameter_sha256
from ml.ppo_multiseed import (CORPUS_FINGERPRINT, CORPUS_STATES, ITERATIONS,
                              TRAINING_SEEDS, WARMSTART_PARAMETER_SHA256,
                              eval_command, seed_dir)


def run(seed):
    root = seed_dir(seed)
    reference = root / "checkpoints" / "iter_000.pt"
    if checkpoint_parameter_sha256(str(reference)) != WARMSTART_PARAMETER_SHA256:
        raise ValueError(f"seed {seed} iteration-0 warm-start hash mismatch")
    (root / "dev").mkdir(parents=True, exist_ok=True)
    for iteration in ITERATIONS:
        for field in ("greedy", "greedy4_random3"):
            subprocess.run(eval_command(seed, iteration, field), check=True)

    checkpoints = [
        str(root / "checkpoints" / f"iter_{iteration:03d}.pt")
        for iteration in ITERATIONS
    ]
    subprocess.run([
        sys.executable, "-m", "ml.policy_drift",
        "--reference", str(reference),
        "--checkpoints", *checkpoints,
        "--json-out", str(root / "policy_drift.json"),
        "--categories-out", str(root / "action_category_drift.json"),
    ], check=True)

    import json
    drift = json.load(open(root / "policy_drift.json"))
    corpus = drift["corpus"]
    if (corpus["states"] != CORPUS_STATES or
            corpus["fingerprint_sha256"] != CORPUS_FINGERPRINT):
        raise ValueError(f"seed {seed} frozen corpus mismatch: {corpus}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    args = parser.parse_args(argv)
    run(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
