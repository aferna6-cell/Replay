"""Write the Experiment 3 (multi-seed PPO budget replication) top-level
reproducibility manifest.

Records the frozen recipe (identical to Experiment 2 except training seed),
the frozen BC+DAgger warm start and its verification, per-seed
reproduction-gate results (iteration-0 checkpoint parameter hash must equal
the warm start), the frozen diagnostic-corpus fingerprint, and package
versions. Reads only already-computed fingerprints/results; does not train
or evaluate anything.

    python scripts/ppo_multiseed_manifest.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_fingerprint import checkpoint_fingerprint          # noqa: E402
from ml.policy_drift import CORPUS_LOBBIES, CORPUS_SEED_BASE      # noqa: E402
from ml.seeds import (DEV_SEED_END, DEV_SEED_START, EVAL_SEED_END,  # noqa: E402
                      EVAL_SEED_START, ppo_episode_seed,
                      check_training_range)
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,  # noqa: E402
                          LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)

DIR = "results/ppo_multiseed_v1"
NEW_SEEDS = [1, 2, 3]
PRIMARY_ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
WARM_START_PARAMETER_SHA256 = "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b"
WARM_START_CHECKPOINT_SHA256 = "bd3a4386329ecf1abc045e90d069816b801aa02d18db04af625e8ef452b0d871"
DIAGNOSTIC_CORPUS_FINGERPRINT = "2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e"


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def main() -> int:
    import numpy
    import torch

    warm = checkpoint_fingerprint("ml/policy_bc.pt")
    warm_ok = warm["parameter_sha256"] == WARM_START_PARAMETER_SHA256
    print(f"Warm-start parameter hash matches frozen Experiment 1/2 value: {warm_ok}")

    seeds_block = {}
    all_gates_pass = warm_ok
    for seed in NEW_SEEDS:
        base = f"{DIR}/seed_{seed}"
        checkpoints = []
        for it in PRIMARY_ITERS:
            path = f"{base}/checkpoints/iter_{it:03d}.pt"
            checkpoints.append({
                "iteration": it, "cumulative_episodes": it * EPISODES_PER_ITER,
                "primary": True, "file": os.path.basename(path),
                **checkpoint_fingerprint(path)})
        iter0 = checkpoints[0]
        gate_passed = iter0["parameter_sha256"] == WARM_START_PARAMETER_SHA256
        all_gates_pass = all_gates_pass and gate_passed

        lo = ppo_episode_seed(seed, 1)
        hi = ppo_episode_seed(seed, 320 * EPISODES_PER_ITER)
        overlaps_reserved = check_training_range(f"seed_{seed}_check", lo, hi)

        seeds_block[str(seed)] = {
            "training_seed": seed,
            "command": (
                f"python -m ml.train_ppo --iters 320 --episodes 16 "
                f"--seed {seed} --shaping 1.0 --shaping-horizon 40 "
                f"--from-bc ml/policy_bc.pt "
                f"--out {base}/final.pt "
                f"--save-iters 0,40,80,160,320 "
                f"--save-dir {base}/checkpoints "
                f"--diag-log {base}/train_diag.jsonl"),
            "seed_scheme": f"ml.seeds.ppo_episode_seed({seed}, k) = {seed}*1000003 + k",
            "episode_seed_span": [lo, hi],
            "overlaps_dev_or_test_reserved_interval": overlaps_reserved,
            "checkpoints": checkpoints,
            "reproduction_gate": {
                "requirement": "iteration-0 checkpoint parameter_sha256 must "
                               "equal the frozen BC+DAgger warm start",
                "warm_start_parameter_sha256": WARM_START_PARAMETER_SHA256,
                "iter0_parameter_sha256": iter0["parameter_sha256"],
                "passed": gate_passed,
            },
        }

    manifest = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": ("Does the transient PPO improvement (and subsequent "
                     "drift/regression) found at training seed 0 in "
                     "Experiment 2 replicate across independent training "
                     "seeds, or was it a lucky excursion of one trajectory?"),
        "single_variable": "PPO training seed (1, 2, 3 new; seed 0 reused "
                           "as-is from Experiment 2)",
        "unchanged_from_experiment_2": [
            "architecture", "optimizer", "learning rate", "weight decay",
            "gamma", "lambda", "PPO clip", "entropy coefficient",
            "value coefficient", "PPO epochs", "batch size",
            "league behavior/schedule", "reward function",
            "shaping magnitude/schedule (--shaping-horizon 40)",
            "episodes per iteration", "observation encoder", "action space",
            "training budget (320 iterations x 16 episodes = 5,120 episodes)",
        ],
        "hyperparameters": {
            "lr": 0.0003, "weight_decay": 0.0001, "optimizer": "AdamW",
            "gamma": GAMMA, "lam": LAM, "clip": CLIP, "entropy": ENTROPY,
            "value_coef": VALUE_COEF, "ppo_epochs": PPO_EPOCHS,
            "minibatch": 256, "grad_clip_norm": 1.0,
            "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX,
        },
        "shaping_schedule": {
            "initial": 1.0, "horizon_iterations": 40,
            "formula": "shaping = 1.0 * max(0, 1 - it / (horizon * 0.7))",
            "iterations_1_to_40": "identical to Experiment 2 / the original "
                                  "baseline PPO schedule",
            "extension_behavior": "shaping stays 0.0 for every iteration "
                                  "after 40 for all three new seeds",
        },
        "warm_start": {
            "file": "ml/policy_bc.pt",
            "checkpoint_sha256": WARM_START_CHECKPOINT_SHA256,
            "parameter_sha256": WARM_START_PARAMETER_SHA256,
            "matches_experiment_2_manifest": warm_ok,
            "historical_training_command":
                "python -m ml.bc --lobbies 150 --epochs 6 --dagger-rounds 2 "
                "--dagger-lobbies 80 --seed 0 --out ml/policy_bc.pt",
            "reproduced_locally": True,
            "reproduction_note": ("ml/policy_bc.pt is gitignored per repo "
                                  "convention; it was not present locally at "
                                  "the start of Experiment 3, so it was "
                                  "reproduced by re-running the exact "
                                  "historical BC+DAgger command above. The "
                                  "reproduced checkpoint's parameter_sha256 "
                                  "matched the historical recorded value "
                                  "exactly (bitwise-identical parameters) "
                                  "before any Experiment 3 training began."),
        },
        "seeds": seeds_block,
        "seed_0_reference": {
            "note": ("Seed 0 is the existing, committed Experiment 2 "
                     "trajectory (results/ppo_budget_v1/) — reused as-is, "
                     "not retrained or altered for Experiment 3."),
            "source": "results/ppo_budget_v1/manifest.json",
        },
        "evaluation": {
            "split": "dev",
            "dev_interval": [DEV_SEED_START, DEV_SEED_END],
            "test_interval_NOT_USED": [EVAL_SEED_START, EVAL_SEED_END],
            "test_usage": "Benchmark v1 TEST was not run at any point in "
                          "this experiment and was not used to select any "
                          "checkpoint",
            "primary_field": {"name": "greedy", "games": 1000,
                              "seed_range": [DEV_SEED_START, DEV_SEED_START + 999]},
            "intermediate_diagnostic_field": {
                "name": "greedy4_random3", "games": 500,
                "seed_range": [DEV_SEED_START, DEV_SEED_START + 499],
                "kind": "dev diagnostic; the 4.5 threshold does NOT apply, "
                        "reused unchanged from Experiment 2"},
            "drift_corpus": {
                "source": "ml.policy_drift.build_corpus (greedy trajectories)",
                "lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE,
                "states": 4440,
                "fingerprint_sha256": DIAGNOSTIC_CORPUS_FINGERPRINT,
                "note": "the exact same frozen corpus used in Experiments "
                        "1 and 2, verified by fingerprint before use, never "
                        "regenerated with different content"},
        },
        "software": {"python": sys.version.split()[0],
                     "torch": torch.__version__, "numpy": numpy.__version__},
        "artifacts": {
            "aggregate": f"{DIR}/aggregate/",
            "cross_seed_summary": f"{DIR}/aggregate/cross_seed_summary.json",
            "paired_results": f"{DIR}/aggregate/paired_results.json",
            "replication_analysis": f"{DIR}/aggregate/replication_analysis.json",
            "plots": f"{DIR}/aggregate/plots/",
            "report": "experiments/ppo_multiseed_replication_v1.md",
        },
        "reproduction_gates_all_passed": all_gates_pass,
        "limitations": [
            "Only 4 total PPO training seeds (0 existing + 1,2,3 new). n=4 "
            "is a small sample for population-level claims about PPO "
            "stochasticity; treat cross-seed statistics as descriptive/"
            "exploratory, not inferential.",
            "Checkpoint binaries are gitignored per repo convention; only "
            "their fingerprints are stored.",
            "DEV evaluation of learned checkpoints is deterministic (argmax "
            "actions, seeded env), but bitwise reproducibility across "
            "different torch versions or hardware is not guaranteed.",
        ],
    }
    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved -> {DIR}/manifest.json")
    print(f"All reproduction gates passed: {all_gates_pass}")
    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
