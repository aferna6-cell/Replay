"""Write the Experiment 3 (multi-seed PPO budget replication) manifest.

Records exactly what produced the multi-seed numbers: code commit, the
per-seed training commands (the Experiment 2 recipe verbatim except for the
seed and output paths), the frozen warm-start fingerprint and its per-seed
iteration-0 reproduction gate, every checkpoint fingerprint, the frozen
drift-corpus fingerprint, the DEV seed ranges and field compositions, the
environment configuration, and package versions.

    python scripts/ppo_multiseed_manifest.py

Checkpoint binaries stay gitignored per repo convention; the manifest
carries their fingerprints so the runs remain identifiable without them.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsbg_coach.bg_env import MAX_TURNS, N_ACTIONS                # noqa: E402
from ml.benchmark import FIELD_SIZE                               # noqa: E402
from ml.dev_benchmark import field_composition                    # noqa: E402
from ml.model_fingerprint import checkpoint_fingerprint           # noqa: E402
from ml.policy_drift import CORPUS_LOBBIES, CORPUS_SEED_BASE      # noqa: E402
from ml.rl_common import MAX_DECISIONS                            # noqa: E402
from ml.seeds import (DEV_SEED_END, DEV_SEED_START, EVAL_SEED_END,  # noqa: E402
                      EVAL_SEED_START, ppo_episode_seed)
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,  # noqa: E402
                          LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)

DIR = "results/ppo_multiseed_v1"
SEED0_DIR = "results/ppo_budget_v1"
NEW_SEEDS = [1, 2, 3]
PRIMARY = [0, 40, 80, 160, 320]
ALL_CKPT = [0, 10, 20, 40, 80, 120, 160, 240, 320]
EPISODES_PER_ITER = 16
ITERATIONS = 320

# Frozen values carried over from the committed Experiment 2 manifest.
WARM_PARAMETER_SHA256 = \
    "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b"
WARM_CHECKPOINT_SHA256 = \
    "bd3a4386329ecf1abc045e90d069816b801aa02d18db04af625e8ef452b0d871"
CORPUS_FINGERPRINT = \
    "2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e"


def train_command(seed: int) -> str:
    return (f"python -m ml.train_ppo --iters {ITERATIONS} --episodes "
            f"{EPISODES_PER_ITER} --seed {seed} --shaping 1.0 "
            f"--shaping-horizon 40 --from-bc ml/policy_bc.pt "
            f"--out {DIR}/seed_{seed}/final.pt "
            f"--save-iters 0,10,20,40,80,120,160,240,320 "
            f"--save-dir {DIR}/seed_{seed}/checkpoints "
            f"--diag-log {DIR}/seed_{seed}/train_diag.jsonl "
            f"--eval-episodes 1")


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
    assert warm["parameter_sha256"] == WARM_PARAMETER_SHA256, \
        "warm start does not match the Experiment 2 manifest"

    greedy0 = json.load(open(f"{DIR}/seed_1/dev/iter000_vs_greedy.json"))
    mixed0 = json.load(
        open(f"{DIR}/seed_1/dev/iter000_vs_greedy4_random3.json"))

    seed_blocks = []
    gates = []
    for seed in NEW_SEEDS:
        checkpoints = []
        for it in ALL_CKPT:
            path = f"{DIR}/seed_{seed}/checkpoints/iter_{it:03d}.pt"
            checkpoints.append({
                "iteration": it,
                "cumulative_episodes": it * EPISODES_PER_ITER,
                "primary": it in PRIMARY,
                "file": os.path.basename(path),
                **checkpoint_fingerprint(path),
            })
        iter0 = next(c for c in checkpoints if c["iteration"] == 0)
        gate = iter0["parameter_sha256"] == WARM_PARAMETER_SHA256
        gates.append(gate)
        lo = ppo_episode_seed(seed, 1)
        hi = ppo_episode_seed(seed, ITERATIONS * EPISODES_PER_ITER)
        seed_blocks.append({
            "training_seed": seed,
            "command": train_command(seed),
            "seed_scheme": f"ml.seeds.ppo_episode_seed({seed}, k) = "
                           f"{seed}*1000003 + k",
            "episode_seed_span": [lo, hi],
            "episode_seeds_outside_dev":
                hi < DEV_SEED_START or lo > DEV_SEED_END,
            "episode_seeds_outside_test":
                hi < EVAL_SEED_START or lo > EVAL_SEED_END,
            "iteration0_reproduction_gate": {
                "requirement": "iteration-0 parameters equal the frozen "
                               "warm start",
                "iter0_parameter_sha256": iter0["parameter_sha256"],
                "warm_start_parameter_sha256": WARM_PARAMETER_SHA256,
                "passed": gate,
            },
            "checkpoints": checkpoints,
        })

    manifest = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget "
                      "Replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": ("Does the Experiment 2 pattern — transient DEV "
                     "improvement around 1,280-2,560 episodes followed by "
                     "decay — reproduce across independent PPO training "
                     "seeds?"),
        "single_variable": "the PPO training seed (1, 2, 3 vs the "
                           "historical seed 0)",
        "unchanged": ["architecture", "optimizer", "learning rate",
                      "weight decay", "gamma", "lambda", "PPO clip",
                      "entropy coefficient", "value coefficient",
                      "PPO epochs", "batch size", "league behavior",
                      "league schedule", "reward function",
                      "shaping magnitude", "shaping schedule",
                      "episodes per iteration", "observation encoder",
                      "action space", "warm-start checkpoint",
                      "DEV evaluation seeds", "drift corpus"],
        "seed0": {
            "source": "committed Experiment 2 artifacts "
                      "(results/ppo_budget_v1) — read as-is, never rerun",
            "manifest": f"{SEED0_DIR}/manifest.json",
            "command": ("python -m ml.train_ppo --iters 320 --episodes 16 "
                        "--seed 0 --shaping 1.0 --shaping-horizon 40 "
                        "--from-bc ml/policy_bc.pt "
                        "--out results/ppo_budget_v1/final.pt "
                        "--save-iters 0,10,20,40,80,120,160,240,320 "
                        "--save-dir results/ppo_budget_v1/checkpoints "
                        "--diag-log results/ppo_budget_v1/train_diag.jsonl"),
        },
        "warm_start": {
            "file": "ml/policy_bc.pt",
            "identical_for_every_seed": True,
            "reproduction_command": ("python -m ml.bc --lobbies 150 "
                                     "--epochs 6 --dagger-rounds 2 "
                                     "--dagger-lobbies 80 --seed 0 "
                                     "--out ml/policy_bc.pt"),
            "expected_parameter_sha256": WARM_PARAMETER_SHA256,
            "expected_checkpoint_sha256": WARM_CHECKPOINT_SHA256,
            **warm,
            "verified_before_training": True,
        },
        "training": {
            "iterations": ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITER,
            "total_episodes": ITERATIONS * EPISODES_PER_ITER,
            "hyperparameters": {
                "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
                "gamma": GAMMA, "lam": LAM, "clip": CLIP, "entropy": ENTROPY,
                "value_coef": VALUE_COEF, "ppo_epochs": PPO_EPOCHS,
                "minibatch": 256, "grad_clip_norm": 1.0,
                "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX,
                "note": "the exact Experiment 2 recipe; nothing tuned"},
            "shaping_schedule": {
                "initial": 1.0, "horizon_iterations": 40,
                "formula": "shaping = 1.0 * max(0, 1 - it / (horizon * 0.7))",
                "extension_behavior": "shaping stays 0.0 after iteration 40; "
                                      "it never depends on the 320-iteration "
                                      "horizon"},
            "parallelism_note": ("seeds 1-3 trained concurrently in separate "
                                 "processes with OMP_NUM_THREADS=1 and "
                                 "MKL_NUM_THREADS=1; each run is "
                                 "self-contained and seeded, so concurrency "
                                 "cannot alter any run's RNG stream"),
            "per_seed": seed_blocks,
        },
        "evaluation": {
            "split": "dev",
            "dev_interval": [DEV_SEED_START, DEV_SEED_END],
            "test_interval_NOT_USED": [EVAL_SEED_START, EVAL_SEED_END],
            "test_usage": ("Benchmark v1 TEST was not run at any point in "
                           "this experiment and was not used to select any "
                           "checkpoint"),
            "primary_field": {
                "name": "greedy", "games": greedy0["games"],
                "seed_range": greedy0["seed_range"],
                "composition": field_composition("greedy"),
                "identical_to_experiment_2": True},
            "intermediate_diagnostic_field": {
                "name": "greedy4_random3", "games": mixed0["games"],
                "seed_range": mixed0["seed_range"],
                "composition": field_composition("greedy4_random3"),
                "kind": "dev diagnostic; the 4.5 threshold does NOT apply",
                "identical_to_experiment_2": True},
            "drift_corpus": {
                "source": "ml.policy_drift.build_corpus (greedy trajectories)",
                "lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE,
                "expected_fingerprint_sha256": CORPUS_FINGERPRINT,
                "note": "the same frozen corpus as Experiments 1-2; its "
                        "fingerprint was verified against the historical "
                        "value before any drift analysis"},
        },
        "environment": {"env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
                        "field_size": FIELD_SIZE, "agent_seat": 0,
                        "n_actions": N_ACTIONS, "max_turns": MAX_TURNS,
                        "max_decisions": MAX_DECISIONS},
        "software": {"python": sys.version.split()[0],
                     "torch": torch.__version__, "numpy": numpy.__version__},
        "artifacts": {
            "per_seed": {str(s): {
                "learning_curve": f"{DIR}/seed_{s}/learning_curve.json",
                "policy_drift": f"{DIR}/seed_{s}/policy_drift.json",
                "action_category_drift":
                    f"{DIR}/seed_{s}/action_category_drift.json",
                "rl_signal": f"{DIR}/seed_{s}/rl_signal.json",
                "train_diag": f"{DIR}/seed_{s}/train_diag.jsonl",
                "checkpoint_metadata": f"{DIR}/seed_{s}/checkpoints.json",
                "dev": f"{DIR}/seed_{s}/dev/",
            } for s in NEW_SEEDS},
            "aggregate": {
                "cross_seed_summary": f"{DIR}/aggregate/cross_seed_summary.json",
                "paired_results": f"{DIR}/aggregate/paired_results.json",
                "replication_analysis":
                    f"{DIR}/aggregate/replication_analysis.json",
                "plots": f"{DIR}/aggregate/plots/"},
            "report": "experiments/ppo_multiseed_replication_v1.md"},
        "limitations": [
            "n=4 training seeds (1 historical + 3 new): the cross-seed level "
            "is descriptive; no population-level effect-size claims.",
            "Checkpoint binaries are gitignored per repo convention; only "
            "their fingerprints are stored. Exact re-verification requires "
            "the binaries or retraining with the recorded commands.",
            "DEV evaluation of learned checkpoints is deterministic (argmax "
            "actions, seeded env), but bitwise reproducibility across "
            "different torch versions or hardware is not guaranteed.",
        ],
    }
    with open(f"{DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    # per-seed checkpoint metadata files
    for block in seed_blocks:
        s = block["training_seed"]
        with open(f"{DIR}/seed_{s}/checkpoints.json", "w") as f:
            json.dump({"training_seed": s,
                       "command": block["command"],
                       "warm_start_parameter_sha256": WARM_PARAMETER_SHA256,
                       "iteration0_reproduction_gate":
                           block["iteration0_reproduction_gate"],
                       "checkpoints": block["checkpoints"]}, f, indent=2)
    ok = all(gates)
    print(f"Saved -> {DIR}/manifest.json (+ per-seed checkpoints.json)")
    print(f"Iteration-0 reproduction gates (all seeds): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
