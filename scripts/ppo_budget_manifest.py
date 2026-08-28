"""Write the Experiment 2 reproducibility manifest.

Records exactly what produced the budget-study numbers: code commit,
training command and seed, the warm-start and per-checkpoint fingerprints
(both parameter-level and raw-artifact), the shaping schedule and its
extension behavior, the DEV seed ranges and field compositions, the
environment configuration, and package versions.

    python scripts/ppo_budget_manifest.py

Checkpoint binaries stay gitignored per repo convention; the manifest
carries their fingerprints so the run remains identifiable without them.
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
                      EVAL_SEED_START)
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,  # noqa: E402
                          LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)

DIR = "results/ppo_budget_v1"
PRIMARY = [0, 40, 80, 160, 320]
ALL_CKPT = [0, 10, 20, 40, 80, 120, 160, 240, 320]
EPISODES_PER_ITER = 16


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

    greedy0 = json.load(open(f"{DIR}/dev/iter000_vs_greedy.json"))
    mixed0 = json.load(open(f"{DIR}/dev/iter000_vs_greedy4_random3.json"))

    checkpoints = []
    for it in ALL_CKPT:
        path = f"{DIR}/checkpoints/iter_{it:03d}.pt"
        checkpoints.append({
            "iteration": it,
            "cumulative_episodes": it * EPISODES_PER_ITER,
            "primary": it in PRIMARY,
            "file": os.path.basename(path),
            **checkpoint_fingerprint(path),
        })

    warm = checkpoint_fingerprint("ml/policy_bc.pt")
    baseline_ppo = checkpoint_fingerprint("ml/policy_ppo.pt")
    ext40 = next(c for c in checkpoints if c["iteration"] == 40)

    manifest = {
        "experiment": "Replay Experiment 2 — PPO Training-Budget Study",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": ("Does more PPO experience produce measurable improvement, "
                     "or only more policy drift?"),
        "single_variable": "PPO training budget (iterations x 16 episodes)",
        "unchanged": ["architecture", "learning rate", "gamma", "lambda",
                      "clip", "entropy coefficient", "value coefficient",
                      "optimizer", "opponent league logic", "reward function"],
        "training": {
            "command": ("python -m ml.train_ppo --iters 320 --episodes 16 "
                        "--seed 0 --shaping 1.0 --shaping-horizon 40 "
                        "--from-bc ml/policy_bc.pt "
                        "--out results/ppo_budget_v1/final.pt "
                        "--save-iters 0,10,20,40,80,120,160,240,320 "
                        "--save-dir results/ppo_budget_v1/checkpoints "
                        "--diag-log results/ppo_budget_v1/train_diag.jsonl"),
            "training_seed": 0,
            "seed_scheme": "ml.seeds.ppo_episode_seed(0, k) = 0*1000003 + k",
            "episodes_per_iteration": EPISODES_PER_ITER,
            "iterations": 320,
            "total_episodes": 320 * EPISODES_PER_ITER,
            "warm_start": {"file": "policy_bc.pt", **warm},
            "hyperparameters": {
                "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
                "gamma": GAMMA, "lam": LAM, "clip": CLIP, "entropy": ENTROPY,
                "value_coef": VALUE_COEF, "ppo_epochs": PPO_EPOCHS,
                "minibatch": 256, "grad_clip_norm": 1.0,
                "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX,
                "note": "all shipped defaults, unchanged from the baseline"},
            "shaping_schedule": {
                "initial": 1.0, "horizon_iterations": 40,
                "formula": "shaping = 1.0 * max(0, 1 - it / (horizon * 0.7))",
                "iterations_1_to_40": "exactly the original baseline values",
                "reaches_zero_at_iteration": 28,
                "extension_behavior": ("shaping stays 0.0 for every iteration "
                                       "after 40, so extending the run does "
                                       "not alter the reward the first 40 "
                                       "iterations saw")},
        },
        "reproduction_gate": {
            "requirement": ("the extended trajectory's iteration-40 parameters "
                            "must equal the historical baseline PPO model"),
            "baseline_ppo_parameter_sha256": baseline_ppo["parameter_sha256"],
            "extended_iter40_parameter_sha256": ext40["parameter_sha256"],
            "passed": (baseline_ppo["parameter_sha256"]
                       == ext40["parameter_sha256"]),
            "note": ("parameter_sha256 is filename-independent; the raw "
                     "checkpoint_sha256 differs between these two files "
                     "purely because torch.save embeds an archive name "
                     "derived from the filename (found in Experiment 1)"),
        },
        "checkpoints": checkpoints,
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
                "composition": field_composition("greedy")},
            "intermediate_diagnostic_field": {
                "name": "greedy4_random3", "games": mixed0["games"],
                "seed_range": mixed0["seed_range"],
                "composition": field_composition("greedy4_random3"),
                "kind": "dev diagnostic; the 4.5 threshold does NOT apply",
                "calibration": ("warm start scored 4.55 with a spread "
                                "distribution on a separate DEV sub-range "
                                "(base 10570000, 100 games) — chosen once, "
                                "not searched over compositions")},
            "drift_corpus": {
                "source": "ml.policy_drift.build_corpus (greedy trajectories)",
                "lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE,
                "note": "the same frozen corpus definition as Experiment 1"},
        },
        "environment": {"env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
                        "field_size": FIELD_SIZE, "agent_seat": 0,
                        "n_actions": N_ACTIONS, "max_turns": MAX_TURNS,
                        "max_decisions": MAX_DECISIONS},
        "software": {"python": sys.version.split()[0],
                     "torch": torch.__version__, "numpy": numpy.__version__},
        "artifacts": {
            "learning_curve": f"{DIR}/learning_curve.json",
            "paired_analysis": f"{DIR}/paired_analysis.json",
            "policy_drift": f"{DIR}/policy_drift.json",
            "action_category_drift": f"{DIR}/action_category_drift.json",
            "rl_signal": f"{DIR}/rl_signal.json",
            "train_diag": f"{DIR}/train_diag.jsonl",
            "plots": f"{DIR}/plots/",
            "report": "experiments/ppo_budget_study_v1.md"},
        "limitations": [
            ("TRAINING SEED 0 ONLY. This is a budget-mechanism study on the "
             "exact historical training trajectory, not a claim about PPO "
             "across random initializations. Replication across independent "
             "training seeds is deliberately deferred until a budget effect "
             "worth replicating is established."),
            ("Checkpoint binaries are gitignored per repo convention; only "
             "their fingerprints are stored. Exact re-verification requires "
             "the binaries or retraining with the recorded command."),
            ("DEV evaluation of learned checkpoints is deterministic (argmax "
             "actions, seeded env), but bitwise reproducibility across "
             "different torch versions or hardware is not guaranteed."),
        ],
    }
    with open(f"{DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    gate = manifest["reproduction_gate"]["passed"]
    print(f"Saved -> {DIR}/manifest.json")
    print(f"Reproduction gate (iter40 == historical baseline): {gate}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
