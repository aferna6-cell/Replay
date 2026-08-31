"""Write the Experiment 4 (PPO Policy Anchoring) reproducibility manifest."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsbg_coach.bg_env import MAX_TURNS, N_ACTIONS
from ml.benchmark import FIELD_SIZE
from ml.dev_benchmark import field_composition
from ml.model_fingerprint import checkpoint_fingerprint
from ml.policy_drift import CORPUS_LOBBIES, CORPUS_SEED_BASE
from ml.rl_common import MAX_DECISIONS
from ml.seeds import DEV_SEED_END, DEV_SEED_START, EVAL_SEED_END, EVAL_SEED_START
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,
                          LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)

DIR = "results/ppo_anchor_v1"
BASELINE_DIR = "results/ppo_budget_v1"
PRIMARY = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
KL_COEF = 0.1


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
    for it in PRIMARY:
        path = f"{DIR}/checkpoints/iter_{it:03d}.pt"
        checkpoints.append({
            "iteration": it,
            "cumulative_episodes": it * EPISODES_PER_ITER,
            "file": os.path.basename(path),
            **checkpoint_fingerprint(path),
        })

    warm = checkpoint_fingerprint("ml/policy_bc.pt")
    baseline_curve = json.load(open(f"{BASELINE_DIR}/learning_curve.json"))

    manifest = {
        "experiment": "Replay Experiment 4 — PPO Policy Anchoring",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": ("Does KL anchoring to the frozen BC prior reduce "
                     "unbounded policy drift while preserving or improving "
                     "DEV placement vs unconstrained PPO at the same budget?"),
        "single_variable": f"KL(pi_BC || pi_theta) penalty coefficient "
                           f"(beta={KL_COEF}; baseline beta=0 is Experiment 2)",
        "unchanged": ["architecture", "learning rate", "gamma", "lambda",
                      "clip", "entropy coefficient", "value coefficient",
                      "optimizer", "opponent league logic", "reward function",
                      "training seed", "training budget", "shaping schedule"],
        "training": {
            "command": (f"python -m ml.train_ppo --iters 320 --episodes 16 "
                        f"--seed 0 --shaping 1.0 --shaping-horizon 40 "
                        f"--from-bc ml/policy_bc.pt --anchor ml/policy_bc.pt "
                        f"--kl-coef {KL_COEF} "
                        f"--out {DIR}/final.pt "
                        f"--save-iters 0,40,80,160,320 "
                        f"--save-dir {DIR}/checkpoints "
                        f"--diag-log {DIR}/train_diag.jsonl"),
            "training_seed": 0,
            "episodes_per_iteration": EPISODES_PER_ITER,
            "iterations": 320,
            "total_episodes": 320 * EPISODES_PER_ITER,
            "warm_start": {"file": "policy_bc.pt", **warm},
            "anchor": {
                "file": "policy_bc.pt",
                "frozen": True,
                "kl_definition": "mean over states of KL(pi_anchor || pi_theta) "
                                 "on legal-action-masked softmax distributions",
                "kl_coef": KL_COEF,
                "note": "beta=0.1 pre-specified before results; not tuned"},
            "hyperparameters": {
                "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
                "gamma": GAMMA, "lam": LAM, "clip": CLIP, "entropy": ENTROPY,
                "value_coef": VALUE_COEF, "ppo_epochs": PPO_EPOCHS,
                "minibatch": 256, "grad_clip_norm": 1.0,
                "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX},
            "shaping_schedule": {
                "initial": 1.0, "horizon_iterations": 40,
                "formula": "shaping = 1.0 * max(0, 1 - it / (horizon * 0.7))",
                "reaches_zero_at_iteration": 28},
        },
        "baseline_comparison": {
            "experiment": "Replay Experiment 2 — PPO Training-Budget Study",
            "artifacts": BASELINE_DIR,
            "difference": "kl_coef=0 (unconstrained PPO); otherwise identical recipe",
            "baseline_warm_start_parameter_sha256":
                baseline_curve["curve"][0]["parameter_sha256"],
        },
        "checkpoints": checkpoints,
        "evaluation": {
            "split": "dev",
            "dev_interval": [DEV_SEED_START, DEV_SEED_END],
            "test_interval_NOT_USED": [EVAL_SEED_START, EVAL_SEED_END],
            "test_usage": "Benchmark v1 TEST was NOT run at any point.",
            "primary_field": {
                "name": "greedy", "games": greedy0["games"],
                "seed_range": greedy0["seed_range"],
                "composition": field_composition("greedy")},
            "intermediate_diagnostic_field": {
                "name": "greedy4_random3", "games": mixed0["games"],
                "seed_range": mixed0["seed_range"],
                "composition": field_composition("greedy4_random3")},
            "drift_corpus": {
                "lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE},
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
            "baseline_comparison": f"{DIR}/baseline_comparison.json",
            "policy_drift": f"{DIR}/policy_drift.json",
            "action_category_drift": f"{DIR}/action_category_drift.json",
            "rl_signal": f"{DIR}/rl_signal.json",
            "train_diag": f"{DIR}/train_diag.jsonl",
            "plots": f"{DIR}/plots/",
            "report": "experiments/ppo_policy_anchoring_v1.md"},
    }
    with open(f"{DIR}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved -> {DIR}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
