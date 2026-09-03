"""Write the Experiment 3 (Multi-Seed PPO Budget Replication) reproducibility manifest.

Records:
- Experiment question, single experimental variable (PPO training seed), unchanged hyperparameters
- Warm-start checkpoint parameter SHA-256 (identical across all seeds)
- Training seeds, commands, seed schemes, and evaluation isolation validation
- Primary checkpoints and their fingerprints for seeds 0, 1, 2, 3
- Evaluation field definitions (1000 games vs 7x greedy, 500 games vs greedy4_random3)
- Software and hardware environment info
"""

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
from ml.seeds import (DEV_SEED_END, DEV_SEED_START, EVAL_SEED_END,
                      EVAL_SEED_START, ppo_episode_seed, overlaps_dev_range,
                      overlaps_eval_range)
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,
                          LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)

BASE_DIR = "results/ppo_multiseed_v1"
AGG_DIR = os.path.join(BASE_DIR, "aggregate")
PLOTS_DIR = os.path.join(AGG_DIR, "plots")
EXP2_DIR = "results/ppo_budget_v1"
SEEDS = [0, 1, 2, 3]
PRIMARY_ITERS = [0, 40, 80, 160, 320]
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

    warm_start_fp = checkpoint_fingerprint("ml/policy_bc.pt")

    per_seed_info = {}
    for s in SEEDS:
        first_seed = ppo_episode_seed(s, 1)
        last_seed = ppo_episode_seed(s, 320 * EPISODES_PER_ITER)

        ckpts = []
        for it in PRIMARY_ITERS:
            if s == 0:
                p = f"{EXP2_DIR}/checkpoints/iter_{it:03d}.pt"
                # If historical pt file is absent, use manifest from exp2
                if not os.path.exists(p):
                    exp2_m = json.load(open(f"{EXP2_DIR}/manifest.json"))
                    c_info = next(c for c in exp2_m["checkpoints"] if c["iteration"] == it)
                    ckpts.append(c_info)
                    continue
            else:
                p = f"{BASE_DIR}/seed_{s}/checkpoints/iter_{it:03d}.pt"

            fp = checkpoint_fingerprint(p) if os.path.exists(p) else {"checkpoint_sha256": None, "parameter_sha256": None}
            ckpts.append({
                "iteration": it,
                "cumulative_episodes": it * EPISODES_PER_ITER,
                "primary": True,
                "file": f"iter_{it:03d}.pt",
                **fp,
            })

        per_seed_info[f"seed_{s}"] = {
            "training_seed": s,
            "episodes_per_iteration": EPISODES_PER_ITER,
            "iterations": 320,
            "total_episodes": 320 * EPISODES_PER_ITER,
            "seed_range": [first_seed, last_seed],
            "overlaps_dev": overlaps_dev_range(first_seed, last_seed),
            "overlaps_test": overlaps_eval_range(first_seed, last_seed),
            "command": (
                f"python -m ml.train_ppo --iters 320 --episodes 16 --seed {s} "
                f"--shaping 1.0 --shaping-horizon 40 --from-bc ml/policy_bc.pt "
                f"--out {BASE_DIR}/seed_{s}/final.pt "
                f"--save-iters 0,40,80,160,320 "
                f"--save-dir {BASE_DIR}/seed_{s}/checkpoints "
                f"--diag-log {BASE_DIR}/seed_{s}/train_diag.jsonl"
                if s > 0 else "historical Experiment 2 (seed 0)"
            ),
            "checkpoints": ckpts,
        }

    manifest = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": (
            "Does the transient PPO improvement around 1,280–2,560 episodes reproduce across "
            "independent PPO training seeds, and does performance then decay with extended training?"
        ),
        "single_variable": "PPO training seed (seeds 0, 1, 2, 3)",
        "unchanged_recipe": [
            "architecture", "optimizer (AdamW)", "learning rate (3e-4)", "weight decay (1e-4)",
            "gamma (0.999)", "lambda (0.95)", "clip (0.2)", "entropy coefficient (0.01)",
            "value coefficient (0.5)", "ppo_epochs (4)", "batch size / minibatch (256)",
            "league behavior (league_every 8, league_max 5)", "reward function",
            "shaping magnitude (1.0)", "shaping schedule (--shaping-horizon 40)",
            "episodes per iteration (16)", "observation encoder", "action space (28)",
        ],
        "warm_start": {
            "file": "ml/policy_bc.pt",
            "parameter_sha256": warm_start_fp["parameter_sha256"],
            "checkpoint_sha256": warm_start_fp["checkpoint_sha256"],
            "identical_across_all_seeds": True,
            "note": "Exact same BC+DAgger checkpoint used to initialize every PPO seed trajectory",
        },
        "seeds": SEEDS,
        "primary_iterations": PRIMARY_ITERS,
        "per_seed_training": per_seed_info,
        "evaluation": {
            "split": "dev",
            "dev_interval": [DEV_SEED_START, DEV_SEED_END],
            "test_interval_LOCKED": [EVAL_SEED_START, EVAL_SEED_END],
            "test_usage": "Benchmark v1 TEST was NOT run at any point during Experiment 3.",
            "primary_field": {
                "name": "greedy",
                "games": 1000,
                "seed_range": [10550000, 10550999],
                "composition": field_composition("greedy"),
            },
            "intermediate_diagnostic_field": {
                "name": "greedy4_random3",
                "games": 500,
                "seed_range": [10550000, 10550499],
                "composition": field_composition("greedy4_random3"),
            },
            "drift_corpus": {
                "source": "ml.policy_drift.build_corpus (greedy trajectories)",
                "lobbies": CORPUS_LOBBIES,
                "seed_base": CORPUS_SEED_BASE,
                "states": 4440,
            },
        },
        "environment": {
            "env": "hsbg_coach.bg_env.BGEnv",
            "n_players": 8,
            "field_size": FIELD_SIZE,
            "agent_seat": 0,
            "n_actions": N_ACTIONS,
            "max_turns": MAX_TURNS,
            "max_decisions": MAX_DECISIONS,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
        "artifacts": {
            "manifest": f"{BASE_DIR}/manifest.json",
            "cross_seed_summary": f"{AGG_DIR}/cross_seed_summary.json",
            "paired_results": f"{AGG_DIR}/paired_results.json",
            "replication_analysis": f"{AGG_DIR}/replication_analysis.json",
            "plots": f"{PLOTS_DIR}/",
            "report": "experiments/ppo_multiseed_replication_v1.md",
        },
        "limitations": [
            "FOUR TOTAL PPO TRAINING SEEDS (N=4). While this establishes that Seed 0's improvement does not robustly generalize, 4 seeds remains a small sample for population-level distributional estimation.",
            "DEV split only. Benchmark v1 TEST set remains locked until an intervention is chosen and validated.",
            "Checkpoint binaries are gitignored per repo convention; parameter fingerprints are committed in full.",
        ],
    }

    out_path = os.path.join(BASE_DIR, "manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
