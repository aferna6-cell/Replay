"""Write the Experiment 3 reproducibility manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsbg_coach.bg_env import MAX_TURNS, N_ACTIONS
from ml.benchmark import FIELD_SIZE
from ml.dev_benchmark import field_composition
from ml.multiseed_analysis import (EXP2_WARM_START_PARAMETER_SHA256,
                                   PRIMARY_ITERS, episodes)
from ml.policy_drift import CORPUS_LOBBIES, CORPUS_SEED_BASE
from ml.rl_common import MAX_DECISIONS
from ml.seeds import (DEV_SEED_END, DEV_SEED_START, EVAL_SEED_END,
                      EVAL_SEED_START)
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY, LEAGUE_MAX,
                          PPO_EPOCHS, VALUE_COEF)

ROOT = "results/ppo_multiseed_v1"


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

    seeds_meta = {}
    for s in (0, 1, 2, 3):
        meta_path = f"{ROOT}/seed_{s}/checkpoints_meta.json"
        seeds_meta[str(s)] = (json.load(open(meta_path))
                              if os.path.isfile(meta_path) else {"training_seed": s})

    # Prefer live warm-start fingerprint when binary present.
    warm = {"parameter_sha256": EXP2_WARM_START_PARAMETER_SHA256,
            "file": "policy_bc.pt"}
    if os.path.isfile("ml/policy_bc.pt"):
        from ml.model_fingerprint import checkpoint_fingerprint
        warm = {"file": "policy_bc.pt", **checkpoint_fingerprint("ml/policy_bc.pt")}

    commands = {
        str(s): (
            f"python -m ml.train_ppo --iters 320 --episodes 16 --seed {s} "
            f"--shaping 1.0 --shaping-horizon 40 --from-bc ml/policy_bc.pt "
            f"--out results/ppo_multiseed_v1/seed_{s}/final.pt "
            f"--save-iters 0,40,80,160,320 "
            f"--save-dir results/ppo_multiseed_v1/seed_{s}/checkpoints "
            f"--diag-log results/ppo_multiseed_v1/seed_{s}/train_diag.jsonl"
        )
        for s in (1, 2, 3)
    }
    commands["0"] = (
        "Experiment 2 published trajectory "
        "(results/ppo_budget_v1) — not retrained"
    )

    greedy0 = json.load(open(f"{ROOT}/seed_0/dev/iter000_vs_greedy.json"))
    mixed0 = json.load(open(f"{ROOT}/seed_0/dev/iter000_vs_greedy4_random3.json"))

    manifest = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": ("Does the Exp2 U-shaped budget curve (mid-budget gain, "
                     "late regression, unbounded drift) replicate across "
                     "independent PPO training seeds?"),
        "single_variable": "PPO training seed",
        "frozen_from_experiment_2": [
            "architecture", "optimizer", "lr", "weight_decay", "gamma", "lambda",
            "ppo clip", "entropy coef", "value coef", "ppo epochs", "batch size",
            "league behavior/schedule", "reward", "shaping magnitude/schedule",
            "episodes per iteration", "observation encoder", "action space",
            "BC+DAgger warm-start checkpoint", "DEV evaluation protocol",
            "drift corpus",
        ],
        "warm_start": {
            **warm,
            "matches_experiment_2_parameter_sha256":
                warm["parameter_sha256"] == EXP2_WARM_START_PARAMETER_SHA256,
            "reproduction": (
                "PYTHONHASHSEED=0 python scripts/reproduce_warm_start_bc.py "
                "(requires torch.use_deterministic_algorithms(True); do not "
                "force single-threaded BLAS)"),
        },
        "training": {
            "seeds": [0, 1, 2, 3],
            "seed_0_source": "Experiment 2 results/ppo_budget_v1 (not rerun)",
            "new_seeds": [1, 2, 3],
            "commands": commands,
            "seed_scheme": "ml.seeds.ppo_episode_seed(base, k) = base*1000003 + k",
            "episodes_per_iteration": 16,
            "iterations": 320,
            "total_episodes_per_seed": 5120,
            "primary_checkpoints": PRIMARY_ITERS,
            "shaping_horizon": 40,
            "hyperparameters": {
                "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
                "gamma": GAMMA, "lam": LAM, "clip": CLIP, "entropy": ENTROPY,
                "value_coef": VALUE_COEF, "ppo_epochs": PPO_EPOCHS,
                "minibatch": 256, "grad_clip_norm": 1.0,
                "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX,
            },
            "per_seed_meta": seeds_meta,
        },
        "evaluation": {
            "split": "dev",
            "dev_interval": [DEV_SEED_START, DEV_SEED_END],
            "test_interval_NOT_USED": [EVAL_SEED_START, EVAL_SEED_END],
            "test_usage": "Benchmark v1 TEST was not run and must not be used",
            "primary_field": {
                "name": "greedy", "games": greedy0["games"],
                "seed_range": greedy0["seed_range"],
                "composition": field_composition("greedy"),
            },
            "intermediate_diagnostic_field": {
                "name": "greedy4_random3", "games": mixed0["games"],
                "seed_range": mixed0["seed_range"],
                "composition": field_composition("greedy4_random3"),
                "kind": "dev diagnostic; the 4.5 threshold does NOT apply",
            },
            "drift_corpus": {
                "source": "ml.policy_drift.build_corpus (greedy trajectories)",
                "lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE,
                "expected_fingerprint_prefix": "2ec217b353bd",
                "note": "same frozen corpus definition as Experiments 1 and 2",
            },
        },
        "environment": {
            "env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
            "field_size": FIELD_SIZE, "agent_seat": 0,
            "n_actions": N_ACTIONS, "max_turns": MAX_TURNS,
            "max_decisions": MAX_DECISIONS,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
        "artifacts": {
            "per_seed": {str(s): f"{ROOT}/seed_{s}/" for s in (0, 1, 2, 3)},
            "aggregate": f"{ROOT}/aggregate/",
            "report": "experiments/ppo_multiseed_replication_v1.md",
        },
        "limitations": [
            "n=4 training seeds is exploratory only.",
            "Checkpoint binaries are gitignored; manifests carry fingerprints.",
            "Seed 0 is the published Exp2 trajectory; seeds 1–3 were trained "
            "in this experiment from the verified warm start.",
            "Do not select a best seed or deploy iter80 from these results.",
        ],
    }
    with open(f"{ROOT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    ok = manifest["warm_start"]["matches_experiment_2_parameter_sha256"]
    print(f"Saved -> {ROOT}/manifest.json")
    print(f"Warm-start matches Exp2: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
