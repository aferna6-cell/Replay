"""Write the Experiment 3 reproducibility manifest.

    python scripts/ppo_multiseed_manifest.py

Records the frozen PPO recipe, the shared warm-start parameter hash, the
three new training commands, seed-isolation checks, DEV seed ranges, and
pointers to Experiment 2 for seed 0 (no copy, no overwrite).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsbg_coach.bg_env import MAX_TURNS, N_ACTIONS  # noqa: E402
from ml.benchmark import FIELD_SIZE  # noqa: E402
from ml.dev_benchmark import field_composition  # noqa: E402
from ml.multiseed_analysis import (  # noqa: E402
    ALL_SEEDS, CORPUS_FINGERPRINT_SHA256, CORPUS_LOBBIES, CORPUS_SEED_BASE,
    CORPUS_STATES, DEV_EVAL_BASE, DEV_EVAL_GAMES, DEV_EVAL_LAST, MIXED_FIELD,
    MIXED_GAMES, MULTI_DIR, PRIMARY_ITERS, SEED0_DIR,
    SHAPING_HORIZON, TOTAL_EPISODES, TRAINING_ITERS,
    WARM_START_PARAMETER_SHA256, EPISODES_PER_ITER, planned_ppo_span,
    seed_dir, training_seeds_isolated, write_json,
)
from ml.rl_common import MAX_DECISIONS  # noqa: E402
from ml.seeds import DEV_SEED_END, DEV_SEED_START, EVAL_SEED_END, EVAL_SEED_START  # noqa: E402
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,  # noqa: E402
                          LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def train_command(seed: int) -> str:
    d = seed_dir(seed)
    return (
        f"python -m ml.train_ppo --iters {TRAINING_ITERS} --episodes "
        f"{EPISODES_PER_ITER} --seed {seed} --shaping 1.0 "
        f"--shaping-horizon {SHAPING_HORIZON} --from-bc ml/policy_bc.pt "
        f"--out {d}/final.pt --save-iters 0,10,20,40,80,120,160,240,320 "
        f"--save-dir {d}/checkpoints --diag-log {d}/train_diag.jsonl"
    )


def main() -> int:
    import numpy
    import torch

    isolation = training_seeds_isolated(ALL_SEEDS)
    if not all(r["isolated"] for r in isolation):
        print("ERROR: a planned PPO seed overlaps DEV/TEST", file=sys.stderr)
        return 1

    seed_records = []
    for s in ALL_SEEDS:
        lo, hi = planned_ppo_span(s)
        rec = {
            "training_seed": s,
            "source": "experiment_2_read_only" if s == 0 else "experiment_3",
            "artifact_dir": seed_dir(s),
            "episode_seed_span": [lo, hi],
            "seed_scheme": f"ml.seeds.ppo_episode_seed({s}, k) = {s}*1000003 + k",
        }
        if s == 0:
            rec["command"] = (
                "python -m ml.train_ppo --iters 320 --episodes 16 --seed 0 "
                "--shaping 1.0 --shaping-horizon 40 --from-bc ml/policy_bc.pt "
                "--out results/ppo_budget_v1/final.pt "
                "--save-iters 0,10,20,40,80,120,160,240,320 "
                "--save-dir results/ppo_budget_v1/checkpoints "
                "--diag-log results/ppo_budget_v1/train_diag.jsonl")
            rec["note"] = ("historical Experiment 2 trajectory; loaded in "
                           "place from results/ppo_budget_v1/; not rerun")
        else:
            rec["command"] = train_command(s)
            ckpt_meta = os.path.join(seed_dir(s), "checkpoints.json")
            if os.path.isfile(ckpt_meta):
                rec["checkpoints"] = __import__("json").load(open(ckpt_meta))
        seed_records.append(rec)

    manifest = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": ("Does the transient PPO improvement around 1,280–2,560 "
                     "episodes reproduce across independent PPO training "
                     "seeds, and does performance then decay with extended "
                     "training?"),
        "single_variable": "PPO training seed",
        "unchanged": [
            "architecture", "optimizer", "learning rate", "weight decay",
            "gamma", "lambda", "PPO clip", "entropy coefficient",
            "value coefficient", "PPO epochs", "batch size", "league behavior",
            "league schedule", "reward function", "shaping magnitude",
            "shaping schedule", "episodes per iteration",
            "observation encoder", "action space",
            "warm-start BC+DAgger checkpoint",
        ],
        "warm_start": {
            "file": "ml/policy_bc.pt",
            "parameter_sha256": WARM_START_PARAMETER_SHA256,
            "note": ("same exact BC + DAgger checkpoint as Experiments 1–2; "
                     "not retrained for seeds 1/2/3; every seed's iteration-0 "
                     "checkpoint must reproduce this parameter hash"),
            "historical_command": (
                "python -m ml.bc --lobbies 150 --epochs 6 --dagger-rounds 2 "
                "--dagger-lobbies 80 --seed 0 --out ml/policy_bc.pt"),
        },
        "training": {
            "episodes_per_iteration": EPISODES_PER_ITER,
            "iterations": TRAINING_ITERS,
            "total_episodes": TOTAL_EPISODES,
            "shaping_horizon": SHAPING_HORIZON,
            "shaping_formula":
                "shaping = 1.0 * max(0, 1 - it / (horizon * 0.7))",
            "hyperparameters": {
                "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
                "gamma": GAMMA, "lam": LAM, "clip": CLIP, "entropy": ENTROPY,
                "value_coef": VALUE_COEF, "ppo_epochs": PPO_EPOCHS,
                "minibatch": 256, "grad_clip_norm": 1.0,
                "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX,
                "note": "exact Experiment 2 / shipped defaults; not tuned",
            },
            "primary_checkpoints": [
                {"iteration": it, "cumulative_episodes": it * EPISODES_PER_ITER}
                for it in PRIMARY_ITERS
            ],
            "seeds": seed_records,
            "seed_isolation": isolation,
        },
        "evaluation": {
            "split": "dev",
            "dev_interval": [DEV_SEED_START, DEV_SEED_END],
            "test_interval_NOT_USED": [EVAL_SEED_START, EVAL_SEED_END],
            "test_usage": ("Benchmark v1 TEST was not run at any point and "
                           "was not used to select any checkpoint"),
            "primary_field": {
                "name": "greedy", "games": DEV_EVAL_GAMES,
                "seed_range": [DEV_EVAL_BASE, DEV_EVAL_LAST],
                "composition": field_composition("greedy"),
                "note": "identical DEV seeds for every checkpoint and seed",
            },
            "intermediate_diagnostic_field": {
                "name": MIXED_FIELD, "games": MIXED_GAMES,
                "seed_range": [DEV_EVAL_BASE, DEV_EVAL_BASE + MIXED_GAMES - 1],
                "composition": field_composition(MIXED_FIELD),
                "kind": "dev diagnostic; the 4.5 threshold does NOT apply",
            },
            "drift_corpus": {
                "lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE,
                "states": CORPUS_STATES,
                "fingerprint_sha256": CORPUS_FINGERPRINT_SHA256,
                "note": "frozen Experiment 1/2 corpus; not regenerated",
            },
        },
        "environment": {"env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
                        "field_size": FIELD_SIZE, "agent_seat": 0,
                        "n_actions": N_ACTIONS, "max_turns": MAX_TURNS,
                        "max_decisions": MAX_DECISIONS},
        "software": {"python": sys.version.split()[0],
                     "torch": torch.__version__, "numpy": numpy.__version__},
        "artifacts": {
            "seed_0_experiment_2": SEED0_DIR,
            "seed_1": f"{MULTI_DIR}/seed_1/",
            "seed_2": f"{MULTI_DIR}/seed_2/",
            "seed_3": f"{MULTI_DIR}/seed_3/",
            "aggregate": f"{MULTI_DIR}/aggregate/",
            "report": "experiments/ppo_multiseed_replication_v1.md",
        },
        "hygiene": {
            "pr_9": "merged into claude/hearthstone-battlegrounds-ml-00rjn5 "
                    "as 8ffe58bedd84f6e09882b988339cc6a0dd4abcf5 "
                    "(Merge pull request #9: Experiment 2 PPO "
                    "training-budget study)",
            "experiment_3_base": "branched from that post-merge canonical tip",
            "experiment_3_branch": "claude/ppo-budget-multiseed-v1",
        },
        "limitations": [
            "Four PPO training seeds total (one historical + three new). "
            "n=4 is small; cross-seed means are descriptive.",
            "Checkpoint binaries are gitignored; fingerprints are stored.",
            "DEV only. Benchmark v1 TEST was never run.",
        ],
    }
    # clean the dummy command assignment for seed 0
    write_json(os.path.join(MULTI_DIR, "manifest.json"), manifest)
    print(f"Saved -> {MULTI_DIR}/manifest.json")
    print(f"Warm-start parameter_sha256: {WARM_START_PARAMETER_SHA256}")
    print("Training seeds isolated from DEV/TEST:",
          all(r["isolated"] for r in isolation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
