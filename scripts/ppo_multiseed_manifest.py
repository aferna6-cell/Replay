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
        f"--shaping-horizon {SHAPING_HORIZON} --eval-episodes 1 "
        f"--from-bc ml/policy_bc.pt "
        f"--out {d}/final.pt --save-iters 0,10,20,40,80,120,160,240,320 "
        f"--save-dir {d}/checkpoints --diag-log {d}/train_diag.jsonl"
    )


def _dev_blob(seed: int, iteration: int, field: str = "greedy"):
    import json
    path = os.path.join(seed_dir(seed), "dev",
                        f"iter{iteration:03d}_vs_{field}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def non_termination_record() -> dict:
    """Which checkpoints could not finish every fixed DEV lobby.

    ``ml.benchmark`` refuses to score an episode that has not terminated
    within MAX_DECISIONS. Recording the stalls keeps a degenerate policy
    visible instead of letting it vanish behind an aborted evaluation.
    """
    rows = []
    for s in ALL_SEEDS:
        for it in PRIMARY_ITERS:
            for field in ("greedy", MIXED_FIELD):
                blob = _dev_blob(s, it, field)
                n = int(blob.get("games_non_terminating", 0))
                if n:
                    rows.append({
                        "training_seed": s, "iteration": it, "field": field,
                        "games_requested": blob.get("games_requested",
                                                    blob["games"]),
                        "games_scored": blob["games"],
                        "games_non_terminating": n,
                        "non_terminating_seeds":
                            blob.get("non_terminating_seeds", [])})
    return {
        "decision_cap": MAX_DECISIONS,
        "affected_checkpoints": rows,
        "n_affected": len(rows),
        "consequence": ("an affected checkpoint's placement average excludes "
                        "the lobbies it could not finish and is therefore "
                        "optimistic; every paired comparison involving it is "
                        "restricted to the lobbies both checkpoints "
                        "finished, and says so in paired_results.json"),
    }


def main() -> int:
    import json
    import numpy
    import torch
    from ml.model_fingerprint import checkpoint_fingerprint

    isolation = training_seeds_isolated(ALL_SEEDS)
    if not all(r["isolated"] for r in isolation):
        print("ERROR: a planned PPO seed overlaps DEV/TEST", file=sys.stderr)
        return 1

    warm = None
    if os.path.isfile("ml/policy_bc.pt"):
        warm = checkpoint_fingerprint("ml/policy_bc.pt")
        if warm["parameter_sha256"] != WARM_START_PARAMETER_SHA256:
            print(f"ERROR: ml/policy_bc.pt parameter hash "
                  f"{warm['parameter_sha256']} is not the frozen Experiment 2 "
                  f"warm start", file=sys.stderr)
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
                with open(ckpt_meta, encoding="utf-8") as f:
                    meta = json.load(f)
                rec["checkpoints"] = meta
                iter0 = next(c for c in meta["checkpoints"]
                             if c["iteration"] == 0)
                rec["iteration0_parameter_sha256"] = iter0["parameter_sha256"]
                rec["iteration0_matches_frozen_warm_start"] = (
                    iter0["parameter_sha256"] == WARM_START_PARAMETER_SHA256)
            drift_path = os.path.join(seed_dir(s), "policy_drift.json")
            if os.path.isfile(drift_path):
                with open(drift_path, encoding="utf-8") as f:
                    corpus = json.load(f)["corpus"]
                rec["drift_corpus_fingerprint_sha256"] = \
                    corpus["fingerprint_sha256"]
                rec["drift_corpus_matches_frozen"] = (
                    corpus["fingerprint_sha256"] == CORPUS_FINGERPRINT_SHA256)
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
            "verified_parameter_sha256": warm and warm["parameter_sha256"],
            "verified_checkpoint_sha256": warm and warm["checkpoint_sha256"],
            "verified_at_manifest_time": warm is not None,
            "note": ("same exact BC + DAgger checkpoint as Experiments 1–2; "
                     "not retrained for seeds 1/2/3; every seed's iteration-0 "
                     "checkpoint must reproduce this parameter hash"),
            "historical_command": (
                "python -m ml.bc --lobbies 150 --epochs 6 --dagger-rounds 2 "
                "--dagger-lobbies 80 --seed 0 --out ml/policy_bc.pt"),
            "reproduction_detail": (
                "7,319 demonstrations + 2,729 + 2,691 DAgger states = 12,739 "
                "total; final imitation accuracy 82.4% — identical to the "
                "historical record"),
            "why_shared_across_seeds": (
                "holding the BC warm start fixed is what makes the PPO "
                "training seed the only variable. Re-running BC per seed "
                "would confound BC randomness with PPO randomness, and the "
                "experiment could no longer attribute curve differences to "
                "PPO training stochasticity."),
        },
        "reproduction_gate": {
            "requirement": ("every seed's iteration-0 checkpoint must carry "
                            "the frozen warm start's parameter_sha256"),
            "expected": WARM_START_PARAMETER_SHA256,
            "per_seed": {str(r["training_seed"]):
                         r.get("iteration0_parameter_sha256")
                         for r in seed_records if r["training_seed"] != 0},
            "passed": all(r.get("iteration0_matches_frozen_warm_start")
                          for r in seed_records if r["training_seed"] != 0),
            "cross_experiment_determinism_check": {
                "claim": ("the iteration-0 checkpoint is the same weights in "
                          "every seed and in Experiment 2, so its 1000-game "
                          "DEV evaluation should reproduce Experiment 2's "
                          "per-game placements exactly despite the different "
                          "torch/numpy build"),
                "experiment_2_iter0_avg_placement":
                    _dev_blob(0, 0)["metrics"]["avg_placement"],
                "experiment_3_iter0_avg_placement":
                    _dev_blob(1, 0)["metrics"]["avg_placement"],
                "per_game_placements_identical":
                    _dev_blob(1, 0)["placements"]
                    == _dev_blob(0, 0)["placements"],
            },
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
                "verified_per_seed": {
                    str(r["training_seed"]):
                        r.get("drift_corpus_fingerprint_sha256")
                    for r in seed_records if r["training_seed"] != 0},
                "note": "frozen Experiment 1/2 corpus; not regenerated",
            },
            "action_categories": ("ml.action_categories — the Experiment 2 "
                                  "mapping, unchanged"),
            "paired_bootstrap": ("ml.analyze_benchmark.paired_diff — "
                                 "deterministic paired percentile bootstrap, "
                                 "10000 resamples, bootstrap seed 0"),
            "non_termination": non_termination_record(),
        },
        "environment": {"env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
                        "field_size": FIELD_SIZE, "agent_seat": 0,
                        "n_actions": N_ACTIONS, "max_turns": MAX_TURNS,
                        "max_decisions": MAX_DECISIONS},
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__, "numpy": numpy.__version__,
            "experiment_2_recorded": {"python": "3.11.15",
                                      "torch": "2.13.0+cu130",
                                      "numpy": "2.4.6"},
            "caveat": (
                "Experiment 3 ran on a different build than Experiment 2 "
                "recorded (CPU-only torch, newer numpy and Python). Seed 0's "
                "numbers are reused from Experiment 2's artifacts and were "
                "NOT recomputed, so any build-dependent difference would sit "
                "between seed 0 and seeds 1–3 rather than inside them. The "
                "one direct check available is reassuring: the iteration-0 "
                "checkpoint — the same weights in all four seeds — "
                "reproduces Experiment 2's 1000 per-game DEV placements "
                "exactly on this build (see reproduction_gate)."),
        },
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
            "FOUR PPO TRAINING SEEDS TOTAL (one historical + three new). "
            "n=4 is a small sample for training variability; cross-seed "
            "means and intervals here are descriptive, not population "
            "estimates for PPO. No extra seed was added after seeing the "
            "results.",
            "Package versions differ from Experiment 2's recorded "
            "environment (see software.caveat); seed 0 is reused, not "
            "recomputed.",
            "Seed 1's 5,120-episode policy could not finish every DEV lobby "
            "(see evaluation.non_termination); its average excludes those "
            "lobbies and is optimistic.",
            "Checkpoint binaries are gitignored; fingerprints are stored.",
            "The greedy4_random3 field is a relative comparison instrument "
            "only — the 4.5 lobby average is not a threshold for it.",
            "Drift, agreement and KL are measured on one frozen 4,440-state "
            "corpus of greedy trajectories, so they describe behavior on "
            "expert-visited states, not on the states a drifted policy "
            "actually reaches.",
            "DEV only. Benchmark v1 TEST was never run, and no checkpoint "
            "was tuned, selected, or deployed.",
        ],
    }
    write_json(os.path.join(MULTI_DIR, "manifest.json"), manifest)
    gate = manifest["reproduction_gate"]
    print(f"Saved -> {MULTI_DIR}/manifest.json")
    print(f"Warm-start parameter_sha256: {WARM_START_PARAMETER_SHA256}")
    print(f"  ml/policy_bc.pt verified: "
          f"{manifest['warm_start']['verified_at_manifest_time']}")
    print(f"Iteration-0 gate (all new seeds == frozen warm start): "
          f"{gate['passed']}")
    print(f"iter0 DEV placements reproduce Experiment 2 exactly: "
          f"{gate['cross_experiment_determinism_check']['per_game_placements_identical']}")
    print("Training seeds isolated from DEV/TEST:",
          all(r["isolated"] for r in isolation))
    nt = manifest["evaluation"]["non_termination"]
    print(f"Checkpoints that could not finish every DEV lobby: "
          f"{nt['n_affected']}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
