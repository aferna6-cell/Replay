"""Write the Experiment 3 reproducibility manifest.

Records exactly what produced the multi-seed replication numbers: code
commit, the per-seed training commands, the frozen BC warm start every seed
started from, the DEV/TEST seed-isolation check for every planned PPO
episode, the (unchanged) hyperparameters and shaping schedule, per-seed
checkpoint fingerprints, the evaluation configuration, the frozen drift
corpus, the environment, and the package versions actually used here.

    python scripts/ppo_multiseed_manifest.py

Checkpoint binaries stay gitignored per repo convention; the manifest
carries their fingerprints so each run remains identifiable without them.
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
from ml.replication import PRIMARY_ITERS, episodes                # noqa: E402
from ml.rl_common import MAX_DECISIONS                            # noqa: E402
from ml import seeds                                              # noqa: E402
from ml.train_ppo import (CLIP, ENTROPY, GAMMA, LAM, LEAGUE_EVERY,  # noqa: E402
                          LEAGUE_MAX, PPO_EPOCHS, VALUE_COEF)

ROOT = "results/ppo_multiseed_v1"
EXP2 = "results/ppo_budget_v1"
TRAINING_SEEDS = [1, 2, 3]
ITERATIONS = 320
EPISODES_PER_ITER = 16
WARMSTART_PARAMETER_SHA256 = (
    "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b")
WARMSTART_CHECKPOINT_SHA256 = (
    "bd3a4386329ecf1abc045e90d069816b801aa02d18db04af625e8ef452b0d871")
CORPUS_FINGERPRINT = (
    "2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e")


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def training_command(seed: int) -> str:
    d = f"{ROOT}/seed_{seed}"
    return (f"python -m ml.train_ppo --iters {ITERATIONS} "
            f"--episodes {EPISODES_PER_ITER} --seed {seed} --shaping 1.0 "
            f"--shaping-horizon 40 --eval-episodes 1 "
            f"--from-bc ml/policy_bc.pt --out {d}/final.pt "
            f"--save-iters 0,10,20,40,80,120,160,240,320 "
            f"--save-dir {d}/checkpoints --diag-log {d}/train_diag.jsonl")


def seed_isolation() -> dict:
    """Every planned PPO episode seed of every training seed, checked against
    BOTH reserved evaluation intervals with the repo's own helpers."""
    rows = []
    for s in TRAINING_SEEDS:
        lo = seeds.ppo_episode_seed(s, 1)
        hi = seeds.ppo_episode_seed(s, ITERATIONS * EPISODES_PER_ITER)
        rows.append({
            "training_seed": s, "first_episode_seed": lo,
            "last_episode_seed": hi,
            "episodes": ITERATIONS * EPISODES_PER_ITER,
            "overlaps_test": seeds.overlaps_eval_range(lo, hi),
            "overlaps_dev": seeds.overlaps_dev_range(lo, hi),
            "check_training_range_warned":
                seeds.check_training_range("ml.train_ppo", lo, hi)})
    return {
        "scheme": "ml.seeds.ppo_episode_seed(base, k) = base * 1000003 + k",
        "verified_with": ["ml.seeds.ppo_episode_seed",
                          "ml.seeds.check_training_range",
                          "ml.seeds.overlaps_eval_range",
                          "ml.seeds.overlaps_dev_range"],
        "test_interval": [seeds.EVAL_SEED_START, seeds.EVAL_SEED_END],
        "dev_interval": [seeds.DEV_SEED_START, seeds.DEV_SEED_END],
        "per_seed": rows,
        "all_clear": not any(r["overlaps_test"] or r["overlaps_dev"]
                             for r in rows),
    }


def main() -> int:
    import numpy
    import torch

    warm = checkpoint_fingerprint("ml/policy_bc.pt")
    if warm["parameter_sha256"] != WARMSTART_PARAMETER_SHA256:
        raise SystemExit(
            f"ml/policy_bc.pt parameter hash {warm['parameter_sha256']} is "
            f"not the frozen Experiment 2 warm start — refusing to write a "
            f"manifest for a different warm start")

    per_seed = {}
    for s in TRAINING_SEEDS:
        with open(f"{ROOT}/seed_{s}/checkpoint_metadata.json",
                  encoding="utf-8") as f:
            meta = json.load(f)
        with open(f"{ROOT}/seed_{s}/policy_drift.json", encoding="utf-8") as f:
            corpus = json.load(f)["corpus"]
        per_seed[str(s)] = {
            "training_command": training_command(s),
            "training_seed": s,
            "iterations": ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITER,
            "total_episodes": ITERATIONS * EPISODES_PER_ITER,
            "iteration0_parameter_sha256": meta["iter0_parameter_sha256"],
            "iteration0_matches_frozen_warm_start":
                meta["iter0_matches_frozen_warm_start"],
            "checkpoints": meta["checkpoints"],
            "drift_corpus_fingerprint_sha256": corpus["fingerprint_sha256"],
            "artifacts": {
                "learning_curve": f"{ROOT}/seed_{s}/learning_curve.json",
                "policy_drift": f"{ROOT}/seed_{s}/policy_drift.json",
                "action_category_drift":
                    f"{ROOT}/seed_{s}/action_category_drift.json",
                "rl_signal": f"{ROOT}/seed_{s}/rl_signal.json",
                "train_diag": f"{ROOT}/seed_{s}/train_diag.jsonl",
                "train_log": f"{ROOT}/seed_{s}/train.log",
                "checkpoint_metadata":
                    f"{ROOT}/seed_{s}/checkpoint_metadata.json",
                "dev": f"{ROOT}/seed_{s}/dev/"},
        }

    greedy0 = json.load(open(f"{ROOT}/seed_1/dev/iter000_vs_greedy.json"))
    mixed0 = json.load(
        open(f"{ROOT}/seed_1/dev/iter000_vs_greedy4_random3.json"))
    exp2_greedy0 = json.load(open(f"{EXP2}/dev/iter000_vs_greedy.json"))

    manifest = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget "
                      "Replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "question": ("Does the transient PPO improvement seen around "
                     "1,280-2,560 episodes on training seed 0 reproduce "
                     "across independent PPO training seeds, and does "
                     "performance then decay with extended training?"),
        "single_variable": "the PPO training seed",
        "unchanged": ["architecture", "optimizer", "learning rate",
                      "weight decay", "gamma", "lambda", "PPO clip",
                      "entropy coefficient", "value coefficient",
                      "PPO epochs", "batch size", "league behavior",
                      "league schedule", "reward function",
                      "shaping magnitude", "shaping schedule",
                      "episodes per iteration", "observation encoder",
                      "action space", "warm start"],
        "replication_unit": (
            "the TRAINING SEED. 4 training seeds x 1000 DEV games are not "
            "4000 independent training experiments; the paired games give "
            "precision within one trained model, the seeds give the sample "
            "on training variability."),
        "training_seeds": {"new_in_experiment_3": TRAINING_SEEDS,
                           "reused_from_experiment_2": [0],
                           "seed_0_source": f"{EXP2} (read-only)"},
        "warm_start": {
            "file": "ml/policy_bc.pt",
            "command": ("python -m ml.bc --lobbies 150 --epochs 6 "
                        "--dagger-rounds 2 --dagger-lobbies 80 --seed 0 "
                        "--out ml/policy_bc.pt"),
            "parameter_sha256": warm["parameter_sha256"],
            "checkpoint_sha256": warm["checkpoint_sha256"],
            "expected_parameter_sha256": WARMSTART_PARAMETER_SHA256,
            "expected_checkpoint_sha256": WARMSTART_CHECKPOINT_SHA256,
            "matches_experiment_2": (
                warm["parameter_sha256"] == WARMSTART_PARAMETER_SHA256
                and warm["checkpoint_sha256"] == WARMSTART_CHECKPOINT_SHA256),
            "reproduction_detail": ("7,319 demonstrations + 2,729 + 2,691 "
                                    "DAgger states = 12,739 total; final "
                                    "imitation accuracy 82.4% — identical to "
                                    "the historical record"),
            "why_frozen": ("every training seed starts from the SAME BC "
                           "warm start, so the experiment isolates PPO "
                           "training randomness. Re-running BC per seed "
                           "would confound BC randomness with PPO "
                           "randomness."),
        },
        "training": {
            "commands": {str(s): training_command(s) for s in TRAINING_SEEDS},
            "experiment_2_reference_command_seed_0": (
                "python -m ml.train_ppo --iters 320 --episodes 16 --seed 0 "
                "--shaping 1.0 --shaping-horizon 40 --from-bc "
                "ml/policy_bc.pt --out results/ppo_budget_v1/final.pt "
                "--save-iters 0,10,20,40,80,120,160,240,320 --save-dir "
                "results/ppo_budget_v1/checkpoints --diag-log "
                "results/ppo_budget_v1/train_diag.jsonl"),
            "hyperparameters": {
                "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
                "gamma": GAMMA, "lam": LAM, "clip": CLIP, "entropy": ENTROPY,
                "value_coef": VALUE_COEF, "ppo_epochs": PPO_EPOCHS,
                "minibatch": 256, "grad_clip_norm": 1.0,
                "league_every": LEAGUE_EVERY, "league_max": LEAGUE_MAX,
                "note": ("copied unchanged from the Experiment 2 manifest; "
                         "all shipped defaults")},
            "shaping_schedule": {
                "initial": 1.0, "horizon_iterations": 40,
                "formula": "shaping = 1.0 * max(0, 1 - it / (horizon * 0.7))",
                "iterations_1_to_40": "exactly the original baseline values",
                "reaches_zero_at_iteration": 28,
                "extension_behavior": ("shaping stays 0.0 for every iteration "
                                       "after 40, identical to Experiment 2")},
            "primary_checkpoints": [
                {"iteration": it, "cumulative_episodes": episodes(it)}
                for it in PRIMARY_ITERS],
            "all_saved_checkpoints": [0, 10, 20, 40, 80, 120, 160, 240, 320],
            "instrumentation_note": (
                "checkpoint snapshotting and diagnostic logging are "
                "RNG-neutral; tests/test_budget_study.py pins the shaping "
                "series against the real trainer and the existing suite "
                "covers the no-perturbation property."),
        },
        "seed_isolation": seed_isolation(),
        "reproduction_gate": {
            "requirement": ("every seed's iteration-0 checkpoint must have "
                            "the frozen warm start's parameter_sha256"),
            "expected": WARMSTART_PARAMETER_SHA256,
            "per_seed": {str(s): per_seed[str(s)]["iteration0_parameter_sha256"]
                         for s in TRAINING_SEEDS},
            "passed": all(per_seed[str(s)]["iteration0_matches_frozen_warm_start"]
                          for s in TRAINING_SEEDS),
            "cross_experiment_determinism_check": {
                "claim": ("the iteration-0 checkpoint is bit-identical to "
                          "Experiment 2's, so its 1000-game DEV evaluation "
                          "should reproduce Experiment 2's placements "
                          "exactly despite the different torch/numpy build"),
                "experiment_2_iter0_avg_placement":
                    exp2_greedy0["metrics"]["avg_placement"],
                "experiment_3_iter0_avg_placement":
                    greedy0["metrics"]["avg_placement"],
                "per_game_placements_identical":
                    greedy0["placements"] == exp2_greedy0["placements"]},
        },
        "per_seed": per_seed,
        "evaluation": {
            "split": "dev",
            "dev_interval": [seeds.DEV_SEED_START, seeds.DEV_SEED_END],
            "test_interval_NOT_USED": [seeds.EVAL_SEED_START,
                                       seeds.EVAL_SEED_END],
            "test_usage": ("Benchmark v1 TEST was not run, read or referenced "
                           "at any point in this experiment, and was not used "
                           "to select any checkpoint."),
            "paired_design": ("every checkpoint of every training seed is "
                              "evaluated on the identical DEV seeds, so "
                              "placements pair game-by-game both within and "
                              "across training seeds"),
            "primary_field": {
                "name": "greedy", "games": greedy0["games"],
                "seed_range": greedy0["seed_range"],
                "composition": field_composition("greedy"),
                "identical_to_experiment_2":
                    greedy0["seed_range"] == exp2_greedy0["seed_range"]},
            "intermediate_diagnostic_field": {
                "name": "greedy4_random3", "games": mixed0["games"],
                "seed_range": mixed0["seed_range"],
                "composition": field_composition("greedy4_random3"),
                "kind": "dev diagnostic; the 4.5 threshold does NOT apply",
                "note": ("reused unchanged from Experiment 2 — no search for "
                         "a new intermediate field was performed")},
            "paired_bootstrap": ("ml.analyze_benchmark.paired_diff — "
                                 "deterministic paired percentile bootstrap, "
                                 "10000 resamples, bootstrap seed 0"),
            "drift_corpus": {
                "source": "ml.policy_drift.build_corpus (greedy trajectories)",
                "lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE,
                "states": 4440,
                "fingerprint_sha256": CORPUS_FINGERPRINT,
                "verified_per_seed": {
                    str(s): per_seed[str(s)]["drift_corpus_fingerprint_sha256"]
                    for s in TRAINING_SEEDS},
                "matches_experiments_1_and_2": all(
                    per_seed[str(s)]["drift_corpus_fingerprint_sha256"]
                    == CORPUS_FINGERPRINT for s in TRAINING_SEEDS),
                "note": "the same frozen corpus as Experiments 1 and 2"},
            "action_categories": ("ml.action_categories — the Experiment 2 "
                                  "mapping, unchanged"),
        },
        "environment": {"env": "hsbg_coach.bg_env.BGEnv", "n_players": 8,
                        "field_size": FIELD_SIZE, "agent_seat": 0,
                        "n_actions": N_ACTIONS, "max_turns": MAX_TURNS,
                        "max_decisions": MAX_DECISIONS},
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "experiment_2_recorded": {"python": "3.11.15",
                                      "torch": "2.13.0+cu130",
                                      "numpy": "2.4.6"},
            "caveat": ("Experiment 3 ran on a different build than "
                       "Experiment 2 recorded (CPU-only torch, newer numpy, "
                       "newer Python). Seed 0's numbers are reused from "
                       "Experiment 2's artifacts and were NOT recomputed "
                       "here, so any build-dependent difference would land "
                       "between seed 0 and seeds 1-3 rather than inside "
                       "them. The one available direct check is reassuring: "
                       "the iteration-0 checkpoint — bit-identical across "
                       "all four seeds — reproduces Experiment 2's 1000 "
                       "per-game DEV placements exactly on this build.")},
        "artifacts": {
            "cross_seed_summary": f"{ROOT}/aggregate/cross_seed_summary.json",
            "paired_results": f"{ROOT}/aggregate/paired_results.json",
            "replication_analysis":
                f"{ROOT}/aggregate/replication_analysis.json",
            "plots": f"{ROOT}/aggregate/plots/",
            "report": "experiments/ppo_multiseed_replication_v1.md",
            "experiment_2_read_only": [f"{EXP2}/manifest.json",
                                       f"{EXP2}/learning_curve.json",
                                       f"{EXP2}/dev/",
                                       "experiments/ppo_budget_study_v1.md"]},
        "limitations": [
            ("FOUR PPO TRAINING SEEDS TOTAL (0, 1, 2, 3). That is a small "
             "sample for a claim about training variability. Cross-seed "
             "means and intervals in this experiment are descriptive; they "
             "are not population estimates for PPO, and no additional seed "
             "was added after seeing the results."),
            ("Package versions differ from Experiment 2's recorded "
             "environment (see software.caveat); seed 0 is reused, not "
             "recomputed."),
            ("Checkpoint binaries are gitignored per repo convention; only "
             "their fingerprints are stored."),
            ("The greedy4_random3 field is a relative comparison instrument "
             "only — the 4.5 lobby average is not a threshold for it."),
            ("Drift, agreement and KL are measured on one frozen 4,440-state "
             "corpus of greedy trajectories; they describe behavior on "
             "expert-visited states, not on the states a drifted policy "
             "actually reaches."),
            ("No hyperparameter was tuned, no checkpoint was selected for "
             "deployment, and Benchmark v1 TEST was not touched."),
        ],
    }
    os.makedirs(ROOT, exist_ok=True)
    with open(f"{ROOT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    gate = manifest["reproduction_gate"]["passed"]
    iso = manifest["seed_isolation"]["all_clear"]
    corpus_ok = manifest["evaluation"]["drift_corpus"][
        "matches_experiments_1_and_2"]
    print(f"Saved -> {ROOT}/manifest.json")
    print(f"Warm start matches Experiment 2: "
          f"{manifest['warm_start']['matches_experiment_2']}")
    print(f"Iteration-0 gate (all seeds == frozen warm start): {gate}")
    print(f"Seed isolation (no TEST/DEV contact): {iso}")
    print(f"Drift corpus == frozen 2ec217b353bd: {corpus_ok}")
    print(f"iter0 DEV placements reproduce Experiment 2 exactly: "
          f"{manifest['reproduction_gate']['cross_experiment_determinism_check']['per_game_placements_identical']}")
    return 0 if (gate and iso and corpus_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
