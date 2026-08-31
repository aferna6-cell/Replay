"""Run one Experiment 3 training seed end-to-end: PPO → DEV eval → drift.

    python scripts/ppo_multiseed_run_seed.py --seed 1

Does not touch Benchmark v1 TEST. Does not overwrite seed 0 / Exp2 artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_fingerprint import checkpoint_fingerprint
from ml.multiseed_analysis import (EXP2_CORPUS_FINGERPRINT_PREFIX,
                                   EXP2_WARM_START_PARAMETER_SHA256,
                                   PRIMARY_ITERS, curve_point, episodes,
                                   load_json, load_seed_greedy, load_seed_mixed,
                                   within_seed_paired)
from ml import seeds as seedmod

ROOT = "results/ppo_multiseed_v1"
WARM = "ml/policy_bc.pt"
SAVE_ITERS = "0,40,80,160,320"


def _run(cmd: list, log_path: str) -> None:
    print("+", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
        log.flush()
        print(proc.stdout, end="")
        if proc.returncode != 0:
            raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def train(seed: int, seed_dir: str) -> None:
    ckpt = f"{seed_dir}/checkpoints"
    os.makedirs(ckpt, exist_ok=True)
    diag = f"{seed_dir}/train_diag.jsonl"
    if os.path.isfile(diag):
        os.remove(diag)
    out = f"{seed_dir}/final.pt"
    log = f"{seed_dir}/train.log"
    open(log, "w").close()
    _run([
        sys.executable, "-m", "ml.train_ppo",
        "--iters", "320", "--episodes", "16",
        "--seed", str(seed),
        "--shaping", "1.0", "--shaping-horizon", "40",
        "--from-bc", WARM,
        "--out", out,
        "--save-iters", SAVE_ITERS,
        "--save-dir", ckpt,
        "--diag-log", diag,
        "--eval-episodes", "1",
    ], log)
    # Reproduction check: iter 0 must equal frozen warm start.
    fp0 = checkpoint_fingerprint(f"{ckpt}/iter_000.pt")
    warm = checkpoint_fingerprint(WARM)
    meta = {
        "training_seed": seed,
        "warm_start": warm,
        "iter0": fp0,
        "iter0_matches_warm_start":
            fp0["parameter_sha256"] == warm["parameter_sha256"],
        "warm_start_matches_exp2":
            warm["parameter_sha256"] == EXP2_WARM_START_PARAMETER_SHA256,
        "checkpoints": {
            it: {
                "iteration": it,
                "cumulative_episodes": episodes(it),
                **checkpoint_fingerprint(f"{ckpt}/iter_{it:03d}.pt"),
            }
            for it in PRIMARY_ITERS
        },
    }
    with open(f"{seed_dir}/checkpoints_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if not meta["iter0_matches_warm_start"]:
        raise SystemExit("iter0 parameter_sha256 != warm start — abort")
    if not meta["warm_start_matches_exp2"]:
        raise SystemExit("warm start does not match Exp2 parameter_sha256 — abort")
    print(f"iter0 == warm start == Exp2: {fp0['parameter_sha256']}")


def evaluate(seed: int, seed_dir: str) -> None:
    ckpt = f"{seed_dir}/checkpoints"
    dev = f"{seed_dir}/dev"
    os.makedirs(dev, exist_ok=True)
    log = f"{seed_dir}/eval.log"
    open(log, "w").close()
    for it in PRIMARY_ITERS:
        path = f"{ckpt}/iter_{it:03d}.pt"
        name = f"seed{seed}_iter{it:03d}"
        for field, games in (("greedy", 1000), ("greedy4_random3", 500)):
            out = f"{dev}/iter{it:03d}_vs_{field}.json"
            _run([
                sys.executable, "-m", "ml.dev_benchmark",
                "--agent", "policy", "--checkpoint", path,
                "--name", name, "--games", str(games),
                "--seed", str(seedmod.DEV_SEED_START),
                "--field", field, "--json-out", out,
            ], log)
            blob = load_json(out)
            # Stamp training-seed / budget metadata into the result.
            fp = checkpoint_fingerprint(path)
            blob["training_seed"] = seed
            blob["ppo_iteration"] = it
            blob["cumulative_episodes"] = episodes(it)
            blob["parameter_sha256"] = fp["parameter_sha256"]
            blob["checkpoint_sha256"] = fp["checkpoint_sha256"]
            with open(out, "w") as f:
                json.dump(blob, f, indent=2)


def drift(seed: int, seed_dir: str) -> None:
    ckpt = f"{seed_dir}/checkpoints"
    paths = [f"{ckpt}/iter_{it:03d}.pt" for it in PRIMARY_ITERS]
    log = f"{seed_dir}/drift.log"
    open(log, "w").close()
    _run([
        sys.executable, "-m", "ml.policy_drift",
        "--reference", paths[0],
        "--checkpoints", *paths,
        "--json-out", f"{seed_dir}/policy_drift.json",
        "--categories-out", f"{seed_dir}/action_category_drift.json",
    ], log)
    blob = load_json(f"{seed_dir}/policy_drift.json")
    fp = blob["corpus"]["fingerprint_sha256"]
    if not fp.startswith(EXP2_CORPUS_FINGERPRINT_PREFIX):
        raise SystemExit(
            f"drift corpus fingerprint {fp[:12]} does not match Exp1/2 "
            f"prefix {EXP2_CORPUS_FINGERPRINT_PREFIX}")
    print(f"corpus fingerprint OK {fp[:12]}")


def summarize_seed(seed: int, seed_dir: str) -> None:
    greedy = load_seed_greedy(seed_dir)
    mixed = load_seed_mixed(seed_dir)
    drift_blob = load_json(f"{seed_dir}/policy_drift.json")
    cats_blob = load_json(f"{seed_dir}/action_category_drift.json")
    drift_by = {r["checkpoint"]: r for r in drift_blob["checkpoints"]}
    cats_by = {r["checkpoint"]: r for r in cats_blob["checkpoints"]}
    curve = []
    for it in PRIMARY_ITERS:
        key = f"iter_{it:03d}.pt"
        curve.append(curve_point(greedy[it], mixed[it], drift_by[key],
                                 cats_by.get(key), it, seed))
    paired = within_seed_paired(greedy)
    diag = [json.loads(l) for l in open(f"{seed_dir}/train_diag.jsonl")]

    def block(rows):
        keys = ("adv_mean", "adv_std", "adv_mean_abs", "adv_frac_positive",
                "adv_frac_negative", "return_mean", "return_std",
                "value_pred_mean", "value_pred_std", "value_explained_variance",
                "placement_std", "shaping_reward_sum", "terminal_reward_sum",
                "entropy", "approx_kl", "clip_frac", "grad_norm",
                "pi_loss", "v_loss")
        return {k: st.mean(r[k] for r in rows if r.get(k) is not None)
                for k in keys}

    blocks = {
        "iters_1_40": block([r for r in diag if r["iter"] <= 40]),
        "iters_41_160": block([r for r in diag if 40 < r["iter"] <= 160]),
        "iters_161_320": block([r for r in diag if r["iter"] > 160]),
    }
    with open(f"{seed_dir}/learning_curve.json", "w") as f:
        json.dump({
            "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
            "training_seed": seed,
            "evaluation_split": "dev",
            "episodes_per_iteration": 16,
            "primary_iterations": PRIMARY_ITERS,
            "curve": curve,
            "paired": paired,
        }, f, indent=2)
    with open(f"{seed_dir}/paired_analysis.json", "w") as f:
        json.dump({
            "method": "deterministic paired percentile bootstrap, 10000 "
                      "resamples, seed 0",
            "note": "positive difference = checkpoint places worse than "
                    "the reference (lower placement is better)",
            "training_seed": seed,
            "contrasts": paired,
        }, f, indent=2)
    with open(f"{seed_dir}/rl_signal.json", "w") as f:
        json.dump({
            "training_seed": seed,
            "definitions": {
                "adv_*": "GAE advantages BEFORE PPO's per-batch normalization",
                "value_explained_variance":
                    "1 - Var(returns - value_preds) / Var(returns)",
            },
            "per_iteration": diag,
            "blocks": blocks,
        }, f, indent=2)
    # Convenience copy of train_diag already present.
    print(f"Seed {seed} summary written under {seed_dir}/")


def materialize_seed0() -> None:
    """Point seed_0 at Experiment 2 committed artifacts (do not retrain)."""
    src = "results/ppo_budget_v1"
    dst = f"{ROOT}/seed_0"
    os.makedirs(f"{dst}/dev", exist_ok=True)
    for name in ("learning_curve.json", "paired_analysis.json",
                 "policy_drift.json", "action_category_drift.json",
                 "rl_signal.json", "train_diag.jsonl", "train.log",
                 "manifest.json"):
        s = f"{src}/{name}"
        if os.path.isfile(s):
            shutil.copy2(s, f"{dst}/{name}")
    for it in PRIMARY_ITERS:
        for field in ("greedy", "greedy4_random3"):
            s = f"{src}/dev/iter{it:03d}_vs_{field}.json"
            if os.path.isfile(s):
                blob = load_json(s)
                blob.setdefault("training_seed", 0)
                blob.setdefault("ppo_iteration", it)
                blob.setdefault("cumulative_episodes", episodes(it))
                # Prefer fingerprints already in Exp2 drift/manifest.
                with open(f"{dst}/dev/iter{it:03d}_vs_{field}.json", "w") as f:
                    json.dump(blob, f, indent=2)
    # Normalize seed_0 learning_curve / paired to Exp3 schema if needed.
    # Rebuild from DEV + drift so schema matches seeds 1-3.
    summarize_seed(0, dst)
    # Restore Exp2 paired contrasts that use a slightly different key set —
    # summarize_seed overwrites with the full Exp3 contrast list (good).
    meta = {
        "training_seed": 0,
        "source": "results/ppo_budget_v1 (Experiment 2 — not retrained)",
        "warm_start_parameter_sha256": EXP2_WARM_START_PARAMETER_SHA256,
        "note": "Seed 0 is the published Exp2 trajectory; checkpoints are "
                "gitignored and not re-derived here.",
    }
    with open(f"{dst}/checkpoints_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True,
                   help="PPO training seed (1,2,3 to train; 0 to materialize Exp2)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-drift", action="store_true")
    a = p.parse_args(argv)

    if a.seed == 0:
        materialize_seed0()
        return 0

    if a.seed not in (1, 2, 3):
        p.error("training seeds for Exp3 are 1, 2, 3 (0 = materialize Exp2)")

    lo = seedmod.ppo_episode_seed(a.seed, 1)
    hi = seedmod.ppo_episode_seed(a.seed, 320 * 16)
    if seedmod.check_training_range(f"exp3-seed{a.seed}", lo, hi):
        raise SystemExit("training seed span overlaps DEV/TEST — refuse")

    warm = checkpoint_fingerprint(WARM)
    if warm["parameter_sha256"] != EXP2_WARM_START_PARAMETER_SHA256:
        raise SystemExit(
            f"refuse: {WARM} parameter_sha256={warm['parameter_sha256']} "
            f"!= Exp2 {EXP2_WARM_START_PARAMETER_SHA256}. "
            f"Run: PYTHONHASHSEED=0 python scripts/reproduce_warm_start_bc.py")

    seed_dir = f"{ROOT}/seed_{a.seed}"
    os.makedirs(seed_dir, exist_ok=True)
    if not a.skip_train:
        train(a.seed, seed_dir)
    if not a.skip_eval:
        evaluate(a.seed, seed_dir)
    if not a.skip_drift:
        drift(a.seed, seed_dir)
    # Summaries need DEV eval + drift JSON; skip on train-only runs.
    if not a.skip_eval and not a.skip_drift:
        summarize_seed(a.seed, seed_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
