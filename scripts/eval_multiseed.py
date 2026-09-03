"""Run DEV evaluations and policy drift analysis for Experiment 3 (Multi-Seed Replication).

For each seed (1, 2, 3) and checkpoint (0, 40, 80, 160, 320):
  1. 1000 DEV games vs 7x greedy (seeds 10,550,000 - 10,550,999)
  2. 500 DEV games vs greedy4_random3 (seeds 10,550,000 - 10,550,499)
  3. Policy drift & action category drift over the frozen 4,440-state corpus.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SEEDS = [1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
BASE_DIR = "results/ppo_multiseed_v1"
DEV_SEED_START = 10550000


def get_single_thread_env():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def run_eval_job(job: dict) -> dict:
    seed = job["seed"]
    it = job["iteration"]
    field = job["field"]
    games = job["games"]
    ckpt_path = job["checkpoint"]
    out_json = job["out_json"]

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    cmd = [
        sys.executable, "-m", "ml.dev_benchmark",
        "--agent", "policy",
        "--checkpoint", ckpt_path,
        "--name", f"Seed{seed}_iter{it:03d}",
        "--field", field,
        "--games", str(games),
        "--seed", str(DEV_SEED_START),
        "--json-out", out_json,
    ]

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=get_single_thread_env())
    elapsed = time.time() - t0
    success = res.returncode == 0
    if not success:
        print(f"Error evaluating seed {seed} iter {it} field {field}:\n{res.stderr}")
    else:
        print(f"[Done] Seed {seed} iter {it:03d} vs {field} ({games} games) in {elapsed:.1f}s")
    return {"job": job, "success": success, "elapsed_s": elapsed}


def run_drift_for_seed(seed: int) -> dict:
    seed_dir = os.path.join(BASE_DIR, f"seed_{seed}")
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    ref_ckpt = os.path.join(ckpt_dir, "iter_000.pt")
    ckpts = [os.path.join(ckpt_dir, f"iter_{it:03d}.pt") for it in ITERS]
    drift_out = os.path.join(seed_dir, "policy_drift.json")
    cats_out = os.path.join(seed_dir, "action_category_drift.json")

    cmd = [
        sys.executable, "-m", "ml.policy_drift",
        "--reference", ref_ckpt,
        "--checkpoints", *ckpts,
        "--json-out", drift_out,
        "--categories-out", cats_out,
    ]

    t0 = time.time()
    print(f"[Drift] Starting policy drift analysis for seed {seed}...")
    res = subprocess.run(cmd, capture_output=True, text=True, env=get_single_thread_env())
    elapsed = time.time() - t0
    success = res.returncode == 0
    if not success:
        print(f"Error in drift analysis for seed {seed}:\n{res.stderr}")
    else:
        print(f"[Drift] Finished drift analysis for seed {seed} in {elapsed:.1f}s")
    return {"seed": seed, "success": success, "elapsed_s": elapsed}


def main():
    t_start = time.time()
    jobs = []
    for s in SEEDS:
        seed_dir = os.path.join(BASE_DIR, f"seed_{s}")
        dev_dir = os.path.join(seed_dir, "dev")
        ckpt_dir = os.path.join(seed_dir, "checkpoints")
        for it in ITERS:
            ckpt_path = os.path.join(ckpt_dir, f"iter_{it:03d}.pt")
            jobs.append({
                "seed": s, "iteration": it, "field": "greedy", "games": 1000,
                "checkpoint": ckpt_path,
                "out_json": os.path.join(dev_dir, f"iter{it:03d}_vs_greedy.json"),
            })
            jobs.append({
                "seed": s, "iteration": it, "field": "greedy4_random3", "games": 500,
                "checkpoint": ckpt_path,
                "out_json": os.path.join(dev_dir, f"iter{it:03d}_vs_greedy4_random3.json"),
            })

    print(f"Total benchmark evaluation jobs: {len(jobs)}")
    with ProcessPoolExecutor(max_workers=4) as executor:
        eval_results = list(executor.map(run_eval_job, jobs))

    all_eval_ok = all(r["success"] for r in eval_results)
    print(f"\nAll benchmark evaluations complete. Success: {all_eval_ok}")

    print("\nRunning drift analysis for all seeds...")
    with ProcessPoolExecutor(max_workers=3) as executor:
        drift_results = list(executor.map(run_drift_for_seed, SEEDS))

    all_drift_ok = all(r["success"] for r in drift_results)
    total_time = time.time() - t_start
    print(f"\nAll evaluations and drift analyses finished in {total_time:.1f}s. Overall success: {all_eval_ok and all_drift_ok}")

    if not (all_eval_ok and all_drift_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
