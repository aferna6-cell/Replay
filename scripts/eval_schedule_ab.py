"""DEV evaluation and policy drift for Experiment 6 scheduled arm."""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.experiment_contract import enforce_runtime_match, load_contract

BASE_DIR = "results/ppo_schedule_v1"
CONTRACT_PATH = os.path.join(BASE_DIR, "contract.json")
SCHEDULE_LABEL = "beta_sched"
SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
DEV_SEED_START = 10_550_000


def single_thread_env():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def run_eval(job: dict) -> dict:
    cmd = [
        sys.executable, "-m", "ml.dev_benchmark",
        "--agent", "policy",
        "--checkpoint", job["checkpoint"],
        "--name", job["name"],
        "--field", job["field"],
        "--games", str(job["games"]),
        "--seed", str(DEV_SEED_START),
        "--json-out", job["out_json"],
    ]
    os.makedirs(os.path.dirname(job["out_json"]), exist_ok=True)
    if os.path.isfile(job["out_json"]):
        return {"job": job, "success": True, "skipped": True}
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env=single_thread_env())
    ok = res.returncode == 0
    if ok:
        print(f"[ok] {job['name']} {job['field']} in {time.time() - t0:.1f}s")
    else:
        print(f"EVAL FAIL {job['name']}: {res.stderr}")
    return {"job": job, "success": ok}


def run_drift(seed: int) -> dict:
    run = os.path.join(BASE_DIR, SCHEDULE_LABEL, f"seed_{seed}")
    ckpt_dir = os.path.join(run, "checkpoints")
    ref = os.path.join(ckpt_dir, "iter_000.pt")
    ckpts = [os.path.join(ckpt_dir, f"iter_{it:03d}.pt") for it in ITERS]
    drift_out = os.path.join(run, "policy_drift.json")
    cats_out = os.path.join(run, "action_category_drift.json")
    cmd = [
        sys.executable, "-m", "ml.policy_drift",
        "--reference", ref,
        "--checkpoints", *ckpts,
        "--json-out", drift_out,
        "--categories-out", cats_out,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env=single_thread_env())
    ok = res.returncode == 0
    if ok:
        print(f"[drift ok] seed {seed}")
    return {"seed": seed, "success": ok}


def main() -> int:
    contract = load_contract(CONTRACT_PATH)
    enforce_runtime_match(contract, strict_commit=False)

    jobs = []
    for seed in SEEDS:
        run = os.path.join(BASE_DIR, SCHEDULE_LABEL, f"seed_{seed}")
        dev_dir = os.path.join(run, "dev")
        ckpt_dir = os.path.join(run, "checkpoints")
        for it in ITERS:
            ckpt = os.path.join(ckpt_dir, f"iter_{it:03d}.pt")
            tag = f"{SCHEDULE_LABEL}_s{seed}_i{it:03d}"
            jobs.append({
                "checkpoint": ckpt, "name": tag, "field": "greedy",
                "games": 1000,
                "out_json": os.path.join(dev_dir, f"iter{it:03d}_vs_greedy.json"),
            })
            jobs.append({
                "checkpoint": ckpt, "name": tag, "field": "greedy4_random3",
                "games": 500,
                "out_json": os.path.join(dev_dir,
                                         f"iter{it:03d}_vs_greedy4_random3.json"),
            })

    print(f"Total DEV eval jobs (scheduled arm): {len(jobs)}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as pool:
        eval_results = list(pool.map(run_eval, jobs))
        drift_results = list(pool.map(run_drift, SEEDS))

    all_ok = (all(r["success"] for r in eval_results)
              and all(r["success"] for r in drift_results))
    print(f"Evaluation finished in {time.time() - t0:.1f}s. Success={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
