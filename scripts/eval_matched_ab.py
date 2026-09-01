"""DEV evaluation and policy drift for Experiment 4b matched A/B runs."""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.experiment_contract import enforce_runtime_match, load_contract

BASE_DIR = "results/ppo_matched_ab_v1"
CONTRACT_PATH = os.path.join(BASE_DIR, "contract.json")
SEEDS = [0, 1, 2, 3]
KL_LABELS = ("beta0", "beta01")
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
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env=single_thread_env())
    ok = res.returncode == 0
    if not ok:
        print(f"EVAL FAIL {job['name']}: {res.stderr}")
    else:
        print(f"[ok] {job['name']} {job['field']} {job['games']}g "
              f"in {time.time() - t0:.1f}s")
    return {"job": job, "success": ok}


def run_drift(kl_label: str, seed: int) -> dict:
    run = os.path.join(BASE_DIR, kl_label, f"seed_{seed}")
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
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env=single_thread_env())
    ok = res.returncode == 0
    if not ok:
        print(f"DRIFT FAIL {kl_label} seed {seed}: {res.stderr}")
    else:
        print(f"[drift ok] {kl_label} seed {seed} in {time.time() - t0:.1f}s")
    return {"kl_label": kl_label, "seed": seed, "success": ok}


def main() -> int:
    contract = load_contract(CONTRACT_PATH)
    enforce_runtime_match(contract, strict_commit=False)

    jobs = []
    for kl_label in KL_LABELS:
        kl_coef = 0.0 if kl_label == "beta0" else 0.1
        for seed in SEEDS:
            run = os.path.join(BASE_DIR, kl_label, f"seed_{seed}")
            dev_dir = os.path.join(run, "dev")
            ckpt_dir = os.path.join(run, "checkpoints")
            for it in ITERS:
                ckpt = os.path.join(ckpt_dir, f"iter_{it:03d}.pt")
                tag = f"{kl_label}_s{seed}_i{it:03d}"
                jobs.append({
                    "checkpoint": ckpt,
                    "name": tag,
                    "field": "greedy",
                    "games": 1000,
                    "out_json": os.path.join(dev_dir,
                                             f"iter{it:03d}_vs_greedy.json"),
                    "kl_coef": kl_coef,
                    "seed": seed,
                    "iteration": it,
                })
                jobs.append({
                    "checkpoint": ckpt,
                    "name": tag,
                    "field": "greedy4_random3",
                    "games": 500,
                    "out_json": os.path.join(dev_dir,
                                             f"iter{it:03d}_vs_greedy4_random3.json"),
                    "kl_coef": kl_coef,
                    "seed": seed,
                    "iteration": it,
                })

    print(f"Total DEV eval jobs: {len(jobs)}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as pool:
        eval_results = list(pool.map(run_eval, jobs))

    drift_jobs = [(kl, s) for kl in KL_LABELS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        drift_results = list(pool.starmap(run_drift, drift_jobs))

    all_ok = (all(r["success"] for r in eval_results)
              and all(r["success"] for r in drift_results))
    print(f"\nEvaluation finished in {time.time() - t0:.1f}s. Success={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
