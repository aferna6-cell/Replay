"""Train PPO seeds 1, 2, and 3 for Experiment 3 (Multi-Seed Replication).

Each seed trains for 320 iterations (5,120 total episodes) with fixed
--shaping-horizon 40 and warm-started from ml/policy_bc.pt.
Checkpoints are saved at iterations 0, 40, 80, 160, 320.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SEEDS = [1, 2, 3]
ITERS = 320
EPISODES = 16
SHAPING_HORIZON = 40
SAVE_ITERS = "0,40,80,160,320"
BASE_DIR = "results/ppo_multiseed_v1"
BC_PATH = "ml/policy_bc.pt"


def train_seed(seed: int) -> dict:
    seed_dir = os.path.join(BASE_DIR, f"seed_{seed}")
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(seed_dir, "final.pt")
    diag_log = os.path.join(seed_dir, "train_diag.jsonl")
    train_log = os.path.join(seed_dir, "train.log")

    # Remove any stale diag log
    if os.path.exists(diag_log):
        os.remove(diag_log)

    cmd = [
        sys.executable, "-m", "ml.train_ppo",
        "--iters", str(ITERS),
        "--episodes", str(EPISODES),
        "--seed", str(seed),
        "--shaping", "1.0",
        "--shaping-horizon", str(SHAPING_HORIZON),
        "--from-bc", BC_PATH,
        "--out", out_path,
        "--save-iters", SAVE_ITERS,
        "--save-dir", ckpt_dir,
        "--diag-log", diag_log,
        "--eval-episodes", "1",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    t0 = time.time()
    print(f"[Seed {seed}] Starting training (320 iters, 5120 episodes)...")
    with open(train_log, "w", encoding="utf-8") as f_log:
        res = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, text=True, env=env)

    elapsed = time.time() - t0
    success = res.returncode == 0
    print(f"[Seed {seed}] Finished in {elapsed:.1f}s (returncode: {res.returncode})")
    return {"seed": seed, "success": success, "elapsed_s": elapsed, "returncode": res.returncode}


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    t_start = time.time()
    print(f"Starting parallel training for seeds {SEEDS}...")
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as executor:
        results = list(executor.map(train_seed, SEEDS))

    all_ok = all(r["success"] for r in results)
    total_time = time.time() - t_start
    print(f"\nAll training runs complete in {total_time:.1f}s. Success: {all_ok}")
    for r in results:
        print(f"  Seed {r['seed']}: success={r['success']}, time={r['elapsed_s']:.1f}s")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
