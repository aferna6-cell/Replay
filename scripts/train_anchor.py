"""Train anchored PPO for Experiment 4 (Policy Anchoring to BC Prior).

Single training seed (0), 320 iterations, frozen shaping horizon 40,
KL penalty toward the frozen BC warm start with pre-specified coefficient.
"""

import os
import subprocess
import sys
import time

DIR = "results/ppo_anchor_v1"
KL_COEF = 0.1
ITERS = 320
EPISODES = 16
SHAPING_HORIZON = 40
SAVE_ITERS = "0,40,80,160,320"
BC_PATH = "ml/policy_bc.pt"


def main() -> int:
    ckpt_dir = os.path.join(DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(DIR, "final.pt")
    diag_log = os.path.join(DIR, "train_diag.jsonl")
    train_log = os.path.join(DIR, "train.log")

    if os.path.exists(diag_log):
        os.remove(diag_log)

    cmd = [
        sys.executable, "-m", "ml.train_ppo",
        "--iters", str(ITERS),
        "--episodes", str(EPISODES),
        "--seed", "0",
        "--shaping", "1.0",
        "--shaping-horizon", str(SHAPING_HORIZON),
        "--from-bc", BC_PATH,
        "--anchor", BC_PATH,
        "--kl-coef", str(KL_COEF),
        "--out", out_path,
        "--save-iters", SAVE_ITERS,
        "--save-dir", ckpt_dir,
        "--diag-log", diag_log,
        "--eval-episodes", "1",
    ]

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"

    t0 = time.time()
    print(f"Starting anchored PPO (kl_coef={KL_COEF}, {ITERS} iters)...")
    with open(train_log, "w", encoding="utf-8") as f_log:
        res = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT,
                             text=True, env=env)
    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s (returncode={res.returncode})")
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
