"""Run DEV evaluations and policy drift for Experiment 4 (Policy Anchoring)."""

import os
import subprocess
import sys
import time

DIR = "results/ppo_anchor_v1"
ITERS = [0, 40, 80, 160, 320]
DEV_SEED_START = 10550000


def single_thread_env():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def run_eval(it: int, field: str, games: int, out_json: str) -> bool:
    ckpt = os.path.join(DIR, "checkpoints", f"iter_{it:03d}.pt")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    cmd = [
        sys.executable, "-m", "ml.dev_benchmark",
        "--agent", "policy",
        "--checkpoint", ckpt,
        "--name", f"Anchor_iter{it:03d}",
        "--field", field,
        "--games", str(games),
        "--seed", str(DEV_SEED_START),
        "--json-out", out_json,
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env=single_thread_env())
    ok = res.returncode == 0
    status = "ok" if ok else "FAILED"
    print(f"[{status}] iter {it:03d} vs {field} ({games} games) "
          f"in {time.time() - t0:.1f}s")
    if not ok:
        print(res.stderr)
    return ok


def run_drift() -> bool:
    ckpt_dir = os.path.join(DIR, "checkpoints")
    ref = os.path.join(ckpt_dir, "iter_000.pt")
    ckpts = [os.path.join(ckpt_dir, f"iter_{it:03d}.pt") for it in ITERS]
    drift_out = os.path.join(DIR, "policy_drift.json")
    cats_out = os.path.join(DIR, "action_category_drift.json")
    cmd = [
        sys.executable, "-m", "ml.policy_drift",
        "--reference", ref,
        "--checkpoints", *ckpts,
        "--json-out", drift_out,
        "--categories-out", cats_out,
    ]
    print("Running policy drift analysis...")
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env=single_thread_env())
    if res.returncode != 0:
        print(res.stderr)
    return res.returncode == 0


def main() -> int:
    t0 = time.time()
    dev_dir = os.path.join(DIR, "dev")
    ok = True
    for it in ITERS:
        ok &= run_eval(it, "greedy", 1000,
                       os.path.join(dev_dir, f"iter{it:03d}_vs_greedy.json"))
        ok &= run_eval(it, "greedy4_random3", 500,
                       os.path.join(dev_dir,
                                    f"iter{it:03d}_vs_greedy4_random3.json"))
    ok &= run_drift()
    print(f"\nEvaluation complete in {time.time() - t0:.1f}s. Success={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
