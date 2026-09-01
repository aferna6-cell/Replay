"""Experiment 4b — matched anchored vs unanchored PPO (8 trajectories).

Freezes one BC warm start and one runtime contract, then for each training
seed in {0,1,2,3} runs β=0.0 and β=0.1 with hard reproducibility gates.

    python scripts/train_matched_ab.py
    python scripts/eval_matched_ab.py
    python scripts/ppo_matched_ab_report.py
    python scripts/ppo_matched_ab_manifest.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.benchmark import make_agent
from ml.dev_benchmark import run_dev_benchmark
from ml.experiment_contract import (ContractViolation, build_contract,
                                    enforce_runtime_match, load_contract,
                                    save_contract, verify_identical_placements,
                                    verify_matched_iter0_pair, verify_warm_start)
from ml.model_fingerprint import checkpoint_fingerprint

BASE_DIR = "results/ppo_matched_ab_v1"
WARM_START = os.path.join(BASE_DIR, "warm_start.pt")
CONTRACT_PATH = os.path.join(BASE_DIR, "contract.json")
BC_SOURCE = "ml/policy_bc.pt"
SEEDS = [0, 1, 2, 3]
ITERS = 320
EPISODES = 16
SHAPING_HORIZON = 40
SAVE_ITERS = "0,40,80,160,320"
KL_ARMS = ((0.0, "beta0"), (0.1, "beta01"))
DEV_SEED_START = 10_550_000


def single_thread_env():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def run_dir(kl_label: str, seed: int) -> str:
    return os.path.join(BASE_DIR, kl_label, f"seed_{seed}")


def prepare_warm_start_and_contract() -> dict:
    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.path.isfile(WARM_START):
        if not os.path.isfile(BC_SOURCE):
            print("Training BC warm start (once for entire experiment)…")
            res = subprocess.run(
                [sys.executable, "-m", "ml.bc", "--lobbies", "150",
                 "--epochs", "6", "--eval-episodes", "1"],
                env=single_thread_env())
            if res.returncode != 0:
                raise SystemExit("BC training failed")
        shutil.copy2(BC_SOURCE, WARM_START)
        print(f"Frozen warm start -> {WARM_START}")

    contract = build_contract(WARM_START)
    save_contract(CONTRACT_PATH, contract)
    warm_record = {
        "path": WARM_START,
        **checkpoint_fingerprint(WARM_START),
    }
    with open(os.path.join(BASE_DIR, "warm_start.json"), "w", encoding="utf-8") as f:
        json.dump(warm_record, f, indent=2)
    print(f"Contract saved -> {CONTRACT_PATH}")
    print(f"expected_warm_start_parameter_sha256="
          f"{contract['expected_warm_start_parameter_sha256']}")
    return contract


def train_arm(seed: int, kl_coef: float, kl_label: str,
              expected_sha: str) -> int:
    out = run_dir(kl_label, seed)
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    diag_log = os.path.join(out, "train_diag.jsonl")
    train_log = os.path.join(out, "train.log")
    if os.path.exists(diag_log):
        os.remove(diag_log)

    cmd = [
        sys.executable, "-m", "ml.train_ppo",
        "--iters", str(ITERS),
        "--episodes", str(EPISODES),
        "--seed", str(seed),
        "--shaping", "1.0",
        "--shaping-horizon", str(SHAPING_HORIZON),
        "--from-bc", WARM_START,
        "--expected-warm-start-sha", expected_sha,
        "--out", os.path.join(out, "final.pt"),
        "--save-iters", SAVE_ITERS,
        "--save-dir", ckpt_dir,
        "--diag-log", diag_log,
        "--eval-episodes", "1",
        "--kl-coef", str(kl_coef),
    ]
    if kl_coef > 0:
        cmd.extend(["--anchor", WARM_START])

    meta = {
        "training_seed": seed,
        "kl_coef": kl_coef,
        "kl_label": kl_label,
        "warm_start_sha256": expected_sha,
    }
    with open(os.path.join(out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    with open(train_log, "w", encoding="utf-8") as f_log:
        res = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT,
                             text=True, env=single_thread_env())
    elapsed = time.time() - t0
    print(f"[{kl_label} seed {seed}] training {elapsed:.1f}s rc={res.returncode}")
    return res.returncode


def eval_iter0_gate(seed: int, kl_label: str) -> str:
    out = run_dir(kl_label, seed)
    dev_dir = os.path.join(out, "dev")
    os.makedirs(dev_dir, exist_ok=True)
    json_out = os.path.join(dev_dir, "iter000_vs_greedy.json")
    if os.path.isfile(json_out):
        return json_out
    ckpt = os.path.join(out, "checkpoints", "iter_000.pt")
    agent = make_agent("policy", checkpoint=ckpt, name=f"{kl_label}_s{seed}_iter000")
    res = run_dev_benchmark(agent, "greedy", 1000, base_seed=DEV_SEED_START)
    from ml.dev_benchmark import dev_result_to_json
    from ml.benchmark import _write_json
    _write_json(json_out, dev_result_to_json(res))
    return json_out


def verify_seed_pair(seed: int, expected_sha: str) -> None:
    beta0_ckpt = os.path.join(run_dir("beta0", seed), "checkpoints", "iter_000.pt")
    beta01_ckpt = os.path.join(run_dir("beta01", seed), "checkpoints", "iter_000.pt")
    verify_matched_iter0_pair(beta0_ckpt, beta01_ckpt, expected_sha)
    p0 = eval_iter0_gate(seed, "beta0")
    p1 = eval_iter0_gate(seed, "beta01")
    verify_identical_placements(p0, p1, label=f"seed {seed} iter0 greedy")
    print(f"[GATE OK] seed {seed}: iter0 hashes match warm start; "
          f"iter0 DEV placements identical ({len(json.load(open(p0))['placements'])} games)")


def train_seed_pair(seed: int) -> dict:
    contract = load_contract(CONTRACT_PATH)
    enforce_runtime_match(contract)
    expected_sha = contract["expected_warm_start_parameter_sha256"]
    verify_warm_start(WARM_START, expected_sha)

    for kl_coef, kl_label in KL_ARMS:
        rc = train_arm(seed, kl_coef, kl_label, expected_sha)
        if rc != 0:
            return {"seed": seed, "success": False, "stage": f"train_{kl_label}",
                    "returncode": rc}

    try:
        verify_seed_pair(seed, expected_sha)
    except ContractViolation as exc:
        print(f"[GATE FAILED] seed {seed}: {exc}", file=sys.stderr)
        return {"seed": seed, "success": False, "stage": "gate", "error": str(exc)}

    return {"seed": seed, "success": True}


def main() -> int:
    contract = prepare_warm_start_and_contract()
    enforce_runtime_match(contract)

    t0 = time.time()
    print(f"Starting matched A/B training for seeds {SEEDS} "
          f"(2 arms × {len(SEEDS)} seeds = {2 * len(SEEDS)} trajectories)…")
    results = []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        futs = {pool.submit(train_seed_pair, s): s for s in SEEDS}
        for fut in as_completed(futs):
            seed = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                print(f"[FAIL] seed {seed}: {exc}", file=sys.stderr)
                results.append({"seed": seed, "success": False,
                                "stage": "exception", "error": str(exc)})

    results.sort(key=lambda r: r["seed"])
    all_ok = all(r.get("success") for r in results)
    elapsed = time.time() - t0
    print(f"\nTraining + gates finished in {elapsed:.1f}s. Success={all_ok}")
    for r in results:
        print(f"  seed {r['seed']}: {r}")

    gate_path = os.path.join(BASE_DIR, "gate_results.json")
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump({"all_passed": all_ok, "per_seed": results,
                   "expected_warm_start_parameter_sha256":
                   contract["expected_warm_start_parameter_sha256"]}, f, indent=2)

    if not all_ok:
        print("STOP: reproducibility gate failed — experiment aborted.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
