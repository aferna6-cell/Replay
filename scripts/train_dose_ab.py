"""Experiment 5 — KL anchoring dose-response (β=0.01 and β=0.03).

Reuses the frozen Experiment 4b contract and warm start. Trains 8 new
trajectories (seeds 0–3 × β ∈ {0.01, 0.03}). β=0.0 and β=0.1 results are
reused from ``results/ppo_matched_ab_v1/``.

    python scripts/train_dose_ab.py
    python scripts/eval_dose_ab.py
    python scripts/ppo_dose_report.py
    python scripts/ppo_dose_manifest.py
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
                                    save_contract, verify_checkpoint_parameter_sha256,
                                    verify_warm_start)
from ml.model_fingerprint import checkpoint_fingerprint

BASE_DIR = "results/ppo_dose_v1"
MATCHED_DIR = "results/ppo_matched_ab_v1"
WARM_START = os.path.join(BASE_DIR, "warm_start.pt")
CONTRACT_PATH = os.path.join(BASE_DIR, "contract.json")
MATCHED_WARM = os.path.join(MATCHED_DIR, "warm_start.pt")
SEEDS = [0, 1, 2, 3]
ITERS = 320
EPISODES = 16
SHAPING_HORIZON = 40
SAVE_ITERS = "0,40,80,160,320"
NEW_ARMS = ((0.01, "beta001"), (0.03, "beta003"))
ALL_KL_VALUES = [0.0, 0.01, 0.03, 0.1]
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
    if not os.path.isfile(MATCHED_WARM):
        raise SystemExit(f"Experiment 4b warm start missing: {MATCHED_WARM}")
    if not os.path.isfile(WARM_START):
        shutil.copy2(MATCHED_WARM, WARM_START)
        print(f"Copied frozen warm start -> {WARM_START}")

    matched = load_contract(os.path.join(MATCHED_DIR, "contract.json"))
    contract = build_contract(WARM_START, kl_coef_values=ALL_KL_VALUES)
    contract["parent_experiment"] = "ppo_matched_ab_v1"
    contract["parent_contract_path"] = os.path.join(MATCHED_DIR, "contract.json")
    contract["reused_arms"] = {
        "beta0": {"kl_coef": 0.0, "source": MATCHED_DIR},
        "beta01": {"kl_coef": 0.1, "source": MATCHED_DIR},
    }
    contract["new_arms"] = {
        label: coef for coef, label in NEW_ARMS
    }
    contract["question"] = (
        "What is the weakest KL anchoring strength that preserves PPO stability "
        "while allowing enough policy movement to beat the BC warm start (~6.550)?"
    )
    save_contract(CONTRACT_PATH, contract)

    warm_record = {
        "path": WARM_START,
        **checkpoint_fingerprint(WARM_START),
    }
    with open(os.path.join(BASE_DIR, "warm_start.json"), "w", encoding="utf-8") as f:
        json.dump(warm_record, f, indent=2)

    expected = contract["expected_warm_start_parameter_sha256"]
    if expected != matched["expected_warm_start_parameter_sha256"]:
        raise ContractViolation(
            "warm-start hash differs from Experiment 4b contract")
    print(f"Contract saved -> {CONTRACT_PATH}")
    print(f"expected_warm_start_parameter_sha256={expected}")
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
        "--anchor", WARM_START,
    ]

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
    _write_json(dev_result_to_json(res), json_out)
    return json_out


def verify_seed_arm(seed: int, kl_label: str, expected_sha: str) -> None:
    ckpt = os.path.join(run_dir(kl_label, seed), "checkpoints", "iter_000.pt")
    verify_checkpoint_parameter_sha256(ckpt, expected_sha, label=f"{kl_label} iter0")
    print(f"[HASH OK] {kl_label} seed {seed}: iter_000 matches warm start")


def train_seed_all_arms(seed: int) -> dict:
    contract = load_contract(CONTRACT_PATH)
    enforce_runtime_match(contract)
    expected_sha = contract["expected_warm_start_parameter_sha256"]
    verify_warm_start(WARM_START, expected_sha)

    for kl_coef, kl_label in NEW_ARMS:
        rc = train_arm(seed, kl_coef, kl_label, expected_sha)
        if rc != 0:
            return {"seed": seed, "success": False, "stage": f"train_{kl_label}",
                    "returncode": rc}

    for _, kl_label in NEW_ARMS:
        try:
            verify_seed_arm(seed, kl_label, expected_sha)
        except ContractViolation as exc:
            return {"seed": seed, "success": False, "stage": f"gate_{kl_label}",
                    "error": str(exc)}

    return {"seed": seed, "success": True, "stage": "train"}


def run_gates_only() -> int:
    contract = load_contract(CONTRACT_PATH)
    enforce_runtime_match(contract, strict_commit=False)
    expected_sha = contract["expected_warm_start_parameter_sha256"]
    verify_warm_start(WARM_START, expected_sha)
    results = []
    for kl_coef, kl_label in NEW_ARMS:
        for seed in SEEDS:
            try:
                verify_seed_arm(seed, kl_label, expected_sha)
                results.append({"kl_label": kl_label, "seed": seed, "hash_ok": True})
            except ContractViolation as exc:
                results.append({"kl_label": kl_label, "seed": seed,
                                "hash_ok": False, "error": str(exc)})
                print(f"[HASH FAIL] {kl_label} seed {seed}: {exc}", file=sys.stderr)
                return 1
    gate_path = os.path.join(BASE_DIR, "gate_results.json")
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump({"all_passed": True, "hash_gates": results,
                   "expected_warm_start_parameter_sha256": expected_sha,
                   "mode": "gates_only"}, f, indent=2)
    print("Gates complete. all_passed=True")
    return 0


def main() -> int:
    if "--gates-only" in sys.argv:
        return run_gates_only()
    contract = prepare_warm_start_and_contract()
    enforce_runtime_match(contract)

    t0 = time.time()
    n_runs = len(NEW_ARMS) * len(SEEDS)
    print(f"Starting dose-response training for seeds {SEEDS} "
          f"({len(NEW_ARMS)} new arms × {len(SEEDS)} seeds = {n_runs} trajectories)…")
    results = []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        futs = {pool.submit(train_seed_all_arms, s): s for s in SEEDS}
        for fut in as_completed(futs):
            seed = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                print(f"[FAIL] seed {seed}: {exc}", file=sys.stderr)
                results.append({"seed": seed, "success": False,
                                "stage": "exception", "error": str(exc)})

    results.sort(key=lambda r: r["seed"])
    train_ok = all(r.get("success") for r in results)
    if not train_ok:
        gate_path = os.path.join(BASE_DIR, "gate_results.json")
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump({"all_passed": False, "per_seed": results,
                       "expected_warm_start_parameter_sha256":
                       contract["expected_warm_start_parameter_sha256"],
                       "stage": "training"}, f, indent=2)
        print("STOP: training failed.", file=sys.stderr)
        return 1

    expected_sha = contract["expected_warm_start_parameter_sha256"]
    print("\nRunning sequential iter-0 placement spot-checks…")
    placement_results = []
    for kl_coef, kl_label in NEW_ARMS:
        for seed in SEEDS:
            try:
                path = eval_iter0_gate(seed, kl_label)
                data = json.load(open(path))
                avg = data["metrics"]["avg_placement"]
                placement_results.append({
                    "kl_label": kl_label, "seed": seed, "success": True,
                    "iter0_avg": avg,
                })
                print(f"[GATE OK] {kl_label} seed {seed}: iter0 avg={avg:.3f}")
            except Exception as exc:
                placement_results.append({
                    "kl_label": kl_label, "seed": seed, "success": False,
                    "error": str(exc),
                })

    all_ok = all(r.get("success") for r in placement_results)
    elapsed = time.time() - t0
    print(f"\nTraining + gates finished in {elapsed:.1f}s. Success={all_ok}")
    for r in results:
        print(f"  seed {r['seed']}: {r}")

    gate_path = os.path.join(BASE_DIR, "gate_results.json")
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump({"all_passed": all_ok, "per_seed": results,
                   "placement_spot_checks": placement_results,
                   "expected_warm_start_parameter_sha256": expected_sha}, f, indent=2)

    if not all_ok:
        print("STOP: reproducibility gate failed — experiment aborted.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
