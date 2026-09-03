"""Experiment 6 — scheduled KL anchoring (β=0.03→0.01) vs fixed β=0.03.

Trains 4 scheduled trajectories (seeds 0–3). Fixed β=0.03 control is reused
from ``results/ppo_dose_v1/beta003/``.

Frozen schedule (1-based iterations):
    1–160:   β = 0.030
    161–320: linear 0.030 → 0.010

    python scripts/train_schedule_ab.py
    python scripts/eval_schedule_ab.py
    python scripts/ppo_schedule_report.py
    python scripts/ppo_schedule_manifest.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.experiment_contract import (ContractViolation, build_contract,
                                    enforce_runtime_match, load_contract,
                                    save_contract, verify_checkpoint_parameter_sha256,
                                    verify_identical_placements, verify_warm_start)
from ml.kl_schedule import EXPERIMENT_6_KL_SCHEDULE, schedule_table
from ml.model_fingerprint import checkpoint_fingerprint

BASE_DIR = "results/ppo_schedule_v1"
DOSE_DIR = "results/ppo_dose_v1"
WARM_START = os.path.join(BASE_DIR, "warm_start.pt")
CONTRACT_PATH = os.path.join(BASE_DIR, "contract.json")
DOSE_WARM = os.path.join(DOSE_DIR, "warm_start.pt")
CONTROL_LABEL = "beta003"
SCHEDULE_LABEL = "beta_sched"
KL_SCHEDULE = EXPERIMENT_6_KL_SCHEDULE
SEEDS = [0, 1, 2, 3]
ITERS = 320
EPISODES = 16
SHAPING_HORIZON = 40
SAVE_ITERS = "0,40,80,160,320"
CHECKPOINTS = (0, 40, 80, 160, 320)


def single_thread_env():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def run_dir(arm_label: str, seed: int) -> str:
    return os.path.join(BASE_DIR, arm_label, f"seed_{seed}")


def control_run_dir(seed: int) -> str:
    return os.path.join(DOSE_DIR, CONTROL_LABEL, f"seed_{seed}")


def prepare_warm_start_and_contract() -> dict:
    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.path.isfile(DOSE_WARM):
        raise SystemExit(f"Experiment 5 warm start missing: {DOSE_WARM}")
    if not os.path.isfile(WARM_START):
        shutil.copy2(DOSE_WARM, WARM_START)

    dose = load_contract(os.path.join(DOSE_DIR, "contract.json"))
    contract = build_contract(WARM_START, kl_coef_values=[0.03])
    contract["parent_experiment"] = "ppo_dose_v1"
    contract["parent_contract_path"] = os.path.join(DOSE_DIR, "contract.json")
    contract["experiment"] = "ppo_schedule_v1"
    contract["question"] = (
        "Can delayed relaxation of KL anchoring (β=0.03 through iter 160, "
        "then linear to β=0.01 by iter 320) preserve fixed-β=0.03 stability "
        "while reliably beating the BC warm start (~6.550)?"
    )
    contract["arms"] = {
        "control": {
            "label": CONTROL_LABEL,
            "kl_mode": "fixed",
            "kl_coef": 0.03,
            "source": DOSE_DIR,
            "reused": True,
        },
        "treatment": {
            "label": SCHEDULE_LABEL,
            "kl_mode": "scheduled",
            "kl_schedule": KL_SCHEDULE,
            "schedule_table": schedule_table(KL_SCHEDULE, CHECKPOINTS),
        },
    }
    contract["success_criteria"] = {
        "bc_baseline": 6.550,
        "max_mean_delta_vs_bc": -0.02,
        "max_scheduled_cross_seed_mean": 6.530,
        "min_seeds_beating_bc": 3,
        "seed_beat_threshold": -0.01,
        "max_worst_seed_delta_vs_bc": 0.05,
        "max_std_ratio_vs_control": 1.5,
    }
    contract["control_code_equivalence"] = {
        "required_before_training": True,
        "shadow_seed": 0,
        "shadow_kl_coef": 0.03,
        "checkpoints_compared": list(CHECKPOINTS),
        "artifact": os.path.join(BASE_DIR, "control_code_equivalence.json"),
    }
    contract["hard_stop"] = (
        "Last PPO experiment on current simulator. If schedule fails, "
        "pivot to Simulator Fidelity Phase 2."
    )
    save_contract(CONTRACT_PATH, contract)

    warm_record = {"path": WARM_START, **checkpoint_fingerprint(WARM_START)}
    with open(os.path.join(BASE_DIR, "warm_start.json"), "w") as f:
        json.dump(warm_record, f, indent=2)

    expected = contract["expected_warm_start_parameter_sha256"]
    if expected != dose["expected_warm_start_parameter_sha256"]:
        raise ContractViolation("warm-start hash differs from Experiment 5")
    print(f"Contract saved -> {CONTRACT_PATH}")
    print(f"KL schedule: {KL_SCHEDULE}")
    print(f"expected_warm_start_parameter_sha256={expected}")
    return contract


def train_scheduled(seed: int, expected_sha: str) -> int:
    out = run_dir(SCHEDULE_LABEL, seed)
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
        "--kl-schedule", KL_SCHEDULE,
        "--anchor", WARM_START,
    ]

    meta = {
        "training_seed": seed,
        "arm": SCHEDULE_LABEL,
        "kl_mode": "scheduled",
        "kl_schedule": KL_SCHEDULE,
        "warm_start_sha256": expected_sha,
    }
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    with open(train_log, "w") as f_log:
        res = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT,
                             text=True, env=single_thread_env())
    print(f"[{SCHEDULE_LABEL} seed {seed}] training {time.time() - t0:.1f}s "
          f"rc={res.returncode}")
    return res.returncode


def verify_seed_pair(seed: int, expected_sha: str) -> None:
    sched_ckpt = os.path.join(run_dir(SCHEDULE_LABEL, seed),
                              "checkpoints", "iter_000.pt")
    ctrl_ckpt = os.path.join(control_run_dir(seed),
                             "checkpoints", "iter_000.pt")
    verify_checkpoint_parameter_sha256(sched_ckpt, expected_sha,
                                       label=f"{SCHEDULE_LABEL} iter0")
    verify_checkpoint_parameter_sha256(ctrl_ckpt, expected_sha,
                                       label=f"{CONTROL_LABEL} iter0")
    ctrl_dev = os.path.join(control_run_dir(seed), "dev",
                            "iter000_vs_greedy.json")
    sched_dev = os.path.join(run_dir(SCHEDULE_LABEL, seed), "dev",
                             "iter000_vs_greedy.json")
    if os.path.isfile(ctrl_dev) and os.path.isfile(sched_dev):
        verify_identical_placements(sched_dev, ctrl_dev,
                                    label=f"seed {seed} iter0 greedy")
    print(f"[GATE OK] seed {seed}: iter_000 hash + warm start verified")


def train_seed(seed: int) -> dict:
    contract = load_contract(CONTRACT_PATH)
    enforce_runtime_match(contract)
    expected_sha = contract["expected_warm_start_parameter_sha256"]
    verify_warm_start(WARM_START, expected_sha)

    rc = train_scheduled(seed, expected_sha)
    if rc != 0:
        return {"seed": seed, "success": False, "stage": "train", "returncode": rc}
    try:
        verify_checkpoint_parameter_sha256(
            os.path.join(run_dir(SCHEDULE_LABEL, seed), "checkpoints",
                         "iter_000.pt"),
            expected_sha, label=f"{SCHEDULE_LABEL} iter0")
    except ContractViolation as exc:
        return {"seed": seed, "success": False, "stage": "gate_hash",
                "error": str(exc)}
    return {"seed": seed, "success": True, "stage": "train"}


def run_control_equivalence_gate() -> bool:
    """Require shadow fixed-β seed-0 hashes to match Experiment 5 control."""
    equiv_path = os.path.join(BASE_DIR, "control_code_equivalence.json")
    if os.path.isfile(equiv_path):
        data = json.load(open(equiv_path))
        if data.get("control_code_equivalence_passed"):
            print(f"Control code equivalence already passed -> {equiv_path}")
            return True

    print("Running shadow fixed β=0.03 seed-0 control-equivalence gate…")
    res = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      "shadow_control_equivalence.py")],
        env=single_thread_env())
    if res.returncode != 0:
        print("STOP: control code equivalence gate failed.", file=sys.stderr)
        return False
    return True


def main() -> int:
    contract = prepare_warm_start_and_contract()
    enforce_runtime_match(contract)

    if not run_control_equivalence_gate():
        with open(os.path.join(BASE_DIR, "gate_results.json"), "w") as f:
            json.dump({"all_passed": False,
                       "stage": "control_code_equivalence_failed",
                       "control_code_equivalence_passed": False}, f, indent=2)
        return 1

    t0 = time.time()
    print(f"Training scheduled arm for seeds {SEEDS} "
          f"(control β=0.03 reused from {DOSE_DIR})…")
    results = []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        futs = {pool.submit(train_seed, s): s for s in SEEDS}
        for fut in as_completed(futs):
            seed = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"seed": seed, "success": False,
                                "stage": "exception", "error": str(exc)})

    results.sort(key=lambda r: r["seed"])
    if not all(r.get("success") for r in results):
        with open(os.path.join(BASE_DIR, "gate_results.json"), "w") as f:
            json.dump({"all_passed": False, "per_seed": results}, f, indent=2)
        print("STOP: training failed.", file=sys.stderr)
        return 1

    expected_sha = contract["expected_warm_start_parameter_sha256"]
    gate_results = []
    for seed in SEEDS:
        try:
            verify_seed_pair(seed, expected_sha)
            gate_results.append({"seed": seed, "success": True})
        except ContractViolation as exc:
            gate_results.append({"seed": seed, "success": False,
                                 "error": str(exc)})

    all_ok = all(r["success"] for r in gate_results)
    equiv = json.load(open(os.path.join(BASE_DIR,
                                         "control_code_equivalence.json")))
    with open(os.path.join(BASE_DIR, "gate_results.json"), "w") as f:
        json.dump({"all_passed": all_ok, "per_seed": results,
                   "hash_gates": gate_results,
                   "control_code_equivalence_passed":
                   equiv.get("control_code_equivalence_passed"),
                   "expected_warm_start_parameter_sha256": expected_sha,
                   "kl_schedule": KL_SCHEDULE}, f, indent=2)

    print(f"\nTraining + gates finished in {time.time() - t0:.1f}s. "
          f"Success={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
