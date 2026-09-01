"""Shadow control-equivalence gate for Experiment 6.

Verifies that fixed β=0.03 training with the current ``train_ppo.py`` produces
bit-identical parameter hashes to Experiment 5's β=0.03 seed-0 checkpoints.

    python scripts/shadow_control_equivalence.py

On success writes ``results/ppo_schedule_v1/control_code_equivalence.json`` with
``control_code_equivalence_passed: true`` and reuses the Experiment 5 control.

On failure: hard stop — retrain both control and treatment under current code.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.experiment_contract import (ContractViolation, enforce_runtime_match,
                                    load_contract, verify_warm_start)
from ml.model_fingerprint import checkpoint_parameter_sha256

BASE_DIR = "results/ppo_schedule_v1"
DOSE_DIR = "results/ppo_dose_v1"
WARM_START = os.path.join(BASE_DIR, "warm_start.pt")
CONTRACT_PATH = os.path.join(BASE_DIR, "contract.json")
DOSE_WARM = os.path.join(DOSE_DIR, "warm_start.pt")
CONTROL_LABEL = "beta003"
SHADOW_DIR = os.path.join(BASE_DIR, "shadow_control_equiv")
EQUIV_PATH = os.path.join(BASE_DIR, "control_code_equivalence.json")
SHADOW_SEED = 0
KL_COEF = 0.03
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


def reference_ckpt(iteration: int) -> str:
    return os.path.join(DOSE_DIR, CONTROL_LABEL, f"seed_{SHADOW_SEED}",
                         "checkpoints", f"iter_{iteration:03d}.pt")


def shadow_ckpt(iteration: int) -> str:
    return os.path.join(SHADOW_DIR, f"seed_{SHADOW_SEED}", "checkpoints",
                         f"iter_{iteration:03d}.pt")


def run_shadow_training(expected_sha: str) -> int:
    out = os.path.join(SHADOW_DIR, f"seed_{SHADOW_SEED}")
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
        "--seed", str(SHADOW_SEED),
        "--shaping", "1.0",
        "--shaping-horizon", str(SHAPING_HORIZON),
        "--from-bc", WARM_START,
        "--expected-warm-start-sha", expected_sha,
        "--out", os.path.join(out, "final.pt"),
        "--save-iters", SAVE_ITERS,
        "--save-dir", ckpt_dir,
        "--diag-log", diag_log,
        "--eval-episodes", "1",
        "--kl-coef", str(KL_COEF),
        "--anchor", WARM_START,
    ]
    meta = {
        "training_seed": SHADOW_SEED,
        "kl_coef": KL_COEF,
        "kl_mode": "fixed",
        "purpose": "control_code_equivalence_shadow",
        "reference": os.path.join(DOSE_DIR, CONTROL_LABEL, f"seed_{SHADOW_SEED}"),
    }
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Shadow fixed β={KL_COEF} seed {SHADOW_SEED} training…")
    t0 = time.time()
    with open(train_log, "w") as f_log:
        res = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT,
                             text=True, env=single_thread_env())
    print(f"Shadow training finished in {time.time() - t0:.1f}s rc={res.returncode}")
    return res.returncode


def verify_checkpoint_hashes() -> tuple[bool, list]:
    rows = []
    all_match = True
    for it in CHECKPOINTS:
        ref = reference_ckpt(it)
        shd = shadow_ckpt(it)
        if not os.path.isfile(ref):
            raise FileNotFoundError(f"reference checkpoint missing: {ref}")
        if not os.path.isfile(shd):
            raise FileNotFoundError(f"shadow checkpoint missing: {shd}")
        ref_sha = checkpoint_parameter_sha256(ref)
        shd_sha = checkpoint_parameter_sha256(shd)
        match = ref_sha == shd_sha
        rows.append({
            "iteration": it,
            "reference_path": ref,
            "shadow_path": shd,
            "reference_parameter_sha256": ref_sha,
            "shadow_parameter_sha256": shd_sha,
            "match": match,
        })
        status = "OK" if match else "MISMATCH"
        print(f"  iter {it:3d}: {status}  ref={ref_sha[:16]}…  shd={shd_sha[:16]}…")
        if not match:
            all_match = False
    return all_match, rows


def load_existing_pass() -> bool:
    if not os.path.isfile(EQUIV_PATH):
        return False
    data = json.load(open(EQUIV_PATH))
    return data.get("control_code_equivalence_passed") is True


def ensure_contract_and_warm_start() -> str:
    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.path.isfile(WARM_START) and os.path.isfile(DOSE_WARM):
        import shutil
        shutil.copy2(DOSE_WARM, WARM_START)
    if not os.path.isfile(WARM_START):
        raise SystemExit(f"warm start missing: {WARM_START}")

    if not os.path.isfile(CONTRACT_PATH):
        from ml.experiment_contract import build_contract, save_contract
        from ml.kl_schedule import EXPERIMENT_6_KL_SCHEDULE, schedule_table
        contract = build_contract(WARM_START, kl_coef_values=[0.03])
        contract["experiment"] = "ppo_schedule_v1"
        contract["control_code_equivalence"] = {
            "shadow_seed": SHADOW_SEED,
            "shadow_kl_coef": KL_COEF,
            "checkpoints_compared": list(CHECKPOINTS),
        }
        save_contract(CONTRACT_PATH, contract)

    contract = load_contract(CONTRACT_PATH)
    enforce_runtime_match(contract)
    return contract["expected_warm_start_parameter_sha256"]


def main() -> int:
    if "--force" not in sys.argv and load_existing_pass():
        print(f"Control equivalence already passed -> {EQUIV_PATH}")
        return 0

    expected_sha = ensure_contract_and_warm_start()
    verify_warm_start(WARM_START, expected_sha)

    rc = run_shadow_training(expected_sha)
    if rc != 0:
        result = {
            "control_code_equivalence_passed": False,
            "stage": "shadow_training",
            "returncode": rc,
            "action": "STOP — retrain both control and treatment under current code",
        }
        with open(EQUIV_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return 1

    print("\nComparing parameter SHA256 at checkpoints vs Experiment 5 β=0.03 seed 0:")
    passed, comparisons = verify_checkpoint_hashes()
    result = {
        "control_code_equivalence_passed": passed,
        "shadow_seed": SHADOW_SEED,
        "kl_coef": KL_COEF,
        "reference_run": os.path.join(DOSE_DIR, CONTROL_LABEL, f"seed_{SHADOW_SEED}"),
        "shadow_run": os.path.join(SHADOW_DIR, f"seed_{SHADOW_SEED}"),
        "checkpoints_compared": list(CHECKPOINTS),
        "comparisons": comparisons,
        "action_on_pass": "Reuse Experiment 5 four-seed β=0.03 control",
        "action_on_fail": (
            "STOP — retrain both control and treatment under current train_ppo.py"
        ),
    }
    with open(EQUIV_PATH, "w") as f:
        json.dump(result, f, indent=2)

    if passed:
        print(f"\ncontrol_code_equivalence_passed=True -> {EQUIV_PATH}")
        return 0
    print("\nCONTROL CODE EQUIVALENCE FAILED — do not reuse Experiment 5 control.",
          file=sys.stderr)
    print("Retrain both control (β=0.03) and treatment under current code.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
