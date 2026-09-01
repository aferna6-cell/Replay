"""Write Experiment 6 reproducibility manifest."""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dev_benchmark import field_composition
from ml.experiment_contract import load_contract
from ml.kl_schedule import EXPERIMENT_6_KL_SCHEDULE, schedule_table
from ml.model_fingerprint import checkpoint_fingerprint

SCHEDULE_DIR = "results/ppo_schedule_v1"
DOSE_DIR = "results/ppo_dose_v1"
CONTRACT_PATH = os.path.join(SCHEDULE_DIR, "contract.json")
SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]


def main() -> int:
    contract = load_contract(CONTRACT_PATH)
    warm = json.load(open(os.path.join(SCHEDULE_DIR, "warm_start.json")))
    equiv_path = os.path.join(SCHEDULE_DIR, "control_code_equivalence.json")
    equiv = (json.load(open(equiv_path))
             if os.path.isfile(equiv_path) else None)
    gate_path = os.path.join(SCHEDULE_DIR, "gate_results.json")
    gate = json.load(open(gate_path)) if os.path.isfile(gate_path) else {}

    runs = []
    for label, base, reused in (
        ("beta003", DOSE_DIR, True),
        ("beta_sched", SCHEDULE_DIR, False),
    ):
        for seed in SEEDS:
            run = os.path.join(base, label, f"seed_{seed}")
            ckpts = []
            for it in ITERS:
                path = os.path.join(run, "checkpoints", f"iter_{it:03d}.pt")
                if os.path.isfile(path):
                    ckpts.append({"iteration": it, **checkpoint_fingerprint(path)})
            meta_path = os.path.join(run, "run_meta.json")
            runs.append({
                "arm_label": label,
                "training_seed": seed,
                "source_dir": base,
                "reused_from_dose_v1": reused,
                "checkpoints": ckpts,
                "run_meta": json.load(open(meta_path))
                if os.path.isfile(meta_path) else None,
            })

    manifest = {
        "experiment": "Replay Experiment 6 — Scheduled KL Anchoring",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": contract.get("question"),
        "kl_schedule": EXPERIMENT_6_KL_SCHEDULE,
        "schedule_at_checkpoints": schedule_table(EXPERIMENT_6_KL_SCHEDULE, tuple(ITERS)),
        "contract": contract,
        "warm_start": warm,
        "control_code_equivalence": equiv,
        "reproducibility_gates": gate,
        "runs": runs,
        "evaluation": {
            "split": "dev",
            "test_usage": "Benchmark v1 TEST was NOT run.",
            "primary_field": {
                **contract["primary_dev_eval"],
                "composition": field_composition("greedy"),
            },
        },
        "success_criteria": contract.get("success_criteria"),
        "hard_stop": contract.get("hard_stop"),
        "artifacts": {
            "contract": CONTRACT_PATH,
            "report": "experiments/ppo_schedule_v1.md",
            "control_source": os.path.join(DOSE_DIR, "beta003/"),
        },
    }
    out = os.path.join(SCHEDULE_DIR, "manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved -> {out}")
    return 0 if gate.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
