"""Write Experiment 5 dose-response reproducibility manifest."""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dev_benchmark import field_composition
from ml.experiment_contract import load_contract
from ml.model_fingerprint import checkpoint_fingerprint
from ml.policy_drift import CORPUS_LOBBIES, CORPUS_SEED_BASE

DOSE_DIR = "results/ppo_dose_v1"
MATCHED_DIR = "results/ppo_matched_ab_v1"
CONTRACT_PATH = os.path.join(DOSE_DIR, "contract.json")
SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]

ALL_ARMS = (
    ("beta0", 0.0, MATCHED_DIR),
    ("beta001", 0.01, DOSE_DIR),
    ("beta003", 0.03, DOSE_DIR),
    ("beta01", 0.1, MATCHED_DIR),
)


def main() -> int:
    contract = load_contract(CONTRACT_PATH)
    warm = json.load(open(os.path.join(DOSE_DIR, "warm_start.json")))
    gate_path = os.path.join(DOSE_DIR, "gate_results.json")
    gate = json.load(open(gate_path)) if os.path.isfile(gate_path) else {}

    runs = []
    for kl_label, kl_coef, base in ALL_ARMS:
        for seed in SEEDS:
            run = os.path.join(base, kl_label, f"seed_{seed}")
            ckpts = []
            for it in ITERS:
                path = os.path.join(run, "checkpoints", f"iter_{it:03d}.pt")
                if os.path.isfile(path):
                    ckpts.append({"iteration": it, **checkpoint_fingerprint(path)})
            meta_path = os.path.join(run, "run_meta.json")
            runs.append({
                "kl_label": kl_label,
                "kl_coef": kl_coef,
                "training_seed": seed,
                "source_dir": base,
                "reused_from_4b": base == MATCHED_DIR,
                "checkpoints": ckpts,
                "run_meta": json.load(open(meta_path))
                if os.path.isfile(meta_path) else None,
            })

    manifest = {
        "experiment": "Replay Experiment 5 — KL Anchoring Dose-Response",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": contract.get("question"),
        "single_variable": "kl_coef across {0.0, 0.01, 0.03, 0.1}",
        "contract": contract,
        "warm_start": warm,
        "reproducibility_gates": gate,
        "runs": runs,
        "evaluation": {
            "split": "dev",
            "test_usage": "Benchmark v1 TEST was NOT run.",
            "primary_field": {
                **contract["primary_dev_eval"],
                "composition": field_composition("greedy"),
            },
            "secondary_field": {
                **contract["secondary_dev_eval"],
                "composition": field_composition("greedy4_random3"),
            },
            "drift_corpus": {
                "lobbies": CORPUS_LOBBIES,
                "seed_base": CORPUS_SEED_BASE,
            },
        },
        "artifacts": {
            "contract": CONTRACT_PATH,
            "gate_results": gate_path,
            "aggregate": os.path.join(DOSE_DIR, "aggregate/"),
            "report": "experiments/ppo_dose_v1.md",
            "reused_4b_results": MATCHED_DIR,
        },
    }
    out = os.path.join(DOSE_DIR, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved -> {out}")
    print(f"all_gates_passed={gate.get('all_passed')}")
    return 0 if gate.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
