"""Write Experiment 4b reproducibility manifest (extends contract.json)."""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dev_benchmark import field_composition
from ml.experiment_contract import load_contract
from ml.model_fingerprint import checkpoint_fingerprint
from ml.policy_drift import CORPUS_LOBBIES, CORPUS_SEED_BASE

BASE_DIR = "results/ppo_matched_ab_v1"
CONTRACT_PATH = os.path.join(BASE_DIR, "contract.json")
SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
KL_ARMS = (("beta0", 0.0), ("beta01", 0.1))


def main() -> int:
    contract = load_contract(CONTRACT_PATH)
    warm = json.load(open(os.path.join(BASE_DIR, "warm_start.json")))
    gate = json.load(open(os.path.join(BASE_DIR, "gate_results.json")))

    runs = []
    for kl_label, kl_coef in KL_ARMS:
        for seed in SEEDS:
            run = os.path.join(BASE_DIR, kl_label, f"seed_{seed}")
            ckpts = []
            for it in ITERS:
                path = os.path.join(run, "checkpoints", f"iter_{it:03d}.pt")
                if os.path.isfile(path):
                    ckpts.append({"iteration": it, **checkpoint_fingerprint(path)})
            runs.append({
                "kl_label": kl_label,
                "kl_coef": kl_coef,
                "training_seed": seed,
                "checkpoints": ckpts,
                "run_meta": json.load(open(os.path.join(run, "run_meta.json")))
                if os.path.isfile(os.path.join(run, "run_meta.json")) else None,
            })

    manifest = {
        "experiment": "Replay Experiment 4b — Matched Anchored vs Unanchored PPO",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": contract.get("question") or (
            "Does β=0.1 reduce cross-seed instability and catastrophic policy "
            "drift while preserving placement vs matched β=0.0 control?"),
        "single_variable_within_seed_pair": "kl_coef (0.0 vs 0.1)",
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
            "gate_results": os.path.join(BASE_DIR, "gate_results.json"),
            "aggregate": os.path.join(BASE_DIR, "aggregate/"),
            "report": "experiments/ppo_matched_ab_v1.md",
        },
    }
    out = os.path.join(BASE_DIR, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved -> {out}")
    print(f"all_gates_passed={gate.get('all_passed')}")
    return 0 if gate.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
