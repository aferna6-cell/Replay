"""Per-seed evaluation pipeline for Experiment 3 (multi-seed PPO replication).

For one new PPO training seed (1, 2, or 3), runs the exact same DEV
evaluation Experiment 2 used for every primary checkpoint {0, 40, 80, 160,
320}: 1000 games vs 7x greedy and 500 games vs the greedy4_random3
diagnostic field, both on DEV seed base 10550000 (``ml/seeds.py``
DEV_SEED_START) — never TEST. Then runs the frozen 4,440-state policy-drift
corpus and action-category confusion analysis against those checkpoints,
exactly like Experiment 2's ``ml.policy_drift`` invocation.

Nothing here trains, tunes, or selects a checkpoint — it only evaluates
checkpoints that already exist on disk.

    python -m scripts.ppo_multiseed_eval --seed 1
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.seeds import DEV_SEED_START  # noqa: E402

ITERS = [0, 40, 80, 160, 320]
GREEDY_GAMES = 1000
MIXED_GAMES = 500


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True,
                   help="PPO training seed whose checkpoints to evaluate "
                        "(1, 2, or 3 for Experiment 3)")
    p.add_argument("--base-dir", default=None,
                   help="override results/ppo_multiseed_v1/seed_<seed>")
    a = p.parse_args(argv)

    base = a.base_dir or f"results/ppo_multiseed_v1/seed_{a.seed}"
    ckpt_dir = os.path.join(base, "checkpoints")
    dev_dir = os.path.join(base, "dev")
    os.makedirs(dev_dir, exist_ok=True)

    for it in ITERS:
        ckpt = os.path.join(ckpt_dir, f"iter_{it:03d}.pt")
        if not os.path.isfile(ckpt):
            raise SystemExit(f"missing checkpoint: {ckpt}")
        run(["python3", "-m", "ml.dev_benchmark", "--agent", "policy",
             "--checkpoint", ckpt, "--name", f"PPO_seed{a.seed}_iter{it}",
             "--games", str(GREEDY_GAMES), "--field", "greedy",
             "--seed", str(DEV_SEED_START), "--quiet",
             "--json-out", os.path.join(dev_dir, f"iter{it:03d}_vs_greedy.json")])
        run(["python3", "-m", "ml.dev_benchmark", "--agent", "policy",
             "--checkpoint", ckpt, "--name", f"PPO_seed{a.seed}_iter{it}",
             "--games", str(MIXED_GAMES), "--field", "greedy4_random3",
             "--seed", str(DEV_SEED_START), "--quiet",
             "--json-out", os.path.join(
                 dev_dir, f"iter{it:03d}_vs_greedy4_random3.json")])

    checkpoints = [os.path.join(ckpt_dir, f"iter_{it:03d}.pt") for it in ITERS]
    run(["python3", "-m", "ml.policy_drift",
         "--reference", checkpoints[0],
         "--checkpoints", *checkpoints,
         "--json-out", os.path.join(base, "policy_drift.json"),
         "--categories-out", os.path.join(base, "action_category_drift.json")])

    print(f"\nSeed {a.seed} DEV evaluation + drift diagnostics complete -> {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
