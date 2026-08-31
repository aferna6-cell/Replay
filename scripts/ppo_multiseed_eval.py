"""DEV evaluation + drift sweep for one Experiment 3 PPO seed.

    python scripts/ppo_multiseed_eval.py --seed 1

Runs the exact Experiment 2 DEV protocol on primary checkpoints:
1000 games vs 7× greedy and 500 vs greedy4_random3, seeds
10,550,000–10,550,999. Then scores the frozen 4,440-state drift corpus
(must match fingerprint 2ec217b353bd…). Never touches TEST seeds.
Never writes into results/ppo_budget_v1/.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dev_benchmark import run_dev_benchmark, save_dev_json  # noqa: E402
from ml.benchmark import make_agent  # noqa: E402
from ml.model_fingerprint import checkpoint_fingerprint  # noqa: E402
from ml.multiseed_analysis import (  # noqa: E402
    CORPUS_FINGERPRINT_SHA256, CORPUS_LOBBIES, CORPUS_SEED_BASE,
    DEV_EVAL_BASE, DEV_EVAL_GAMES, MIXED_FIELD, MIXED_GAMES, PRIMARY_ITERS,
    WARM_START_PARAMETER_SHA256, assert_corpus_fingerprint,
    assert_warmstart_hash, seed_dir, write_json,
)
from ml.policy_drift import main as drift_main  # noqa: E402
from ml.seeds import EVAL_SEED_START  # noqa: E402


def _ckpt(directory: str, it: int) -> str:
    return os.path.join(directory, "checkpoints", f"iter_{it:03d}.pt")


def evaluate_seed(seed: int, directory: str | None = None,
                  quiet: bool = False) -> None:
    directory = directory or seed_dir(seed)
    if seed == 0:
        raise ValueError("seed 0 is Experiment 2 — do not re-evaluate it here")
    if DEV_EVAL_BASE <= EVAL_SEED_START <= DEV_EVAL_BASE + DEV_EVAL_GAMES - 1:
        raise RuntimeError("DEV eval range collided with TEST — aborting")

    warm = checkpoint_fingerprint(_ckpt(directory, 0))
    assert_warmstart_hash(warm["parameter_sha256"])
    print(f"seed {seed}: iter0 parameter_sha256 = {warm['parameter_sha256']}")
    print(f"  matches frozen warm start: {warm['parameter_sha256'] == WARM_START_PARAMETER_SHA256}")

    meta = []
    for it in PRIMARY_ITERS:
        path = _ckpt(directory, it)
        fp = checkpoint_fingerprint(path)
        meta.append({"iteration": it, "file": os.path.basename(path),
                     "training_seed": seed, **fp})
        for field, games in (("greedy", DEV_EVAL_GAMES),
                             (MIXED_FIELD, MIXED_GAMES)):
            out = os.path.join(directory, "dev",
                               f"iter{it:03d}_vs_{field}.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            agent = make_agent("policy", path, f"iter{it:03d}")
            print(f"  eval iter {it:3d}  {field}  {games} games  "
                  f"seeds {DEV_EVAL_BASE}-{DEV_EVAL_BASE + games - 1}")
            res = run_dev_benchmark(agent, field, games, DEV_EVAL_BASE,
                                    progress=not quiet)
            # persist fingerprints on the result JSON
            blob_path = out
            save_dev_json(res, blob_path)
            import json
            with open(blob_path, encoding="utf-8") as f:
                blob = json.load(f)
            blob["training_seed"] = seed
            blob["ppo_iteration"] = it
            blob["cumulative_training_episodes"] = it * 16
            blob["parameter_sha256"] = fp["parameter_sha256"]
            blob["checkpoint_sha256"] = fp["checkpoint_sha256"]
            with open(blob_path, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2)
                f.write("\n")
    write_json(os.path.join(directory, "checkpoints.json"), {
        "training_seed": seed,
        "warm_start_parameter_sha256": WARM_START_PARAMETER_SHA256,
        "iter0_matches_warm_start":
            warm["parameter_sha256"] == WARM_START_PARAMETER_SHA256,
        "checkpoints": meta,
    })

    ckpts = [_ckpt(directory, it) for it in PRIMARY_ITERS]
    drift_out = os.path.join(directory, "policy_drift.json")
    cats_out = os.path.join(directory, "action_category_drift.json")
    drift_main([
        "--reference", _ckpt(directory, 0),
        "--checkpoints", *ckpts,
        "--lobbies", str(CORPUS_LOBBIES),
        "--corpus-seed", str(CORPUS_SEED_BASE),
        "--json-out", drift_out,
        "--categories-out", cats_out,
    ])
    import json
    drift = json.load(open(drift_out))
    assert_corpus_fingerprint(drift["corpus"]["fingerprint_sha256"])
    print(f"  corpus fingerprint {drift['corpus']['fingerprint_sha256'][:12]} "
          f"== historical {CORPUS_FINGERPRINT_SHA256[:12]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Experiment 3 DEV eval + drift")
    p.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--dir")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    evaluate_seed(a.seed, a.dir, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
