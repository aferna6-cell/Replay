"""Evaluate the Experiment 3 per-seed PPO checkpoints on the DEV split.

One invocation per training seed (or several at once — the frozen drift
corpus is then built only once and shared, which is safe because it is
deterministic given its seeds):

    python scripts/ppo_multiseed_eval.py --seeds 1 2 3

For every primary checkpoint (iterations 0, 40, 80, 160, 320) it runs the
*identical* evaluation Experiment 2 ran:

  * primary field   — 1000 DEV games vs 7x greedy, seeds 10,550,000-10,550,999
  * secondary field — 500 DEV games vs greedy4_random3, seeds 10,550,000-…499

Both use exactly the same seeds for every checkpoint and every training seed,
so placements pair game-by-game across the whole experiment.

Then it scores the frozen 4,440-state diagnostic corpus (fingerprint
2ec217b353bd, unchanged since Experiment 1) for policy drift and
action-category drift, writing the same JSON schemas Experiment 2 used.

DEV ONLY. Benchmark v1 TEST is never run, read, or referenced here; the seed
range is validated against the reserved DEV interval by ml/dev_benchmark.py.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.benchmark import make_agent                               # noqa: E402
from ml.dev_benchmark import run_dev_benchmark, save_dev_json     # noqa: E402
from ml.model_fingerprint import checkpoint_fingerprint           # noqa: E402
from ml.replication import PRIMARY_ITERS, episodes                # noqa: E402
from ml.seeds import DEV_SEED_START                               # noqa: E402

ROOT = "results/ppo_multiseed_v1"
WARMSTART_PARAMETER_SHA256 = (
    "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b")
# The frozen diagnostic corpus shared by Experiments 1, 2 and 3.
CORPUS_FINGERPRINT_PREFIX = "2ec217b353bd"
GREEDY_GAMES = 1000
MIXED_GAMES = 500
ALL_CKPT = [0, 10, 20, 40, 80, 120, 160, 240, 320]


def seed_dir(seed: int) -> str:
    return f"{ROOT}/seed_{seed}"


def ckpt_path(seed: int, iteration: int) -> str:
    return f"{seed_dir(seed)}/checkpoints/iter_{iteration:03d}.pt"


def checkpoint_metadata(seed: int) -> Dict:
    """Fingerprints for every saved checkpoint of one training seed, with the
    warm-start reproduction check the protocol requires at iteration 0."""
    rows = []
    for it in ALL_CKPT:
        path = ckpt_path(seed, it)
        if not os.path.isfile(path):
            continue
        rows.append({"iteration": it, "cumulative_episodes": episodes(it),
                     "primary": it in PRIMARY_ITERS,
                     "file": os.path.basename(path),
                     **checkpoint_fingerprint(path)})
    iter0 = next(r for r in rows if r["iteration"] == 0)
    return {
        "training_seed": seed,
        "warm_start_parameter_sha256": WARMSTART_PARAMETER_SHA256,
        "iter0_parameter_sha256": iter0["parameter_sha256"],
        "iter0_matches_frozen_warm_start":
            iter0["parameter_sha256"] == WARMSTART_PARAMETER_SHA256,
        "checkpoints": rows,
    }


def run_dev_evals(seed: int, iterations: List[int]) -> None:
    for it in iterations:
        path = ckpt_path(seed, it)
        for field, games in (("greedy", GREEDY_GAMES),
                             ("greedy4_random3", MIXED_GAMES)):
            out = f"{seed_dir(seed)}/dev/iter{it:03d}_vs_{field}.json"
            agent = make_agent("policy", path, f"s{seed}_iter{it:03d}")
            res = run_dev_benchmark(agent, field, games, DEV_SEED_START)
            save_dev_json(res, out)
            print(f"  seed {seed} iter{it:>3} {field:<15} "
                  f"avg {res.metrics['avg_placement']:.3f}  -> {out}")


def run_drift(seed: int, iterations: List[int], tensors, states_n: int,
              fingerprint: str) -> None:
    from ml.action_categories import confusion, top_transitions
    from ml.policy_drift import (CORPUS_LOBBIES, CORPUS_SEED_BASE,
                                 drift_metrics, policy_outputs)
    from ml.policy_net import load_policy

    ref_path = ckpt_path(seed, 0)
    ref = load_policy(ref_path)
    logits_ref, _ = policy_outputs(ref, tensors)
    expert_acts = [int(x) for x in tensors[5].tolist()]
    ref_acts = [int(x) for x in logits_ref.argmax(dim=-1).tolist()]

    rows, categories = [], []
    for it in iterations:
        path = ckpt_path(seed, it)
        net = load_policy(path)
        logits_k, values_k = policy_outputs(net, tensors)
        rows.append({"checkpoint": os.path.basename(path),
                     "iteration": it, "cumulative_episodes": episodes(it),
                     **checkpoint_fingerprint(path),
                     **drift_metrics(logits_k, values_k, logits_ref, tensors)})
        k_acts = [int(x) for x in logits_k.argmax(dim=-1).tolist()]
        vs_expert = confusion(expert_acts, k_acts)
        vs_warm = confusion(ref_acts, k_acts)
        categories.append({
            "checkpoint": os.path.basename(path), "iteration": it,
            "cumulative_episodes": episodes(it),
            "vs_expert": {**vs_expert,
                          "top_transitions": top_transitions(vs_expert)},
            "vs_warmstart": {**vs_warm,
                             "top_transitions": top_transitions(vs_warm)}})
        print(f"  seed {seed} iter{it:>3} expert "
              f"{rows[-1]['expert_agreement']:.3f}  warmstart "
              f"{rows[-1]['warmstart_agreement']:.3f}  KL "
              f"{rows[-1]['kl_from_warmstart_mean']:.4f}")

    corpus = {"lobbies": CORPUS_LOBBIES, "seed_base": CORPUS_SEED_BASE,
              "states": states_n, "fingerprint_sha256": fingerprint,
              "source": "greedy seat-0 trajectories via ml.bc.collect",
              "split": "dev (diagnostic sub-range; never TEST seeds)",
              "frozen_since": "Experiment 1; identical corpus in 2 and 3",
              "expected_fingerprint_prefix": CORPUS_FINGERPRINT_PREFIX}
    reference = {"checkpoint": os.path.basename(ref_path),
                 **checkpoint_fingerprint(ref_path)}
    with open(f"{seed_dir(seed)}/policy_drift.json", "w",
              encoding="utf-8") as f:
        json.dump({"training_seed": seed, "corpus": corpus,
                   "reference": reference,
                   "kl_definition": "mean over states of KL(pi_reference || "
                                    "pi_k) on legal-action-masked softmax "
                                    "distributions",
                   "checkpoints": rows}, f, indent=2)
    with open(f"{seed_dir(seed)}/action_category_drift.json", "w",
              encoding="utf-8") as f:
        json.dump({"training_seed": seed, "corpus": corpus,
                   "reference": reference,
                   "categories": ("action index -> decision category, "
                                  "derived from hsbg_coach.bg_env"),
                   "checkpoints": categories}, f, indent=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ppo_multiseed_eval",
        description="DEV evaluation + drift scoring for Experiment 3 "
                    "PPO training seeds (never touches Benchmark v1 TEST)")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--iters", type=int, nargs="+", default=PRIMARY_ITERS)
    p.add_argument("--skip-dev", action="store_true")
    p.add_argument("--skip-drift", action="store_true")
    a = p.parse_args(argv)

    for seed in a.seeds:
        meta = checkpoint_metadata(seed)
        path = f"{seed_dir(seed)}/checkpoint_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if not meta["iter0_matches_frozen_warm_start"]:
            raise SystemExit(
                f"seed {seed}: iteration-0 parameter hash "
                f"{meta['iter0_parameter_sha256']} != the frozen warm start "
                f"{WARMSTART_PARAMETER_SHA256} — refusing to evaluate")
        print(f"seed {seed}: iteration-0 == frozen warm start OK -> {path}")

    if not a.skip_dev:
        for seed in a.seeds:
            run_dev_evals(seed, a.iters)

    if not a.skip_drift:
        from ml.policy_drift import build_corpus, corpus_tensors
        print("Building the frozen diagnostic corpus (100 greedy lobbies, "
              "DEV sub-range)…")
        states, fingerprint = build_corpus()
        print(f"  {len(states)} states, fingerprint {fingerprint[:12]}")
        if not fingerprint.startswith(CORPUS_FINGERPRINT_PREFIX):
            raise SystemExit(
                f"drift corpus fingerprint {fingerprint[:12]} != the frozen "
                f"{CORPUS_FINGERPRINT_PREFIX} used by Experiments 1-2 — "
                f"refusing to score a different corpus")
        tensors = corpus_tensors(states)
        for seed in a.seeds:
            run_drift(seed, a.iters, tensors, len(states), fingerprint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
