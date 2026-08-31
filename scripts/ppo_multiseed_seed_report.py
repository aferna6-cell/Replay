"""Assemble one Experiment 3 seed's artifacts from that seed's eval/drift JSON.

    python scripts/ppo_multiseed_seed_report.py --seed 1

Reads ``results/ppo_multiseed_v1/seed_{S}/`` DEV evaluations, policy drift,
action-category drift, and ``train_diag.jsonl``. Writes learning_curve.json,
paired_results.json (the nine pre-specified pairs), and rl_signal.json.
Does not touch Experiment 2's ``results/ppo_budget_v1/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.multiseed_analysis import (  # noqa: E402
    EPISODES_PER_ITER, MIXED_FIELD, MULTI_DIR, PRIMARY_ITERS,
    assert_eval_seeds_match_experiment2, assert_warmstart_hash,
    classify_ushape, episodes, load_dev_result, load_json,
    load_within_seed_paired, pair_key, rl_block_means, seed_dir, write_json,
)


def assemble_seed(seed: int, directory: str | None = None) -> dict:
    directory = directory or seed_dir(seed)
    if seed == 0:
        raise ValueError("seed 0 is the Experiment 2 reference — do not "
                         "reassemble or overwrite results/ppo_budget_v1/")

    greedy = {it: load_dev_result(directory, it, "greedy")
              for it in PRIMARY_ITERS}
    mixed = {it: load_dev_result(directory, it, MIXED_FIELD)
             for it in PRIMARY_ITERS}
    for blob in greedy.values():
        assert_eval_seeds_match_experiment2(blob, "greedy")
    for blob in mixed.values():
        assert_eval_seeds_match_experiment2(blob, MIXED_FIELD)

    drift_blob = load_json(os.path.join(directory, "policy_drift.json"))
    cats_blob = load_json(os.path.join(directory, "action_category_drift.json"))
    drift = {r["checkpoint"]: r for r in drift_blob["checkpoints"]}
    cats = {c["checkpoint"]: c for c in cats_blob["checkpoints"]}
    diag = [json.loads(l) for l in open(os.path.join(directory, "train_diag.jsonl"))]

    curve = []
    for it in PRIMARY_ITERS:
        d = drift[f"iter_{it:03d}.pt"]
        g, m = greedy[it]["metrics"], mixed[it]["metrics"]
        ce = cats[f"iter_{it:03d}.pt"]["vs_expert"]
        if it == 0:
            assert_warmstart_hash(d["parameter_sha256"])
        curve.append({
            "iteration": it, "cumulative_episodes": episodes(it),
            "training_seed": seed,
            "greedy_avg": g["avg_placement"],
            "greedy_ci95": greedy[it]["avg_placement_ci95"],
            "greedy_median": g["median_placement"],
            "greedy_std": g["std_placement"],
            "greedy_top4": g["top4_rate"], "greedy_win": g["win_rate"],
            "greedy_placement_counts": g["placement_counts"],
            "mixed_avg": m["avg_placement"],
            "mixed_ci95": mixed[it]["avg_placement_ci95"],
            "mixed_top4": m["top4_rate"], "mixed_win": m["win_rate"],
            "expert_agreement": d["expert_agreement"],
            "warmstart_agreement": d["warmstart_agreement"],
            "kl_from_warmstart": d["kl_from_warmstart_mean"],
            "corpus_entropy": d["entropy_mean"],
            "value_mean": d["value_mean"], "value_std": d["value_std"],
            "parameter_sha256": d["parameter_sha256"],
            "checkpoint_sha256": d["checkpoint_sha256"],
            "expert_disagreement_by_category":
                ce["disagreement_share_by_category"],
            "drift_contribution_by_category": ce["contribution_to_total_drift"],
        })

    paired = load_within_seed_paired(directory)
    placements = {c["iteration"]: c["greedy_avg"] for c in curve}
    shape = classify_ushape(placements, paired)
    blocks = rl_block_means(diag)

    learning = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "evaluation_split": "dev",
        "training_seed": seed,
        "episodes_per_iteration": EPISODES_PER_ITER,
        "primary_iterations": list(PRIMARY_ITERS),
        "greedy_games": greedy[0]["games"],
        "mixed_games": mixed[0]["games"],
        "dev_seed_range_greedy": greedy[0]["seed_range"],
        "dev_seed_range_mixed": mixed[0]["seed_range"],
        "curve": curve,
        "paired_keys": list(paired),
        "ushape": shape,
    }
    write_json(os.path.join(directory, "learning_curve.json"), learning)
    write_json(os.path.join(directory, "paired_results.json"), {
        "training_seed": seed,
        "method": "deterministic paired percentile bootstrap, "
                  "10000 resamples, seed 0",
        "note": "positive difference = first checkpoint places worse "
                "(lower placement is better)",
        "pairs": paired,
    })
    write_json(os.path.join(directory, "rl_signal.json"), {
        "training_seed": seed,
        "definitions": {
            "adv_*": "GAE advantages BEFORE PPO's per-batch normalization",
            "value_explained_variance":
                "1 - Var(returns - value_preds) / Var(returns)",
            "shaping_reward_sum/terminal_reward_sum":
                "the two reward sources separated per iteration",
        },
        "per_iteration": diag,
        "blocks": blocks,
    })
    return learning


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Assemble one Experiment 3 seed")
    p.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--dir", help="override seed directory")
    a = p.parse_args(argv)
    learning = assemble_seed(a.seed, a.dir)
    print(f"seed {a.seed}  source {a.dir or os.path.join(MULTI_DIR, f'seed_{a.seed}')}")
    print(f"{'iter':>5} {'episodes':>9} {'GreedyAvg':>10} {'Expert%':>8} "
          f"{'WarmSt%':>8} {'KL':>7}")
    for c in learning["curve"]:
        print(f"{c['iteration']:>5} {c['cumulative_episodes']:>9} "
              f"{c['greedy_avg']:>10.3f} "
              f"{100 * c['expert_agreement']:>7.1f}% "
              f"{100 * c['warmstart_agreement']:>7.1f}% "
              f"{c['kl_from_warmstart']:>7.4f}")
    print("ushape:", learning["ushape"]["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
