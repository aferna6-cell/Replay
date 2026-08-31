"""Build per-seed Experiment 3 artifacts: manifest, learning_curve.json,
rl_signal.json.

Reads only committed result JSON for one new PPO training seed (already
evaluated by ``scripts/ppo_multiseed_eval.py``): the DEV evaluations, the
policy-drift / action-category outputs, and the per-iteration training
diagnostics. Verifies the reproduction gate (iteration-0 checkpoint's
parameter hash must equal the frozen BC+DAgger warm start) and writes
nothing that isn't already recorded elsewhere.

    python -m scripts.ppo_multiseed_seed_artifacts --seed 1 \
        --warm-start-parameter-sha256 094417bd...
"""

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_fingerprint import checkpoint_fingerprint  # noqa: E402

ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16


def episodes(it):
    return it * EPISODES_PER_ITER


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warm-start-parameter-sha256", required=True,
                   help="the frozen BC+DAgger warm-start parameter_sha256 "
                        "every seed's iteration-0 checkpoint must reproduce")
    p.add_argument("--base-dir", default=None)
    a = p.parse_args(argv)

    base = a.base_dir or f"results/ppo_multiseed_v1/seed_{a.seed}"
    ckpt_dir = os.path.join(base, "checkpoints")
    dev_dir = os.path.join(base, "dev")

    checkpoints = []
    for it in ITERS:
        path = os.path.join(ckpt_dir, f"iter_{it:03d}.pt")
        fp = checkpoint_fingerprint(path)
        checkpoints.append({
            "iteration": it, "cumulative_episodes": episodes(it),
            "primary": True, "file": os.path.basename(path), **fp})

    iter0 = checkpoints[0]
    gate_passed = iter0["parameter_sha256"] == a.warm_start_parameter_sha256
    print(f"Reproduction gate (seed {a.seed} iter0 == frozen warm start): "
          f"{gate_passed}")
    if not gate_passed:
        print(f"  iter0 parameter_sha256 = {iter0['parameter_sha256']}")
        print(f"  expected warm start    = {a.warm_start_parameter_sha256}")
        return 1

    greedy = {it: json.load(open(os.path.join(
                  dev_dir, f"iter{it:03d}_vs_greedy.json"))) for it in ITERS}
    mixed = {it: json.load(open(os.path.join(
                 dev_dir, f"iter{it:03d}_vs_greedy4_random3.json")))
             for it in ITERS}
    drift = {r["checkpoint"]: r for r in
             json.load(open(os.path.join(base, "policy_drift.json")))["checkpoints"]}
    cats = {c["checkpoint"]: c for c in
            json.load(open(os.path.join(
                base, "action_category_drift.json")))["checkpoints"]}
    diag = [json.loads(l) for l in open(os.path.join(base, "train_diag.jsonl"))]

    curve = []
    for it in ITERS:
        d = drift[f"iter_{it:03d}.pt"]
        g, m = greedy[it]["metrics"], mixed[it]["metrics"]
        ce = cats[f"iter_{it:03d}.pt"]["vs_expert"]
        curve.append({
            "iteration": it, "cumulative_episodes": episodes(it),
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

    def block(rows):
        keys = ("adv_mean", "adv_std", "adv_mean_abs", "adv_frac_positive",
                "adv_frac_negative", "return_mean", "return_std",
                "value_pred_mean", "value_pred_std",
                "value_explained_variance", "placement_std",
                "shaping_reward_sum", "terminal_reward_sum", "entropy",
                "approx_kl", "clip_frac", "grad_norm", "pi_loss", "v_loss")
        return {k: st.mean(r[k] for r in rows if r.get(k) is not None)
                for k in keys}

    blocks = {"iters_1_40": block([r for r in diag if r["iter"] <= 40]),
              "iters_41_160": block([r for r in diag if 40 < r["iter"] <= 160]),
              "iters_161_320": block([r for r in diag if r["iter"] > 160])}

    learning_curve = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "evaluation_split": "dev",
        "training_seed": a.seed,
        "episodes_per_iteration": EPISODES_PER_ITER,
        "primary_iterations": ITERS,
        "greedy_games": greedy[0]["games"],
        "mixed_games": mixed[0]["games"],
        "dev_seed_range_greedy": greedy[0]["seed_range"],
        "warm_start_parameter_sha256": a.warm_start_parameter_sha256,
        "reproduction_gate_passed": gate_passed,
        "curve": curve,
    }
    rl_signal = {
        "definitions": {
            "adv_*": "GAE advantages BEFORE PPO's per-batch normalization",
            "value_explained_variance":
                "1 - Var(returns - value_preds) / Var(returns); 1 = perfect, "
                "0 = no better than predicting the mean, <0 = worse",
            "shaping_reward_sum/terminal_reward_sum":
                "the two reward sources separated per iteration"},
        "training_seed": a.seed,
        "per_iteration": diag, "blocks": blocks,
    }

    with open(os.path.join(base, "learning_curve.json"), "w") as f:
        json.dump(learning_curve, f, indent=2)
    with open(os.path.join(base, "rl_signal.json"), "w") as f:
        json.dump(rl_signal, f, indent=2)
    with open(os.path.join(base, "checkpoints.json"), "w") as f:
        json.dump({"training_seed": a.seed, "checkpoints": checkpoints}, f,
                  indent=2)

    print(f"Saved -> {base}/learning_curve.json, rl_signal.json, "
          f"checkpoints.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
