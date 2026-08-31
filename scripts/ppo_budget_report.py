"""Assemble the Experiment 2 (PPO training-budget study) artifacts.

Reads only committed result JSON — the DEV evaluations, policy drift, action
categories, and per-iteration training diagnostics — and emits the learning
curve, the paired budget comparisons, the RL-signal summary, and the plots.
Re-runnable; computes nothing that isn't already in those files.

    python scripts/ppo_budget_report.py

DEV split only. Benchmark v1 TEST seeds are never read or run here.
"""

import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import compare_pair, load_result       # noqa: E402

DIR = "results/ppo_budget_v1"
ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
# Pre-specified before results were seen.
PAIRS_VS_0 = [40, 80, 160, 320]
PAIRS_VS_40 = [80, 160, 320]


def episodes(it):
    return it * EPISODES_PER_ITER


def _boot(diffs, seed, B=10000):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs)
                   for _ in range(B))
    return {"mean": st.mean(diffs),
            "ci95": [means[int(0.025 * B) - 1], means[int(0.975 * B)]]}


def main() -> int:
    greedy = {it: load_result(f"{DIR}/dev/iter{it:03d}_vs_greedy.json")
              for it in ITERS}
    mixed = {it: load_result(f"{DIR}/dev/iter{it:03d}_vs_greedy4_random3.json")
             for it in ITERS}
    drift = {r["checkpoint"]: r for r in
             json.load(open(f"{DIR}/policy_drift.json"))["checkpoints"]}
    cats = {c["checkpoint"]: c for c in
            json.load(open(f"{DIR}/action_category_drift.json"))["checkpoints"]}
    diag = [json.loads(l) for l in open(f"{DIR}/train_diag.jsonl")]
    by_iter = {r["iter"]: r for r in diag}

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

    hdr = (f"{'iter':>5} {'episodes':>9} {'GreedyAvg':>10} {'95% CI':>18} "
           f"{'Top4':>7} {'MixedAvg':>9} {'Expert%':>8} {'WarmSt%':>8} {'KL':>7}")
    print("PPO budget study — DEV learning curve "
          "(1000 games vs 7x greedy; 500 vs greedy4_random3)")
    print(hdr)
    print("-" * len(hdr))
    for c in curve:
        print(f"{c['iteration']:>5} {c['cumulative_episodes']:>9} "
              f"{c['greedy_avg']:>10.3f} "
              f"{'[%.3f, %.3f]' % (c['greedy_ci95']['low'], c['greedy_ci95']['high']):>18} "
              f"{100 * c['greedy_top4']:>6.1f}% {c['mixed_avg']:>9.3f} "
              f"{100 * c['expert_agreement']:>7.1f}% "
              f"{100 * c['warmstart_agreement']:>7.1f}% "
              f"{c['kl_from_warmstart']:>7.4f}")

    # --- paired comparisons ---------------------------------------------------
    paired = {"vs_iter0_greedy": [], "vs_iter40_greedy": [],
              "vs_iter0_mixed": []}
    print("\nPaired DEV comparisons (positive = checkpoint WORSE than reference)")
    for label, ref, targets, src, key in [
            ("vs iter 0 (greedy)", 0, PAIRS_VS_0, greedy, "vs_iter0_greedy"),
            ("vs iter 40 (greedy)", 40, PAIRS_VS_40, greedy, "vs_iter40_greedy"),
            ("vs iter 0 (mixed)", 0, PAIRS_VS_0, mixed, "vs_iter0_mixed")]:
        print(f"  {label}:")
        for it in targets:
            row = compare_pair(src[it], src[ref], seed=0)
            row["iteration"], row["reference_iteration"] = it, ref
            paired[key].append(row)
            zero = row["ci95"][0] <= 0 <= row["ci95"][1]
            print(f"    iter{it:>4} - iter{ref:<3} {row['mean_diff']:>+8.3f} "
                  f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]  "
                  f"{'no clear difference' if zero else row['verdict']}")

    # Pre-specified trend across the primary budgets (per-seed slope on
    # log2 episodes, so equal budget doublings are equally spaced).
    import math
    xs = [math.log2(episodes(it)) for it in ITERS[1:]]      # iter 0 has 0 eps
    xm = st.mean(xs)
    sxx = sum((x - xm) ** 2 for x in xs)
    P = {it: greedy[it]["placements"] for it in ITERS}
    n = len(P[0])
    slopes = []
    for gme in range(n):
        ys = [P[it][gme] for it in ITERS[1:]]
        ym = st.mean(ys)
        slopes.append(sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx)
    trend = _boot(slopes, seed=1)
    print(f"\n  trend per doubling of episodes (640->5120): "
          f"{trend['mean']:+.4f} 95% CI "
          f"[{trend['ci95'][0]:+.4f}, {trend['ci95'][1]:+.4f}]")

    # --- RL signal summary ----------------------------------------------------
    def block(rows):
        return {k: st.mean(r[k] for r in rows if r.get(k) is not None)
                for k in ("adv_mean", "adv_std", "adv_mean_abs",
                          "adv_frac_positive", "adv_frac_negative",
                          "return_mean", "return_std", "value_pred_mean",
                          "value_pred_std", "value_explained_variance",
                          "placement_std", "shaping_reward_sum",
                          "terminal_reward_sum", "entropy", "approx_kl",
                          "clip_frac", "grad_norm", "pi_loss", "v_loss")}
    blocks = {"iters_1_40": block([r for r in diag if r["iter"] <= 40]),
              "iters_41_160": block([r for r in diag if 40 < r["iter"] <= 160]),
              "iters_161_320": block([r for r in diag if r["iter"] > 160])}
    print("\nRL signal / optimization diagnostics (means per training block)")
    print(f"{'metric':>26} {'1-40':>10} {'41-160':>10} {'161-320':>10}")
    for k in ("adv_mean_abs", "adv_std", "adv_frac_positive",
              "value_explained_variance", "return_std", "placement_std",
              "entropy", "clip_frac", "grad_norm", "approx_kl",
              "shaping_reward_sum", "terminal_reward_sum"):
        print(f"{k:>26} {blocks['iters_1_40'][k]:>10.4f} "
              f"{blocks['iters_41_160'][k]:>10.4f} "
              f"{blocks['iters_161_320'][k]:>10.4f}")

    out = {"experiment": "Replay Experiment 2 — PPO Training-Budget Study",
           "evaluation_split": "dev",
           "training_seed": 0, "episodes_per_iteration": EPISODES_PER_ITER,
           "primary_iterations": ITERS,
           "greedy_games": greedy[0]["games"],
           "mixed_games": mixed[0]["games"],
           "dev_seed_range_greedy": greedy[0]["seed_range"],
           "curve": curve, "paired": paired,
           "trend_per_episode_doubling": trend,
           "rl_signal_blocks": blocks}
    with open(f"{DIR}/learning_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(f"{DIR}/paired_analysis.json", "w") as f:
        json.dump({"method": "deterministic paired percentile bootstrap, "
                             "10000 resamples, seed 0",
                   "note": "positive difference = checkpoint places worse "
                           "than the reference (lower placement is better)",
                   **paired,
                   "trend_per_episode_doubling": trend}, f, indent=2)
    with open(f"{DIR}/rl_signal.json", "w") as f:
        json.dump({"definitions": {
            "adv_*": "GAE advantages BEFORE PPO's per-batch normalization",
            "value_explained_variance":
                "1 - Var(returns - value_preds) / Var(returns); 1 = perfect, "
                "0 = no better than predicting the mean, <0 = worse",
            "shaping_reward_sum/terminal_reward_sum":
                "the two reward sources separated per iteration"},
            "per_iteration": diag, "blocks": blocks}, f, indent=2)
    print(f"\nSaved -> {DIR}/learning_curve.json, paired_analysis.json, "
          f"rl_signal.json")
    _plots(curve, diag, cats)
    return 0


def _plots(curve, diag, cats) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(f"{DIR}/plots", exist_ok=True)
    eps = [c["cumulative_episodes"] for c in curve]

    def budget_fig(name, ylabel, title, series, invert=False, ref=None):
        f, ax = plt.subplots(figsize=(7.2, 4.3))
        for label, ys, style in series:
            ax.plot(eps, ys, style, label=label, linewidth=1.8, markersize=6)
        if ref is not None:
            ax.axhline(ref, color="#888", linestyle=":", linewidth=1.2,
                       label="warm start (0 episodes)")
        ax.set_xlabel("cumulative PPO training episodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e}\n(it {c['iteration']})"
                            for e, c in zip(eps, curve)], fontsize=8)
        if invert:
            ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(f"{DIR}/plots/{name}.png", dpi=140)
        plt.close(f)

    budget_fig("A_dev_placement_greedy", "DEV avg placement (lower is better)",
               "A. DEV placement vs training budget — 7x greedy (1000 games)",
               [("PPO checkpoint", [c["greedy_avg"] for c in curve], "o-")],
               invert=True, ref=curve[0]["greedy_avg"])
    budget_fig("B_dev_placement_mixed", "DEV avg placement (lower is better)",
               "B. DEV placement vs budget — greedy4_random3 diagnostic (500)",
               [("PPO checkpoint", [c["mixed_avg"] for c in curve], "o-")],
               invert=True, ref=curve[0]["mixed_avg"])
    budget_fig("C_expert_agreement", "agreement with greedy expert",
               "C. Expert-action agreement vs training budget",
               [("argmax == greedy expert",
                 [c["expert_agreement"] for c in curve], "o-")])
    budget_fig("D_drift", "agreement / KL",
               "D. Warm-start agreement and KL vs training budget",
               [("argmax == iter-0 argmax",
                 [c["warmstart_agreement"] for c in curve], "o-"),
                ("mean KL(pi_0 || pi_k)",
                 [c["kl_from_warmstart"] for c in curve], "s--")])

    # E. signal diagnostics over the full 320 iterations
    f, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    it = [r["iter"] for r in diag]
    for ax, (key, label) in zip(axes.flat, [
            ("adv_mean_abs", "mean |raw advantage|"),
            ("value_explained_variance", "value explained variance"),
            ("entropy", "policy entropy (training batches)"),
            ("clip_frac", "PPO clip fraction")]):
        ys = [r.get(key) for r in diag]
        ax.plot(it, ys, linewidth=0.9, color="#39c")
        if key == "value_explained_variance":
            ax.axhline(0, color="#888", linestyle=":", linewidth=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.25)
    f.suptitle("E. RL signal and optimization diagnostics over 320 iterations",
               fontsize=11)
    f.tight_layout()
    f.savefig(f"{DIR}/plots/E_rl_signal.png", dpi=140)
    plt.close(f)

    # F. action-category disagreement vs the expert, by budget
    from ml.action_categories import CATEGORIES
    f, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.8 / len(curve)
    xs = range(len(CATEGORIES))
    for i, c in enumerate(curve):
        shares = [(c["expert_disagreement_by_category"].get(cat) or 0.0)
                  for cat in CATEGORIES]
        ax.bar([x + i * width for x in xs], shares, width,
               label=f"{c['cumulative_episodes']} eps")
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("share of expert decisions changed")
    ax.set_title("F. Where the policy disagrees with the greedy expert, "
                 "by decision category and budget", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{DIR}/plots/F_category_disagreement.png", dpi=140)
    plt.close(f)
    print(f"Saved plots -> {DIR}/plots/")


if __name__ == "__main__":
    raise SystemExit(main())
