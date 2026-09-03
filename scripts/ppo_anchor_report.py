"""Assemble Experiment 4 (PPO Policy Anchoring) artifacts and baseline comparison.

Reads anchored DEV results and compares against the committed Experiment 2
(unconstrained PPO, kl_coef=0) baseline at identical checkpoints.

    python scripts/ppo_anchor_report.py

DEV split only. Benchmark v1 TEST seeds are never read or run here.
"""

import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import compare_pair, load_result

DIR = "results/ppo_anchor_v1"
BASELINE_DIR = "results/ppo_budget_v1"
ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
KL_COEF = 0.1
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
    base_greedy = {it: load_result(
        f"{BASELINE_DIR}/dev/iter{it:03d}_vs_greedy.json") for it in ITERS}
    base_mixed = {it: load_result(
        f"{BASELINE_DIR}/dev/iter{it:03d}_vs_greedy4_random3.json")
        for it in ITERS}
    drift = {r["checkpoint"]: r for r in
             json.load(open(f"{DIR}/policy_drift.json"))["checkpoints"]}
    base_drift = {r["checkpoint"]: r for r in
                  json.load(open(f"{BASELINE_DIR}/policy_drift.json"))
                  ["checkpoints"]}
    cats = {c["checkpoint"]: c for c in
            json.load(open(f"{DIR}/action_category_drift.json"))["checkpoints"]}
    diag = [json.loads(l) for l in open(f"{DIR}/train_diag.jsonl")]

    curve = []
    for it in ITERS:
        d = drift[f"iter_{it:03d}.pt"]
        bd = base_drift[f"iter_{it:03d}.pt"]
        g, m = greedy[it]["metrics"], mixed[it]["metrics"]
        bg = base_greedy[it]["metrics"]
        ce = cats[f"iter_{it:03d}.pt"]["vs_expert"]
        curve.append({
            "iteration": it, "cumulative_episodes": episodes(it),
            "greedy_avg": g["avg_placement"],
            "greedy_ci95": greedy[it]["avg_placement_ci95"],
            "greedy_top4": g["top4_rate"], "greedy_win": g["win_rate"],
            "mixed_avg": m["avg_placement"],
            "mixed_ci95": mixed[it]["avg_placement_ci95"],
            "expert_agreement": d["expert_agreement"],
            "warmstart_agreement": d["warmstart_agreement"],
            "kl_from_warmstart": d["kl_from_warmstart_mean"],
            "baseline_greedy_avg": bg["avg_placement"],
            "baseline_kl_from_warmstart": bd["kl_from_warmstart_mean"],
            "baseline_expert_agreement": bd["expert_agreement"],
            "parameter_sha256": d["parameter_sha256"],
            "expert_disagreement_by_category":
                ce["disagreement_share_by_category"],
        })

    print("Experiment 4 — anchored vs Experiment 2 baseline (1000 DEV greedy games)")
    hdr = (f"{'iter':>5} {'eps':>6} {'Anchored':>10} {'Baseline':>10} "
           f"{'Delta':>8} {'KL anch':>8} {'KL base':>8} {'Exp% anch':>10}")
    print(hdr)
    print("-" * len(hdr))
    for c in curve:
        delta = c["greedy_avg"] - c["baseline_greedy_avg"]
        print(f"{c['iteration']:>5} {c['cumulative_episodes']:>6} "
              f"{c['greedy_avg']:>10.3f} {c['baseline_greedy_avg']:>10.3f} "
              f"{delta:>+8.3f} {c['kl_from_warmstart']:>8.4f} "
              f"{c['baseline_kl_from_warmstart']:>8.4f} "
              f"{100 * c['expert_agreement']:>9.1f}%")

    paired = {"vs_iter0_greedy": [], "vs_iter40_greedy": [],
              "vs_iter0_mixed": []}
    print("\nAnchored paired comparisons (positive = checkpoint WORSE)")
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

    baseline_cmp = []
    print("\nAnchored vs unconstrained baseline at same iteration (paired):")
    for it in ITERS:
        row = compare_pair(greedy[it], base_greedy[it], seed=0)
        row["iteration"] = it
        row["field"] = "greedy"
        baseline_cmp.append(row)
        zero = row["ci95"][0] <= 0 <= row["ci95"][1]
        print(f"  iter {it:>3}: anchored - baseline {row['mean_diff']:>+8.3f} "
              f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]  "
              f"{'no clear difference' if zero else row['verdict']}")

    def block(rows):
        keys = ("adv_mean_abs", "adv_std", "adv_frac_positive",
                "value_explained_variance", "entropy", "approx_kl",
                "anchor_kl", "clip_frac", "grad_norm", "pi_loss", "v_loss")
        return {k: st.mean(r[k] for r in rows if r.get(k) is not None)
                for k in keys}
    blocks = {"iters_1_40": block([r for r in diag if r["iter"] <= 40]),
              "iters_41_160": block([r for r in diag if 40 < r["iter"] <= 160]),
              "iters_161_320": block([r for r in diag if r["iter"] > 160])}

    out = {
        "experiment": "Replay Experiment 4 — PPO Policy Anchoring",
        "evaluation_split": "dev",
        "kl_coef": KL_COEF,
        "baseline_experiment": "Replay Experiment 2 — PPO Training-Budget Study",
        "training_seed": 0,
        "primary_iterations": ITERS,
        "curve": curve,
        "paired": paired,
        "vs_baseline_paired": baseline_cmp,
        "rl_signal_blocks": blocks,
    }
    with open(f"{DIR}/learning_curve.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(f"{DIR}/paired_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"method": "deterministic paired percentile bootstrap, "
                             "10000 resamples, seed 0",
                   "note": "positive difference = first agent places worse",
                   **paired}, f, indent=2)
    with open(f"{DIR}/baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"method": "paired anchored vs Experiment 2 at same iteration",
                   "comparisons": baseline_cmp}, f, indent=2)
    with open(f"{DIR}/rl_signal.json", "w", encoding="utf-8") as f:
        json.dump({"per_iteration": diag, "blocks": blocks}, f, indent=2)

    _plots(curve, diag)
    print(f"\nSaved -> {DIR}/learning_curve.json, baseline_comparison.json")
    return 0


def _plots(curve, diag) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = os.path.join(DIR, "plots")
    os.makedirs(plots, exist_ok=True)
    eps = [c["cumulative_episodes"] for c in curve]

    def fig(name, ylabel, title, series, invert=False):
        f, ax = plt.subplots(figsize=(7.2, 4.3))
        for label, ys, style in series:
            ax.plot(eps, ys, style, label=label, linewidth=1.8, markersize=6)
        ax.set_xlabel("cumulative PPO training episodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(eps)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        if invert:
            ax.invert_yaxis()
        f.tight_layout()
        f.savefig(os.path.join(plots, name), dpi=140)
        plt.close(f)

    fig("A_dev_placement_greedy", "DEV avg placement (lower is better)",
        "A. Anchored vs baseline DEV placement — 7x greedy",
        [("Anchored (kl=0.1)", [c["greedy_avg"] for c in curve], "o-"),
         ("Baseline unconstrained", [c["baseline_greedy_avg"] for c in curve],
          "s--")], invert=True)
    fig("B_kl_from_warmstart", "KL(pi_BC || pi_k)",
        "B. Policy drift vs warm start — anchored vs baseline",
        [("Anchored", [c["kl_from_warmstart"] for c in curve], "o-"),
         ("Baseline", [c["baseline_kl_from_warmstart"] for c in curve], "s--")])
    fig("C_expert_agreement", "agreement with greedy expert",
        "C. Expert agreement — anchored vs baseline",
        [("Anchored", [c["expert_agreement"] for c in curve], "o-"),
         ("Baseline", [c["baseline_expert_agreement"] for c in curve], "s--")])

    f, ax = plt.subplots(figsize=(7.2, 4.3))
    iters = [r["iter"] for r in diag]
    ax.plot(iters, [r.get("anchor_kl", 0) for r in diag], label="anchor KL loss",
            linewidth=0.9)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean KL(pi_BC || pi_theta) in loss")
    ax.set_title("D. Training-time anchor KL penalty (beta=0.1)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(os.path.join(plots, "D_training_anchor_kl.png"), dpi=140)
    plt.close(f)
    print(f"Saved plots -> {plots}/")


if __name__ == "__main__":
    raise SystemExit(main())
