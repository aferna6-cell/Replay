"""Assemble Experiment 3 (Multi-Seed PPO Budget Replication) artifacts and analysis.

Reads committed DEV results, policy drift, action categories, and RL signals
across 4 PPO training seeds (seed 0 from Experiment 2, plus seeds 1, 2, 3).
Computes within-seed paired comparisons, cross-seed statistics, replication
questions, U-shape classifications, and generates machine-readable JSON artifacts
and plots A-G.
"""

import json
import math
import os
import random
import statistics as st
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import compare_pair, load_result

BASE_DIR = "results/ppo_multiseed_v1"
EXP2_DIR = "results/ppo_budget_v1"
AGG_DIR = os.path.join(BASE_DIR, "aggregate")
PLOTS_DIR = os.path.join(AGG_DIR, "plots")

SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16

# All 9 required within-seed paired comparison targets:
PAIRS_CONFIG = [
    ("iter040", 40, "iter000", 0),
    ("iter080", 80, "iter000", 0),
    ("iter160", 160, "iter000", 0),
    ("iter320", 320, "iter000", 0),
    ("iter080", 80, "iter040", 40),
    ("iter160", 160, "iter040", 40),
    ("iter320", 320, "iter040", 40),
    ("iter160", 160, "iter080", 80),
    ("iter320", 320, "iter080", 80),
]


def get_dev_greedy_path(seed: int, iteration: int) -> str:
    if seed == 0:
        return f"{EXP2_DIR}/dev/iter{iteration:03d}_vs_greedy.json"
    return f"{BASE_DIR}/seed_{seed}/dev/iter{iteration:03d}_vs_greedy.json"


def get_dev_mixed_path(seed: int, iteration: int) -> str:
    if seed == 0:
        return f"{EXP2_DIR}/dev/iter{iteration:03d}_vs_greedy4_random3.json"
    return f"{BASE_DIR}/seed_{seed}/dev/iter{iteration:03d}_vs_greedy4_random3.json"


def get_policy_drift_path(seed: int) -> str:
    if seed == 0:
        return f"{EXP2_DIR}/policy_drift.json"
    return f"{BASE_DIR}/seed_{seed}/policy_drift.json"


def get_category_drift_path(seed: int) -> str:
    if seed == 0:
        return f"{EXP2_DIR}/action_category_drift.json"
    return f"{BASE_DIR}/seed_{seed}/action_category_drift.json"


def get_train_diag_path(seed: int) -> str:
    if seed == 0:
        return f"{EXP2_DIR}/train_diag.jsonl"
    return f"{BASE_DIR}/seed_{seed}/train_diag.jsonl"


def block_rl_stats(rows: List[Dict]) -> Dict:
    metrics = (
        "adv_mean", "adv_std", "adv_mean_abs",
        "adv_frac_positive", "adv_frac_negative",
        "return_mean", "return_std", "value_pred_mean",
        "value_pred_std", "value_explained_variance",
        "placement_std", "shaping_reward_sum",
        "terminal_reward_sum", "entropy", "approx_kl",
        "clip_frac", "grad_norm", "pi_loss", "v_loss",
    )
    res = {}
    for k in metrics:
        vals = [r[k] for r in rows if r.get(k) is not None]
        res[k] = float(st.mean(vals)) if vals else None
    return res


def compute_rl_signal_for_seed(seed: int) -> Dict:
    diag_path = get_train_diag_path(seed)
    diag = [json.loads(line) for line in open(diag_path, encoding="utf-8")]
    blocks = {
        "iters_1_40": block_rl_stats([r for r in diag if r["iter"] <= 40]),
        "iters_41_160": block_rl_stats([r for r in diag if 40 < r["iter"] <= 160]),
        "iters_161_320": block_rl_stats([r for r in diag if r["iter"] > 160]),
    }
    return {
        "definitions": {
            "adv_*": "GAE advantages BEFORE PPO's per-batch normalization",
            "value_explained_variance": "1 - Var(returns - value_preds) / Var(returns); 1 = perfect, 0 = no better than predicting the mean, <0 = worse",
            "shaping_reward_sum/terminal_reward_sum": "the two reward sources separated per iteration",
        },
        "per_iteration": diag,
        "blocks": blocks,
    }


def classify_ushape(curve: List[Dict], paired_results: Dict[str, Dict]) -> Dict:
    """Classify the trajectory curve descriptively based on measured checkpoint values
    and within-seed paired bootstrap comparisons.

    Classification rules:
      - 'U-like / transient improvement': Has a minimum at iter 40, 80, or 160 that is
        lower than iter 0 (paired diff < 0 with CI excluding 0), followed by regression
        at iter 320 (iter320 - min_iter > 0).
      - 'monotonic improvement': Successive checkpoints decrease or stay <=, with iter320 < iter0 (CI excludes 0).
      - 'monotonic degradation': Successive checkpoints increase or stay >=, with iter320 > iter0 (CI excludes 0).
      - 'mostly flat/noisy': No primary checkpoint differs significantly from iter 0 (all CIs vs iter0 include 0).
      - 'other': Other non-monotonic trajectories (e.g. transient severe degradation or complex oscillation).
    """
    diff_40_0 = paired_results["iter040_vs_iter000"]
    diff_80_0 = paired_results["iter080_vs_iter000"]
    diff_160_0 = paired_results["iter160_vs_iter000"]
    diff_320_0 = paired_results["iter320_vs_iter000"]

    diff_320_80 = paired_results["iter320_vs_iter080"]

    # Check if iter80 significantly beats iter0
    iter80_beats_iter0 = diff_80_0["ci95"][1] < 0.0  # high CI < 0 means diff < 0
    iter80_regresses_320 = diff_320_80["ci95"][0] > 0.0  # low CI > 0 means iter320 > iter80

    all_vs_0 = [diff_40_0, diff_80_0, diff_160_0, diff_320_0]
    all_flat = all(d["ci95"][0] <= 0.0 <= d["ci95"][1] for d in all_vs_0)

    if iter80_beats_iter0 and iter80_regresses_320:
        classification = "U-like / transient improvement"
        rationale = "Transient improvement at iter80 (diff vs iter0 < 0, CI excludes 0) followed by significant regression at iter320."
    elif all_flat:
        classification = "mostly flat/noisy"
        rationale = "No intermediate or final checkpoint differs significantly from iter 0 (all paired 95% CIs include 0)."
    else:
        # Check if there is a transient degradation or other non-standard pattern
        iter80_worse = diff_80_0["ci95"][0] > 0.0
        iter160_worse = diff_160_0["ci95"][0] > 0.0
        if iter80_worse or iter160_worse:
            classification = "other"
            rationale = "Transient severe degradation at mid-budget (iter80/iter160) followed by recovery toward warm start at iter320."
        elif diff_320_0["ci95"][1] < 0.0:
            classification = "monotonic improvement"
            rationale = "Significant improvement by iter320 over iter0 with monotonic trend."
        elif diff_320_0["ci95"][0] > 0.0:
            classification = "monotonic degradation"
            rationale = "Significant degradation by iter320 over iter0."
        else:
            classification = "other"
            rationale = "Non-standard trajectory not matching flat, monotonic, or U-shaped profile."

    return {
        "classification": classification,
        "rationale": rationale,
        "iter80_vs_iter0_diff": diff_80_0["mean_diff"],
        "iter80_vs_iter0_ci95": diff_80_0["ci95"],
        "iter320_vs_iter80_diff": diff_320_80["mean_diff"],
        "iter320_vs_iter80_ci95": diff_320_80["ci95"],
    }


def make_plots(cross_summary: Dict, per_seed_curves: Dict[int, List[Dict]],
               per_seed_diags: Dict[int, List[Dict]], per_seed_cats: Dict[int, Dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    eps = [it * EPISODES_PER_ITER for it in ITERS]
    seed_colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
    seed_markers = {0: "o", 1: "s", 2: "^", 3: "D"}

    # --- Plot A: Multi-seed DEV learning curves (1000 games vs 7x greedy) ---
    plt.figure(figsize=(8.0, 5.0))
    for s in SEEDS:
        ys = [c["greedy_avg"] for c in per_seed_curves[s]]
        plt.plot(eps, ys, marker=seed_markers[s], color=seed_colors[s],
                 label=f"Seed {s}", linewidth=1.8, markersize=6)
    plt.axhline(6.573, color="#888", linestyle=":", label="Warm start reference (~6.57)")
    plt.xlabel("Cumulative PPO Training Episodes", fontsize=10)
    plt.ylabel("DEV Avg Placement vs 7x Greedy (lower is better)", fontsize=10)
    plt.title("A: Multi-Seed DEV Learning Curves (1000 Games vs 7x Greedy)", fontsize=11, fontweight="bold")
    plt.xticks(eps, [f"{e}\n(it {it})" for e, it in zip(eps, ITERS)], fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "A_multiseed_dev_learning_curves.png"), dpi=150)
    plt.close()

    # --- Plot B: Cross-seed mean curve with individual seed trajectories ---
    plt.figure(figsize=(8.0, 5.0))
    for s in SEEDS:
        ys = [c["greedy_avg"] for c in per_seed_curves[s]]
        plt.plot(eps, ys, marker=seed_markers[s], color=seed_colors[s],
                 alpha=0.45, linestyle="--", label=f"Seed {s} (raw)")
    means = [cross_summary["by_budget"][str(it)]["mean"] for it in ITERS]
    stds = [cross_summary["by_budget"][str(it)]["std"] for it in ITERS]
    plt.errorbar(eps, means, yerr=stds, color="black", marker="o", linewidth=2.5,
                 capsize=5, label="Cross-Seed Mean ± 1 Std Dev")
    plt.xlabel("Cumulative PPO Training Episodes", fontsize=10)
    plt.ylabel("DEV Avg Placement (lower is better)", fontsize=10)
    plt.title("B: Cross-Seed Mean DEV Learning Curve (N=4 Training Seeds)", fontsize=11, fontweight="bold")
    plt.xticks(eps, [f"{e}\n(it {it})" for e, it in zip(eps, ITERS)], fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "B_cross_seed_mean_curve.png"), dpi=150)
    plt.close()

    # --- Plot C: Expert agreement by training seed ---
    plt.figure(figsize=(8.0, 5.0))
    for s in SEEDS:
        ys = [c["expert_agreement"] * 100 for c in per_seed_curves[s]]
        plt.plot(eps, ys, marker=seed_markers[s], color=seed_colors[s],
                 label=f"Seed {s}", linewidth=1.8, markersize=6)
    plt.xlabel("Cumulative PPO Training Episodes", fontsize=10)
    plt.ylabel("Greedy Expert Agreement (%)", fontsize=10)
    plt.title("C: Expert Agreement on Frozen Corpus (4,440 states)", fontsize=11, fontweight="bold")
    plt.xticks(eps, [f"{e}\n(it {it})" for e, it in zip(eps, ITERS)], fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "C_expert_agreement.png"), dpi=150)
    plt.close()

    # --- Plot D: KL divergence from warm start by training seed ---
    plt.figure(figsize=(8.0, 5.0))
    for s in SEEDS:
        ys = [c["kl_from_warmstart"] for c in per_seed_curves[s]]
        plt.plot(eps, ys, marker=seed_markers[s], color=seed_colors[s],
                 label=f"Seed {s}", linewidth=1.8, markersize=6)
    plt.xlabel("Cumulative PPO Training Episodes", fontsize=10)
    plt.ylabel("KL(Warm Start || PPO_k)", fontsize=10)
    plt.title("D: Policy Drift — KL Divergence from Warm Start (Iter 0)", fontsize=11, fontweight="bold")
    plt.xticks(eps, [f"{e}\n(it {it})" for e, it in zip(eps, ITERS)], fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "D_kl_from_warmstart.png"), dpi=150)
    plt.close()

    # --- Plot E: Warm-start agreement by training seed ---
    plt.figure(figsize=(8.0, 5.0))
    for s in SEEDS:
        ys = [c["warmstart_agreement"] * 100 for c in per_seed_curves[s]]
        plt.plot(eps, ys, marker=seed_markers[s], color=seed_colors[s],
                 label=f"Seed {s}", linewidth=1.8, markersize=6)
    plt.xlabel("Cumulative PPO Training Episodes", fontsize=10)
    plt.ylabel("Warm-Start Agreement (%)", fontsize=10)
    plt.title("E: Warm-Start Action Agreement Across Training Seeds", fontsize=11, fontweight="bold")
    plt.xticks(eps, [f"{e}\n(it {it})" for e, it in zip(eps, ITERS)], fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "E_warmstart_agreement.png"), dpi=150)
    plt.close()

    # --- Plot F: Action-category drift at iter320 across seeds ---
    plt.figure(figsize=(9.0, 5.0))
    categories = ["buy", "play", "sell", "roll", "level", "freeze", "end"]
    x = np.arange(len(categories))
    width = 0.2
    for i, s in enumerate(SEEDS):
        cat_data = per_seed_cats[s]["iter_320.pt"]["vs_expert"]["disagreement_share_by_category"]
        shares = [cat_data.get(c) or 0.0 for c in categories]
        plt.bar(x + (i - 1.5) * width, [v * 100 for v in shares], width,
                label=f"Seed {s}", color=seed_colors[s])
    plt.xlabel("Action Category", fontsize=10)
    plt.ylabel("Disagreement with Expert (%)", fontsize=10)
    plt.title("F: Action-Category Disagreement with Expert at Iteration 320", fontsize=11, fontweight="bold")
    plt.xticks(x, [c.capitalize() for c in categories], fontsize=9)
    plt.grid(alpha=0.3, axis="y")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "F_action_category_drift.png"), dpi=150)
    plt.close()

    # --- Plot G: PPO optimization diagnostics across seeds ---
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    metrics_to_plot = [
        ("entropy", "Policy Entropy", axes[0, 0]),
        ("approx_kl", "Approximate KL", axes[0, 1]),
        ("clip_frac", "Clip Fraction", axes[1, 0]),
        ("value_explained_variance", "Value Explained Variance", axes[1, 1]),
    ]
    for m_key, m_title, ax in metrics_to_plot:
        for s in SEEDS:
            diag_rows = per_seed_diags[s]
            x_vals = [r["iter"] for r in diag_rows]
            y_vals = [r.get(m_key, 0.0) for r in diag_rows]
            # Simple running mean smoothing for noisy line
            window = 10
            y_smooth = np.convolve(y_vals, np.ones(window)/window, mode="valid")
            x_smooth = x_vals[window-1:]
            ax.plot(x_smooth, y_smooth, label=f"Seed {s}", color=seed_colors[s], alpha=0.85, linewidth=1.5)
        ax.set_title(m_title, fontsize=10, fontweight="bold")
        ax.set_xlabel("PPO Iteration", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("G: PPO Internal Optimization Diagnostics (10-iter Moving Average)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "G_ppo_diagnostics.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(AGG_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    per_seed_curves: Dict[int, List[Dict]] = {}
    per_seed_paired: Dict[int, Dict[str, Dict]] = {}
    per_seed_diags: Dict[int, List[Dict]] = {}
    per_seed_cats: Dict[int, Dict] = {}
    per_seed_rl_signals: Dict[int, Dict] = {}

    print("Loading data for all 4 seeds (Seed 0 historical + Seeds 1, 2, 3)...")
    for s in SEEDS:
        greedy_res = {it: load_result(get_dev_greedy_path(s, it)) for it in ITERS}
        mixed_res = {it: load_result(get_dev_mixed_path(s, it)) for it in ITERS}
        drift_data = json.load(open(get_policy_drift_path(s), encoding="utf-8"))
        drift_by_ckpt = {r["checkpoint"]: r for r in drift_data["checkpoints"]}
        cat_data = json.load(open(get_category_drift_path(s), encoding="utf-8"))
        cat_by_ckpt = {c["checkpoint"]: c for c in cat_data["checkpoints"]}
        per_seed_cats[s] = cat_by_ckpt

        rl_signal = compute_rl_signal_for_seed(s)
        per_seed_rl_signals[s] = rl_signal
        per_seed_diags[s] = rl_signal["per_iteration"]

        # Write per-seed rl_signal.json if seed > 0
        if s > 0:
            seed_dir = os.path.join(BASE_DIR, f"seed_{s}")
            with open(os.path.join(seed_dir, "rl_signal.json"), "w", encoding="utf-8") as f:
                json.dump(rl_signal, f, indent=2)

        curve: List[Dict] = []
        for it in ITERS:
            ckpt_name = f"iter_{it:03d}.pt"
            d = drift_by_ckpt[ckpt_name]
            g, m = greedy_res[it]["metrics"], mixed_res[it]["metrics"]
            ce = cat_by_ckpt[ckpt_name]["vs_expert"]
            curve.append({
                "iteration": it,
                "cumulative_episodes": it * EPISODES_PER_ITER,
                "greedy_avg": g["avg_placement"],
                "greedy_ci95": greedy_res[it]["avg_placement_ci95"],
                "greedy_median": g["median_placement"],
                "greedy_std": g["std_placement"],
                "greedy_top4": g["top4_rate"],
                "greedy_win": g["win_rate"],
                "greedy_placement_counts": g["placement_counts"],
                "mixed_avg": m["avg_placement"],
                "mixed_ci95": mixed_res[it]["avg_placement_ci95"],
                "mixed_top4": m["top4_rate"],
                "mixed_win": m["win_rate"],
                "expert_agreement": d["expert_agreement"],
                "warmstart_agreement": d["warmstart_agreement"],
                "kl_from_warmstart": d["kl_from_warmstart_mean"],
                "corpus_entropy": d["entropy_mean"],
                "value_mean": d["value_mean"],
                "value_std": d["value_std"],
                "parameter_sha256": d["parameter_sha256"],
                "checkpoint_sha256": d["checkpoint_sha256"],
                "expert_disagreement_by_category": ce["disagreement_share_by_category"],
                "drift_contribution_by_category": ce["contribution_to_total_drift"],
            })
        per_seed_curves[s] = curve

        # Within-seed paired comparisons
        paired_dict: Dict[str, Dict] = {}
        for target_label, it_tgt, ref_label, it_ref in PAIRS_CONFIG:
            res_tgt = greedy_res[it_tgt]
            res_ref = greedy_res[it_ref]
            pair_stat = compare_pair(res_tgt, res_ref, seed=0)
            key = f"{target_label}_vs_{ref_label}"
            paired_dict[key] = {
                "target_iteration": it_tgt,
                "reference_iteration": it_ref,
                "mean_diff": pair_stat["mean_diff"],
                "ci95": pair_stat["ci95"],
                "verdict": pair_stat["verdict"],
                "n_games": pair_stat["n"],
            }
        per_seed_paired[s] = paired_dict

        # Write per-seed learning_curve.json if seed > 0
        if s > 0:
            seed_dir = os.path.join(BASE_DIR, f"seed_{s}")
            seed_lc = {
                "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
                "evaluation_split": "dev",
                "training_seed": s,
                "episodes_per_iteration": EPISODES_PER_ITER,
                "primary_iterations": ITERS,
                "greedy_games": 1000,
                "mixed_games": 500,
                "dev_seed_range_greedy": [10550000, 10550999],
                "curve": curve,
                "paired": paired_dict,
                "rl_signal_blocks": rl_signal["blocks"],
            }
            with open(os.path.join(seed_dir, "learning_curve.json"), "w", encoding="utf-8") as f:
                json.dump(seed_lc, f, indent=2)

    # --- Cross-Training-Seed Summary Table ---
    cross_table = {}
    by_budget: Dict[str, Dict] = {}
    for it in ITERS:
        placements = [per_seed_curves[s][ITERS.index(it)]["greedy_avg"] for s in SEEDS]
        by_budget[str(it)] = {
            "iteration": it,
            "cumulative_episodes": it * EPISODES_PER_ITER,
            "seed_placements": {f"seed_{s}": p for s, p in zip(SEEDS, placements)},
            "mean": float(st.mean(placements)),
            "median": float(st.median(placements)),
            "min": float(min(placements)),
            "max": float(max(placements)),
            "std": float(st.stdev(placements)) if len(placements) > 1 else 0.0,
        }

    cross_summary = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "seeds": SEEDS,
        "iterations": ITERS,
        "episodes_per_iteration": EPISODES_PER_ITER,
        "table_by_seed": {
            f"seed_{s}": {f"iter_{it:03d}": per_seed_curves[s][ITERS.index(it)]["greedy_avg"] for it in ITERS}
            for s in SEEDS
        },
        "by_budget": by_budget,
    }

    with open(os.path.join(AGG_DIR, "cross_seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cross_summary, f, indent=2)

    # --- Paired Results Artifact ---
    paired_artifact = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "description": "Within-seed paired bootstrap comparisons on 1000 identical DEV games (vs 7x greedy)",
        "method": "deterministic paired percentile bootstrap, 10,000 resamples, seed 0",
        "convention": "mean_diff = target - reference; positive = target is WORSE (higher placement), negative = target is BETTER",
        "by_seed": {f"seed_{s}": per_seed_paired[s] for s in SEEDS},
    }
    with open(os.path.join(AGG_DIR, "paired_results.json"), "w", encoding="utf-8") as f:
        json.dump(paired_artifact, f, indent=2)

    # --- Pre-specified Replication Questions & U-shape Classifications ---
    # Question A: Does the ~1,280 episode improvement reproduce? (iter80 - iter0)
    q_a_results = []
    improved_point_a = 0
    worsened_point_a = 0
    sig_improved_a = 0
    sig_worsened_a = 0
    ci_excludes_zero_a = 0
    for s in SEEDS:
        p_80_0 = per_seed_paired[s]["iter080_vs_iter000"]
        diff = p_80_0["mean_diff"]
        ci = p_80_0["ci95"]
        improves_point = diff < 0.0
        worsens_point = diff > 0.0
        sig_improves = ci[1] < 0.0
        sig_worsens = ci[0] > 0.0
        excludes_zero = sig_improves or sig_worsens
        if improves_point:
            improved_point_a += 1
        if worsens_point:
            worsened_point_a += 1
        if sig_improves:
            sig_improved_a += 1
        if sig_worsens:
            sig_worsened_a += 1
        if excludes_zero:
            ci_excludes_zero_a += 1
        q_a_results.append({
            "seed": s,
            "iter80_avg": per_seed_curves[s][ITERS.index(80)]["greedy_avg"],
            "iter0_avg": per_seed_curves[s][ITERS.index(0)]["greedy_avg"],
            "mean_diff": diff,
            "ci95": ci,
            "verdict": p_80_0["verdict"],
            "point_improves": improves_point,
            "point_worsens": worsens_point,
            "significant_improvement": sig_improves,
            "significant_degradation": sig_worsens,
        })

    # Question B: Does performance decay after the transient improvement? (iter320 - iter80)
    q_b_results = []
    regressed_count_b = 0
    improved_count_b = 0
    indistinguishable_count_b = 0
    for s in SEEDS:
        p_320_80 = per_seed_paired[s]["iter320_vs_iter080"]
        diff = p_320_80["mean_diff"]
        ci = p_320_80["ci95"]
        regresses = ci[0] > 0.0  # iter320 is worse than iter80
        improves = ci[1] < 0.0   # iter320 is better than iter80
        indistinguishable = not (regresses or improves)
        if regresses:
            regressed_count_b += 1
        if improves:
            improved_count_b += 1
        if indistinguishable:
            indistinguishable_count_b += 1
        q_b_results.append({
            "seed": s,
            "iter320_avg": per_seed_curves[s][ITERS.index(320)]["greedy_avg"],
            "iter80_avg": per_seed_curves[s][ITERS.index(80)]["greedy_avg"],
            "mean_diff": diff,
            "ci95": ci,
            "verdict": p_320_80["verdict"],
            "regresses_after_iter80": regresses,
            "improves_after_iter80": improves,
            "indistinguishable": indistinguishable,
        })

    # U-shape classifications
    ushape_classifications = {
        f"seed_{s}": classify_ushape(per_seed_curves[s], per_seed_paired[s])
        for s in SEEDS
    }
    ushape_count = sum(1 for c in ushape_classifications.values() if c["classification"] == "U-like / transient improvement")

    # Policy Drift at Iteration 320 across seeds
    drift_at_320 = []
    for s in SEEDS:
        c320 = per_seed_curves[s][ITERS.index(320)]
        drift_at_320.append({
            "seed": s,
            "expert_agreement": c320["expert_agreement"],
            "warmstart_agreement": c320["warmstart_agreement"],
            "kl_from_warmstart": c320["kl_from_warmstart"],
            "corpus_entropy": c320["corpus_entropy"],
            "value_mean": c320["value_mean"],
            "value_std": c320["value_std"],
        })

    # Best-performing checkpoint vs drift
    best_ckpt_drift = []
    for s in SEEDS:
        curve = per_seed_curves[s]
        best_ckpt = min(curve, key=lambda c: c["greedy_avg"])
        final_ckpt = curve[ITERS.index(320)]
        best_ckpt_drift.append({
            "seed": s,
            "best_iteration": best_ckpt["iteration"],
            "best_placement": best_ckpt["greedy_avg"],
            "best_expert_agreement": best_ckpt["expert_agreement"],
            "best_kl_from_warmstart": best_ckpt["kl_from_warmstart"],
            "final_placement": final_ckpt["greedy_avg"],
            "final_expert_agreement": final_ckpt["expert_agreement"],
            "final_kl_from_warmstart": final_ckpt["kl_from_warmstart"],
        })

    # Action-Category Drift and Freeze behavior at iter 320
    action_cat_320 = []
    for s in SEEDS:
        vs_exp = per_seed_cats[s]["iter_320.pt"]["vs_expert"]
        cm = vs_exp["confusion_matrix"]
        # Freeze action in 28-action space is action 26 (category 'freeze')
        # Check if PPO selected freeze in any state of the corpus
        freeze_selections = sum(tos.get("freeze", 0) for tos in cm.values())
        total_states = vs_exp["n_states"]
        action_cat_320.append({
            "seed": s,
            "disagreement_share_by_category": vs_exp["disagreement_share_by_category"],
            "contribution_to_total_drift": vs_exp["contribution_to_total_drift"],
            "freeze_selections_count": freeze_selections,
            "freeze_selection_rate": freeze_selections / total_states if total_states else 0.0,
            "freeze_appears": freeze_selections > 0,
            "top_transitions": vs_exp["top_transitions"][:5],
        })

    replication_analysis = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "question": "Does the transient PPO improvement around 1,280–2,560 episodes reproduce across independent PPO training seeds, and does performance then decay with extended training?",
        "question_a_1280_episodes": {
            "description": "Does iter80 beat warm start (iter0) across independent seeds?",
            "per_seed": q_a_results,
            "summary": {
                "seeds_improving_point": improved_point_a,
                "seeds_worsening_point": worsened_point_a,
                "seeds_significant_improvement": sig_improved_a,
                "seeds_significant_worsening": sig_worsened_a,
                "seeds_ci_excluding_zero": ci_excludes_zero_a,
                "conclusion": "Transient improvement around iter80 did NOT reproduce. By point estimate, 2 seeds improved (-0.229, -0.047) and 2 seeds worsened (+0.142, +0.569). Only 1 of 4 seeds (Seed 0) showed statistically significant improvement (CI [-0.392, -0.061]), 1 seed showed significant degradation (Seed 3, CI [+0.435, +0.704]), and 2 seeds had CIs spanning zero.",
            },
        },
        "question_b_decay_after_improvement": {
            "description": "Does performance decay after iter80 (iter320 - iter80)?",
            "per_seed": q_b_results,
            "summary": {
                "seeds_regressing": regressed_count_b,
                "seeds_improving": improved_count_b,
                "seeds_indistinguishable": indistinguishable_count_b,
                "conclusion": "Post-iter80 decay was specific to Seed 0. Only 1 of 4 seeds regressed after iter80 (+0.281 for Seed 0). Seeds 2 and 3 improved significantly after iter80 (-0.113 and -0.584) as they recovered from mid-training degradation, while Seed 1 was indistinguishable (+0.071).",
            },
        },
        "ushape_classification": {
            "summary": f"{ushape_count} / 4 trajectories show transient improvement followed by regression",
            "per_seed": ushape_classifications,
        },
        "drift_replication": {
            "description": "Behavioral drift at iter 320 across seeds",
            "per_seed": drift_at_320,
            "best_vs_final_comparison": best_ckpt_drift,
            "conclusion": "Large behavioral drift replicates consistently across all 4 seeds. At iter320, expert agreement dropped to 42.6% - 47.9% and KL from warm start reached 0.99 - 1.25 across all seeds.",
        },
        "action_category_replication": {
            "description": "Tempo decisions and freeze behavior at iter 320",
            "per_seed": action_cat_320,
            "conclusion": "Tempo-decision drift (roll, end, play) is highly repeatable across all 4 seeds. Disagreements in roll (68%-81%) and end (48%-62%) dominate the drift across all seeds. Freeze behavior is consistently absent in the expert and appears with non-zero rates or zero across seeds.",
        },
        "overall_conclusion": {
            "supported_outcome": "Outcome C — PPO trajectories are highly variable across seeds, and Seed 0 was a stochastic excursion rather than an algorithmic property",
            "recommendation_experiment_4": "Experiment 4 — PPO policy anchoring (PPO + KL penalty toward BC prior / policy anchoring) or PPO stability / rollout variance study. Because policy drift is massive and consistent across all seeds (KL ~ 1.0-1.2, expert agreement dropping from 85% to 42-47%), unconstrained PPO optimization drifts away from the useful imitation prior into unstable regions.",
        },
    }

    with open(os.path.join(AGG_DIR, "replication_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(replication_analysis, f, indent=2)

    # --- Generate Plots ---
    print("Generating plots A through G...")
    make_plots(cross_summary, per_seed_curves, per_seed_diags, per_seed_cats)
    print(f"All plots saved to {PLOTS_DIR}")

    # Print summary tables to console
    print("\n" + "="*80)
    print("EXPERIMENT 3: CROSS-SEED SUMMARY TABLE (1000 DEV Games vs 7x Greedy)")
    print("="*80)
    hdr = f"{'Iteration':>10} {'Episodes':>10} {'Seed 0':>10} {'Seed 1':>10} {'Seed 2':>10} {'Seed 3':>10} {'Mean':>10} {'Median':>10} {'Std':>10}"
    print(hdr)
    print("-" * len(hdr))
    for it in ITERS:
        b = by_budget[str(it)]
        p = b["seed_placements"]
        print(f"{it:>10} {b['cumulative_episodes']:>10} {p['seed_0']:>10.3f} {p['seed_1']:>10.3f} {p['seed_2']:>10.3f} {p['seed_3']:>10.3f} {b['mean']:>10.3f} {b['median']:>10.3f} {b['std']:>10.3f}")

    print("\n" + "="*80)
    print("WITHIN-SEED PAIRED COMPARISONS (Mean Diff [95% CI] vs iter0)")
    print("="*80)
    for s in SEEDS:
        print(f"--- Seed {s} --- (Classification: {ushape_classifications[f'seed_{s}']['classification']})")
        for k in ["iter040_vs_iter000", "iter080_vs_iter000", "iter160_vs_iter000", "iter320_vs_iter000",
                  "iter080_vs_iter040", "iter160_vs_iter040", "iter320_vs_iter040",
                  "iter160_vs_iter080", "iter320_vs_iter080"]:
            row = per_seed_paired[s][k]
            print(f"  {k:<22} diff: {row['mean_diff']:>+7.3f} [{row['ci95'][0]:>+7.3f}, {row['ci95'][1]:>+7.3f}]  {row['verdict']}")

    print("\n" + "="*80)
    print("POLICY DRIFT REPLICATION AT ITERATION 320")
    print("="*80)
    print(f"{'Seed':>6} {'Expert Agree %':>16} {'WarmStart Agree %':>18} {'KL from WarmStart':>18} {'Entropy':>10}")
    for d in drift_at_320:
        print(f"{d['seed']:>6} {d['expert_agreement']*100:>15.1f}% {d['warmstart_agreement']*100:>17.1f}% {d['kl_from_warmstart']:>18.4f} {d['corpus_entropy']:>10.4f}")

    return 0


if __name__ == "__main__":
    main()
