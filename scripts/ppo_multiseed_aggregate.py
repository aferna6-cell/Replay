"""Cross-seed aggregation and plots for Experiment 3.

Reads committed JSON only — seed 0 from Experiment 2
(``results/ppo_budget_v1/``), seeds 1–3 from
``results/ppo_multiseed_v1/seed_{S}/``. Writes aggregate JSON and plots
A–G. Plot series are taken from those files; no numeric literals.

    python scripts/ppo_multiseed_aggregate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.action_categories import CATEGORIES  # noqa: E402
from ml.multiseed_analysis import (  # noqa: E402
    ALL_SEEDS, MIXED_FIELD, MULTI_DIR, PRIMARY_ITERS, SEED0_DIR,
    assemble_replication, episodes, load_rl_signal, load_seed_bundle,
    outcome_and_recommendation, pair_key, write_json,
)


def load_bundles(seeds=ALL_SEEDS):
    return {s: load_seed_bundle(s) for s in seeds}


def plot_from_json(bundles, analysis, out_dir: str) -> list:
    """Machine-generated plots. All series come from ``bundles`` / ``analysis``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []
    eps = [episodes(it) for it in PRIMARY_ITERS]
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
    markers = {0: "o", 1: "s", 2: "D", 3: "^"}

    def save(fig, name):
        path = os.path.join(out_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    def series(key):
        return {s: [_curve(bundles[s], it)[key] for it in PRIMARY_ITERS]
                for s in bundles}

    def _curve(bundle, it):
        return next(c for c in bundle["curve"]["curve"] if c["iteration"] == it)

    # A. Multi-seed DEV learning curves
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    greedy = series("greedy_avg")
    for s, ys in greedy.items():
        ax.plot(eps, ys, color=colors[s], marker=markers[s],
                linewidth=1.8, markersize=6, label=f"seed {s}")
    ax.invert_yaxis()
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("DEV avg placement (lower is better)")
    ax.set_title("A. Multi-seed DEV learning curves — 7× greedy (1000 games)",
                 fontsize=11)
    ax.set_xticks(eps)
    ax.set_xticklabels([f"{e}\n(it {it})" for e, it in zip(eps, PRIMARY_ITERS)],
                       fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "A_multiseed_dev_curves.png")

    # B. Mean across seeds + individual points/lines
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for s, ys in greedy.items():
        ax.plot(eps, ys, color=colors[s], marker=markers[s],
                linewidth=0.9, alpha=0.65, markersize=5, label=f"seed {s}")
    means = [analysis["cross_seed_summary"]["by_iteration"][str(it)]["mean"]
             for it in PRIMARY_ITERS]
    ax.plot(eps, means, color="#111", linewidth=2.4, marker="o",
            markersize=7, label="mean across seeds")
    ax.invert_yaxis()
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("DEV avg placement (lower is better)")
    ax.set_title("B. Cross-seed mean DEV curve (individual seeds visible)",
                 fontsize=11)
    ax.set_xticks(eps)
    ax.set_xticklabels([f"{e}\n(it {it})" for e, it in zip(eps, PRIMARY_ITERS)],
                       fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "B_cross_seed_mean.png")

    # C / D / E. agreement and KL
    for name, key, ylabel, title in [
            ("C_expert_agreement.png", "expert_agreement",
             "greedy expert-action agreement",
             "C. Expert agreement by training seed"),
            ("D_kl_from_warmstart.png", "kl_from_warmstart",
             "mean KL(π₀ || πₖ)",
             "D. KL from warm start by training seed"),
            ("E_warmstart_agreement.png", "warmstart_agreement",
             "agreement with iteration-0 argmax",
             "E. Warm-start agreement by training seed")]:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for s, ys in series(key).items():
            ax.plot(eps, ys, color=colors[s], marker=markers[s],
                    linewidth=1.8, markersize=6, label=f"seed {s}")
        ax.set_xlabel("cumulative PPO training episodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e}" for e in eps], fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        save(fig, name)

    # F. action-category drift at iter320
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    width = 0.8 / max(1, len(bundles))
    xs = range(len(CATEGORIES))
    cat_rows = analysis["category_replication"]["per_seed"]
    for i, row in enumerate(cat_rows):
        shares = [(row["disagreement_share_by_category"].get(cat) or 0.0)
                  for cat in CATEGORIES]
        ax.bar([x + i * width for x in xs], shares, width,
               color=colors[row["training_seed"]],
               label=f"seed {row['training_seed']}")
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("share of expert decisions changed")
    ax.set_title("F. Action-category disagreement at iter 320, by seed",
                 fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    save(fig, "F_category_drift_iter320.png")

    # G. PPO optimization diagnostics across seeds
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))
    keys = (("entropy", "policy entropy"),
            ("approx_kl", "approximate KL"),
            ("clip_frac", "PPO clip fraction"),
            ("value_explained_variance", "value explained variance"))
    for ax, (key, label) in zip(axes.flat, keys):
        for s, bundle in bundles.items():
            rl = load_rl_signal(bundle["source_dir"])
            it = [r["iter"] for r in rl["per_iteration"]]
            ys = [r.get(key) for r in rl["per_iteration"]]
            ax.plot(it, ys, color=colors[s], linewidth=0.8, alpha=0.85,
                    label=f"seed {s}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.25)
    fig.suptitle("G. PPO optimization diagnostics across training seeds",
                 fontsize=11)
    axes[0, 0].legend(fontsize=7)
    save(fig, "G_rl_diagnostics.png")
    return written


def main(argv=None) -> int:
    bundles = load_bundles()
    analysis = assemble_replication(bundles)
    decision = outcome_and_recommendation(analysis)
    analysis["outcome"] = decision

    agg = os.path.join(MULTI_DIR, "aggregate")
    os.makedirs(agg, exist_ok=True)
    write_json(os.path.join(agg, "cross_seed_summary.json"),
               analysis["cross_seed_summary"])
    write_json(os.path.join(agg, "paired_results.json"), {
        "method": "deterministic paired percentile bootstrap, 10000 resamples, seed 0",
        "note": "positive = first checkpoint worse; replication unit = training seed",
        "within_seed": {str(s): bundles[s]["paired"] for s in bundles},
        "question_a_iter80_minus_iter0":
            analysis["question_a_1280_episode_replication"],
        "question_b_iter320_minus_iter80":
            analysis["question_b_late_regression"],
    })
    write_json(os.path.join(agg, "replication_analysis.json"), analysis)

    plots = os.path.join(agg, "plots")
    written = plot_from_json(bundles, analysis, plots)

    print("Cross-seed greedy averages (DEV, 1000 games vs 7× greedy)")
    hdr = f"{'iter':>5} {'eps':>6}" + "".join(f"{'s'+str(s):>8}" for s in bundles)
    hdr += f"{'mean':>8} {'std':>7}"
    print(hdr)
    for it in PRIMARY_ITERS:
        row = analysis["cross_seed_summary"]["by_iteration"][str(it)]
        cells = "".join(f"{row['per_seed'][s]:8.3f}" for s in bundles)
        std = row["std"] if row["std"] is not None else float("nan")
        print(f"{it:>5} {episodes(it):>6}{cells}{row['mean']:8.3f}{std:7.3f}")

    print("\nQuestion A  iter80 − iter0")
    for r in analysis["question_a_1280_episode_replication"]["per_seed"]:
        print(f"  seed {r['training_seed']}: {r['mean_diff']:+.3f}  "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['direction']}")
    print("\nQuestion B  iter320 − iter80")
    for r in analysis["question_b_late_regression"]["per_seed"]:
        print(f"  seed {r['training_seed']}: {r['mean_diff']:+.3f}  "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['direction']}")
    print("\nU-shape:", analysis["ushape"]["statement"])
    for s, sh in analysis["ushape"]["per_seed"].items():
        print(f"  seed {s}: {sh['label']}")
    print("\n" + decision["outcome_text"])
    print(decision["recommendation"])
    print("Plots:", ", ".join(written))
    print(f"Seed 0 loaded from {SEED0_DIR} (read-only)")
    print(f"Mixed diagnostic field: {MIXED_FIELD}")
    print(f"Pair keys checked: {pair_key(80, 0)}, {pair_key(320, 80)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
