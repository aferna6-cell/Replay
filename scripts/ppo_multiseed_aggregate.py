"""Aggregate Experiment 3 multi-seed results + plots A–G.

    python scripts/ppo_multiseed_aggregate.py

Reads only committed/per-seed JSON under results/ppo_multiseed_v1/.
Never touches Benchmark v1 TEST.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.action_categories import CATEGORIES
from ml.multiseed_analysis import (PRIMARY_ITERS, classify_u_shape,
                                   load_json, load_seed_greedy,
                                   replication_questions, summarize_across_seeds,
                                   within_seed_paired)

ROOT = "results/ppo_multiseed_v1"
SEEDS = [0, 1, 2, 3]
AGG = f"{ROOT}/aggregate"


def seed_dir(s: int) -> str:
    return f"{ROOT}/seed_{s}"


def main() -> int:
    os.makedirs(f"{AGG}/plots", exist_ok=True)
    per_seed_curves = {}
    per_seed_paired = {}
    per_seed_u = {}
    per_seed_drift320 = {}
    per_seed_cats320 = {}
    per_seed_rl = {}

    for s in SEEDS:
        d = seed_dir(s)
        if not os.path.isdir(d):
            raise SystemExit(f"missing {d}")
        lc = load_json(f"{d}/learning_curve.json")
        # Prefer Exp3 schema; rebuild paired from DEV if needed.
        greedy = load_seed_greedy(d)
        paired = within_seed_paired(greedy)
        per_seed_paired[s] = paired
        per_seed_u[s] = classify_u_shape(paired)
        curve = lc.get("curve")
        if not curve or "training_seed" not in curve[0]:
            # rebuild via run_seed summarize path is preferred; fall back
            raise SystemExit(f"{d}/learning_curve.json missing Exp3 curve schema")
        per_seed_curves[s] = curve

        drift = load_json(f"{d}/policy_drift.json")
        cats = load_json(f"{d}/action_category_drift.json")
        per_seed_drift320[s] = next(
            r for r in drift["checkpoints"] if r["checkpoint"] == "iter_320.pt")
        per_seed_cats320[s] = next(
            r for r in cats["checkpoints"] if r["checkpoint"] == "iter_320.pt")
        rl = load_json(f"{d}/rl_signal.json")
        per_seed_rl[s] = rl.get("blocks", {})

    cross = summarize_across_seeds(per_seed_curves)
    repl = replication_questions(per_seed_paired)

    paired_out = {
        "method": "within-seed paired percentile bootstrap over 1000 DEV games; "
                  "training seed is the replication unit",
        "note": "positive = first checkpoint worse than reference",
        "per_seed": {str(s): rows for s, rows in per_seed_paired.items()},
        "u_shape_classification": {str(s): u for s, u in per_seed_u.items()},
    }
    with open(f"{AGG}/paired_results.json", "w") as f:
        json.dump(paired_out, f, indent=2)

    # Action-category replication at iter320 (tempo + freeze appearance)
    cat_rep = {}
    for s in SEEDS:
        ce = per_seed_cats320[s]["vs_expert"]
        shares = ce["disagreement_share_by_category"]
        top = ce.get("top_transitions", [])
        freeze_from_expert = sum(
            r["count"] for r in top if r.get("to") == "freeze")
        # also count freeze mass in confusion matrix if present
        cm = ce.get("confusion_matrix", {})
        freeze_count = sum(row.get("freeze", 0) for row in cm.values())
        cat_rep[str(s)] = {
            "expert_agreement": ce["overall_agreement"],
            "disagreement_share_by_category": shares,
            "tempo_shares": {k: shares.get(k) for k in ("roll", "end", "play")},
            "freeze_appearances_in_confusion": freeze_count,
            "top_transitions": top[:8],
        }

    # Drift replication at iter320 + relationship best-ckpt vs later drift
    drift_rep = {}
    for s in SEEDS:
        curve = per_seed_curves[s]
        best = min(curve, key=lambda c: c["greedy_avg"])
        late = next(c for c in curve if c["iteration"] == 320)
        mid = next(c for c in curve if c["iteration"] == 80)
        d320 = per_seed_drift320[s]
        drift_rep[str(s)] = {
            "iter320_expert_agreement": d320["expert_agreement"],
            "iter320_warmstart_agreement": d320["warmstart_agreement"],
            "iter320_kl": d320["kl_from_warmstart_mean"],
            "best_checkpoint_iteration": best["iteration"],
            "best_greedy_avg": best["greedy_avg"],
            "iter80_greedy_avg": mid["greedy_avg"],
            "iter320_greedy_avg": late["greedy_avg"],
            "descriptive_note": (
                "relationship between best-checkpoint placement and later "
                "drift is descriptive only (n=4)"),
        }

    replication = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "n_training_seeds": 4,
        "inferential_caution": repl["inferential_caution"],
        "questions": repl,
        "u_shape_by_seed": {str(s): u for s, u in per_seed_u.items()},
        "seed0_u_shape_replicated": sum(
            1 for u in per_seed_u.values()
            if u["class"] == "u_like_transient_improvement") >= 2,
        "n_seeds_u_like": sum(
            1 for u in per_seed_u.values()
            if u["class"] == "u_like_transient_improvement"),
        "drift_at_iter320": drift_rep,
        "action_category_at_iter320": cat_rep,
        "rl_signal_blocks_by_seed": {str(s): b for s, b in per_seed_rl.items()},
    }
    with open(f"{AGG}/replication_analysis.json", "w") as f:
        json.dump(replication, f, indent=2)

    summary = {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "training_seeds": SEEDS,
        "primary_iterations": PRIMARY_ITERS,
        "cross_seed_by_budget": {str(k): v for k, v in cross.items()},
        "per_seed_curves": {str(s): c for s, c in per_seed_curves.items()},
        "caution": "n=4 is exploratory; do not hide individual seeds",
    }
    with open(f"{AGG}/cross_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_tables(cross, per_seed_paired, per_seed_u, repl)
    _plots(per_seed_curves, per_seed_cats320, per_seed_rl)
    print(f"Saved aggregate JSON + plots under {AGG}/")
    return 0


def _print_tables(cross, paired, u_shape, repl) -> None:
    print("\nCross-seed DEV greedy avg by budget (individuals + mean)")
    print(f"{'iter':>5} {'eps':>6} {'mean':>7} {'med':>7} {'min':>7} {'max':>7} "
          f"{'std':>6}  per-seed")
    for it in PRIMARY_ITERS:
        r = cross[it]
        seeds = " ".join(f"s{p['training_seed']}={p['greedy_avg']:.3f}"
                         for p in r["per_seed"])
        print(f"{it:>5} {r['cumulative_episodes']:>6} "
              f"{r['greedy_avg_mean']:>7.3f} {r['greedy_avg_median']:>7.3f} "
              f"{r['greedy_avg_min']:>7.3f} {r['greedy_avg_max']:>7.3f} "
              f"{r['greedy_avg_std']:>6.3f}  {seeds}")

    print("\nWithin-seed paired (positive = first worse)")
    for s, rows in paired.items():
        print(f"  seed {s}  class={u_shape[s]['class']}")
        for r in rows:
            z = "excl0" if not (r["ci95"][0] <= 0 <= r["ci95"][1]) else "incl0"
            print(f"    {r['label']:<16} {r['mean_diff']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {z}")

    qa, qb = repl["question_A_iter80_minus_iter0"], repl["question_B_iter320_minus_iter80"]
    print(f"\nQ-A iter80−iter0: improve={qa['seeds_improve']} "
          f"worsen={qa['seeds_worsen']} clear={qa['seeds_ci_excludes_zero']}")
    print(f"Q-B iter320−iter80: regress={qb['seeds_regress']} "
          f"continue={qb['seeds_continue_improve']} "
          f"clear={qb['seeds_ci_excludes_zero']}")


def _plots(curves, cats320, rl_blocks) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [it * 16 for it in PRIMARY_ITERS]
    colors = {0: "#1f4e79", 1: "#c45c26", 2: "#2a7f62", 3: "#6b4c9a"}

    # A. multi-seed DEV greedy curves
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    for s, curve in curves.items():
        ys = [c["greedy_avg"] for c in curve]
        ax.plot(eps, ys, "o-", color=colors[s], label=f"seed {s}",
                linewidth=1.6, markersize=5)
    ax.invert_yaxis()
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("DEV avg placement (lower better)")
    ax.set_title("A. Multi-seed DEV placement vs budget — 7× greedy (1000 games)")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/A_multiseed_dev_greedy.png", dpi=140)
    plt.close(f)

    # B. mean + individuals
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    for s, curve in curves.items():
        ax.plot(eps, [c["greedy_avg"] for c in curve], "-", color=colors[s],
                alpha=0.45, linewidth=1.2)
    means = [st.mean(curves[s][i]["greedy_avg"] for s in curves)
             for i in range(len(PRIMARY_ITERS))]
    ax.plot(eps, means, "o-", color="#111", linewidth=2.2, markersize=6,
            label="mean across seeds")
    ax.invert_yaxis()
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("DEV avg placement (lower better)")
    ax.set_title("B. Cross-seed mean DEV curve with individual seeds")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/B_mean_with_individuals.png", dpi=140)
    plt.close(f)

    # C. expert agreement
    f, ax = plt.subplots(figsize=(7.6, 4.4))
    for s, curve in curves.items():
        ax.plot(eps, [c["expert_agreement"] for c in curve], "o-",
                color=colors[s], label=f"seed {s}", linewidth=1.5)
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("greedy-expert argmax agreement")
    ax.set_title("C. Expert-action agreement vs budget (multi-seed)")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/C_expert_agreement.png", dpi=140)
    plt.close(f)

    # D. KL from warm start
    f, ax = plt.subplots(figsize=(7.6, 4.4))
    for s, curve in curves.items():
        ax.plot(eps, [c["kl_from_warmstart"] for c in curve], "s-",
                color=colors[s], label=f"seed {s}", linewidth=1.5)
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("mean KL(π₀ ‖ πₖ)")
    ax.set_title("D. KL from warm start vs budget (multi-seed)")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/D_kl_from_warmstart.png", dpi=140)
    plt.close(f)

    # E. warm-start agreement
    f, ax = plt.subplots(figsize=(7.6, 4.4))
    for s, curve in curves.items():
        ax.plot(eps, [c["warmstart_agreement"] for c in curve], "o-",
                color=colors[s], label=f"seed {s}", linewidth=1.5)
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("argmax agreement with iter-0")
    ax.set_title("E. Warm-start agreement vs budget (multi-seed)")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/E_warmstart_agreement.png", dpi=140)
    plt.close(f)

    # F. action-category drift at iter320
    f, ax = plt.subplots(figsize=(8.6, 4.6))
    width = 0.8 / len(SEEDS)
    xs = range(len(CATEGORIES))
    for i, s in enumerate(SEEDS):
        shares = cats320[s]["vs_expert"]["disagreement_share_by_category"]
        vals = [(shares.get(cat) or 0.0) for cat in CATEGORIES]
        ax.bar([x + i * width for x in xs], vals, width,
               color=colors[s], label=f"seed {s}")
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("share of expert decisions changed")
    ax.set_title("F. Action-category disagreement vs expert at iter 320")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/F_category_disagreement_iter320.png", dpi=140)
    plt.close(f)

    # G. PPO opt diagnostics — mean |adv|, EV, entropy, clip across seeds
    f, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    metrics = [
        ("adv_mean_abs", "mean |raw advantage|"),
        ("value_explained_variance", "value explained variance"),
        ("entropy", "policy entropy"),
        ("clip_frac", "PPO clip fraction"),
    ]
    block_names = ["iters_1_40", "iters_41_160", "iters_161_320"]
    x = [1, 2, 3]
    for ax, (key, title) in zip(axes.flat, metrics):
        for s in SEEDS:
            ys = [rl_blocks[s][b][key] for b in block_names]
            ax.plot(x, ys, "o-", color=colors[s], label=f"seed {s}",
                    linewidth=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["1–40", "41–160", "161–320"])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    f.suptitle("G. RL signal / PPO optimization diagnostics by training block",
               fontsize=11)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/G_rl_signal_blocks.png", dpi=140)
    plt.close(f)
    print(f"Saved plots -> {AGG}/plots/")


if __name__ == "__main__":
    raise SystemExit(main())
