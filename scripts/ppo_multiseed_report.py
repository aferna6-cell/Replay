"""Assemble the Experiment 3 (multi-seed PPO budget replication) analysis.

Reads only committed result JSON — Experiment 2's seed-0 artifacts (read
only, never rewritten) and the seed 1/2/3 artifacts produced by
``scripts/ppo_multiseed_eval.py`` — and emits the per-seed learning curves,
the within-seed paired comparisons, the cross-training-seed summary, the
pre-specified replication analysis, and the plots.

    python scripts/ppo_multiseed_report.py

DEV split only. Benchmark v1 TEST is never read or run.

The replication unit is the TRAINING SEED. The 1000 paired DEV games make one
trained model's placement precise; four training trajectories is the entire
sample that speaks to training variability, and it is a small one.
"""

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.replication import (COMPARISONS, PRIMARY_ITERS,          # noqa: E402
                            SHAPE_CLASSES, SHAPE_RULE_DOC, build_curve,
                            build_plot_data, category_replication,
                            classify_curve, cross_seed_summary,
                            describe, drift_vs_performance,
                            effect_across_seeds, episodes, exploratory_ci,
                            load_seed_bundle, paired_table, rl_blocks,
                            spearman)

ROOT = "results/ppo_multiseed_v1"
AGG = f"{ROOT}/aggregate"
EXP2 = "results/ppo_budget_v1"
NEW_SEEDS = [1, 2, 3]
ALL_SEEDS = [0] + NEW_SEEDS


def _bundle(seed):
    if seed == 0:            # Experiment 2 artifacts — read only
        return load_seed_bundle(0, f"{EXP2}/dev", f"{EXP2}/policy_drift.json",
                                f"{EXP2}/action_category_drift.json",
                                f"{EXP2}/train_diag.jsonl")
    d = f"{ROOT}/seed_{seed}"
    return load_seed_bundle(seed, f"{d}/dev", f"{d}/policy_drift.json",
                            f"{d}/action_category_drift.json",
                            f"{d}/train_diag.jsonl")


def _print_curve(seed, curve):
    hdr = (f"{'iter':>5} {'episodes':>9} {'GreedyAvg':>10} {'95% CI':>18} "
           f"{'Top4':>7} {'Win':>6} {'MixedAvg':>9} {'Expert%':>8} "
           f"{'WarmSt%':>8} {'KL':>7}")
    print(f"\n--- training seed {seed} — DEV learning curve "
          f"(1000 games vs 7x greedy; 500 vs greedy4_random3) ---")
    print(hdr)
    print("-" * len(hdr))
    for c in curve:
        ci = c["greedy_ci95"]
        print(f"{c['iteration']:>5} {c['cumulative_episodes']:>9} "
              f"{c['greedy_avg']:>10.3f} "
              f"{'[%.3f, %.3f]' % (ci['low'], ci['high']):>18} "
              f"{100 * c['greedy_top4']:>6.1f}% {100 * c['greedy_win']:>5.1f}% "
              f"{c['mixed_avg']:>9.3f} {100 * c['expert_agreement']:>7.1f}% "
              f"{100 * c['warmstart_agreement']:>7.1f}% "
              f"{c['kl_from_warmstart']:>7.4f}")


def _print_paired(seed, rows):
    print(f"\n  paired DEV comparisons, training seed {seed} "
          f"(positive = first checkpoint WORSE)")
    for r in rows:
        print(f"    {r['label']:<18} {r['mean_diff']:>+8.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"{r['significance']}")


def main() -> int:
    bundles = {s: _bundle(s) for s in ALL_SEEDS}
    curves = {s: build_curve(b) for s, b in bundles.items()}
    paired = {s: paired_table(b["greedy"]) for s, b in bundles.items()}
    paired_mixed = {s: paired_table(b["mixed"],
                                    comparisons=[(i, 0) for i in
                                                 PRIMARY_ITERS[1:]])
                    for s, b in bundles.items()}
    blocks = {s: rl_blocks(b["diag"]) for s, b in bundles.items()}

    # --- per-seed artifacts (seed 0's Experiment 2 files are NOT touched) ---
    for s in NEW_SEEDS:
        _print_curve(s, curves[s])
        _print_paired(s, paired[s])
        out = {
            "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget "
                          "Replication",
            "evaluation_split": "dev",
            "training_seed": s,
            "episodes_per_iteration": 16,
            "primary_iterations": PRIMARY_ITERS,
            "greedy_games": bundles[s]["greedy"][0]["games"],
            "mixed_games": bundles[s]["mixed"][0]["games"],
            "dev_seed_range_greedy": bundles[s]["greedy"][0]["seed_range"],
            "dev_seed_range_mixed": bundles[s]["mixed"][0]["seed_range"],
            "curve": curves[s],
            "paired_greedy": paired[s],
            "paired_mixed_vs_iter0": paired_mixed[s],
        }
        with open(f"{ROOT}/seed_{s}/learning_curve.json", "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        with open(f"{ROOT}/seed_{s}/rl_signal.json", "w",
                  encoding="utf-8") as f:
            json.dump({"training_seed": s, "definitions": {
                "adv_*": "GAE advantages BEFORE PPO's per-batch normalization",
                "value_explained_variance":
                    "1 - Var(returns - value_preds) / Var(returns)",
                "shaping_reward_sum/terminal_reward_sum":
                    "the two reward sources separated per iteration"},
                "per_iteration": bundles[s]["diag"], "blocks": blocks[s]},
                f, indent=2)

    # --- cross-training-seed summary ---------------------------------------
    avg_by_seed = {s: {r["iteration"]: r["greedy_avg"] for r in curves[s]}
                   for s in ALL_SEEDS}
    mixed_by_seed = {s: {r["iteration"]: r["mixed_avg"] for r in curves[s]}
                     for s in ALL_SEEDS}
    summary = cross_seed_summary(avg_by_seed)
    summary["greedy_field"] = summary.pop("per_budget")
    summary["mixed_diagnostic_field"] = cross_seed_summary(
        mixed_by_seed)["per_budget"]
    summary["per_seed_curves"] = {
        str(s): [{k: r[k] for k in
                  ("iteration", "cumulative_episodes", "greedy_avg",
                   "greedy_ci95", "greedy_top4", "greedy_win", "mixed_avg",
                   "expert_agreement", "warmstart_agreement",
                   "kl_from_warmstart", "corpus_entropy")}
                 for r in curves[s]] for s in ALL_SEEDS}
    summary["caveat"] = (
        f"{len(ALL_SEEDS)} training seeds is a small sample. The per-budget "
        f"mean/median/sd below describe these four trajectories; they are "
        f"not population estimates for PPO.")

    print("\n=== cross-training-seed DEV placement (7x greedy, 1000 games) ===")
    hdr = (f"{'episodes':>9} " + " ".join(f"{'seed ' + str(s):>9}"
                                          for s in ALL_SEEDS)
           + f" {'mean':>8} {'median':>8} {'min':>8} {'max':>8} {'sd':>7}")
    print(hdr)
    print("-" * len(hdr))
    for row in summary["greedy_field"]:
        print(f"{row['cumulative_episodes']:>9} "
              + " ".join(f"{row['by_seed'][str(s)]:>9.3f}" for s in ALL_SEEDS)
              + f" {row['mean']:>8.3f} {row['median']:>8.3f} "
                f"{row['min']:>8.3f} {row['max']:>8.3f} {row['sd']:>7.3f}")

    # --- pre-specified replication questions --------------------------------
    q_a = effect_across_seeds(paired, 80, 0)
    q_a["question"] = ("Question A — does the 1,280-episode improvement seen "
                       "on training seed 0 reproduce on independent seeds?")
    q_b = effect_across_seeds(paired, 320, 80)
    q_b["question"] = ("Question B — does extended training regress relative "
                       "to the 1,280-episode checkpoint?")
    all_effects = {f"iter{t}_minus_iter{r}": effect_across_seeds(paired, t, r)
                   for t, r in COMPARISONS}

    shapes = {s: classify_curve(avg_by_seed[s], paired[s]) for s in ALL_SEEDS}
    n_u = sum(1 for s in ALL_SEEDS
              if shapes[s]["shape_class"] == SHAPE_CLASSES[0])
    n_point_transient = sum(1 for s in ALL_SEEDS
                            if shapes[s]["descriptive"]
                            ["point_estimate_transient"])

    print("\n=== pre-specified replication questions ===")
    for q in (q_a, q_b):
        print(f"\n{q['question']}")
        print(f"  ({q['sign_convention']})")
        for r in q["per_seed"]:
            print(f"    seed {r['training_seed']}: {r['mean_diff']:>+7.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
                  f"{r['significance']}")
        print(f"    point estimates: {q['n_point_estimate_better']} better / "
              f"{q['n_point_estimate_worse']} worse; CIs excluding zero: "
              f"{q['n_ci_excludes_zero_better']} better, "
              f"{q['n_ci_excludes_zero_worse']} worse, "
              f"{q['n_ci_includes_zero']} inconclusive")

    print("\n=== curve shape (pre-specified automated rule) ===")
    for s in ALL_SEEDS:
        print(f"  seed {s}: {shapes[s]['shape_class']} — "
              f"{shapes[s]['reason']}")
    print(f"  {n_u} / {len(ALL_SEEDS)} trajectories show transient "
          f"improvement followed by regression (significant, by the rule); "
          f"{n_point_transient} / {len(ALL_SEEDS)} by point estimates alone")

    # --- drift replication ---------------------------------------------------
    drift_rows = []
    for s in ALL_SEEDS:
        last = next(r for r in curves[s] if r["iteration"] == 320)
        drift_rows.append({"training_seed": s,
                           "expert_agreement": last["expert_agreement"],
                           "warmstart_agreement": last["warmstart_agreement"],
                           "kl_from_warmstart": last["kl_from_warmstart"],
                           "corpus_entropy": last["corpus_entropy"],
                           "value_mean": last["value_mean"],
                           "value_std": last["value_std"]})
    pooled = [(r["greedy_avg"], r["expert_agreement"], r["kl_from_warmstart"])
              for s in ALL_SEEDS for r in curves[s]]
    drift = {
        "at_iteration_320": drift_rows,
        "across_seeds": {k: describe([r[k] for r in drift_rows]) for k in
                         ("expert_agreement", "warmstart_agreement",
                          "kl_from_warmstart", "corpus_entropy")},
        "per_seed_trajectories": {
            str(s): [{"iteration": r["iteration"],
                      "cumulative_episodes": r["cumulative_episodes"],
                      "expert_agreement": r["expert_agreement"],
                      "warmstart_agreement": r["warmstart_agreement"],
                      "kl_from_warmstart": r["kl_from_warmstart"],
                      "corpus_entropy": r["corpus_entropy"]}
                     for r in curves[s]] for s in ALL_SEEDS},
        "best_checkpoint_vs_later": {str(s): drift_vs_performance(curves[s])
                                     for s in ALL_SEEDS},
        "pooled_association": {
            "n_points": len(pooled),
            "spearman_placement_vs_expert_agreement":
                spearman([p[0] for p in pooled], [p[1] for p in pooled]),
            "spearman_placement_vs_kl":
                spearman([p[0] for p in pooled], [p[2] for p in pooled]),
            "note": ("20 (seed, budget) points that are not independent — "
                     "five budgets share a trajectory. Descriptive "
                     "association only; no causal claim.")},
    }
    print("\n=== drift at 5,120 episodes, by training seed ===")
    print(f"{'seed':>5} {'expert%':>9} {'warmstart%':>11} {'KL':>8} "
          f"{'entropy':>9}")
    for r in drift_rows:
        print(f"{r['training_seed']:>5} {100 * r['expert_agreement']:>8.1f}% "
              f"{100 * r['warmstart_agreement']:>10.1f}% "
              f"{r['kl_from_warmstart']:>8.4f} {r['corpus_entropy']:>9.4f}")

    # --- action-category replication ----------------------------------------
    cats320 = {s: category_replication(bundles[s]["categories"][320])
               for s in ALL_SEEDS}
    cat_out = {
        "iteration": 320, "cumulative_episodes": episodes(320),
        "expert_never_freezes_note": (
            "Experiment 2 found the greedy expert selects `freeze` in 0 of "
            "the 4,440 corpus states; any freeze below is an action the "
            "policy introduced."),
        "buy_artifact_note": (
            "Experiment 2 showed `buy` disagreement is a warm-start artifact "
            "(~0.74 at iteration 0, flat across budgets) — the clone picks a "
            "different shop slot. Compare `buy` against each seed's own "
            "iteration 0, not against zero."),
        "by_seed": {str(s): cats320[s] for s in ALL_SEEDS},
        "buy_disagreement_by_budget": {
            str(s): {str(r["iteration"]):
                     r["expert_disagreement_by_category"]["buy"]
                     for r in curves[s]} for s in ALL_SEEDS},
        "tempo_disagreement_by_budget": {
            str(s): {str(r["iteration"]): {
                cat: r["expert_disagreement_by_category"][cat]
                for cat in ("roll", "end", "play")}
                for r in curves[s]} for s in ALL_SEEDS},
        "freeze_across_seeds": {
            "n_seeds_with_freeze": sum(1 for s in ALL_SEEDS
                                       if cats320[s]["freeze_appears"]),
            "n_seeds": len(ALL_SEEDS),
            "by_seed": {str(s): {"freeze_selections":
                                 cats320[s]["freeze_selections"],
                                 "freeze_rate": cats320[s]["freeze_rate"]}
                        for s in ALL_SEEDS}},
    }
    print("\n=== action-category drift at 5,120 episodes ===")
    print(f"{'seed':>5} {'agree':>7} {'roll':>7} {'end':>7} {'play':>7} "
          f"{'buy':>7} {'freeze picks':>13}")
    for s in ALL_SEEDS:
        c = cats320[s]
        d = c["disagreement_share_by_category"]
        print(f"{s:>5} {c['overall_agreement']:>7.3f} "
              + " ".join(f"{(d.get(k) or 0.0):>7.3f}"
                         for k in ("roll", "end", "play", "buy"))
              + f" {c['freeze_selections']:>13}")

    # --- RL-signal comparison across seeds -----------------------------------
    rl = {"blocks_by_seed": {str(s): blocks[s] for s in ALL_SEEDS},
          "across_seeds": {
              name: {k: describe([blocks[s][name][k] for s in ALL_SEEDS])
                     for k in ("rollout_avg_placement", "entropy", "approx_kl",
                               "clip_frac", "grad_norm", "adv_mean_abs",
                               "adv_frac_positive",
                               "value_explained_variance", "return_std",
                               "placement_std", "v_loss", "pi_loss")}
              for name in blocks[0]},
          "note": ("do internal PPO metrics separate the trajectories that "
                   "improved from the ones that did not? compared against "
                   "the per-seed outcomes in replication_questions")}
    print("\n=== RL-signal block means (iters 161-320) by training seed ===")
    keys = ("rollout_avg_placement", "entropy", "approx_kl", "clip_frac",
            "grad_norm", "adv_mean_abs", "value_explained_variance")
    print(f"{'seed':>5} " + " ".join(f"{k[:11]:>12}" for k in keys))
    for s in ALL_SEEDS:
        b = blocks[s]["iters_161_320"]
        print(f"{s:>5} " + " ".join(f"{b[k]:>12.4f}" for k in keys))

    # --- write the aggregate artifacts --------------------------------------
    os.makedirs(AGG, exist_ok=True)
    with open(f"{AGG}/cross_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(f"{AGG}/paired_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": ("deterministic paired percentile bootstrap, 10000 "
                       "resamples, seed 0 — the same implementation "
                       "Experiment 2 used (ml.analyze_benchmark)"),
            "note": ("positive difference = the first checkpoint places "
                     "worse than the reference (lower placement is better). "
                     "Every comparison is within one training seed, over the "
                     "identical 1000 DEV game seeds."),
            "comparisons": [f"iter{t} - iter{r}" for t, r in COMPARISONS],
            "greedy_by_seed": {str(s): paired[s] for s in ALL_SEEDS},
            "mixed_vs_iter0_by_seed": {str(s): paired_mixed[s]
                                       for s in ALL_SEEDS},
            "across_seeds": all_effects}, f, indent=2)
    with open(f"{AGG}/replication_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget "
                          "Replication",
            "evaluation_split": "dev",
            "replication_unit": (
                "the TRAINING SEED. 4 seeds x 1000 DEV games are NOT 4000 "
                "independent training experiments: the 1000 paired games "
                "give precision within a trained model, the 4 seeds give "
                "the (small) sample on training variability."),
            "training_seeds": ALL_SEEDS,
            "seed_0_source": "Experiment 2 (results/ppo_budget_v1), read-only",
            "replication_questions": {"question_A": q_a, "question_B": q_b},
            "shape_rule": SHAPE_RULE_DOC,
            "shape_classes": SHAPE_CLASSES,
            "shape_by_seed": {str(s): shapes[s] for s in ALL_SEEDS},
            "shape_summary": {
                "n_transient_improvement_then_regression": n_u,
                "n_seeds": len(ALL_SEEDS),
                "statement": (f"{n_u} / {len(ALL_SEEDS)} trajectories show "
                              f"transient improvement followed by regression"),
                "n_point_estimate_transient": n_point_transient,
                "class_counts": {c: sum(1 for s in ALL_SEEDS
                                        if shapes[s]["shape_class"] == c)
                                 for c in SHAPE_CLASSES}},
            "drift_replication": drift,
            "action_category_replication": cat_out,
            "rl_signal": rl,
            "exploratory_cross_seed_effects": {
                k: v["across_seed_effect"] for k, v in all_effects.items()},
        }, f, indent=2)
    print(f"\nSaved -> {AGG}/cross_seed_summary.json, paired_results.json, "
          f"replication_analysis.json")
    for s in NEW_SEEDS:
        print(f"Saved -> {ROOT}/seed_{s}/learning_curve.json, rl_signal.json")

    plot_data = build_plot_data(curves, cats320, blocks)
    _plots(plot_data)
    return 0


# --- plots: rendered from build_plot_data, never from typed-in numbers -------
_STYLES = ["o-", "s-", "^-", "d-"]
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _plots(d, out: str = f"{AGG}/plots") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    eps, seeds = d["episodes"], d["training_seeds"]

    def xaxis(ax):
        ax.set_xlabel("cumulative PPO training episodes")
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e}\n(it {i})"
                            for e, i in zip(eps, d["iterations"])], fontsize=8)
        ax.grid(alpha=0.25)

    def per_seed_fig(name, key, ylabel, title, invert=False):
        f, ax = plt.subplots(figsize=(7.6, 4.5))
        for i, s in enumerate(seeds):
            ax.plot(eps, d[key][s], _STYLES[i % len(_STYLES)],
                    color=_COLORS[i % len(_COLORS)],
                    label=f"training seed {s}"
                          + (" (Experiment 2)" if s == 0 else ""),
                    linewidth=1.7, markersize=5.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        xaxis(ax)
        if invert:
            ax.invert_yaxis()
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(f"{out}/{name}.png", dpi=140)
        plt.close(f)

    per_seed_fig("A_dev_learning_curves", "greedy_avg",
                 "DEV avg placement (lower is better)",
                 "A. DEV placement vs training budget, one line per PPO "
                 "training seed (1000 games vs 7x greedy, unsmoothed)",
                 invert=True)

    # B. mean across seeds with every seed still visible
    f, ax = plt.subplots(figsize=(7.6, 4.5))
    for i, s in enumerate(seeds):
        ax.plot(eps, d["greedy_avg"][s], _STYLES[i % len(_STYLES)],
                color=_COLORS[i % len(_COLORS)], alpha=0.55, linewidth=1.1,
                markersize=5, label=f"seed {s}")
    ax.plot(eps, d["greedy_mean_across_seeds"], "k-", linewidth=2.6,
            marker="o", markersize=7,
            label=f"mean of {len(seeds)} training seeds")
    ax.set_ylabel("DEV avg placement (lower is better)")
    ax.set_title(f"B. Mean across {len(seeds)} training seeds with every "
                 f"individual trajectory overlaid", fontsize=11)
    xaxis(ax)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{out}/B_mean_with_seed_overlay.png", dpi=140)
    plt.close(f)

    per_seed_fig("C_expert_agreement", "expert_agreement",
                 "agreement with the greedy expert",
                 "C. Expert-action agreement on the frozen 4,440-state "
                 "corpus, by training seed")
    per_seed_fig("D_kl_from_warmstart", "kl_from_warmstart",
                 "mean KL(pi_warmstart || pi_k)",
                 "D. KL from the BC warm start, by training seed")
    per_seed_fig("E_warmstart_agreement", "warmstart_agreement",
                 "argmax == warm-start argmax",
                 "E. Warm-start action agreement, by training seed")

    # F. action-category drift at the final budget
    f, ax = plt.subplots(figsize=(9.0, 4.6))
    cats = d["categories"]
    width = 0.8 / len(seeds)
    xs = range(len(cats))
    for i, s in enumerate(seeds):
        ax.bar([x + i * width for x in xs],
               d["category_disagreement_iter320"][s], width,
               color=_COLORS[i % len(_COLORS)],
               label=f"seed {s} (freeze picks "
                     f"{d['freeze_selections_iter320'][s]})")
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(cats)
    ax.set_ylabel("share of expert decisions changed")
    ax.set_title("F. Where each training seed's 5,120-episode policy "
                 "disagrees with the greedy expert", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{out}/F_category_drift_iter320.png", dpi=140)
    plt.close(f)

    # G. PPO optimization diagnostics per training block, per seed
    keys = [("rollout_avg_placement", "rollout placement (league field)"),
            ("entropy", "policy entropy (training batches)"),
            ("approx_kl", "approximate KL per update"),
            ("clip_frac", "PPO clip fraction"),
            ("value_explained_variance", "value explained variance"),
            ("adv_mean_abs", "mean |raw advantage|")]
    names = d["rl_block_names"]
    f, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    xs = range(len(names))
    width = 0.8 / len(seeds)
    for ax, (key, label) in zip(axes.flat, keys):
        for i, s in enumerate(seeds):
            ax.bar([x + i * width for x in xs],
                   [d["rl_blocks"][s][n][key] for n in names], width,
                   color=_COLORS[i % len(_COLORS)], label=f"seed {s}")
        ax.set_xticks([x + 0.4 - width / 2 for x in xs])
        ax.set_xticklabels([n.replace("iters_", "").replace("_", "-")
                            for n in names], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes.flat[0].legend(fontsize=8)
    f.suptitle("G. PPO optimization diagnostics per training block, by "
               "training seed", fontsize=11)
    f.tight_layout()
    f.savefig(f"{out}/G_rl_diagnostics.png", dpi=140)
    plt.close(f)
    print(f"Saved plots -> {out}/")


if __name__ == "__main__":
    raise SystemExit(main())
