"""Assemble the Experiment 3 (multi-seed PPO budget replication) artifacts.

Reads only committed result JSON — the per-seed DEV evaluations, policy
drift, action categories, and per-iteration training diagnostics — and emits
per-seed learning curves, the cross-seed summary, the pre-specified paired
comparisons, the replication analysis (Questions A and B, U-shape
classification), and the plots. Re-runnable; computes nothing that isn't
already in those files.

    python scripts/ppo_multiseed_report.py

Seed 0 rows come from the committed Experiment 2 artifacts in
``results/ppo_budget_v1`` — they are read, never recomputed or overwritten.
DEV split only. Benchmark v1 TEST seeds are never read or run here.

U-SHAPE CLASSIFICATION RULE (pre-specified: written and committed before any
seed 1-3 DEV evaluation existed; see the git history of this file)
=====================================================================
Inputs per seed: the five primary DEV greedy averages g(0), g(40), g(80),
g(160), g(320) and the nine pre-specified paired-bootstrap comparisons
(placement diff, positive = first checkpoint WORSE; "clear" = the paired 95%
CI excludes zero). Categories, evaluated strictly in this order:

1. "U-like transient improvement" — some mid checkpoint m in {40, 80} is
   clearly BETTER than iteration 0 (diff(m-0) CI entirely < 0) AND iteration
   320 is clearly WORSE than that same m (diff(320-m) CI entirely > 0). The
   improvement must come and go within the measured trajectory. (m = 160 is
   excluded from this clause only because diff(320-160) is not one of the
   nine pre-specified comparisons, so the rule never consults it.)
2. "monotonic improvement" — iteration 320 is clearly better than iteration
   0 (diff(320-0) CI entirely < 0) and no mid checkpoint in {40, 80} is
   clearly better than iteration 320.
3. "monotonic degradation" — iteration 320 is clearly worse than iteration 0
   (diff(320-0) CI entirely > 0) and no mid checkpoint is clearly better
   than iteration 0.
4. "mostly flat / noisy" — none of the nine paired comparisons has a CI
   excluding zero.
5. "other" — anything else (mixed clear signals that fit none of the above).
"""

import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import compare_pair, load_result       # noqa: E402

SEED0_DIR = "results/ppo_budget_v1"          # committed Experiment 2 output
DIR = "results/ppo_multiseed_v1"
AGG = f"{DIR}/aggregate"
SEEDS = [0, 1, 2, 3]
NEW_SEEDS = [1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
# The nine pre-specified within-seed comparisons (target, reference):
# positive difference = target checkpoint WORSE than reference.
PAIRS = [(40, 0), (80, 0), (160, 0), (320, 0),
         (80, 40), (160, 40), (320, 40),
         (160, 80), (320, 80)]
MID_ITERS = [40, 80, 160]        # mids that can be compared against iter 0
U_MIDS = [40, 80]                # mids with a pre-specified (320 - m) pair


def episodes(it):
    return it * EPISODES_PER_ITER


def seed_dir(seed):
    return SEED0_DIR if seed == 0 else f"{DIR}/seed_{seed}"


def load_seed(seed):
    """All committed inputs for one training seed."""
    d = seed_dir(seed)
    greedy = {it: load_result(f"{d}/dev/iter{it:03d}_vs_greedy.json")
              for it in ITERS}
    mixed = {it: load_result(f"{d}/dev/iter{it:03d}_vs_greedy4_random3.json")
             for it in ITERS}
    drift_blob = json.load(open(f"{d}/policy_drift.json"))
    drift = {r["checkpoint"]: r for r in drift_blob["checkpoints"]}
    cat_blob = json.load(open(f"{d}/action_category_drift.json"))
    cats = {c["checkpoint"]: c for c in cat_blob["checkpoints"]}
    diag = [json.loads(l) for l in open(f"{d}/train_diag.jsonl")]
    return {"greedy": greedy, "mixed": mixed, "drift": drift,
            "drift_corpus": drift_blob["corpus"], "cats": cats, "diag": diag}


def build_curve(seed, data):
    """Per-checkpoint learning-curve rows, Experiment 2 schema plus seed."""
    curve = []
    for it in ITERS:
        d = data["drift"][f"iter_{it:03d}.pt"]
        g = data["greedy"][it]["metrics"]
        m = data["mixed"][it]["metrics"]
        ce = data["cats"][f"iter_{it:03d}.pt"]["vs_expert"]
        curve.append({
            "training_seed": seed,
            "iteration": it, "cumulative_episodes": episodes(it),
            "greedy_avg": g["avg_placement"],
            "greedy_ci95": data["greedy"][it]["avg_placement_ci95"],
            "greedy_median": g["median_placement"],
            "greedy_std": g["std_placement"],
            "greedy_top4": g["top4_rate"], "greedy_win": g["win_rate"],
            "greedy_placement_counts": g["placement_counts"],
            "mixed_avg": m["avg_placement"],
            "mixed_ci95": data["mixed"][it]["avg_placement_ci95"],
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
    return curve


def paired_table(data):
    """The nine pre-specified within-seed paired comparisons."""
    rows = []
    for target, ref in PAIRS:
        row = compare_pair(data["greedy"][target], data["greedy"][ref], seed=0)
        row["iteration"] = target
        row["reference_iteration"] = ref
        row["label"] = f"iter{target}-iter{ref}"
        lo, hi = row["ci95"]
        row["ci_excludes_zero"] = not (lo <= 0 <= hi)
        rows.append(row)
    return rows


def _pair(rows, target, ref):
    return next(r for r in rows
                if r["iteration"] == target and r["reference_iteration"] == ref)


def _clearly_negative(row):
    return row["ci95"][1] < 0


def _clearly_positive(row):
    return row["ci95"][0] > 0


def classify_curve(paired_rows):
    """Apply the documented U-shape rule (see module docstring). Returns
    (category, human-readable justification)."""
    transient_mids = [m for m in U_MIDS
                      if _clearly_negative(_pair(paired_rows, m, 0))
                      and _clearly_positive(_pair(paired_rows, 320, m))]
    if transient_mids:
        return ("U-like transient improvement",
                f"mid checkpoint(s) {transient_mids} clearly better than "
                f"iter0 AND iter320 clearly worse than the same mid "
                f"checkpoint(s)")
    end = _pair(paired_rows, 320, 0)
    if _clearly_negative(end) and not any(
            _clearly_positive(_pair(paired_rows, 320, m)) for m in U_MIDS):
        return ("monotonic improvement",
                "iter320 clearly better than iter0 and no mid checkpoint "
                "clearly better than iter320")
    if _clearly_positive(end) and not any(
            _clearly_negative(_pair(paired_rows, m, 0)) for m in MID_ITERS):
        return ("monotonic degradation",
                "iter320 clearly worse than iter0 and no mid checkpoint "
                "clearly better than iter0")
    if not any(r["ci_excludes_zero"] for r in paired_rows):
        return ("mostly flat / noisy",
                "none of the nine paired comparisons excludes zero")
    return ("other",
            "clear signals present but they fit none of the first four "
            "patterns")


def cross_seed_stats(values):
    """Descriptive spread of one budget's placement across the four seeds."""
    return {"mean": st.mean(values), "median": st.median(values),
            "min": min(values), "max": max(values),
            "std": st.stdev(values),
            "n_seeds": len(values),
            "note": "n=4 training seeds — descriptive only; sample std"}


def main() -> int:
    data = {s: load_seed(s) for s in SEEDS}

    # Cross-run controls: every seed must have scored the identical DEV games
    # from an identical warm start.
    fp0 = data[0]["drift_corpus"]["fingerprint_sha256"]
    warm0 = data[0]["drift"]["iter_000.pt"]["parameter_sha256"]
    for s in SEEDS:
        assert data[s]["drift_corpus"]["fingerprint_sha256"] == fp0, \
            f"seed {s} scored a different drift corpus"
        assert data[s]["drift"]["iter_000.pt"]["parameter_sha256"] == warm0, \
            f"seed {s} iteration 0 differs from the frozen warm start"
        for it in ITERS:
            assert (data[s]["greedy"][it]["seed_range"]
                    == data[0]["greedy"][it]["seed_range"]), \
                f"seed {s} iter {it}: different DEV evaluation seeds"
            assert (data[s]["mixed"][it]["seed_range"]
                    == data[0]["mixed"][it]["seed_range"])

    # --- per-seed learning curves + rl signal ------------------------------
    curves = {}
    for s in SEEDS:
        curve = build_curve(s, data[s])
        curves[s] = curve
        if s != 0:                       # seed-0 artifacts stay Experiment 2's
            with open(f"{DIR}/seed_{s}/learning_curve.json", "w") as f:
                json.dump({"experiment": "Replay Experiment 3 — Multi-Seed "
                                         "PPO Budget Replication",
                           "evaluation_split": "dev",
                           "training_seed": s,
                           "episodes_per_iteration": EPISODES_PER_ITER,
                           "primary_iterations": ITERS,
                           "greedy_games": data[s]["greedy"][0]["games"],
                           "mixed_games": data[s]["mixed"][0]["games"],
                           "dev_seed_range_greedy":
                               data[s]["greedy"][0]["seed_range"],
                           "curve": curve}, f, indent=2)
            with open(f"{DIR}/seed_{s}/rl_signal.json", "w") as f:
                json.dump({"definitions": {
                    "adv_*": "GAE advantages BEFORE PPO's per-batch "
                             "normalization",
                    "value_explained_variance":
                        "1 - Var(returns - value_preds) / Var(returns)",
                    "shaping_reward_sum/terminal_reward_sum":
                        "the two reward sources separated per iteration"},
                    "training_seed": s,
                    "per_iteration": data[s]["diag"]}, f, indent=2)

    # --- cross-seed budget table -------------------------------------------
    budget_table = []
    for it in ITERS:
        per_seed = {s: curves[s][ITERS.index(it)]["greedy_avg"] for s in SEEDS}
        budget_table.append({
            "iteration": it, "cumulative_episodes": episodes(it),
            "greedy_avg_by_seed": per_seed,
            "across_seeds": cross_seed_stats(list(per_seed.values())),
            "mixed_avg_by_seed":
                {s: curves[s][ITERS.index(it)]["mixed_avg"] for s in SEEDS},
        })

    hdr = (f"{'iter':>5} {'episodes':>9} " +
           " ".join(f"{'seed ' + str(s):>9}" for s in SEEDS) +
           f" {'mean':>8} {'median':>8} {'min':>7} {'max':>7} {'std':>7}")
    print("Experiment 3 — cross-seed DEV budget table "
          "(1000 games vs 7x greedy; lower placement is better)")
    print(hdr)
    print("-" * len(hdr))
    for row in budget_table:
        a = row["across_seeds"]
        print(f"{row['iteration']:>5} {row['cumulative_episodes']:>9} " +
              " ".join(f"{row['greedy_avg_by_seed'][s]:>9.3f}" for s in SEEDS) +
              f" {a['mean']:>8.3f} {a['median']:>8.3f} {a['min']:>7.3f} "
              f"{a['max']:>7.3f} {a['std']:>7.3f}")

    # --- within-seed paired comparisons ------------------------------------
    paired = {}
    print("\nWithin-seed paired DEV comparisons "
          "(positive = first checkpoint WORSE; paired bootstrap, 10000 "
          "resamples, seed 0)")
    for s in SEEDS:
        paired[s] = paired_table(data[s])
        print(f"  training seed {s}:")
        for r in paired[s]:
            mark = "*" if r["ci_excludes_zero"] else " "
            print(f"    {r['label']:>14} {r['mean_diff']:>+8.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {mark}")

    # --- replication questions ----------------------------------------------
    qa_rows, qb_rows = {}, {}
    for s in SEEDS:
        qa = _pair(paired[s], 80, 0)
        qa_rows[s] = {"mean_diff": qa["mean_diff"], "ci95": qa["ci95"],
                      "improved": qa["mean_diff"] < 0,
                      "clearly_improved": _clearly_negative(qa),
                      "clearly_worse": _clearly_positive(qa)}
        qb = _pair(paired[s], 320, 80)
        qb_rows[s] = {"mean_diff": qb["mean_diff"], "ci95": qb["ci95"],
                      "regressed": qb["mean_diff"] > 0,
                      "clearly_regressed": _clearly_positive(qb),
                      "clearly_kept_improving": _clearly_negative(qb)}

    question_a = {
        "definition": "iter80 - iter0 paired DEV placement difference per "
                      "seed (negative = iteration 80 better)",
        "per_seed": qa_rows,
        "n_seeds_improving": sum(r["improved"] for r in qa_rows.values()),
        "n_seeds_worsening": sum(not r["improved"] for r in qa_rows.values()),
        "n_ci_excluding_zero_improvement":
            sum(r["clearly_improved"] for r in qa_rows.values()),
        "n_ci_excluding_zero_worsening":
            sum(r["clearly_worse"] for r in qa_rows.values()),
    }
    question_b = {
        "definition": "iter320 - iter80 paired DEV placement difference per "
                      "seed (positive = the trajectory regressed after "
                      "iteration 80)",
        "per_seed": qb_rows,
        "n_regressing": sum(r["regressed"] for r in qb_rows.values()),
        "n_clearly_regressing":
            sum(r["clearly_regressed"] for r in qb_rows.values()),
        "n_clearly_kept_improving":
            sum(r["clearly_kept_improving"] for r in qb_rows.values()),
        "n_indistinguishable":
            sum(not r["clearly_regressed"] and not r["clearly_kept_improving"]
                for r in qb_rows.values()),
    }

    classification = {}
    for s in SEEDS:
        cat, why = classify_curve(paired[s])
        classification[s] = {"category": cat, "justification": why}
    n_transient = sum(1 for c in classification.values()
                      if c["category"] == "U-like transient improvement")

    # --- drift replication at iteration 320 --------------------------------
    drift320 = {}
    for s in SEEDS:
        row = curves[s][ITERS.index(320)]
        best_it = min(ITERS, key=lambda it: curves[s][ITERS.index(it)]
                      ["greedy_avg"])
        best = curves[s][ITERS.index(best_it)]
        drift320[s] = {
            "expert_agreement": row["expert_agreement"],
            "warmstart_agreement": row["warmstart_agreement"],
            "kl_from_warmstart": row["kl_from_warmstart"],
            "corpus_entropy": row["corpus_entropy"],
            "best_checkpoint_iteration": best_it,
            "best_checkpoint_expert_agreement": best["expert_agreement"],
            "best_checkpoint_kl": best["kl_from_warmstart"],
            "best_has_higher_expert_agreement_than_iter320":
                best["expert_agreement"] > row["expert_agreement"],
            "best_has_lower_kl_than_iter320":
                best["kl_from_warmstart"] < row["kl_from_warmstart"],
        }

    print("\nQuestion A (iter80 - iter0): "
          f"{question_a['n_seeds_improving']}/4 seeds improve, "
          f"{question_a['n_ci_excluding_zero_improvement']} with CI "
          f"excluding zero")
    print("Question B (iter320 - iter80): "
          f"{question_b['n_clearly_regressing']}/4 seeds clearly regress")
    for s in SEEDS:
        print(f"  seed {s}: {classification[s]['category']}")
    print(f"U-shape summary: {n_transient} / 4 trajectories show transient "
          f"improvement followed by regression")

    # --- write aggregates ----------------------------------------------------
    os.makedirs(AGG, exist_ok=True)
    with open(f"{AGG}/cross_seed_summary.json", "w") as f:
        json.dump({
            "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget "
                          "Replication",
            "evaluation_split": "dev",
            "replication_unit": "the PPO training seed (n=4: seed 0 from "
                                "Experiment 2 plus new seeds 1, 2, 3); the "
                                "1000 paired DEV games per checkpoint give "
                                "within-model precision only. 4x1000 games "
                                "are NOT 4000 independent experiments.",
            "seeds": SEEDS, "primary_iterations": ITERS,
            "episodes_per_iteration": EPISODES_PER_ITER,
            "seed0_source": "committed Experiment 2 artifacts "
                            "(results/ppo_budget_v1), read as-is",
            "budget_table": budget_table,
            "per_seed_curves": {str(s): curves[s] for s in SEEDS},
            "drift_at_iter320": {str(s): drift320[s] for s in SEEDS},
        }, f, indent=2)
    with open(f"{AGG}/paired_results.json", "w") as f:
        json.dump({
            "method": "deterministic paired percentile bootstrap, 10000 "
                      "resamples, seed 0, over the identical 1000 DEV games",
            "note": "positive difference = the first (later) checkpoint "
                    "places WORSE than the reference (lower placement is "
                    "better)",
            "pairs": [f"iter{t}-iter{r}" for t, r in PAIRS],
            "per_seed": {str(s): paired[s] for s in SEEDS},
        }, f, indent=2)
    with open(f"{AGG}/replication_analysis.json", "w") as f:
        json.dump({
            "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget "
                          "Replication",
            "classification_rule": "see the docstring of "
                                   "scripts/ppo_multiseed_report.py — "
                                   "committed before any seed 1-3 DEV "
                                   "evaluation existed",
            "question_a_1280_episode_improvement": question_a,
            "question_b_post_transient_decay": question_b,
            "u_shape_classification": {str(s): classification[s]
                                       for s in SEEDS},
            "u_shape_summary": f"{n_transient} / 4 trajectories show "
                               f"transient improvement followed by "
                               f"regression",
            "inferential_caution": "n=4 training seeds: descriptive only; "
                                   "no population-level effect-size claims. "
                                   "Any cross-seed aggregate is exploratory "
                                   "and unstable at this n.",
        }, f, indent=2)
    print(f"\nSaved -> {AGG}/cross_seed_summary.json, paired_results.json, "
          f"replication_analysis.json")
    _plots(curves, {s: data[s]["diag"] for s in SEEDS},
           {s: data[s]["cats"] for s in SEEDS})
    return 0


SEED_STYLE = {0: ("#555555", "o"), 1: ("#1f77b4", "s"),
              2: ("#d62728", "^"), 3: ("#2ca02c", "D")}


def _plots(curves, diags, cats) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(f"{AGG}/plots", exist_ok=True)
    eps = [episodes(it) for it in ITERS]

    def per_seed_fig(name, key, ylabel, title, invert=False):
        f, ax = plt.subplots(figsize=(7.6, 4.5))
        for s in SEEDS:
            color, marker = SEED_STYLE[s]
            ys = [c[key] for c in curves[s]]
            ax.plot(eps, ys, marker + "-", color=color, label=f"seed {s}",
                    linewidth=1.6, markersize=5.5)
        ax.set_xlabel("cumulative PPO training episodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e}\n(it {it})" for e, it in zip(eps, ITERS)],
                           fontsize=8)
        if invert:
            ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(f"{AGG}/plots/{name}.png", dpi=140)
        plt.close(f)

    # A. multi-seed DEV learning curves (unsmoothed)
    per_seed_fig("A_dev_learning_curves_by_seed", "greedy_avg",
                 "DEV avg placement (lower is better)",
                 "A. DEV placement vs training budget — 7x greedy "
                 "(1000 games), one line per training seed", invert=True)

    # B. mean across seeds with the individual seeds visible
    f, ax = plt.subplots(figsize=(7.6, 4.5))
    for s in SEEDS:
        color, marker = SEED_STYLE[s]
        ax.plot(eps, [c["greedy_avg"] for c in curves[s]], marker + "-",
                color=color, alpha=0.35, linewidth=1.0, markersize=4.5,
                label=f"seed {s}")
    means = [st.mean(curves[s][i]["greedy_avg"] for s in SEEDS)
             for i in range(len(ITERS))]
    ax.plot(eps, means, "o-", color="#000", linewidth=2.4, markersize=7,
            label="mean of 4 seeds (descriptive)")
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("DEV avg placement (lower is better)")
    ax.set_title("B. Cross-seed mean DEV placement (n=4 seeds — "
                 "descriptive only)", fontsize=11)
    ax.set_xticks(eps)
    ax.set_xticklabels([f"{e}\n(it {it})" for e, it in zip(eps, ITERS)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/B_dev_mean_across_seeds.png", dpi=140)
    plt.close(f)

    # C / D / E — drift metrics per seed
    per_seed_fig("C_expert_agreement_by_seed", "expert_agreement",
                 "agreement with greedy expert",
                 "C. Expert-action agreement vs training budget, by seed")
    per_seed_fig("D_kl_from_warmstart_by_seed", "kl_from_warmstart",
                 "mean KL(pi_0 || pi_k)",
                 "D. KL from the frozen warm start vs training budget, "
                 "by seed")
    per_seed_fig("E_warmstart_agreement_by_seed", "warmstart_agreement",
                 "agreement with the warm start",
                 "E. Warm-start action agreement vs training budget, by seed")

    # F. action-category drift at iteration 320 across seeds
    from ml.action_categories import CATEGORIES
    f, ax = plt.subplots(figsize=(8.8, 4.6))
    width = 0.8 / len(SEEDS)
    xs = range(len(CATEGORIES))
    for i, s in enumerate(SEEDS):
        ce = cats[s]["iter_320.pt"]["vs_expert"]
        shares = [(ce["disagreement_share_by_category"].get(c) or 0.0)
                  for c in CATEGORIES]
        color, _ = SEED_STYLE[s]
        ax.bar([x + i * width for x in xs], shares, width, color=color,
               label=f"seed {s}")
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("share of expert decisions changed")
    ax.set_title("F. Expert disagreement at iteration 320 by decision "
                 "category, across training seeds", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/F_category_drift_iter320_by_seed.png", dpi=140)
    plt.close(f)

    # G. PPO optimization diagnostics across seeds
    f, axes = plt.subplots(2, 2, figsize=(11.5, 6.6))
    for ax, (key, label) in zip(axes.flat, [
            ("adv_mean_abs", "mean |raw advantage|"),
            ("value_explained_variance", "value explained variance"),
            ("entropy", "policy entropy (training batches)"),
            ("approx_kl", "approx KL per update")]):
        for s in SEEDS:
            color, _ = SEED_STYLE[s]
            it = [r["iter"] for r in diags[s]]
            ys = [r.get(key) for r in diags[s]]
            ax.plot(it, ys, linewidth=0.8, color=color, alpha=0.8,
                    label=f"seed {s}")
        if key == "value_explained_variance":
            ax.axhline(0, color="#888", linestyle=":", linewidth=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    f.suptitle("G. RL signal and optimization diagnostics over 320 "
               "iterations, across training seeds", fontsize=11)
    f.tight_layout()
    f.savefig(f"{AGG}/plots/G_rl_diagnostics_by_seed.png", dpi=140)
    plt.close(f)
    print(f"Saved plots -> {AGG}/plots/")


if __name__ == "__main__":
    raise SystemExit(main())
