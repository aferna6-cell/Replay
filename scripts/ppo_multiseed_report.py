"""Cross-seed aggregation for Experiment 3 (multi-seed PPO budget replication).

Reads only committed result JSON — seed 0's Experiment 2 artifacts
(``results/ppo_budget_v1/``, used as-is and never modified) plus seeds 1-3's
Experiment 3 artifacts (``results/ppo_multiseed_v1/seed_*/``) — and writes
the cross-seed paired-comparison tables, the budget summary table, the
replication analysis (Questions A/B, U-shape classification, drift and
action-category replication), and plots A-G. Nothing here re-runs games or
re-trains anything; it only aggregates existing DEV evaluations.

    python -m scripts.ppo_multiseed_report
"""

import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.action_categories import CATEGORIES              # noqa: E402
from ml.analyze_benchmark import compare_pair, load_result  # noqa: E402

SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
OUT_DIR = "results/ppo_multiseed_v1/aggregate"

# The 9 pre-registered per-seed paired comparisons (Step 10 of the task
# spec), each a (target_iteration, reference_iteration) pair. Convention:
# mean_diff = target - reference; positive = target places WORSE (higher
# placement number) than the reference; negative = target places BETTER.
PAIR_SPECS = [(40, 0), (80, 0), (160, 0), (320, 0),
              (80, 40), (160, 40), (320, 40),
              (160, 80), (320, 80)]
# Not one of the 9 pre-registered comparisons; used only internally to
# classify U-shaped trajectories whose peak is at iteration 160 (where the
# pre-registered list has no direct 320-vs-160 comparison).
SUPPLEMENTARY_PAIR = (320, 160)


def seed_dir(seed):
    return "results/ppo_budget_v1" if seed == 0 else f"results/ppo_multiseed_v1/seed_{seed}"


def dev_path(seed, it, field):
    return os.path.join(seed_dir(seed), "dev", f"iter{it:03d}_vs_{field}.json")


def load_greedy(seed, it):
    return load_result(dev_path(seed, it, "greedy"))


def load_mixed(seed, it):
    return load_result(dev_path(seed, it, "greedy4_random3"))


def load_drift(seed):
    return {r["checkpoint"]: r for r in
            json.load(open(os.path.join(seed_dir(seed), "policy_drift.json")))["checkpoints"]}


def load_categories(seed):
    return {c["checkpoint"]: c for c in
            json.load(open(os.path.join(
                seed_dir(seed), "action_category_drift.json")))["checkpoints"]}


def load_diag(seed):
    return [json.loads(l) for l in
            open(os.path.join(seed_dir(seed), "train_diag.jsonl"))]


# --- paired comparisons -------------------------------------------------------

def seed_paired_table(seed):
    """The 9 pre-registered paired comparisons for one seed, plus the
    supplementary (320, 160) comparison used only for U-shape
    classification. Bootstrap seed fixed at 0 (deterministic), matching
    Experiment 2's convention."""
    greedy = {it: load_greedy(seed, it) for it in ITERS}
    rows = []
    for target, ref in PAIR_SPECS:
        row = compare_pair(greedy[target], greedy[ref], seed=0)
        row["target_iteration"], row["reference_iteration"] = target, ref
        row["target_episodes"] = target * EPISODES_PER_ITER
        row["reference_episodes"] = ref * EPISODES_PER_ITER
        rows.append(row)
    supp = compare_pair(greedy[SUPPLEMENTARY_PAIR[0]], greedy[SUPPLEMENTARY_PAIR[1]], seed=0)
    supp["target_iteration"], supp["reference_iteration"] = SUPPLEMENTARY_PAIR
    supp["note"] = "supplementary — not one of the 9 pre-registered comparisons"
    return rows, supp


# --- U-shape classification ---------------------------------------------------

def classify_trajectory(curve_by_iter, pairwise_by_pair):
    """Classify one seed's DEV placement trajectory across the 5 primary
    checkpoints {0, 40, 80, 160, 320}, using only the pre-registered paired
    comparisons vs iteration 0 for significance and the raw average
    placements for monotonicity. Documented rule, applied in this fixed
    order (first match wins):

      1. "mostly flat/noisy" — none of iter{40,80,160,320} differs from
         iter0 with a paired-bootstrap 95% CI excluding zero.
      2. "monotonic improvement" — avg placement is non-increasing across
         all 5 checkpoints (0 >= 40 >= 80 >= 160 >= 320, i.e. steadily
         better or flat) AND iter320 is significantly better than iter0.
      3. "monotonic degradation" — mirror of (2): avg placement is
         non-decreasing across all 5 checkpoints AND iter320 is
         significantly worse than iter0.
      4. "U-like/transient improvement" — some interior checkpoint (40, 80,
         or 160) is significantly BETTER than iter0, but iter320 is NOT
         significantly better than iter0 (the gain does not persist to the
         end of training).
      5. "other" — anything the above four don't cover (e.g. a sustained
         but non-monotonic improvement, or a significant early regression
         followed by later recovery).

    Returns (label, evidence dict).
    """
    P = {it: curve_by_iter[it] for it in ITERS}

    def sig(target, ref):
        row = pairwise_by_pair[(target, ref)]
        lo, hi = row["ci95"]
        if hi < 0:
            return "better"     # target significantly better (lower) than ref
        if lo > 0:
            return "worse"
        return "none"

    sig_vs0 = {b: sig(b, 0) for b in (40, 80, 160, 320)}

    monotonic_improve = P[0] >= P[40] >= P[80] >= P[160] >= P[320]
    monotonic_degrade = P[0] <= P[40] <= P[80] <= P[160] <= P[320]

    evidence = {"avg_placement": P, "sig_vs_iter0": sig_vs0,
                "monotonic_improve_raw": monotonic_improve,
                "monotonic_degrade_raw": monotonic_degrade}

    if all(v == "none" for v in sig_vs0.values()):
        return "mostly flat/noisy", evidence
    if monotonic_improve and sig_vs0[320] == "better":
        return "monotonic improvement", evidence
    if monotonic_degrade and sig_vs0[320] == "worse":
        return "monotonic degradation", evidence
    if any(sig_vs0[b] == "better" for b in (40, 80, 160)) and sig_vs0[320] != "better":
        return "U-like/transient improvement", evidence
    return "other", evidence


# --- main ----------------------------------------------------------------------

def main() -> int:
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

    # --- per-seed curves (raw DEV placements + drift + categories) ---------
    curves = {}          # seed -> {iter: {...}}
    diag_by_seed = {}
    for seed in SEEDS:
        drift = load_drift(seed)
        cats = load_categories(seed)
        row = {}
        for it in ITERS:
            g = load_greedy(seed, it)
            m = load_mixed(seed, it)
            d = drift[f"iter_{it:03d}.pt"]
            ce = cats[f"iter_{it:03d}.pt"]["vs_expert"]
            freeze_ct = sum(tos.get("freeze", 0)
                            for tos in ce["confusion_matrix"].values())
            row[it] = {
                "greedy_avg": g["metrics"]["avg_placement"],
                "greedy_ci95": g["avg_placement_ci95"],
                "greedy_median": g["metrics"]["median_placement"],
                "greedy_std": g["metrics"]["std_placement"],
                "greedy_top4": g["metrics"]["top4_rate"],
                "greedy_win": g["metrics"]["win_rate"],
                "mixed_avg": m["metrics"]["avg_placement"],
                "mixed_ci95": m["avg_placement_ci95"],
                "expert_agreement": d["expert_agreement"],
                "warmstart_agreement": d["warmstart_agreement"],
                "kl_from_warmstart": d["kl_from_warmstart_mean"],
                "entropy_mean": d["entropy_mean"],
                "value_mean": d["value_mean"], "value_std": d["value_std"],
                "n_states": ce["n_states"],
                "n_freeze_selections": freeze_ct,
                "freeze_rate": freeze_ct / ce["n_states"],
                "disagreement_share_by_category":
                    ce["disagreement_share_by_category"],
                "contribution_to_total_drift": ce["contribution_to_total_drift"],
            }
        curves[seed] = row
        diag_by_seed[seed] = load_diag(seed)

    # --- per-seed paired tables ---------------------------------------------
    paired_by_seed = {}
    supplementary_by_seed = {}
    for seed in SEEDS:
        rows, supp = seed_paired_table(seed)
        paired_by_seed[seed] = rows
        supplementary_by_seed[seed] = supp

    # --- cross-seed summary table --------------------------------------------
    summary_rows = []
    for it in ITERS:
        vals = [curves[s][it]["greedy_avg"] for s in SEEDS]
        summary_rows.append({
            "iteration": it, "cumulative_episodes": it * EPISODES_PER_ITER,
            "per_seed": {str(s): curves[s][it]["greedy_avg"] for s in SEEDS},
            "mean": st.mean(vals), "median": st.median(vals),
            "min": min(vals), "max": max(vals),
            "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })
    cross_seed_summary = {
        "note": ("n=4 training seeds; std/mean across seeds is descriptive, "
                 "not a population estimate. Seed 0 is the existing "
                 "Experiment 2 trajectory (reused as-is); seeds 1-3 are new."),
        "seeds": SEEDS, "primary_iterations": ITERS,
        "episodes_per_iteration": EPISODES_PER_ITER,
        "table": summary_rows,
    }

    # --- Question A: iter80 - iter0 ------------------------------------------
    qa = []
    for seed in SEEDS:
        row = next(r for r in paired_by_seed[seed]
                   if r["target_iteration"] == 80 and r["reference_iteration"] == 0)
        lo, hi = row["ci95"]
        excludes_zero = lo > 0 or hi < 0
        qa.append({"seed": seed, "mean_diff": row["mean_diff"], "ci95": row["ci95"],
                   "improves": row["mean_diff"] < 0, "worsens": row["mean_diff"] > 0,
                   "ci_excludes_zero": excludes_zero,
                   "direction_if_significant":
                       ("improvement" if excludes_zero and row["mean_diff"] < 0
                        else "regression" if excludes_zero else None)})
    question_a = {
        "question": "iter80 - iter0 (1,280 episodes vs warm start); "
                    "seed 0 reference value was -0.229",
        "per_seed": qa,
        "n_improve": sum(1 for r in qa if r["improves"]),
        "n_worsen": sum(1 for r in qa if r["worsens"]),
        "n_ci_excludes_zero": sum(1 for r in qa if r["ci_excludes_zero"]),
    }

    # --- Question B: iter320 - iter80 ----------------------------------------
    qb = []
    for seed in SEEDS:
        row = next(r for r in paired_by_seed[seed]
                   if r["target_iteration"] == 320 and r["reference_iteration"] == 80)
        lo, hi = row["ci95"]
        excludes_zero = lo > 0 or hi < 0
        qb.append({"seed": seed, "mean_diff": row["mean_diff"], "ci95": row["ci95"],
                   "regresses": row["mean_diff"] > 0,
                   "continues_improving": row["mean_diff"] < 0,
                   "ci_excludes_zero": excludes_zero,
                   "statistically_indistinguishable": not excludes_zero})
    question_b = {
        "question": "iter320 - iter80 (does the long-training regression "
                    "replicate?); seed 0 showed clear regression",
        "per_seed": qb,
        "n_regress": sum(1 for r in qb if r["regresses"]),
        "n_continue_improving": sum(1 for r in qb if r["continues_improving"]),
        "n_indistinguishable": sum(1 for r in qb if r["statistically_indistinguishable"]),
    }

    # --- U-shape classification -----------------------------------------------
    ushape = {}
    for seed in SEEDS:
        curve_by_iter = {it: curves[seed][it]["greedy_avg"] for it in ITERS}
        pairwise = {(r["target_iteration"], r["reference_iteration"]):
                    r for r in paired_by_seed[seed]}
        pairwise[SUPPLEMENTARY_PAIR] = supplementary_by_seed[seed]
        label, evidence = classify_trajectory(curve_by_iter, pairwise)
        ushape[seed] = {"label": label, "evidence": evidence}
    n_transient = sum(1 for s in SEEDS
                      if ushape[s]["label"] == "U-like/transient improvement")

    # --- drift replication -----------------------------------------------------
    drift_replication = {"per_seed_iter320": {}, "notes": []}
    best_vs_later = []
    for seed in SEEDS:
        c320 = curves[seed][320]
        drift_replication["per_seed_iter320"][str(seed)] = {
            "expert_agreement": c320["expert_agreement"],
            "warmstart_agreement": c320["warmstart_agreement"],
            "kl_from_warmstart": c320["kl_from_warmstart"],
            "entropy_mean": c320["entropy_mean"],
        }
        # best checkpoint among {40,80,160,320} by DEV avg placement
        cand = {it: curves[seed][it]["greedy_avg"] for it in (40, 80, 160, 320)}
        best_it = min(cand, key=cand.get)
        later = [it for it in (40, 80, 160, 320) if it > best_it]
        if later:
            best_vs_later.append({
                "seed": seed, "best_iteration": best_it,
                "best_expert_agreement": curves[seed][best_it]["expert_agreement"],
                "best_kl": curves[seed][best_it]["kl_from_warmstart"],
                "later_mean_expert_agreement":
                    st.mean(curves[seed][it]["expert_agreement"] for it in later),
                "later_mean_kl":
                    st.mean(curves[seed][it]["kl_from_warmstart"] for it in later),
                "best_has_higher_expert_agreement":
                    curves[seed][best_it]["expert_agreement"] >
                    st.mean(curves[seed][it]["expert_agreement"] for it in later),
                "best_has_lower_kl":
                    curves[seed][best_it]["kl_from_warmstart"] <
                    st.mean(curves[seed][it]["kl_from_warmstart"] for it in later),
            })
        else:
            best_vs_later.append({"seed": seed, "best_iteration": best_it,
                                  "later_checkpoints": "none (320 is best)"})
    drift_replication["best_checkpoint_vs_later"] = best_vs_later

    def pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return None
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0 or vy <= 0:
            return None
        return cov / math.sqrt(vx * vy)

    all_expert = [curves[s][it]["expert_agreement"] for s in SEEDS for it in ITERS]
    all_kl = [curves[s][it]["kl_from_warmstart"] for s in SEEDS for it in ITERS]
    all_place = [curves[s][it]["greedy_avg"] for s in SEEDS for it in ITERS]
    drift_replication["descriptive_correlation_expert_agreement_vs_placement"] = pearson(all_expert, all_place)
    drift_replication["descriptive_correlation_kl_vs_placement"] = pearson(all_kl, all_place)
    drift_replication["notes"].append(
        "Correlations are descriptive only (n=20 checkpoint-observations "
        "across 4 seeds, not independent samples) — no causal claim.")
    drift_replication["large_drift_consistent_across_seeds"] = all(
        curves[s][320]["kl_from_warmstart"] > curves[s][0]["kl_from_warmstart"]
        for s in SEEDS)

    # --- action-category replication -------------------------------------------
    action_cat_replication = {"per_seed_iter320": {}}
    for seed in SEEDS:
        c320 = curves[seed][320]
        action_cat_replication["per_seed_iter320"][str(seed)] = {
            "disagreement_share_by_category": c320["disagreement_share_by_category"],
            "contribution_to_total_drift": c320["contribution_to_total_drift"],
            "freeze_appears": c320["n_freeze_selections"] > 0,
            "freeze_selections": c320["n_freeze_selections"],
            "freeze_rate": c320["freeze_rate"],
            "n_states": c320["n_states"],
        }
    tempo_categories = ("roll", "end", "play")
    tempo_dominant = []
    for seed in SEEDS:
        contrib = action_cat_replication["per_seed_iter320"][str(seed)]["contribution_to_total_drift"]
        top_cat = max(contrib, key=lambda c: contrib[c] or 0.0)
        tempo_dominant.append(top_cat in tempo_categories)
    action_cat_replication["tempo_pattern_repeats_n_of_4"] = sum(tempo_dominant)
    action_cat_replication["freeze_appears_n_of_4"] = sum(
        1 for seed in SEEDS
        if action_cat_replication["per_seed_iter320"][str(seed)]["freeze_appears"])

    # --- outcome determination --------------------------------------------------
    replication_analysis = {
        "cross_seed_summary_ref": "results/ppo_multiseed_v1/aggregate/cross_seed_summary.json",
        "question_a_iter80_minus_iter0": question_a,
        "question_b_iter320_minus_iter80": question_b,
        "u_shape_classification": {str(s): ushape[s]["label"] for s in SEEDS},
        "u_shape_evidence": {str(s): ushape[s]["evidence"] for s in SEEDS},
        "n_transient_improvement_of_4": n_transient,
        "drift_replication": drift_replication,
        "action_category_replication": action_cat_replication,
    }

    # --- write JSON artifacts -----------------------------------------------
    with open(f"{OUT_DIR}/cross_seed_summary.json", "w") as f:
        json.dump(cross_seed_summary, f, indent=2)
    with open(f"{OUT_DIR}/paired_results.json", "w") as f:
        json.dump({
            "method": "deterministic paired percentile bootstrap, 10000 "
                      "resamples, bootstrap seed 0 (per ml.analyze_benchmark)",
            "convention": "mean_diff = target - reference; positive = "
                          "target places WORSE (higher placement) than "
                          "reference; negative = target places BETTER",
            "pair_specs": PAIR_SPECS,
            "by_seed": {str(s): paired_by_seed[s] for s in SEEDS},
            "supplementary_320_vs_160_by_seed":
                {str(s): supplementary_by_seed[s] for s in SEEDS},
        }, f, indent=2)
    with open(f"{OUT_DIR}/replication_analysis.json", "w") as f:
        json.dump(replication_analysis, f, indent=2)

    print(f"Saved -> {OUT_DIR}/cross_seed_summary.json, paired_results.json, "
          f"replication_analysis.json")

    _print_tables(cross_seed_summary, paired_by_seed, question_a, question_b, ushape)
    _plots(curves, diag_by_seed)
    return 0


def _print_tables(cross_seed_summary, paired_by_seed, question_a, question_b, ushape):
    print("\nCross-seed DEV avg placement (vs 7x greedy, 1000 games)")
    hdr = f"{'seed':>6}" + "".join(f"{'iter'+str(it):>10}" for it in ITERS)
    print(hdr)
    for s in SEEDS:
        vals = "".join(f"{cross_seed_summary['table'][i]['per_seed'][str(s)]:>10.3f}"
                       for i in range(len(ITERS)))
        print(f"{s:>6}{vals}")
    stat_hdr = f"{'stat':>6}" + "".join(f"{'iter'+str(it):>10}" for it in ITERS)
    print(stat_hdr)
    for stat in ("mean", "median", "min", "max", "std"):
        vals = "".join(f"{cross_seed_summary['table'][i][stat]:>10.3f}"
                       for i in range(len(ITERS)))
        print(f"{stat:>6}{vals}")

    print(f"\nQuestion A (iter80-iter0): improve={question_a['n_improve']} "
          f"worsen={question_a['n_worsen']} "
          f"ci_excludes_zero={question_a['n_ci_excludes_zero']}")
    for r in question_a["per_seed"]:
        print(f"  seed {r['seed']}: {r['mean_diff']:+.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    print(f"\nQuestion B (iter320-iter80): regress={question_b['n_regress']} "
          f"continue_improving={question_b['n_continue_improving']} "
          f"indistinguishable={question_b['n_indistinguishable']}")
    for r in question_b["per_seed"]:
        print(f"  seed {r['seed']}: {r['mean_diff']:+.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")

    print("\nU-shape classification:")
    for s in SEEDS:
        print(f"  seed {s}: {ushape[s]['label']}")


def _plots(curves, diag_by_seed) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [it * EPISODES_PER_ITER for it in ITERS]
    seed_colors = {0: "#888888", 1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    outdir = f"{OUT_DIR}/plots"

    # A. multi-seed DEV learning curves
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    for s in SEEDS:
        ys = [curves[s][it]["greedy_avg"] for it in ITERS]
        ax.plot(eps, ys, "o-", label=f"seed {s}", color=seed_colors[s], linewidth=1.8)
    ax.invert_yaxis()
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("DEV avg placement (lower is better)")
    ax.set_title("A. Multi-seed DEV learning curves — 7x greedy (1000 games)")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    f.tight_layout()
    f.savefig(f"{outdir}/A_multiseed_learning_curves.png", dpi=140)
    plt.close(f)

    # B. mean across seeds with individual seed points/lines visible
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    for s in SEEDS:
        ys = [curves[s][it]["greedy_avg"] for it in ITERS]
        ax.plot(eps, ys, "o--", color=seed_colors[s], alpha=0.55,
                linewidth=1.1, markersize=5, label=f"seed {s}")
    means = [st.mean(curves[s][it]["greedy_avg"] for s in SEEDS) for it in ITERS]
    stds = [st.stdev(curves[s][it]["greedy_avg"] for s in SEEDS) for it in ITERS]
    ax.errorbar(eps, means, yerr=stds, color="black", linewidth=2.4,
               marker="s", markersize=7, capsize=4, label="mean ± std (n=4)")
    ax.invert_yaxis()
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("DEV avg placement (lower is better)")
    ax.set_title("B. Mean across seeds (individual seeds NOT hidden)")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{outdir}/B_mean_with_seed_variance.png", dpi=140)
    plt.close(f)

    # C. expert agreement by seed
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    for s in SEEDS:
        ys = [curves[s][it]["expert_agreement"] for it in ITERS]
        ax.plot(eps, ys, "o-", color=seed_colors[s], label=f"seed {s}")
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("agreement with greedy expert")
    ax.set_title("C. Expert-action agreement by seed")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    f.tight_layout()
    f.savefig(f"{outdir}/C_expert_agreement_by_seed.png", dpi=140)
    plt.close(f)

    # D. KL from warm start by seed
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    for s in SEEDS:
        ys = [curves[s][it]["kl_from_warmstart"] for it in ITERS]
        ax.plot(eps, ys, "o-", color=seed_colors[s], label=f"seed {s}")
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("mean KL(pi_0 || pi_k)")
    ax.set_title("D. KL divergence from warm start, by seed")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    f.tight_layout()
    f.savefig(f"{outdir}/D_kl_from_warmstart_by_seed.png", dpi=140)
    plt.close(f)

    # E. warm-start agreement by seed
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    for s in SEEDS:
        ys = [curves[s][it]["warmstart_agreement"] for it in ITERS]
        ax.plot(eps, ys, "o-", color=seed_colors[s], label=f"seed {s}")
    ax.set_xlabel("cumulative PPO training episodes")
    ax.set_ylabel("argmax == iter-0 argmax")
    ax.set_title("E. Warm-start action agreement, by seed")
    ax.set_xticks(eps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    f.tight_layout()
    f.savefig(f"{outdir}/E_warmstart_agreement_by_seed.png", dpi=140)
    plt.close(f)

    # F. action-category drift at iter320 across seeds
    f, ax = plt.subplots(figsize=(9.0, 4.8))
    width = 0.8 / len(SEEDS)
    xs = range(len(CATEGORIES))
    for i, s in enumerate(SEEDS):
        shares = [(curves[s][320]["disagreement_share_by_category"].get(cat) or 0.0)
                  for cat in CATEGORIES]
        ax.bar([x + i * width for x in xs], shares, width,
               label=f"seed {s}", color=seed_colors[s])
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("share of expert decisions changed")
    ax.set_title("F. Action-category disagreement at iter320, across seeds")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{outdir}/F_action_category_drift_iter320.png", dpi=140)
    plt.close(f)

    # G. PPO optimization diagnostics across seeds (faceted)
    metrics = [("pi_loss", "policy loss"), ("v_loss", "value loss"),
               ("entropy", "entropy"), ("approx_kl", "approx KL"),
               ("clip_frac", "clip fraction"), ("grad_norm", "grad norm")]
    f, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (key, label) in zip(axes.flat, metrics):
        for s in SEEDS:
            diag = diag_by_seed[s]
            it_ = [r["iter"] for r in diag]
            ys = [r.get(key) for r in diag]
            ax.plot(it_, ys, linewidth=0.9, color=seed_colors[s], label=f"seed {s}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    f.suptitle("G. PPO optimization diagnostics across seeds (320 iterations)",
              fontsize=12)
    f.tight_layout()
    f.savefig(f"{outdir}/G_rl_diagnostics_across_seeds.png", dpi=140)
    plt.close(f)

    print(f"Saved plots -> {outdir}/")


if __name__ == "__main__":
    raise SystemExit(main())
