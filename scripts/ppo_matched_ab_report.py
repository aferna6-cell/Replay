"""Assemble Experiment 4b matched A/B analysis and cross-seed statistics.

    python scripts/ppo_matched_ab_report.py
"""

import json
import os
import statistics as st
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import compare_pair, load_result
from ml.experiment_contract import load_contract

BASE_DIR = "results/ppo_matched_ab_v1"
AGG_DIR = os.path.join(BASE_DIR, "aggregate")
PLOTS_DIR = os.path.join(AGG_DIR, "plots")
SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
PAIRED_ITERS = [40, 80, 160, 320]
KL_ARMS = (("beta0", 0.0), ("beta01", 0.1))


def run_path(kl_label: str, seed: int) -> str:
    return os.path.join(BASE_DIR, kl_label, f"seed_{seed}")


def dev_greedy(kl_label: str, seed: int, it: int) -> str:
    return os.path.join(run_path(kl_label, seed), "dev",
                         f"iter{it:03d}_vs_greedy.json")


def cross_seed_stats(values: List[float]) -> Dict:
    if not values:
        return {}
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "std": st.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "worst": max(values),  # higher placement is worse
        "n": len(values),
    }


def category_disagreement(kl_label: str, seed: int, it: int) -> Dict:
    ck = f"iter_{it:03d}.pt"
    cats = json.load(open(os.path.join(run_path(kl_label, seed),
                                       "action_category_drift.json")))
    row = next(c for c in cats["checkpoints"] if c["checkpoint"] == ck)
    vs = row["vs_expert"]
    by_cat = vs["disagreement_share_by_category"]
    return {
        "roll": by_cat.get("roll") or 0.0,
        "end": by_cat.get("end") or 0.0,
        "play": by_cat.get("play") or 0.0,
        "freeze": by_cat.get("freeze"),
    }


def main() -> int:
    os.makedirs(AGG_DIR, exist_ok=True)
    contract = load_contract(os.path.join(BASE_DIR, "contract.json"))

    # --- per-run curve rows ---------------------------------------------------
    curves = {label: [] for label, _ in KL_ARMS}
    for kl_label, kl_coef in KL_ARMS:
        for seed in SEEDS:
            drift_rows = {r["checkpoint"]: r for r in
                          json.load(open(os.path.join(run_path(kl_label, seed),
                                                      "policy_drift.json")))
                          ["checkpoints"]}
            for it in ITERS:
                g = load_result(dev_greedy(kl_label, seed, it))
                m = g["metrics"]
                d = drift_rows[f"iter_{it:03d}.pt"]
                cats = category_disagreement(kl_label, seed, it)
                curves[kl_label].append({
                    "training_seed": seed,
                    "kl_coef": kl_coef,
                    "iteration": it,
                    "cumulative_episodes": it * EPISODES_PER_ITER,
                    "greedy_avg": m["avg_placement"],
                    "greedy_ci95": g["avg_placement_ci95"],
                    "greedy_median": m["median_placement"],
                    "greedy_top4": m["top4_rate"],
                    "greedy_win": m["win_rate"],
                    "placement_counts": m["placement_counts"],
                    "parameter_sha256": d["parameter_sha256"],
                    "checkpoint_sha256": d["checkpoint_sha256"],
                    "expert_agreement": d["expert_agreement"],
                    "warmstart_agreement": d["warmstart_agreement"],
                    "kl_from_warmstart": d["kl_from_warmstart_mean"],
                    "corpus_entropy": d["entropy_mean"],
                    "value_mean": d["value_mean"],
                    "value_std": d["value_std"],
                    "roll_disagreement": cats["roll"],
                    "end_disagreement": cats["end"],
                    "play_disagreement": cats["play"],
                    "freeze_emergence": cats["freeze"],
                })

    # --- cross-seed summary at each iteration ---------------------------------
    cross = {}
    for it in ITERS:
        cross[str(it)] = {}
        for kl_label, kl_coef in KL_ARMS:
            avgs = [r["greedy_avg"] for r in curves[kl_label]
                    if r["iteration"] == it]
            kls = [r["kl_from_warmstart"] for r in curves[kl_label]
                   if r["iteration"] == it]
            exps = [r["expert_agreement"] for r in curves[kl_label]
                    if r["iteration"] == it]
            warms = [r["warmstart_agreement"] for r in curves[kl_label]
                     if r["iteration"] == it]
            cross[str(it)][kl_label] = {
                "kl_coef": kl_coef,
                "placement": cross_seed_stats(avgs),
                "kl_from_warmstart": cross_seed_stats(kls),
                "expert_agreement": cross_seed_stats(exps),
                "warmstart_agreement": cross_seed_stats(warms),
                "seed_placements": {
                    f"seed_{r['training_seed']}": r["greedy_avg"]
                    for r in curves[kl_label] if r["iteration"] == it
                },
            }

    # --- per-seed paired anchor − unconstrained --------------------------------
    paired_by_seed = {}
    for seed in SEEDS:
        paired_by_seed[f"seed_{seed}"] = {}
        for it in PAIRED_ITERS:
            anchored = load_result(dev_greedy("beta01", seed, it))
            base = load_result(dev_greedy("beta0", seed, it))
            row = compare_pair(anchored, base, seed=0)
            row["iteration"] = it
            row["training_seed"] = seed
            row["anchored_avg"] = anchored["metrics"]["avg_placement"]
            row["unconstrained_avg"] = base["metrics"]["avg_placement"]
            row["convention"] = ("positive mean_diff = anchored places worse; "
                                 "negative = anchored places better")
            paired_by_seed[f"seed_{seed}"][f"iter{it:03d}"] = row

    # --- console tables -------------------------------------------------------
    print("=" * 88)
    print("EXPERIMENT 4b: MATCHED β=0.1 vs β=0.0 — PRIMARY DEV (1000 games vs greedy)")
    print("=" * 88)
    hdr = (f"{'Iter':>5} {'Seed':>5} {'β=0.1':>8} {'β=0.0':>8} {'Δ(anch-base)':>12} "
           f"{'KL β=0.1':>9} {'KL β=0.0':>9} {'Exp% β=0.1':>11}")
    print(hdr)
    print("-" * len(hdr))
    for it in ITERS:
        for seed in SEEDS:
            a = next(r for r in curves["beta01"]
                     if r["training_seed"] == seed and r["iteration"] == it)
            b = next(r for r in curves["beta0"]
                     if r["training_seed"] == seed and r["iteration"] == it)
            delta = a["greedy_avg"] - b["greedy_avg"]
            print(f"{it:>5} {seed:>5} {a['greedy_avg']:>8.3f} {b['greedy_avg']:>8.3f} "
                  f"{delta:>+12.3f} {a['kl_from_warmstart']:>9.4f} "
                  f"{b['kl_from_warmstart']:>9.4f} {100 * a['expert_agreement']:>10.1f}%")

    print("\nCROSS-SEED SUMMARY (greedy avg placement; lower is better)")
    print(f"{'Iter':>5} {'Arm':>8} {'Mean':>8} {'Median':>8} {'Std':>8} "
          f"{'Worst':>8} {'Mean KL':>9} {'Mean Exp%':>10}")
    for it in ITERS:
        for kl_label, _ in KL_ARMS:
            c = cross[str(it)][kl_label]
            p, k, e = c["placement"], c["kl_from_warmstart"], c["expert_agreement"]
            print(f"{it:>5} {kl_label:>8} {p['mean']:>8.3f} {p['median']:>8.3f} "
                  f"{p['std']:>8.3f} {p['worst']:>8.3f} {k['mean']:>9.4f} "
                  f"{100 * e['mean']:>9.1f}%")

    print("\nPER-SEED PAIRED: anchored − unconstrained at iter 40/80/160/320")
    for seed in SEEDS:
        print(f"  Seed {seed}:")
        for it in PAIRED_ITERS:
            row = paired_by_seed[f"seed_{seed}"][f"iter{it:03d}"]
            print(f"    iter {it:>3}: Δ={row['mean_diff']:>+7.3f} "
                  f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]  {row['verdict']}")

    # --- outcome classification -----------------------------------------------
    it320 = cross["320"]
    p_std_ratio = (it320["beta0"]["placement"]["std"]
                   / max(it320["beta01"]["placement"]["std"], 1e-9))
    kl_ratio = it320["beta0"]["kl_from_warmstart"]["mean"] / max(
        it320["beta01"]["kl_from_warmstart"]["mean"], 1e-9)
    mean_delta_320 = st.mean(
        paired_by_seed[f"seed_{s}"]["iter320"]["mean_diff"] for s in SEEDS)
    worse_seeds_320 = sum(
        1 for s in SEEDS
        if paired_by_seed[f"seed_{s}"]["iter320"]["mean_diff"] > 0
        and paired_by_seed[f"seed_{s}"]["iter320"]["ci95"][0] > 0)
    better_seeds_320 = sum(
        1 for s in SEEDS
        if paired_by_seed[f"seed_{s}"]["iter320"]["mean_diff"] < 0
        and paired_by_seed[f"seed_{s}"]["iter320"]["ci95"][1] < 0)

    if (it320["beta01"]["kl_from_warmstart"]["mean"] < 0.15
            and p_std_ratio > 1.5
            and mean_delta_320 < 0.15):
        outcome = "A"
        outcome_text = ("Anchoring stabilizes drift/variance without clearly "
                        "hurting mean placement")
        exp5 = ("Experiment 5 — weaker / scheduled anchoring (anneal β after "
                "iter 40–80 to recover exploration while keeping stability)")
    elif (it320["beta01"]["kl_from_warmstart"]["mean"] < 0.15
          and worse_seeds_320 >= 2):
        outcome = "B"
        outcome_text = ("Anchoring controls drift but consistently hurts "
                        "placement across matched seeds")
        exp5 = ("Experiment 5 — fixed-beta dose study (pre-specify "
                "β ∈ {0.01, 0.03, 0.10}; no wide tuning)")
    else:
        outcome = "C"
        outcome_text = ("Anchoring does not materially improve cross-seed "
                        "stability vs matched unconstrained control")
        exp5 = ("Shift focus to simulator/data fidelity rather than PPO "
                "anchoring tuning")

    analysis = {
        "experiment": "Replay Experiment 4b — Matched Anchored vs Unanchored PPO",
        "evaluation_split": "dev",
        "contract": {
            "expected_warm_start_parameter_sha256":
                contract["expected_warm_start_parameter_sha256"],
            "runtime_fingerprint_sha256":
                contract["runtime_fingerprint_sha256"],
            "ppo_config_hash_sha256": contract["ppo_config_hash_sha256"],
            "code_commit": contract["code_commit"],
        },
        "curves": curves,
        "cross_seed_by_iteration": cross,
        "paired_by_seed": paired_by_seed,
        "iter320_summary": {
            "beta0": it320["beta0"],
            "beta01": it320["beta01"],
            "placement_std_ratio_beta0_over_beta01": p_std_ratio,
            "kl_mean_ratio_beta0_over_beta01": kl_ratio,
            "mean_paired_delta_anchored_minus_unconstrained": mean_delta_320,
            "seeds_anchored_worse_at_320": worse_seeds_320,
            "seeds_anchored_better_at_320": better_seeds_320,
        },
        "outcome_classification": {
            "label": outcome,
            "summary": outcome_text,
            "recommended_experiment_5": exp5,
        },
    }

    with open(os.path.join(AGG_DIR, "cross_seed_summary.json"), "w") as f:
        json.dump(cross, f, indent=2)
    with open(os.path.join(AGG_DIR, "paired_by_seed.json"), "w") as f:
        json.dump(paired_by_seed, f, indent=2)
    with open(os.path.join(AGG_DIR, "matched_ab_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    _plots(cross, curves)
    print(f"\nOutcome {outcome}: {outcome_text}")
    print(f"Recommended Experiment 5: {exp5}")
    return 0


def _plots(cross, curves) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    iters = ITERS

    # A: cross-seed mean placement
    f, ax = plt.subplots(figsize=(7.5, 4.5))
    for kl_label, style in (("beta0", "s--"), ("beta01", "o-")):
        ys = [cross[str(it)][kl_label]["placement"]["mean"] for it in iters]
        ax.plot(iters, ys, style, label=f"{kl_label}", linewidth=1.8, markersize=6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("cross-seed mean DEV placement (lower is better)")
    ax.set_title("A. Cross-seed mean placement — matched β=0 vs β=0.1")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "A_cross_seed_mean_placement.png"), dpi=140)
    plt.close(f)

    # B: cross-seed placement std (stability)
    f, ax = plt.subplots(figsize=(7.5, 4.5))
    for kl_label, style in (("beta0", "s--"), ("beta01", "o-")):
        ys = [cross[str(it)][kl_label]["placement"]["std"] for it in iters]
        ax.plot(iters, ys, style, label=f"{kl_label}", linewidth=1.8, markersize=6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("cross-seed placement std dev")
    ax.set_title("B. Cross-seed placement variance — stability metric")
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "B_cross_seed_placement_std.png"), dpi=140)
    plt.close(f)

    # C: mean KL from warm start
    f, ax = plt.subplots(figsize=(7.5, 4.5))
    for kl_label, style in (("beta0", "s--"), ("beta01", "o-")):
        ys = [cross[str(it)][kl_label]["kl_from_warmstart"]["mean"] for it in iters]
        ax.plot(iters, ys, style, label=f"{kl_label}", linewidth=1.8, markersize=6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean KL(π_BC ‖ π_k) across seeds")
    ax.set_title("C. Policy drift — cross-seed mean KL from warm start")
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "C_cross_seed_kl.png"), dpi=140)
    plt.close(f)

    # D: per-seed iter320 placement
    f, ax = plt.subplots(figsize=(8, 4.5))
    xs = list(range(len(SEEDS)))
    w = 0.35
    b0 = [cross["320"]["beta0"]["seed_placements"][f"seed_{s}"] for s in SEEDS]
    b1 = [cross["320"]["beta01"]["seed_placements"][f"seed_{s}"] for s in SEEDS]
    ax.bar([x - w / 2 for x in xs], b0, w, label="β=0.0")
    ax.bar([x + w / 2 for x in xs], b1, w, label="β=0.1")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("DEV avg placement (iter 320)")
    ax.set_title("D. Per-seed iter-320 placement — matched arms")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "D_per_seed_iter320.png"), dpi=140)
    plt.close(f)
    print(f"Saved plots -> {PLOTS_DIR}/")


if __name__ == "__main__":
    raise SystemExit(main())
