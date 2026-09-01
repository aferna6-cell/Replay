"""Assemble Experiment 5 dose-response analysis across all four β values.

Reuses β=0.0 and β=0.1 from Experiment 4b; combines with new β=0.01 and
β=0.03 runs from ``results/ppo_dose_v1/``.

    python scripts/ppo_dose_report.py
"""

import json
import os
import statistics as st
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import load_result
from ml.experiment_contract import load_contract

DOSE_DIR = "results/ppo_dose_v1"
MATCHED_DIR = "results/ppo_matched_ab_v1"
AGG_DIR = os.path.join(DOSE_DIR, "aggregate")
PLOTS_DIR = os.path.join(AGG_DIR, "plots")
SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16

# (label, kl_coef, base_dir)
ALL_ARMS: Tuple[Tuple[str, float, str], ...] = (
    ("beta0", 0.0, MATCHED_DIR),
    ("beta001", 0.01, DOSE_DIR),
    ("beta003", 0.03, DOSE_DIR),
    ("beta01", 0.1, MATCHED_DIR),
)


def run_path(base: str, kl_label: str, seed: int) -> str:
    return os.path.join(base, kl_label, f"seed_{seed}")


def dev_greedy(base: str, kl_label: str, seed: int, it: int) -> str:
    return os.path.join(run_path(base, kl_label, seed), "dev",
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
        "worst": max(values),
        "best": min(values),
        "range": max(values) - min(values),
        "n": len(values),
    }


def category_disagreement(base: str, kl_label: str, seed: int, it: int) -> Dict:
    ck = f"iter_{it:03d}.pt"
    cats = json.load(open(os.path.join(run_path(base, kl_label, seed),
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


def bc_baseline() -> float:
    """Cross-seed mean iter-0 placement (untouched BC warm start)."""
    avgs = []
    for seed in SEEDS:
        g = load_result(dev_greedy(MATCHED_DIR, "beta0", seed, 0))
        avgs.append(g["metrics"]["avg_placement"])
    return st.mean(avgs)


def main() -> int:
    os.makedirs(AGG_DIR, exist_ok=True)
    contract = load_contract(os.path.join(DOSE_DIR, "contract.json"))
    bc_avg = bc_baseline()

    curves = {label: [] for label, _, _ in ALL_ARMS}
    for kl_label, kl_coef, base in ALL_ARMS:
        for seed in SEEDS:
            drift_rows = {r["checkpoint"]: r for r in
                          json.load(open(os.path.join(run_path(base, kl_label, seed),
                                                      "policy_drift.json")))
                          ["checkpoints"]}
            for it in ITERS:
                g = load_result(dev_greedy(base, kl_label, seed, it))
                m = g["metrics"]
                d = drift_rows[f"iter_{it:03d}.pt"]
                cats = category_disagreement(base, kl_label, seed, it)
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
                    "delta_vs_bc": m["avg_placement"] - bc_avg,
                    "source_dir": base,
                })

    cross = {}
    for it in ITERS:
        cross[str(it)] = {}
        for kl_label, kl_coef, _ in ALL_ARMS:
            rows = [r for r in curves[kl_label] if r["iteration"] == it]
            avgs = [r["greedy_avg"] for r in rows]
            kls = [r["kl_from_warmstart"] for r in rows]
            exps = [r["expert_agreement"] for r in rows]
            warms = [r["warmstart_agreement"] for r in rows]
            cross[str(it)][kl_label] = {
                "kl_coef": kl_coef,
                "placement": cross_seed_stats(avgs),
                "kl_from_warmstart": cross_seed_stats(kls),
                "expert_agreement": cross_seed_stats(exps),
                "warmstart_agreement": cross_seed_stats(warms),
                "delta_vs_bc": cross_seed_stats([a - bc_avg for a in avgs]),
                "seed_placements": {
                    f"seed_{r['training_seed']}": r["greedy_avg"] for r in rows
                },
            }

    # Per-seed vs BC at iter 320
    vs_bc = {}
    for seed in SEEDS:
        vs_bc[f"seed_{seed}"] = {}
        for kl_label, kl_coef, _ in ALL_ARMS:
            row = next(r for r in curves[kl_label]
                       if r["training_seed"] == seed and r["iteration"] == 320)
            vs_bc[f"seed_{seed}"][kl_label] = {
                "kl_coef": kl_coef,
                "placement": row["greedy_avg"],
                "delta_vs_bc": row["greedy_avg"] - bc_avg,
                "kl_from_warmstart": row["kl_from_warmstart"],
                "expert_agreement": row["expert_agreement"],
            }

    it320 = cross["320"]
    it80 = cross["80"]
    it160 = cross["160"]

    # Outcome classification
    beats_bc_arms = []
    for kl_label, kl_coef, _ in ALL_ARMS:
        p = it320[kl_label]["placement"]
        if p["mean"] < bc_avg and p["worst"] < bc_avg + 0.05:
            seeds_beat = sum(
                1 for s in SEEDS
                if vs_bc[f"seed_{s}"][kl_label]["delta_vs_bc"] < -0.01)
            if seeds_beat >= 3:
                beats_bc_arms.append((kl_label, kl_coef, p["mean"], p["std"]))

    # Stability: compare std at 80/160 vs beta0
    beta0_std_80 = it80["beta0"]["placement"]["std"]
    beta0_std_160 = it160["beta0"]["placement"]["std"]
    stability_rank = []
    for kl_label, kl_coef, _ in ALL_ARMS:
        if kl_label == "beta0":
            continue
        s80 = it80[kl_label]["placement"]["std"]
        s160 = it160[kl_label]["placement"]["std"]
        s320 = it320[kl_label]["placement"]["std"]
        kl_mean = it320[kl_label]["kl_from_warmstart"]["mean"]
        stability_rank.append({
            "kl_label": kl_label,
            "kl_coef": kl_coef,
            "std_80": s80,
            "std_160": s160,
            "std_320": s320,
            "mean_placement_320": it320[kl_label]["placement"]["mean"],
            "kl_mean_320": kl_mean,
            "std_ratio_vs_beta0_80": s80 / max(beta0_std_80, 1e-9),
            "std_ratio_vs_beta0_160": s160 / max(beta0_std_160, 1e-9),
        })
    stability_rank.sort(key=lambda x: (x["mean_placement_320"], x["std_320"]))

    if beats_bc_arms:
        outcome = "A"
        best = min(beats_bc_arms, key=lambda x: x[2])
        outcome_text = (
            f"At least one fixed β ({best[1]}) reliably beats BC warm start "
            f"(mean={best[2]:.3f}, std={best[3]:.3f}) across seeds")
        next_action = (
            "Experiment 6 — scheduled/adaptive KL anchoring: anneal from the "
            f"best fixed β ({best[1]}) after iter 40–80 to recover exploration "
            "while retaining stability. Do NOT run until approved."
        )
    elif any(r["mean_placement_320"] < it320["beta01"]["placement"]["mean"]
             and r["std_320"] <= it320["beta01"]["placement"]["std"] * 1.2
             for r in stability_rank):
        outcome = "B"
        outcome_text = (
            "Intermediate β improves the stability/exploration tradeoff vs "
            "β=0.1 but no fixed β reliably beats BC warm start")
        next_action = (
            "Consider ONE scheduled-anchoring experiment starting from the best "
            "intermediate β. Do NOT start broad hyperparameter tuning."
        )
    else:
        outcome = "C"
        outcome_text = (
            "No fixed β reliably outperforms BC warm start (~6.550). "
            "KL regularization stabilizes PPO but current objective lacks "
            "useful headroom for consistent improvement.")
        next_action = (
            "STOP fixed-PPO tuning. Pivot to Simulator Fidelity Phase 2: "
            "late-game scaling, real card effects, board growth calibration, "
            "composition quality, hero/trinket/anomaly context."
        )

    # Console output
    print("=" * 96)
    print("EXPERIMENT 5: KL ANCHORING DOSE-RESPONSE — PRIMARY DEV (1000 games vs greedy)")
    print(f"BC warm-start baseline (iter 0 cross-seed mean): {bc_avg:.3f}")
    print("=" * 96)
    hdr = (f"{'Iter':>5} {'β':>6} {'Mean':>8} {'Std':>8} {'Worst':>8} "
           f"{'Δ vs BC':>9} {'Mean KL':>9} {'Exp%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for it in ITERS:
        for kl_label, kl_coef, _ in ALL_ARMS:
            c = cross[str(it)][kl_label]
            p, k, e, d = (c["placement"], c["kl_from_warmstart"],
                          c["expert_agreement"], c["delta_vs_bc"])
            print(f"{it:>5} {kl_coef:>6.2f} {p['mean']:>8.3f} {p['std']:>8.3f} "
                  f"{p['worst']:>8.3f} {d['mean']:>+9.3f} {k['mean']:>9.4f} "
                  f"{100 * e['mean']:>7.1f}%")

    print(f"\nITER-320 PER-SEED vs BC (Δ = placement − {bc_avg:.3f}; negative = better)")
    print(f"{'Seed':>6}", end="")
    for kl_label, kl_coef, _ in ALL_ARMS:
        print(f"  β={kl_coef:<4.2f} Δ", end="")
    print()
    for seed in SEEDS:
        print(f"{seed:>6}", end="")
        for kl_label, _, _ in ALL_ARMS:
            d = vs_bc[f"seed_{seed}"][kl_label]["delta_vs_bc"]
            print(f"  {d:>+8.3f}", end="")
        print()

    print("\nSTABILITY RANKING (non-zero β, iter 320)")
    for r in stability_rank:
        print(f"  β={r['kl_coef']:.2f}: mean={r['mean_placement_320']:.3f} "
              f"std={r['std_320']:.3f} KL={r['kl_mean_320']:.4f}")

    analysis = {
        "experiment": "Replay Experiment 5 — KL Anchoring Dose-Response",
        "evaluation_split": "dev",
        "bc_warm_start_baseline": bc_avg,
        "contract": {
            "expected_warm_start_parameter_sha256":
                contract["expected_warm_start_parameter_sha256"],
            "runtime_fingerprint_sha256":
                contract["runtime_fingerprint_sha256"],
            "ppo_config_hash_sha256": contract["ppo_config_hash_sha256"],
        },
        "arms": [{"label": l, "kl_coef": c, "source": b} for l, c, b in ALL_ARMS],
        "curves": curves,
        "cross_seed_by_iteration": cross,
        "per_seed_vs_bc_iter320": vs_bc,
        "stability_ranking_iter320": stability_rank,
        "iter320_summary": {kl: it320[kl] for kl, _, _ in ALL_ARMS},
        "beats_bc_arms": [
            {"label": l, "kl_coef": c, "mean": m, "std": s}
            for l, c, m, s in beats_bc_arms
        ],
        "outcome_classification": {
            "label": outcome,
            "summary": outcome_text,
            "recommended_next_action": next_action,
        },
    }

    with open(os.path.join(AGG_DIR, "cross_seed_summary.json"), "w") as f:
        json.dump(cross, f, indent=2)
    with open(os.path.join(AGG_DIR, "per_seed_vs_bc.json"), "w") as f:
        json.dump(vs_bc, f, indent=2)
    with open(os.path.join(AGG_DIR, "dose_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    _plots(cross, bc_avg)
    print(f"\nOutcome {outcome}: {outcome_text}")
    print(f"Recommended next action: {next_action}")
    return 0


def _plots(cross, bc_avg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    styles = {
        "beta0": ("s--", 0.0),
        "beta001": ("^-", 0.01),
        "beta003": ("d-", 0.03),
        "beta01": ("o-", 0.1),
    }

    # A: cross-seed mean placement
    f, ax = plt.subplots(figsize=(8, 4.5))
    for kl_label, (style, coef) in styles.items():
        ys = [cross[str(it)][kl_label]["placement"]["mean"] for it in ITERS]
        ax.plot(ITERS, ys, style, label=f"β={coef}", linewidth=1.8, markersize=6)
    ax.axhline(bc_avg, color="green", linestyle=":", label=f"BC baseline ({bc_avg:.3f})")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("cross-seed mean DEV placement (lower is better)")
    ax.set_title("A. Dose-response — cross-seed mean placement")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "A_cross_seed_mean_placement.png"), dpi=140)
    plt.close(f)

    # B: cross-seed std
    f, ax = plt.subplots(figsize=(8, 4.5))
    for kl_label, (style, coef) in styles.items():
        ys = [cross[str(it)][kl_label]["placement"]["std"] for it in ITERS]
        ax.plot(ITERS, ys, style, label=f"β={coef}", linewidth=1.8, markersize=6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("cross-seed placement std dev")
    ax.set_title("B. Cross-seed placement variance")
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "B_cross_seed_placement_std.png"), dpi=140)
    plt.close(f)

    # C: mean KL
    f, ax = plt.subplots(figsize=(8, 4.5))
    for kl_label, (style, coef) in styles.items():
        ys = [cross[str(it)][kl_label]["kl_from_warmstart"]["mean"] for it in ITERS]
        ax.plot(ITERS, ys, style, label=f"β={coef}", linewidth=1.8, markersize=6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean KL(π_BC ‖ π_k) across seeds")
    ax.set_title("C. Policy drift — cross-seed mean KL from warm start")
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "C_cross_seed_kl.png"), dpi=140)
    plt.close(f)

    # D: expert agreement
    f, ax = plt.subplots(figsize=(8, 4.5))
    for kl_label, (style, coef) in styles.items():
        ys = [100 * cross[str(it)][kl_label]["expert_agreement"]["mean"]
              for it in ITERS]
        ax.plot(ITERS, ys, style, label=f"β={coef}", linewidth=1.8, markersize=6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("expert agreement (%)")
    ax.set_title("D. Cross-seed mean expert agreement")
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "D_cross_seed_expert_agreement.png"), dpi=140)
    plt.close(f)

    # E: iter-320 per-seed bar chart
    f, ax = plt.subplots(figsize=(10, 4.5))
    xs = list(range(len(SEEDS)))
    n_arms = len(ALL_ARMS)
    w = 0.8 / n_arms
    for i, (kl_label, coef, _) in enumerate(ALL_ARMS):
        ys = [cross["320"][kl_label]["seed_placements"][f"seed_{s}"] for s in SEEDS]
        offset = (i - (n_arms - 1) / 2) * w
        ax.bar([x + offset for x in xs], ys, w, label=f"β={coef}")
    ax.axhline(bc_avg, color="green", linestyle=":", label=f"BC ({bc_avg:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("DEV avg placement (iter 320)")
    ax.set_title("E. Per-seed iter-320 placement — dose-response")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "E_per_seed_iter320.png"), dpi=140)
    plt.close(f)
    print(f"Saved plots -> {PLOTS_DIR}/")


if __name__ == "__main__":
    raise SystemExit(main())
