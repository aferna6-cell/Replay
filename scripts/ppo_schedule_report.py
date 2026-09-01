"""Assemble Experiment 6 scheduled vs fixed β=0.03 analysis.

Control (fixed β=0.03) is reused from ``results/ppo_dose_v1/beta003/``.
Treatment is the scheduled arm in ``results/ppo_schedule_v1/beta_sched/``.

    python scripts/ppo_schedule_report.py
"""

import json
import os
import statistics as st
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import compare_pair, load_result
from ml.experiment_contract import load_contract
from ml.kl_schedule import EXPERIMENT_6_KL_SCHEDULE, schedule_table

SCHEDULE_DIR = "results/ppo_schedule_v1"
DOSE_DIR = "results/ppo_dose_v1"
AGG_DIR = os.path.join(SCHEDULE_DIR, "aggregate")
PLOTS_DIR = os.path.join(AGG_DIR, "plots")
CONTROL_LABEL = "beta003"
SCHEDULE_LABEL = "beta_sched"
SEEDS = [0, 1, 2, 3]
ITERS = [0, 40, 80, 160, 320]
BC_BASELINE = 6.550
PAIRED_ITERS = [40, 80, 160, 320]


def run_path(base: str, label: str, seed: int) -> str:
    return os.path.join(base, label, f"seed_{seed}")


def dev_greedy(base: str, label: str, seed: int, it: int) -> str:
    return os.path.join(run_path(base, label, seed), "dev",
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


def load_arm_curves(base: str, label: str, kl_mode: str,
                    kl_coef: float = 0.03) -> List[dict]:
    rows = []
    for seed in SEEDS:
        drift = {r["checkpoint"]: r for r in
                 json.load(open(os.path.join(run_path(base, label, seed),
                                              "policy_drift.json")))
                 ["checkpoints"]}
        for it in ITERS:
            g = load_result(dev_greedy(base, label, seed, it))
            d = drift[f"iter_{it:03d}.pt"]
            rows.append({
                "training_seed": seed,
                "arm": label,
                "kl_mode": kl_mode,
                "kl_coef_fixed": kl_coef if kl_mode == "fixed" else None,
                "iteration": it,
                "greedy_avg": g["metrics"]["avg_placement"],
                "greedy_ci95": g["avg_placement_ci95"],
                "greedy_top4": g["metrics"]["top4_rate"],
                "greedy_win": g["metrics"]["win_rate"],
                "kl_from_warmstart": d["kl_from_warmstart_mean"],
                "expert_agreement": d["expert_agreement"],
                "warmstart_agreement": d["warmstart_agreement"],
                "delta_vs_bc": g["metrics"]["avg_placement"] - BC_BASELINE,
            })
    return rows


def evaluate_success(control_320: Dict, sched_320: Dict,
                     per_seed: Dict) -> Dict:
    """Apply pre-specified Experiment 6 success criteria."""
    ctrl = control_320["placement"]
    sched = sched_320["placement"]
    seeds_beat = sum(
        1 for s in SEEDS
        if per_seed[f"seed_{s}"]["scheduled"]["delta_vs_bc"] < -0.01)
    worst_vs_bc = max(per_seed[f"seed_{s}"]["scheduled"]["delta_vs_bc"]
                      for s in SEEDS)
    std_ratio = sched["std"] / max(ctrl["std"], 1e-9)

    checks = {
        "mean_below_bc": sched["mean"] < BC_BASELINE,
        "mean_improvement_vs_bc": BC_BASELINE - sched["mean"],
        "seeds_beating_bc_ge_3": seeds_beat >= 3,
        "seeds_beating_bc_count": seeds_beat,
        "no_catastrophic_seed": worst_vs_bc <= 0.05,
        "variance_near_control": std_ratio <= 1.5,
        "std_ratio_vs_control": std_ratio,
        "scheduled_beats_control_mean": sched["mean"] < ctrl["mean"],
    }
    checks["all_three_criteria"] = (
        checks["mean_below_bc"]
        and checks["seeds_beating_bc_ge_3"]
        and checks["no_catastrophic_seed"]
        and checks["variance_near_control"]
    )
    return checks


def main() -> int:
    os.makedirs(AGG_DIR, exist_ok=True)
    contract = load_contract(os.path.join(SCHEDULE_DIR, "contract.json"))

    control = load_arm_curves(DOSE_DIR, CONTROL_LABEL, "fixed", 0.03)
    scheduled = load_arm_curves(SCHEDULE_DIR, SCHEDULE_LABEL, "scheduled")

    cross = {}
    for it in ITERS:
        cross[str(it)] = {}
        for name, curves, mode in (
            ("control", control, "fixed"),
            ("scheduled", scheduled, "scheduled"),
        ):
            subset = [r for r in curves if r["iteration"] == it]
            avgs = [r["greedy_avg"] for r in subset]
            kls = [r["kl_from_warmstart"] for r in subset]
            cross[str(it)][name] = {
                "kl_mode": mode,
                "placement": cross_seed_stats(avgs),
                "kl_from_warmstart": cross_seed_stats(kls),
                "delta_vs_bc": cross_seed_stats([a - BC_BASELINE for a in avgs]),
                "seed_placements": {
                    f"seed_{r['training_seed']}": r["greedy_avg"] for r in subset
                },
            }

    per_seed = {}
    for seed in SEEDS:
        per_seed[f"seed_{seed}"] = {}
        for it in ITERS:
            c = next(r for r in control
                     if r["training_seed"] == seed and r["iteration"] == it)
            s = next(r for r in scheduled
                     if r["training_seed"] == seed and r["iteration"] == it)
            per_seed[f"seed_{seed}"][f"iter{it:03d}"] = {
                "control": c["greedy_avg"],
                "scheduled": s["greedy_avg"],
                "delta_sched_minus_control": s["greedy_avg"] - c["greedy_avg"],
            }
        per_seed[f"seed_{seed}"]["iter320_vs_bc"] = {
            "control": {
                "placement": next(r for r in control
                                  if r["training_seed"] == seed
                                  and r["iteration"] == 320)["greedy_avg"],
                "delta_vs_bc": next(r for r in control
                                    if r["training_seed"] == seed
                                    and r["iteration"] == 320)["delta_vs_bc"],
            },
            "scheduled": {
                "placement": next(r for r in scheduled
                                  if r["training_seed"] == seed
                                  and r["iteration"] == 320)["greedy_avg"],
                "delta_vs_bc": next(r for r in scheduled
                                    if r["training_seed"] == seed
                                    and r["iteration"] == 320)["delta_vs_bc"],
            },
        }

    paired = {}
    for seed in SEEDS:
        paired[f"seed_{seed}"] = {}
        for it in PAIRED_ITERS:
            sched_r = load_result(dev_greedy(SCHEDULE_DIR, SCHEDULE_LABEL, seed, it))
            ctrl_r = load_result(dev_greedy(DOSE_DIR, CONTROL_LABEL, seed, it))
            row = compare_pair(sched_r, ctrl_r, seed=0)
            row["iteration"] = it
            row["scheduled_avg"] = sched_r["metrics"]["avg_placement"]
            row["control_avg"] = ctrl_r["metrics"]["avg_placement"]
            paired[f"seed_{seed}"][f"iter{it:03d}"] = row

    success = evaluate_success(cross["320"]["control"], cross["320"]["scheduled"],
                               per_seed)

    if success["all_three_criteria"]:
        outcome = "SUCCESS"
        next_action = (
            "Freeze the scheduled PPO recipe. Replicate on additional seeds "
            "if needed, then eventual TEST confirmation once simulator and "
            "training procedure are frozen."
        )
    else:
        outcome = "STOP"
        next_action = (
            "STOP PPO tuning on the current simulator. Pivot to Simulator "
            "Fidelity Phase 2: late-game scaling, real card effects, board "
            "growth calibration, composition quality, hero/trinket/anomaly "
            "context. Do NOT run further PPO coefficient experiments."
        )

    # Console
    print("=" * 88)
    print("EXPERIMENT 6: SCHEDULED β=0.03→0.01 vs FIXED β=0.03")
    print(f"BC baseline: {BC_BASELINE:.3f}  |  Schedule: {EXPERIMENT_6_KL_SCHEDULE}")
    print("=" * 88)
    print(f"{'Iter':>5} {'Arm':>10} {'Mean':>8} {'Std':>8} {'Δ vs BC':>9} {'KL':>9}")
    for it in ITERS:
        for arm in ("control", "scheduled"):
            c = cross[str(it)][arm]
            print(f"{it:>5} {arm:>10} {c['placement']['mean']:>8.3f} "
                  f"{c['placement']['std']:>8.3f} "
                  f"{c['delta_vs_bc']['mean']:>+9.3f} "
                  f"{c['kl_from_warmstart']['mean']:>9.4f}")

    print("\nITER-320 PER-SEED vs BC")
    for seed in SEEDS:
        row = per_seed[f"seed_{seed}"]["iter320_vs_bc"]
        print(f"  seed {seed}: control Δ={row['control']['delta_vs_bc']:+.3f}  "
              f"scheduled Δ={row['scheduled']['delta_vs_bc']:+.3f}")

    print("\nSUCCESS CRITERIA (scheduled arm at iter 320)")
    for k, v in success.items():
        print(f"  {k}: {v}")

    analysis = {
        "experiment": "Replay Experiment 6 — Scheduled KL Anchoring",
        "evaluation_split": "dev",
        "bc_baseline": BC_BASELINE,
        "kl_schedule": EXPERIMENT_6_KL_SCHEDULE,
        "schedule_at_checkpoints": schedule_table(EXPERIMENT_6_KL_SCHEDULE, tuple(ITERS)),
        "contract": contract,
        "control_source": DOSE_DIR,
        "curves": {"control": control, "scheduled": scheduled},
        "cross_seed_by_iteration": cross,
        "per_seed": per_seed,
        "paired_scheduled_minus_control": paired,
        "success_evaluation": success,
        "outcome": outcome,
        "recommended_next_action": next_action,
    }

    with open(os.path.join(AGG_DIR, "cross_seed_summary.json"), "w") as f:
        json.dump(cross, f, indent=2)
    with open(os.path.join(AGG_DIR, "schedule_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    _plots(cross)
    print(f"\nOutcome: {outcome}")
    print(f"Next action: {next_action}")
    return 0


def _plots(cross) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    f, ax = plt.subplots(figsize=(8, 4.5))
    for arm, style in (("control", "o-"), ("scheduled", "s--")):
        ys = [cross[str(it)][arm]["placement"]["mean"] for it in ITERS]
        ax.plot(ITERS, ys, style, label=arm, linewidth=1.8, markersize=6)
    ax.axhline(BC_BASELINE, color="green", linestyle=":", label=f"BC ({BC_BASELINE})")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("cross-seed mean placement")
    ax.set_title("Experiment 6 — scheduled vs fixed β=0.03")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(PLOTS_DIR, "mean_placement.png"), dpi=140)
    plt.close(f)
    print(f"Saved plots -> {PLOTS_DIR}/")


if __name__ == "__main__":
    raise SystemExit(main())
