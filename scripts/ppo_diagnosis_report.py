"""Assemble the Experiment 1 (PPO degradation diagnosis) artifacts.

Reads only committed result JSON — the DEV evaluations, the policy-drift
metrics, and the per-iteration training diagnostics — and emits the learning
curve table, the paired iteration-vs-warm-start comparisons, and the plots.
Re-runnable: it computes nothing that isn't already in those files.

    python scripts/ppo_diagnosis_report.py

DEV split only. No Benchmark v1 TEST seeds are read or run here.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.analyze_benchmark import compare_pair, load_result       # noqa: E402

DIR = "results/ppo_diagnosis_v1"
ITERS = [0, 1, 2, 4, 8, 12, 16, 24, 32, 40]
PAIRED_ITERS = [1, 4, 8, 16, 24, 40]


def dev_path(it: int, field: str) -> str:
    return f"{DIR}/dev/iter{it:03d}_vs_{field}.json"


def main() -> int:
    greedy = {it: load_result(dev_path(it, "greedy")) for it in ITERS}
    random_f = {it: load_result(dev_path(it, "random")) for it in ITERS}
    drift = {r["checkpoint"]: r for r in
             json.load(open(f"{DIR}/policy_drift.json"))["checkpoints"]}
    diag = {r["iter"]: r for r in
            (json.loads(l) for l in open(f"{DIR}/train_diag.jsonl"))}

    # --- learning curve table -------------------------------------------------
    curve = []
    for it in ITERS:
        d = drift[f"iter_{it:03d}.pt"]
        g, r = greedy[it]["metrics"], random_f[it]["metrics"]
        curve.append({
            "iteration": it,
            "greedy_avg": g["avg_placement"],
            "greedy_ci95": greedy[it]["avg_placement_ci95"],
            "greedy_top4": g["top4_rate"],
            "greedy_win": g["win_rate"],
            "random_avg": r["avg_placement"],
            "random_top4": r["top4_rate"],
            "expert_agreement": d["expert_agreement"],
            "warmstart_agreement": d["warmstart_agreement"],
            "kl_from_warmstart": d["kl_from_warmstart_mean"],
            "value_mean": d["value_mean"],
            "value_std": d["value_std"],
            "entropy": diag[it]["entropy"] if it in diag else None,
            "approx_kl": diag[it]["approx_kl"] if it in diag else None,
            "clip_frac": diag[it]["clip_frac"] if it in diag else None,
            "grad_norm": diag[it]["grad_norm"] if it in diag else None,
            "rollout_avg_placement": (diag[it]["rollout_avg_placement"]
                                      if it in diag else None),
            "league_size": diag[it]["league_size"] if it in diag else 0,
            "shaping": diag[it]["shaping"] if it in diag else 1.0,
        })

    hdr = (f"{'iter':>5} {'GreedyAvg':>10} {'Top4':>7} {'RandAvg':>8} "
           f"{'Expert%':>8} {'WarmSt%':>8} {'KL':>8} {'Value':>14}")
    print("PPO DEV learning curve (200 games/checkpoint, DEV seeds, "
          "vs 7x field)")
    print(hdr)
    print("-" * len(hdr))
    for c in curve:
        print(f"{c['iteration']:>5} {c['greedy_avg']:>10.3f} "
              f"{100 * c['greedy_top4']:>6.1f}% {c['random_avg']:>8.3f} "
              f"{100 * c['expert_agreement']:>7.1f}% "
              f"{100 * c['warmstart_agreement']:>7.1f}% "
              f"{c['kl_from_warmstart']:>8.4f} "
              f"{c['value_mean']:>+7.3f}±{c['value_std']:<6.3f}")

    # --- paired comparisons vs iteration 0 ------------------------------------
    pairs = []
    print("\nPaired DEV comparisons vs iteration 0 (greedy field; "
          "positive = checkpoint WORSE than warm start)")
    print(f"{'compare':>16} {'mean diff':>10} {'95% CI (paired)':>22}   verdict")
    for it in PAIRED_ITERS:
        row = compare_pair(greedy[it], greedy[0], seed=0)
        row["iteration"] = it
        pairs.append(row)
        print(f"{'iter%d vs iter0' % it:>16} {row['mean_diff']:>+10.3f} "
              f"{'[%+.3f, %+.3f]' % tuple(row['ci95']):>22}   "
              f"{'no clear difference' if row['ci95'][0] <= 0 <= row['ci95'][1] else row['verdict']}")

    # random field, same pairing, for the tradeoff question
    pairs_random = []
    for it in PAIRED_ITERS:
        row = compare_pair(random_f[it], random_f[0], seed=0)
        row["iteration"] = it
        pairs_random.append(row)

    out = {"experiment": "Replay Experiment 1 — PPO Degradation Diagnosis",
           "evaluation_split": "dev",
           "dev_games_per_checkpoint": greedy[0]["games"],
           "dev_seed_range": greedy[0]["seed_range"],
           "curve": curve,
           "paired_vs_iter0_greedy": pairs,
           "paired_vs_iter0_random": pairs_random}
    with open(f"{DIR}/learning_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {DIR}/learning_curve.json")

    _plots(curve)
    return 0


def _plots(curve) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(f"{DIR}/plots", exist_ok=True)
    it = [c["iteration"] for c in curve]

    def fig(name, ylabel, title, series, invert=False, hline=None):
        f, ax = plt.subplots(figsize=(7, 4.2))
        for label, ys, style in series:
            ax.plot(it, ys, style, label=label, linewidth=1.8, markersize=5)
        if hline is not None:
            ax.axhline(hline[0], color="#888", linestyle=":", linewidth=1.2,
                       label=hline[1])
        # League snapshots enter at iteration 8 (LEAGUE_EVERY) — mark, don't claim.
        ax.axvline(8, color="#c44", linestyle="--", linewidth=1,
                   alpha=0.6, label="first league snapshot (iter 8)")
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        if invert:
            ax.invert_yaxis()          # lower placement is better
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(f"{DIR}/plots/{name}.png", dpi=140)
        plt.close(f)

    fig("dev_placement_greedy", "DEV avg placement (lower is better)",
        "DEV placement vs PPO iteration — 7x greedy field",
        [("PPO checkpoint", [c["greedy_avg"] for c in curve], "o-")],
        invert=True, hline=(curve[0]["greedy_avg"], "warm start (iter 0)"))
    fig("dev_placement_random", "DEV avg placement (lower is better)",
        "DEV placement vs PPO iteration — 7x random field",
        [("PPO checkpoint", [c["random_avg"] for c in curve], "o-")],
        invert=True, hline=(curve[0]["random_avg"], "warm start (iter 0)"))
    fig("expert_agreement", "agreement with greedy expert",
        "Expert-action agreement vs PPO iteration (frozen DEV corpus)",
        [("argmax == greedy expert",
          [c["expert_agreement"] for c in curve], "o-")])
    fig("policy_drift", "agreement / KL",
        "Policy drift from warm start vs PPO iteration",
        [("argmax == iter-0 argmax",
          [c["warmstart_agreement"] for c in curve], "o-"),
         ("mean KL(pi_0 || pi_k)",
          [c["kl_from_warmstart"] for c in curve], "s--")])
    print(f"Saved plots -> {DIR}/plots/")


if __name__ == "__main__":
    raise SystemExit(main())
