"""Assemble Replay Experiment 3 from committed JSON artifacts only.

Seed 0 is read exclusively from the committed Experiment 2 artifacts. Seeds
1-3 are the frozen-recipe replications. Checkpoint files are never consulted.
"""

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.analyze_benchmark import compare_pair, load_result  # noqa: E402
from ml.ppo_multiseed import (ALL_SEEDS, CORPUS_FINGERPRINT, CORPUS_STATES,
                              EPISODES_PER_ITERATION, EXPERIMENT2_DIR,
                              EXPERIMENT_DIR, GREEDY_GAMES, ITERATIONS,
                              MIXED_GAMES, REPORT_PATH,
                              WARMSTART_PARAMETER_SHA256)  # noqa: E402

FIELDS = ("greedy", "greedy4_random3")
PAIRS = ((40, 0), (80, 0), (160, 0), (320, 0),
         (80, 40), (160, 40), (320, 40), (320, 80))
BLOCKS = ((1, 40), (41, 160), (161, 320))
SIGNAL_KEYS = (
    "adv_mean", "adv_std", "adv_mean_abs", "adv_frac_positive",
    "adv_frac_zero", "adv_frac_negative", "return_mean", "return_std",
    "value_pred_mean", "value_pred_std", "value_explained_variance",
    "placement_mean", "placement_std", "shaping_reward_sum",
    "terminal_reward_sum", "entropy", "approx_kl", "clip_frac", "grad_norm",
    "pi_loss", "v_loss",
)


def _paths(seed, iteration, field):
    root = EXPERIMENT2_DIR if seed == 0 else EXPERIMENT_DIR / f"seed_{seed}"
    return root / "dev" / f"iter{iteration:03d}_vs_{field}.json"


def _load_eval(seed, iteration, field):
    path = _paths(seed, iteration, field)
    if path.exists():
        result = load_result(str(path))
        result["integrity"] = {
            "games_requested": result["games"],
            "games_completed": result["games"],
            "games_unfinished": 0,
            "mean_placement_sensitivity_bounds": [
                result["metrics"]["avg_placement"],
                result["metrics"]["avg_placement"],
            ],
        }
        return result
    raw = json.load(open(path.with_name(path.stem + "_integrity.json")))
    return {
        "agent": raw["agent"],
        "games": raw["games_requested"],
        "placements": raw["placements_nullable"],
        "metrics": raw["complete_case_metrics"],
        "avg_placement_ci95": None,
        "integrity": {
            k: raw[k] for k in (
                "games_requested", "games_completed", "games_unfinished",
                "mean_placement_sensitivity_bounds", "failures",
                "integrity_note",
            )
        },
    }


def _paired_eval(a, b, seed):
    """Canonical paired bootstrap, or labeled complete-case bootstrap."""
    if all(x is not None for x in a["placements"] + b["placements"]):
        return compare_pair(a, b, seed=seed)
    pairs = [(x, y) for x, y in zip(a["placements"], b["placements"])
             if x is not None and y is not None]
    diffs = [x - y for x, y in pairs]
    rng = random.Random(seed)
    resamples = 10000
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs)))
                   for _ in range(resamples))
    known_sum = sum(diffs)
    missing_refs = [
        y for x, y in zip(a["placements"], b["placements"]) if x is None
    ]
    n = len(a["placements"])
    return {
        "a": a["agent"], "b": b["agent"], "n": n,
        "n_complete_pairs": len(diffs),
        "n_unfinished": n - len(diffs),
        "mean_diff": st.mean(diffs),
        "ci95": [means[int(.025 * resamples) - 1],
                 means[int(.975 * resamples)]],
        "method": "paired percentile bootstrap on completed pairs",
        "resamples": resamples, "bootstrap_seed": seed,
        "mean_diff_sensitivity_bounds": [
            (known_sum + sum(1 - y for y in missing_refs)) / n,
            (known_sum + sum(8 - y for y in missing_refs)) / n,
        ],
        "verdict": "integrity-censored; no placement imputed",
    }


def _drift(seed):
    root = EXPERIMENT2_DIR if seed == 0 else EXPERIMENT_DIR / f"seed_{seed}"
    d = json.load(open(root / "policy_drift.json"))
    c = json.load(open(root / "action_category_drift.json"))
    corpus = d["corpus"]
    if (corpus["states"] != CORPUS_STATES or
            corpus["fingerprint_sha256"] != CORPUS_FINGERPRINT):
        raise ValueError(f"seed {seed} drift corpus does not match Experiment 2")
    if d["reference"]["parameter_sha256"] != WARMSTART_PARAMETER_SHA256:
        raise ValueError(f"seed {seed} warm-start hash mismatch")
    return (
        {int(x["checkpoint"][5:8]): x for x in d["checkpoints"]},
        {int(x["checkpoint"][5:8]): x for x in c["checkpoints"]},
    )


def _block(rows, lo, hi):
    selected = [r for r in rows if lo <= r["iter"] <= hi]
    return {
        k: st.mean(r[k] for r in selected if r.get(k) is not None)
        for k in SIGNAL_KEYS
    }


def _seed_boot(values, seed=20260831, resamples=10000):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values)))
                   for _ in range(resamples))
    return {
        "n_training_seeds": len(values),
        "mean": st.mean(values),
        "sd": st.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "ci95_seed_bootstrap": [
            means[int(0.025 * resamples)],
            means[int(0.975 * resamples) - 1],
        ],
        "resamples": resamples,
        "bootstrap_seed": seed,
    }


def build():
    evaluations = {}
    drift_rows, cat_rows, signals = {}, {}, {}
    for seed in ALL_SEEDS:
        evaluations[seed] = {
            field: {it: _load_eval(seed, it, field)
                    for it in ITERATIONS}
            for field in FIELDS
        }
        drift_rows[seed], cat_rows[seed] = _drift(seed)
        diag_path = (EXPERIMENT2_DIR / "train_diag.jsonl" if seed == 0 else
                     EXPERIMENT_DIR / f"seed_{seed}" / "train_diag.jsonl")
        diag = [json.loads(line) for line in open(diag_path)]
        if len(diag) != 320 or [r["iter"] for r in diag] != list(range(1, 321)):
            raise ValueError(f"seed {seed} diagnostics are not iterations 1..320")
        signals[seed] = {
            f"iters_{lo}_{hi}": _block(diag, lo, hi) for lo, hi in BLOCKS
        }

    curves, paired = {}, {}
    for seed in ALL_SEEDS:
        curves[seed] = []
        paired[seed] = {}
        for it in ITERATIONS:
            g = evaluations[seed]["greedy"][it]
            m = evaluations[seed]["greedy4_random3"][it]
            d = drift_rows[seed][it]
            curves[seed].append({
                "iteration": it,
                "cumulative_episodes": it * EPISODES_PER_ITERATION,
                "greedy_avg": g["metrics"]["avg_placement"],
                "greedy_ci95": g["avg_placement_ci95"],
                "greedy_median": g["metrics"]["median_placement"],
                "greedy_top4": g["metrics"]["top4_rate"],
                "greedy_win": g["metrics"]["win_rate"],
                "greedy_placement_counts": g["metrics"]["placement_counts"],
                "mixed_avg": m["metrics"]["avg_placement"],
                "mixed_ci95": m["avg_placement_ci95"],
                "mixed_top4": m["metrics"]["top4_rate"],
                "mixed_win": m["metrics"]["win_rate"],
                "expert_agreement": d["expert_agreement"],
                "warmstart_agreement": d["warmstart_agreement"],
                "kl_from_warmstart": d["kl_from_warmstart_mean"],
                "kl_from_warmstart_p95": d["kl_from_warmstart_p95"],
                "entropy": d["entropy_mean"],
                "value_mean": d["value_mean"],
                "value_std": d["value_std"],
                "parameter_sha256": d["parameter_sha256"],
                "checkpoint_sha256": d["checkpoint_sha256"],
                "greedy_integrity": g["integrity"],
                "mixed_integrity": m["integrity"],
            })
        for field in FIELDS:
            paired[seed][field] = {}
            for target, reference in PAIRS:
                row = _paired_eval(evaluations[seed][field][target],
                                   evaluations[seed][field][reference], seed)
                paired[seed][field][f"iter{target}_vs_iter{reference}"] = row

    aggregate_curves = []
    for it in ITERATIONS:
        row = {"iteration": it,
               "cumulative_episodes": it * EPISODES_PER_ITERATION}
        for key in ("greedy_avg", "mixed_avg", "expert_agreement",
                    "warmstart_agreement", "kl_from_warmstart"):
            vals = [next(x for x in curves[s] if x["iteration"] == it)[key]
                    for s in ALL_SEEDS]
            row[key] = _seed_boot(vals, seed=1000 + it)
        aggregate_curves.append(row)

    aggregate_effects = {}
    for field in FIELDS:
        for target, reference in PAIRS:
            key = f"{field}_iter{target}_vs_iter{reference}"
            effects = [
                paired[s][field][f"iter{target}_vs_iter{reference}"]["mean_diff"]
                for s in ALL_SEEDS
            ]
            repl = effects[1:]
            aggregate_effects[key] = {
                "all_seeds_0_3": _seed_boot(effects, seed=2000 + target + reference),
                "replication_seeds_1_3": _seed_boot(
                    repl, seed=3000 + target + reference),
                "per_seed_mean_diff": {str(s): effects[s] for s in ALL_SEEDS},
            }

    u_shape = {}
    for seed in ALL_SEEDS:
        gain = paired[seed]["greedy"]["iter80_vs_iter0"]
        decay = paired[seed]["greedy"]["iter320_vs_iter80"]
        censored = bool(gain.get("n_unfinished") or decay.get("n_unfinished"))
        u_shape[seed] = {
            "iter80_better_than_iter0_point_estimate": gain["mean_diff"] < 0,
            "iter80_better_than_iter0_ci_excludes_zero": gain["ci95"][1] < 0,
            "iter320_worse_than_iter80_point_estimate": decay["mean_diff"] > 0,
            "iter320_worse_than_iter80_ci_excludes_zero": decay["ci95"][0] > 0,
            "u_shape_point_estimate": gain["mean_diff"] < 0 and decay["mean_diff"] > 0,
            "integrity_censored": censored,
            "strict_u_shape": (not censored and gain["ci95"][1] < 0
                               and decay["ci95"][0] > 0),
        }
    replication = [u_shape[s] for s in (1, 2, 3)]
    counts = {
        k: sum(bool(x[k]) for x in replication) for k in replication[0]
    }

    action_iter320 = {}
    for seed in ALL_SEEDS:
        x = cat_rows[seed][320]
        action_iter320[seed] = {
            "vs_expert": x["vs_expert"],
            "vs_warmstart": x["vs_warmstart"],
            "freeze": {
                "expert_reference_count":
                    x["vs_expert"]["reference_category_counts"]["freeze"],
                "expert_to_freeze": sum(
                    row.get("freeze", 0)
                    for cat, row in x["vs_expert"]["confusion_matrix"].items()
                    if cat != "freeze"
                ),
                "warmstart_to_freeze": sum(
                    row.get("freeze", 0)
                    for cat, row in x["vs_warmstart"]["confusion_matrix"].items()
                    if cat != "freeze"
                ),
            },
        }

    return {
        "experiment": "Replay Experiment 3 — Multi-seed PPO Budget Replication",
        "evaluation_split": "DEV only; TEST never used",
        "seeds": list(ALL_SEEDS),
        "new_training_seeds": [1, 2, 3],
        "seed0_source": "committed Experiment 2 artifacts only",
        "iterations": list(ITERATIONS),
        "episodes_per_iteration": EPISODES_PER_ITERATION,
        "greedy_games": GREEDY_GAMES,
        "mixed_games": MIXED_GAMES,
        "corpus": {"states": CORPUS_STATES,
                   "fingerprint_sha256": CORPUS_FINGERPRINT},
        "warmstart_parameter_sha256": WARMSTART_PARAMETER_SHA256,
        "curves": {str(k): v for k, v in curves.items()},
        "paired": {str(k): v for k, v in paired.items()},
        "aggregate_curves": aggregate_curves,
        "aggregate_effects": aggregate_effects,
        "u_shape_classification": {
            "definition": ("point U-shape: iter80 mean placement is lower than "
                           "iter0 and iter320 is higher than iter80; strict "
                           "U-shape requires both paired 95% CIs to exclude 0"),
            "per_seed": {str(k): v for k, v in u_shape.items()},
            "replication_seed_counts_out_of_3": counts,
        },
        "action_category_iter320": {str(k): v for k, v in action_iter320.items()},
        "rl_signal_blocks": {str(k): v for k, v in signals.items()},
    }


def _plots_from_committed_json(path):
    """Plots deliberately reload their sole source from the written JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.load(open(path))
    out = EXPERIMENT_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#666666", "#0072B2", "#D55E00", "#009E73"]
    eps = [x["cumulative_episodes"] for x in data["curves"]["0"]]

    def lines(filename, keys, ylabel, title, invert=False):
        fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.2))
        axes = [axes] if len(keys) == 1 else axes
        for ax, (key, label) in zip(axes, keys):
            for seed, color in zip(data["seeds"], colors):
                ys = [x[key] for x in data["curves"][str(seed)]]
                ax.plot(eps, ys, "o-", label=f"seed {seed}", color=color)
            ax.set_xlabel("cumulative PPO episodes")
            ax.set_ylabel(ylabel)
            ax.set_title(label)
            ax.grid(alpha=.25)
            if invert:
                ax.invert_yaxis()
        axes[0].legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)

    lines("A_multiseed_dev_curves.png",
          [("greedy_avg", "7× greedy"), ("mixed_avg", "greedy4_random3")],
          "DEV average placement (lower is better)",
          "Experiment 3 placement curves", invert=True)
    lines("B_multiseed_drift_curves.png",
          [("expert_agreement", "expert agreement"),
           ("warmstart_agreement", "warm-start agreement"),
           ("kl_from_warmstart", "KL(warm start || checkpoint)")],
          "metric", "Experiment 3 policy drift")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = ("adv_mean_abs", "value_explained_variance", "entropy", "clip_frac")
    labels = list(data["rl_signal_blocks"]["0"])
    for ax, metric in zip(axes.flat, metrics):
        for seed, color in zip(data["seeds"], colors):
            ys = [data["rl_signal_blocks"][str(seed)][b][metric] for b in labels]
            ax.plot(labels, ys, "o-", label=f"seed {seed}", color=color)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("Experiment 3 RL-signal comparison")
    fig.tight_layout()
    fig.savefig(out / "C_multiseed_rl_signal.png", dpi=140)
    plt.close(fig)


def _fmt_effect(row):
    text = (f"{row['mean_diff']:+.3f} "
            f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]")
    if row.get("n_unfinished"):
        bounds = row["mean_diff_sensitivity_bounds"]
        text += (f" complete-case ({row['n_complete_pairs']}/{row['n']}); "
                 f"sensitivity [{bounds[0]:+.3f}, {bounds[1]:+.3f}]")
    return text


def _report(data):
    count = data["u_shape_classification"]["replication_seed_counts_out_of_3"]
    point_n, strict_n = count["u_shape_point_estimate"], count["strict_u_shape"]
    lines = [
        "# Replay Experiment 3 — Multi-seed PPO Budget Replication",
        "",
        "DEV only; Benchmark v1 TEST was never used. Seed 0 is the committed "
        "Experiment 2 trajectory; seeds 1–3 are new frozen-recipe replications.",
        "",
        "## Protocol",
        "",
        f"- Warm-start parameter SHA256: `{WARMSTART_PARAMETER_SHA256}`.",
        f"- Frozen corpus: {CORPUS_STATES:,} states, `{CORPUS_FINGERPRINT}`.",
        "- PPO: 16 episodes/iteration, 320 iterations, shaping horizon 40; "
        "checkpoints 0/40/80/160/320.",
        "- DEV: 1,000 paired games vs 7× greedy and 500 paired games vs fixed "
        "`greedy4_random3`, starting at seed 10,550,000.",
        "",
        "## Full budget table",
        "",
        "| iter | episodes | seed | greedy avg | top-4 | win | mixed avg | unfinished G/M | expert | warm-start | KL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in data["seeds"]:
        for x in data["curves"][str(seed)]:
            lines.append(
                f"| {x['iteration']} | {x['cumulative_episodes']} | {seed} | "
                f"{x['greedy_avg']:.3f} | {x['greedy_top4']:.1%} | "
                f"{x['greedy_win']:.1%} | {x['mixed_avg']:.3f} | "
                f"{x['greedy_integrity']['games_unfinished']}/"
                f"{x['mixed_integrity']['games_unfinished']} | "
                f"{x['expert_agreement']:.1%} | {x['warmstart_agreement']:.1%} | "
                f"{x['kl_from_warmstart']:.3f} |")
    for field in FIELDS:
        lines += ["", f"## Paired effects — {field}", "",
                  "Positive is worse placement. Each CI is a paired 10,000-resample bootstrap.",
                  "", "| seed | comparison | effect [95% CI] |",
                  "|---:|---|---|"]
        for seed in data["seeds"]:
            for target, reference in PAIRS:
                row = data["paired"][str(seed)][field][
                    f"iter{target}_vs_iter{reference}"]
                lines.append(
                    f"| {seed} | iter{target} − iter{reference} | "
                    f"{_fmt_effect(row)} |")
    lines += [
        "",
        "## Replication result",
        "",
        f"Among new seeds 1–3, the iter80-vs-iter0 improvement occurs by point "
        f"estimate in {count['iter80_better_than_iter0_point_estimate']}/3 "
        f"({count['iter80_better_than_iter0_ci_excludes_zero']}/3 with CI excluding "
        f"zero). Iter320 regresses from iter80 in "
        f"{count['iter320_worse_than_iter80_point_estimate']}/3 by point estimate "
        f"({count['iter320_worse_than_iter80_ci_excludes_zero']}/3 with CI excluding "
        f"zero). The pre-documented U shape appears in {point_n}/3 by point "
        f"estimate and {strict_n}/3 under the strict paired-CI definition.",
        "",
        "Cross-seed descriptive summaries and seed-level bootstrap intervals are "
        "in `aggregate.json`; with only four trajectories, these intervals are "
        "descriptive and not a population-level asymptotic claim.",
        "",
        "## Drift, action categories, and RL signal",
        "",
        "The full budget table above is the expert/warm-start/KL curve. "
        "Iteration-320 category summaries follow; the committed aggregate keeps "
        "the complete confusion matrices.",
        "",
        "| seed | expert agreement | warm-start agreement | expert→freeze | warm-start→freeze | largest expert transitions |",
        "|---:|---:|---:|---:|---:|---|",
        "",
    ]
    for seed in data["seeds"]:
        curve320 = next(x for x in data["curves"][str(seed)]
                        if x["iteration"] == 320)
        cat = data["action_category_iter320"][str(seed)]
        transitions = ", ".join(
            f"{x['from']}→{x['to']} {x['count']}"
            for x in cat["vs_expert"]["top_transitions"][:3])
        lines.append(
            f"| {seed} | {curve320['expert_agreement']:.1%} | "
            f"{curve320['warmstart_agreement']:.1%} | "
            f"{cat['freeze']['expert_to_freeze']} | "
            f"{cat['freeze']['warmstart_to_freeze']} | {transitions} |")
    lines += [
        "",
        "RL-signal means by training block (raw advantages are measured before "
        "normalization):",
        "",
        "| seed | block | mean abs adv | positive adv | value EV | return SD | placement SD | entropy | approx KL | clip frac |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in data["seeds"]:
        for block, values in data["rl_signal_blocks"][str(seed)].items():
            lines.append(
                f"| {seed} | {block} | {values['adv_mean_abs']:.3f} | "
                f"{values['adv_frac_positive']:.3f} | "
                f"{values['value_explained_variance']:.3f} | "
                f"{values['return_std']:.3f} | {values['placement_std']:.3f} | "
                f"{values['entropy']:.3f} | {values['approx_kl']:.4f} | "
                f"{values['clip_frac']:.3f} |")
    lines += [
        "",
        "Cross-seed primary-effect summaries:",
        "",
        "| comparison | group | mean | SD | range | seed-bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target, reference in ((80, 0), (320, 80)):
        effect = data["aggregate_effects"][
            f"greedy_iter{target}_vs_iter{reference}"]
        for key, label in (("all_seeds_0_3", "seeds 0–3"),
                           ("replication_seeds_1_3", "seeds 1–3")):
            x = effect[key]
            lines.append(
                f"| iter{target} − iter{reference} | {label} | "
                f"{x['mean']:+.3f} | {x['sd']:.3f} | "
                f"[{x['min']:+.3f}, {x['max']:+.3f}] | "
                f"[{x['ci95_seed_bootstrap'][0]:+.3f}, "
                f"{x['ci95_seed_bootstrap'][1]:+.3f}] |")
    lines += [
        "",
        "Seed 1 iteration 320 had 5/1,000 unfinished greedy games and 2/500 "
        "unfinished mixed games at the unchanged 400-decision integrity cap. "
        "They remain null, never imputed; its reported means/CIs are labeled "
        "complete-case and paired effects include best/worst-case bounds.",
        "",
        "All remaining raw-advantage, return, critic, entropy, KL, clipping, "
        "gradient, loss, placement, reward-source, category, and hash fields "
        "are machine-recorded in `aggregate.json`.",
        "",
        "## Outcome and next experiment",
        "",
    ]
    if point_n >= 2:
        outcome = "A"
        conclusion = "the transient iter80 gain and later decay replicate in a majority of new seeds"
        recommendation = ("Experiment 4 recommendation: test a single fixed "
                          "warm-start KL anchor coefficient against this frozen "
                          "multi-seed protocol.")
    else:
        outcome = "B"
        conclusion = "the seed-0 U shape does not replicate in a majority of new seeds"
        recommendation = ("Experiment 4 recommendation: test a single fixed "
                          "warm-start KL anchor coefficient against this frozen "
                          "multi-seed protocol.")
    lines += [
        f"Selected **Outcome {outcome}**: {conclusion}.",
        "",
        recommendation,
        "",
        "No PPO tuning, checkpoint selection, Experiment 4 execution, or TEST "
        "evaluation was performed.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    return outcome


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    data = build()
    aggregate = EXPERIMENT_DIR / "aggregate.json"
    aggregate.write_text(json.dumps(data, indent=2) + "\n")
    (EXPERIMENT_DIR / "paired_analysis.json").write_text(json.dumps({
        "method": "per-seed paired percentile bootstrap, 10000 resamples",
        "positive_difference": "checkpoint places worse than reference",
        "paired": data["paired"],
        "cross_seed": data["aggregate_effects"],
        "u_shape_classification": data["u_shape_classification"],
    }, indent=2) + "\n")
    (EXPERIMENT_DIR / "rl_signal.json").write_text(json.dumps({
        "definitions": {k: "same definition as Experiment 2 ml.train_ppo"
                        for k in SIGNAL_KEYS},
        "blocks": data["rl_signal_blocks"],
    }, indent=2) + "\n")
    _plots_from_committed_json(aggregate)
    outcome = _report(data)
    print(f"Saved Experiment 3 aggregate artifacts; selected Outcome {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
