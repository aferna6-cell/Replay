"""Cross-training-seed replication analysis for the PPO budget curve.

Experiment 2 measured one PPO trajectory (training seed 0) and found a
non-monotonic "U" — a significant improvement at 1,280 episodes that decayed
by 5,120. Experiment 3 repeats the identical frozen recipe from independent
PPO training seeds. This module holds the analysis primitives so they can be
unit-tested away from the orchestration script:

  * ``paired_table``      — the nine pre-specified within-seed paired
    comparisons, computed with the *same* deterministic paired bootstrap
    Experiment 2 used (``ml.analyze_benchmark.compare_pair``).
  * ``classify_curve``    — the pre-specified, automated shape rule that
    labels one seed's budget curve. Written and committed before any
    Experiment 3 evaluation result existed.
  * ``cross_seed_summary``— descriptive statistics across training seeds.

REPLICATION UNIT. The unit of replication is the *training seed*, not the
DEV game. The 1000 paired DEV games measure one trained model precisely; the
handful of training seeds is the only sample that speaks to training
variability. Nothing here pools games across seeds into one n=4000 sample.
"""

import json
import math
import os
import statistics as st
from typing import Dict, List, Optional, Sequence

from .analyze_benchmark import compare_pair, load_result

PRIMARY_ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16

# Pre-specified before any Experiment 3 result was measured: (target,
# reference); positive difference = the target checkpoint places WORSE.
COMPARISONS = [(40, 0), (80, 0), (160, 0), (320, 0),
               (80, 40), (160, 40), (320, 40),
               (160, 80), (320, 80)]

# --- the pre-specified shape rule ---------------------------------------------
# Tolerances are part of the rule and were fixed in advance.
FLAT_TOL = 0.02          # placements: ties allowed when ranking monotonicity
FLAT_RANGE = 0.15        # placements: span of the five budgets to call "flat"

SHAPE_CLASSES = ["U-like / transient improvement", "monotonic improvement",
                 "monotonic degradation", "mostly flat/noisy", "other"]

SHAPE_RULE_DOC = """\
Inputs: the five primary-budget DEV average placements (lower is better) and
the nine within-seed paired-bootstrap comparisons. A comparison (a - b) is
called `better` when its 95% CI lies entirely below 0 (a places better than
b), `worse` when it lies entirely above 0, and `none` when it contains 0.
The first matching rule wins:

1. U-like / transient improvement — some interior budget i in {40, 80, 160}
   is significantly BETTER than the warm start (i - 0 is `better`) AND
   iteration 320 is significantly WORSE than that same i (320 - i is
   `worse`). A real gain that a longer budget then gives back.
2. monotonic improvement — 320 - 0 is `better` and the five averages are
   non-increasing in budget (each step may rise by at most FLAT_TOL=0.02).
3. monotonic degradation — 320 - 0 is `worse` and the five averages are
   non-decreasing in budget (each step may fall by at most FLAT_TOL=0.02).
4. mostly flat/noisy — none of the nine CIs excludes zero AND the span
   max-min of the five averages is at most FLAT_RANGE=0.15 placements.
5. other — anything else; the significant comparisons are listed so the
   curve is never silently forced into one of the four labels above.
"""


def episodes(iteration: int) -> int:
    return iteration * EPISODES_PER_ITER


def significance(ci: Sequence[float]) -> str:
    """`better` / `worse` / `none` for a paired difference CI (a - b), where
    placement is lower-is-better so a negative interval favors `a`."""
    lo, hi = float(ci[0]), float(ci[1])
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "none"


def paired_table(results_by_iter: Dict[int, Dict], boot_seed: int = 0,
                 comparisons: Optional[Sequence] = None) -> List[Dict]:
    """The pre-specified within-seed paired comparisons for one training seed.

    ``results_by_iter`` maps a PPO iteration to a loaded DEV result JSON. The
    comparisons run on the identical 1000 DEV game seeds, so the placement
    lists pair game-by-game (``compare_pair`` re-checks that).
    """
    rows = []
    for target, ref in (comparisons or COMPARISONS):
        row = compare_pair(results_by_iter[target], results_by_iter[ref],
                           seed=boot_seed)
        row["iteration"] = target
        row["reference_iteration"] = ref
        row["label"] = f"iter{target} - iter{ref}"
        row["significance"] = significance(row["ci95"])
        rows.append(row)
    return rows


def _non_increasing(values: Sequence[float], tol: float = FLAT_TOL) -> bool:
    return all(b <= a + tol for a, b in zip(values, values[1:]))


def _non_decreasing(values: Sequence[float], tol: float = FLAT_TOL) -> bool:
    return all(b >= a - tol for a, b in zip(values, values[1:]))


def classify_curve(avg_by_iter: Dict[int, float],
                   paired_rows: Sequence[Dict]) -> Dict:
    """Label one training seed's budget curve with the rule in SHAPE_RULE_DOC.

    Returns the class, the reason, the significant comparisons it used, and a
    purely descriptive point-estimate reading (which budget was best, and
    whether iteration 320 gave that back) that is reported alongside the
    class but never used to decide it.
    """
    sig = {(r["iteration"], r["reference_iteration"]): r["significance"]
           for r in paired_rows}
    values = [avg_by_iter[i] for i in PRIMARY_ITERS]
    significant = [f"{r['label']} {r['significance']}" for r in paired_rows
                   if r["significance"] != "none"]

    best_iter = min(PRIMARY_ITERS, key=lambda i: avg_by_iter[i])
    descriptive = {
        "best_budget_iteration": best_iter,
        "best_budget_episodes": episodes(best_iter),
        "range": max(values) - min(values),
        # point-estimate reading only; no significance is claimed here
        "point_estimate_transient": bool(
            best_iter in (40, 80, 160)
            and avg_by_iter[320] > avg_by_iter[best_iter]
            and avg_by_iter[0] > avg_by_iter[best_iter]),
    }

    transient = [i for i in (40, 80, 160)
                 if sig.get((i, 0)) == "better" and sig.get((320, i)) == "worse"]
    if transient:
        return {"shape_class": SHAPE_CLASSES[0],
                "reason": (f"iter{transient[0]} is significantly better than "
                           f"the warm start and iter320 is significantly "
                           f"worse than iter{transient[0]}"),
                "transient_peak_iterations": transient,
                "significant_comparisons": significant,
                "descriptive": descriptive}
    if sig.get((320, 0)) == "better" and _non_increasing(values):
        return {"shape_class": SHAPE_CLASSES[1],
                "reason": ("placement is non-increasing across all five "
                           "budgets and iter320 is significantly better than "
                           "the warm start"),
                "transient_peak_iterations": [],
                "significant_comparisons": significant,
                "descriptive": descriptive}
    if sig.get((320, 0)) == "worse" and _non_decreasing(values):
        return {"shape_class": SHAPE_CLASSES[2],
                "reason": ("placement is non-decreasing across all five "
                           "budgets and iter320 is significantly worse than "
                           "the warm start"),
                "transient_peak_iterations": [],
                "significant_comparisons": significant,
                "descriptive": descriptive}
    if not significant and descriptive["range"] <= FLAT_RANGE:
        return {"shape_class": SHAPE_CLASSES[3],
                "reason": (f"no paired CI excludes zero and the five budgets "
                           f"span only {descriptive['range']:.3f} placements "
                           f"(<= {FLAT_RANGE})"),
                "transient_peak_iterations": [],
                "significant_comparisons": [],
                "descriptive": descriptive}
    return {"shape_class": SHAPE_CLASSES[4],
            "reason": ("no pre-specified shape matched; the significant "
                       "comparisons are listed verbatim"),
            "transient_peak_iterations": [],
            "significant_comparisons": significant,
            "descriptive": descriptive}


# --- cross-training-seed description ------------------------------------------
def describe(values: Sequence[float]) -> Dict:
    """Mean / median / min / max / sample sd of a handful of training seeds."""
    vals = [float(v) for v in values]
    return {"n": len(vals), "mean": st.mean(vals), "median": st.median(vals),
            "min": min(vals), "max": max(vals),
            "sd": st.stdev(vals) if len(vals) > 1 else 0.0}


# Student-t 97.5th percentiles for the tiny n this experiment can reach.
_T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}


def exploratory_ci(values: Sequence[float]) -> Dict:
    """EXPLORATORY cross-training-seed interval. With a handful of training
    trajectories this is unstable and is reported as a descriptive spread
    indicator only — never as a population claim about PPO."""
    vals = [float(v) for v in values]
    n = len(vals)
    d = describe(vals)
    if n < 2:
        return {**d, "ci95": None, "label": "exploratory",
                "caveat": "fewer than two training seeds; no interval"}
    t = _T975.get(n, 1.96)
    half = t * d["sd"] / math.sqrt(n)
    return {**d, "ci95": [d["mean"] - half, d["mean"] + half],
            "se": d["sd"] / math.sqrt(n), "t_multiplier": t,
            "label": "exploratory",
            "caveat": (f"n={n} training trajectories: this interval is "
                       f"unstable and must not be read as a population "
                       f"effect for PPO")}


def cross_seed_summary(avg_by_seed: Dict[int, Dict[int, float]],
                       iterations: Optional[Sequence[int]] = None) -> Dict:
    """Per-budget description across training seeds, plus the raw per-seed
    values so no individual trajectory is hidden behind an average."""
    iters = list(iterations or PRIMARY_ITERS)
    seeds = sorted(avg_by_seed)
    rows = []
    for it in iters:
        vals = [avg_by_seed[s][it] for s in seeds]
        rows.append({"iteration": it, "cumulative_episodes": episodes(it),
                     "by_seed": {str(s): avg_by_seed[s][it] for s in seeds},
                     **describe(vals)})
    return {"training_seeds": seeds, "iterations": iters,
            "replication_unit": "training seed (not the individual DEV game)",
            "per_budget": rows}


def effect_across_seeds(paired_by_seed: Dict[int, Sequence[Dict]],
                        target: int, reference: int) -> Dict:
    """One pre-specified comparison gathered across training seeds: every
    seed's effect and CI, plus how many improved / worsened / were
    inconclusive by the within-seed paired CI."""
    seeds = sorted(paired_by_seed)
    per_seed = []
    for s in seeds:
        row = next(r for r in paired_by_seed[s]
                   if r["iteration"] == target
                   and r["reference_iteration"] == reference)
        per_seed.append({"training_seed": s, "mean_diff": row["mean_diff"],
                         "ci95": row["ci95"],
                         "significance": row["significance"]})
    diffs = [r["mean_diff"] for r in per_seed]
    sigs = [r["significance"] for r in per_seed]
    return {
        "comparison": f"iter{target} - iter{reference}",
        "sign_convention": ("positive = iter%d places WORSE than iter%d"
                            % (target, reference)),
        "per_seed": per_seed,
        "n_point_estimate_better": sum(1 for d in diffs if d < 0),
        "n_point_estimate_worse": sum(1 for d in diffs if d > 0),
        "n_ci_excludes_zero_better": sigs.count("better"),
        "n_ci_excludes_zero_worse": sigs.count("worse"),
        "n_ci_includes_zero": sigs.count("none"),
        "across_seed_effect": exploratory_ci(diffs),
    }


# --- artifact loading ---------------------------------------------------------
RL_BLOCKS = {"iters_1_40": (1, 40), "iters_41_160": (41, 160),
             "iters_161_320": (161, 320)}
RL_METRICS = ("rollout_avg_placement", "adv_mean", "adv_std", "adv_mean_abs",
              "adv_frac_positive", "adv_frac_negative", "adv_frac_zero",
              "return_mean", "return_std", "value_pred_mean",
              "value_pred_std", "value_explained_variance", "placement_std",
              "shaping_reward_sum", "terminal_reward_sum", "entropy",
              "approx_kl", "clip_frac", "grad_norm", "pi_loss", "v_loss",
              "steps", "league_size")


def iteration_of(checkpoint_name: str) -> int:
    """80 from 'iter_080.pt' — Experiment 2's drift rows key on the filename."""
    stem = os.path.basename(checkpoint_name).split(".")[0]
    return int(stem.split("_")[-1])


def load_seed_bundle(training_seed: int, dev_dir: str, drift_path: str,
                     categories_path: str, diag_path: str,
                     iterations: Optional[Sequence[int]] = None) -> Dict:
    """Every committed artifact for one training seed, keyed by iteration.

    Reads only files; computes nothing. Experiment 2's seed-0 artifacts are
    loaded through this same function and are never written back.
    """
    iters = list(iterations or PRIMARY_ITERS)
    drift = {iteration_of(r["checkpoint"]): r
             for r in json.load(open(drift_path, encoding="utf-8"))["checkpoints"]}
    cats = {iteration_of(r["checkpoint"]): r
            for r in json.load(open(categories_path,
                                    encoding="utf-8"))["checkpoints"]}
    with open(diag_path, encoding="utf-8") as f:
        diag = [json.loads(line) for line in f if line.strip()]
    return {
        "training_seed": training_seed,
        "iterations": iters,
        "greedy": {it: load_result(f"{dev_dir}/iter{it:03d}_vs_greedy.json")
                   for it in iters},
        "mixed": {it: load_result(
            f"{dev_dir}/iter{it:03d}_vs_greedy4_random3.json") for it in iters},
        "drift": {it: drift[it] for it in iters},
        "categories": {it: cats[it] for it in iters},
        "diag": diag,
    }


def build_curve(bundle: Dict) -> List[Dict]:
    """One row per primary budget: DEV placement on both fields plus the
    drift diagnostics measured on the frozen corpus."""
    rows = []
    for it in bundle["iterations"]:
        g, m = bundle["greedy"][it], bundle["mixed"][it]
        d, c = bundle["drift"][it], bundle["categories"][it]["vs_expert"]
        rows.append({
            "training_seed": bundle["training_seed"],
            "iteration": it, "cumulative_episodes": episodes(it),
            "greedy_avg": g["metrics"]["avg_placement"],
            "greedy_ci95": g["avg_placement_ci95"],
            "greedy_median": g["metrics"]["median_placement"],
            "greedy_std": g["metrics"]["std_placement"],
            "greedy_top4": g["metrics"]["top4_rate"],
            "greedy_win": g["metrics"]["win_rate"],
            "greedy_placement_counts": g["metrics"]["placement_counts"],
            "greedy_games": g["games"], "greedy_seed_range": g["seed_range"],
            "mixed_avg": m["metrics"]["avg_placement"],
            "mixed_ci95": m["avg_placement_ci95"],
            "mixed_top4": m["metrics"]["top4_rate"],
            "mixed_win": m["metrics"]["win_rate"],
            "mixed_games": m["games"], "mixed_seed_range": m["seed_range"],
            "expert_agreement": d["expert_agreement"],
            "warmstart_agreement": d["warmstart_agreement"],
            "kl_from_warmstart": d["kl_from_warmstart_mean"],
            "corpus_entropy": d["entropy_mean"],
            "value_mean": d["value_mean"], "value_std": d["value_std"],
            "parameter_sha256": d["parameter_sha256"],
            "checkpoint_sha256": d["checkpoint_sha256"],
            "expert_disagreement_by_category":
                c["disagreement_share_by_category"],
            "drift_contribution_by_category": c["contribution_to_total_drift"],
        })
    return rows


def rl_blocks(diag: Sequence[Dict]) -> Dict:
    """Experiment 2's per-training-block means, recomputed per seed."""
    out = {}
    for name, (lo, hi) in RL_BLOCKS.items():
        rows = [r for r in diag if lo <= r["iter"] <= hi]
        out[name] = {"iterations": [lo, hi], "n": len(rows), **{
            k: (st.mean(r[k] for r in rows if r.get(k) is not None)
                if any(r.get(k) is not None for r in rows) else None)
            for k in RL_METRICS}}
    return out


# --- drift / category replication ---------------------------------------------
def freeze_stats(category_row: Dict) -> Dict:
    """How often the checkpoint picks `freeze` on the frozen corpus — the
    action Experiment 2 found the greedy expert never takes."""
    conf = category_row["vs_expert"]
    matrix = conf["confusion_matrix"]
    chosen = sum(tos.get("freeze", 0) for tos in matrix.values())
    n = conf["n_states"]
    return {"freeze_selections": chosen, "n_states": n,
            "freeze_rate": chosen / n if n else None,
            "freeze_appears": chosen > 0,
            "expert_freeze_states":
                conf["reference_category_counts"].get("freeze", 0)}


def category_replication(category_row: Dict) -> Dict:
    """The expert -> PPO category picture Experiment 2 reported, per seed."""
    conf = category_row["vs_expert"]
    return {
        "iteration": category_row.get("iteration"),
        "overall_agreement": conf["overall_agreement"],
        "n_disagreements": conf["n_disagreements"],
        "confusion_matrix": conf["confusion_matrix"],
        "expert_category_counts": conf["reference_category_counts"],
        "disagreement_share_by_category":
            conf["disagreement_share_by_category"],
        "contribution_to_total_drift": conf["contribution_to_total_drift"],
        "top_transitions": conf.get("top_transitions", [])[:6],
        **freeze_stats(category_row),
    }


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation, ties averaged. Descriptive association only — with
    five budgets per seed it cannot support a causal reading."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def drift_vs_performance(curve: Sequence[Dict]) -> Dict:
    """Within one seed: is the best-placing budget also closer to the expert
    and to its warm start than the budgets that follow it?

    Association only. Nothing here identifies a direction of causation — a
    checkpoint could place well *because* it stayed near the prior, or drift
    and placement could both track a third property of the trajectory.
    """
    rows = sorted(curve, key=lambda r: r["iteration"])
    placements = [r["greedy_avg"] for r in rows]
    best = min(rows, key=lambda r: r["greedy_avg"])
    later = [r for r in rows if r["iteration"] > best["iteration"]]
    return {
        "best_iteration": best["iteration"],
        "best_greedy_avg": best["greedy_avg"],
        "best_expert_agreement": best["expert_agreement"],
        "best_kl_from_warmstart": best["kl_from_warmstart"],
        "later_mean_expert_agreement":
            st.mean(r["expert_agreement"] for r in later) if later else None,
        "later_mean_kl_from_warmstart":
            st.mean(r["kl_from_warmstart"] for r in later) if later else None,
        "best_has_higher_expert_agreement_than_later":
            (best["expert_agreement"]
             > st.mean(r["expert_agreement"] for r in later)) if later else None,
        "best_has_lower_kl_than_later":
            (best["kl_from_warmstart"]
             < st.mean(r["kl_from_warmstart"] for r in later)) if later else None,
        "spearman_placement_vs_expert_agreement":
            spearman(placements, [r["expert_agreement"] for r in rows]),
        "spearman_placement_vs_kl":
            spearman(placements, [r["kl_from_warmstart"] for r in rows]),
        "note": ("descriptive association across five budgets within one "
                 "training seed; no causal claim"),
    }


# --- plot inputs --------------------------------------------------------------
def build_plot_data(curves_by_seed: Dict[int, Sequence[Dict]],
                    categories_iter320: Dict[int, Dict],
                    rl_blocks_by_seed: Dict[int, Dict]) -> Dict:
    """Every numeric series the plots draw, derived purely from the loaded
    result artifacts. The plotting code renders this and nothing else, so no
    measured value is ever typed into a figure script."""
    from .action_categories import CATEGORIES
    seeds = sorted(curves_by_seed)
    iters = [r["iteration"] for r in sorted(curves_by_seed[seeds[0]],
                                            key=lambda r: r["iteration"])]
    eps = [episodes(i) for i in iters]

    def series(key):
        return {s: [r[key] for r in sorted(curves_by_seed[s],
                                           key=lambda r: r["iteration"])]
                for s in seeds}

    greedy = series("greedy_avg")
    mean_curve = [st.mean(greedy[s][i] for s in seeds)
                  for i in range(len(iters))]
    return {
        "training_seeds": seeds, "iterations": iters, "episodes": eps,
        "greedy_avg": greedy,
        "greedy_mean_across_seeds": mean_curve,
        "mixed_avg": series("mixed_avg"),
        "expert_agreement": series("expert_agreement"),
        "warmstart_agreement": series("warmstart_agreement"),
        "kl_from_warmstart": series("kl_from_warmstart"),
        "categories": list(CATEGORIES),
        "category_disagreement_iter320": {
            s: [(categories_iter320[s]["disagreement_share_by_category"]
                 .get(c) or 0.0) for c in CATEGORIES] for s in seeds},
        "freeze_selections_iter320": {
            s: categories_iter320[s]["freeze_selections"] for s in seeds},
        "rl_blocks": {s: rl_blocks_by_seed[s] for s in seeds},
        "rl_block_names": list(RL_BLOCKS),
    }
