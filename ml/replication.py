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

import math
import statistics as st
from typing import Dict, List, Optional, Sequence

from .analyze_benchmark import compare_pair

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
