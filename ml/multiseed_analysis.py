"""Cross-seed aggregation for Experiment 3 (multi-seed PPO budget replication).

Pure functions over committed per-seed JSON — no training, no DEV games.
Training seed is the replication unit; within-seed paired comparisons reuse
``ml.analyze_benchmark.compare_pair`` over the identical 1000 DEV seeds.
"""

from __future__ import annotations

import json
import math
import statistics as st
from typing import Dict, List, Optional, Sequence, Tuple

from ml.analyze_benchmark import compare_pair, load_result

PRIMARY_ITERS = [0, 40, 80, 160, 320]
EPISODES_PER_ITER = 16
# Within-seed paired contrasts (first − second); positive = first worse.
PAIRED_CONTRASTS = [
    (40, 0), (80, 0), (160, 0), (320, 0),
    (80, 40), (160, 40), (320, 40),
    (160, 80), (320, 80),
]

EXP2_WARM_START_PARAMETER_SHA256 = (
    "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b"
)
EXP2_CORPUS_FINGERPRINT_PREFIX = "2ec217b353bd"


def episodes(it: int) -> int:
    return it * EPISODES_PER_ITER


def load_seed_greedy(seed_dir: str) -> Dict[int, dict]:
    return {it: load_result(f"{seed_dir}/dev/iter{it:03d}_vs_greedy.json")
            for it in PRIMARY_ITERS}


def load_seed_mixed(seed_dir: str) -> Dict[int, dict]:
    return {it: load_result(f"{seed_dir}/dev/iter{it:03d}_vs_greedy4_random3.json")
            for it in PRIMARY_ITERS}


def within_seed_paired(greedy: Dict[int, dict],
                       bootstrap_seed: int = 0) -> List[dict]:
    """Paired bootstrap for each pre-specified contrast on one training seed."""
    rows = []
    for a, b in PAIRED_CONTRASTS:
        row = compare_pair(greedy[a], greedy[b], seed=bootstrap_seed)
        row["iteration"] = a
        row["reference_iteration"] = b
        row["label"] = f"iter{a}-iter{b}"
        rows.append(row)
    return rows


def ci_excludes_zero(ci: Sequence[float]) -> bool:
    return not (ci[0] <= 0 <= ci[1])


def classify_u_shape(paired_rows: Sequence[dict]) -> Dict:
    """Documented rule for the Exp2-style U / transient-improvement shape.

    Looks at the pre-specified contrasts:
      * improve_mid: iter80−iter0  (negative = better at mid budget)
      * regress_late: iter320−iter80 (positive = worse at long budget)
      * mid_vs_early: iter80−iter40
      * late_vs_0: iter320−iter0

    Classes:
      - ``u_like_transient_improvement``: mid improves vs iter0 (CI excludes 0)
        AND late regresses vs mid (CI excludes 0), matching Exp2 seed 0.
      - ``monotonic_improvement``: later budgets keep beating earlier ones
        without a late regression vs mid.
      - ``monotonic_degradation``: later budgets keep losing to earlier ones.
      - ``mostly_flat_noisy``: no contrast vs iter0 excludes zero.
      - ``other``: anything else.
    """
    by = {(r["iteration"], r["reference_iteration"]): r for r in paired_rows}
    mid = by[(80, 0)]
    late_vs_mid = by[(320, 80)]
    late_vs_0 = by[(320, 0)]
    early = by[(40, 0)]

    mid_better = mid["mean_diff"] < 0 and ci_excludes_zero(mid["ci95"])
    mid_worse = mid["mean_diff"] > 0 and ci_excludes_zero(mid["ci95"])
    late_regresses = (late_vs_mid["mean_diff"] > 0
                      and ci_excludes_zero(late_vs_mid["ci95"]))
    late_improves_vs_mid = (late_vs_mid["mean_diff"] < 0
                            and ci_excludes_zero(late_vs_mid["ci95"]))
    any_vs0_clear = any(
        ci_excludes_zero(by[(it, 0)]["ci95"]) for it in (40, 80, 160, 320))

    if mid_better and late_regresses:
        cls = "u_like_transient_improvement"
    elif mid_better and late_improves_vs_mid:
        cls = "monotonic_improvement"
    elif (mid_worse or (early["mean_diff"] > 0
                        and ci_excludes_zero(early["ci95"]))) and not mid_better:
        # progressive worsening without a clear mid recovery
        if (late_vs_0["mean_diff"] > 0 and ci_excludes_zero(late_vs_0["ci95"])
                and early["mean_diff"] > 0
                and ci_excludes_zero(early["ci95"])):
            cls = "monotonic_degradation"
        elif not any_vs0_clear:
            cls = "mostly_flat_noisy"
        else:
            cls = "other"
    elif not any_vs0_clear:
        cls = "mostly_flat_noisy"
    else:
        cls = "other"

    return {
        "class": cls,
        "rule": ("u_like iff iter80−iter0 improves (CI excl 0) AND "
                 "iter320−iter80 regresses (CI excl 0); else monotonic "
                 "improvement / degradation / mostly_flat_noisy / other"),
        "iter80_minus_iter0": {
            "mean_diff": mid["mean_diff"], "ci95": mid["ci95"],
            "ci_excludes_zero": ci_excludes_zero(mid["ci95"]),
        },
        "iter320_minus_iter80": {
            "mean_diff": late_vs_mid["mean_diff"], "ci95": late_vs_mid["ci95"],
            "ci_excludes_zero": ci_excludes_zero(late_vs_mid["ci95"]),
        },
    }


def curve_point(greedy: dict, mixed: dict, drift_row: dict,
                cat_row: Optional[dict], it: int,
                training_seed: int) -> dict:
    g, m = greedy["metrics"], mixed["metrics"]
    ce = (cat_row or {}).get("vs_expert", {})
    return {
        "training_seed": training_seed,
        "iteration": it,
        "cumulative_episodes": episodes(it),
        "greedy_avg": g["avg_placement"],
        "greedy_ci95": greedy["avg_placement_ci95"],
        "greedy_median": g["median_placement"],
        "greedy_std": g.get("std_placement", g.get("placement_std")),
        "greedy_top4": g["top4_rate"],
        "greedy_win": g["win_rate"],
        "greedy_placement_counts": g["placement_counts"],
        "mixed_avg": m["avg_placement"],
        "mixed_ci95": mixed["avg_placement_ci95"],
        "mixed_top4": m["top4_rate"],
        "mixed_win": m["win_rate"],
        "expert_agreement": drift_row["expert_agreement"],
        "warmstart_agreement": drift_row["warmstart_agreement"],
        "kl_from_warmstart": drift_row["kl_from_warmstart_mean"],
        "corpus_entropy": drift_row["entropy_mean"],
        "value_mean": drift_row["value_mean"],
        "value_std": drift_row["value_std"],
        "parameter_sha256": drift_row["parameter_sha256"],
        "checkpoint_sha256": drift_row["checkpoint_sha256"],
        "expert_disagreement_by_category":
            ce.get("disagreement_share_by_category"),
        "drift_contribution_by_category":
            ce.get("contribution_to_total_drift"),
    }


def summarize_across_seeds(per_seed_curves: Dict[int, List[dict]]) -> dict:
    """Per-budget mean/median/min/max/std across training seeds; keep individuals."""
    by_iter = {}
    for it in PRIMARY_ITERS:
        vals = []
        for seed, curve in sorted(per_seed_curves.items()):
            pt = next(c for c in curve if c["iteration"] == it)
            vals.append({"training_seed": seed, "greedy_avg": pt["greedy_avg"],
                         "expert_agreement": pt["expert_agreement"],
                         "warmstart_agreement": pt["warmstart_agreement"],
                         "kl_from_warmstart": pt["kl_from_warmstart"],
                         "mixed_avg": pt["mixed_avg"]})
        avgs = [v["greedy_avg"] for v in vals]
        by_iter[it] = {
            "iteration": it,
            "cumulative_episodes": episodes(it),
            "n_seeds": len(avgs),
            "greedy_avg_mean": st.mean(avgs),
            "greedy_avg_median": st.median(avgs),
            "greedy_avg_min": min(avgs),
            "greedy_avg_max": max(avgs),
            "greedy_avg_std": st.pstdev(avgs) if len(avgs) > 1 else 0.0,
            "per_seed": vals,
            "caution": "n=4 training seeds is exploratory only",
        }
    return by_iter


def replication_questions(per_seed_paired: Dict[int, List[dict]]) -> dict:
    """Question A (iter80−iter0) and Question B (iter320−iter80)."""
    a_improve, a_worsen, a_clear = [], [], []
    b_regress, b_improve, b_clear = [], [], []
    for seed, rows in sorted(per_seed_paired.items()):
        by = {(r["iteration"], r["reference_iteration"]): r for r in rows}
        mid = by[(80, 0)]
        late = by[(320, 80)]
        if mid["mean_diff"] < 0:
            a_improve.append(seed)
        elif mid["mean_diff"] > 0:
            a_worsen.append(seed)
        if ci_excludes_zero(mid["ci95"]):
            a_clear.append(seed)
        if late["mean_diff"] > 0:
            b_regress.append(seed)
        elif late["mean_diff"] < 0:
            b_improve.append(seed)
        if ci_excludes_zero(late["ci95"]):
            b_clear.append(seed)
    return {
        "question_A_iter80_minus_iter0": {
            "n_seeds": len(per_seed_paired),
            "seeds_improve": a_improve,
            "seeds_worsen": a_worsen,
            "seeds_ci_excludes_zero": a_clear,
            "n_improve": len(a_improve),
            "n_worsen": len(a_worsen),
            "n_ci_excludes_zero": len(a_clear),
        },
        "question_B_iter320_minus_iter80": {
            "n_seeds": len(per_seed_paired),
            "seeds_regress": b_regress,
            "seeds_continue_improve": b_improve,
            "seeds_ci_excludes_zero": b_clear,
            "n_regress": len(b_regress),
            "n_continue_improve": len(b_improve),
            "n_ci_excludes_zero": len(b_clear),
        },
        "inferential_caution": (
            "n=4 training seeds is exploratory only — do not treat counts "
            "as confirmatory frequency estimates"),
    }


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
