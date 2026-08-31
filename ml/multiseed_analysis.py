"""Experiment 3 — multi-seed PPO budget replication analysis.

Pure functions over committed result JSON. Nothing here trains, evaluates,
or mutates checkpoints. Seed 0 is loaded from the Experiment 2 artifact
tree (``results/ppo_budget_v1/``) and is never overwritten.

    python -c "from ml.multiseed_analysis import main; raise SystemExit(main())"

The training seed — not an individual DEV game — is the replication unit
for claims about PPO stochasticity. Within-seed paired CIs use the 1000
identical DEV games; cross-seed summaries treat n=4 trajectories.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ml.action_categories import CATEGORIES
from ml.analyze_benchmark import compare_pair, load_result
from ml.seeds import (DEV_SEED_END, DEV_SEED_START, EVAL_SEED_END,
                      EVAL_SEED_START, check_training_range,
                      overlaps_dev_range, overlaps_eval_range,
                      ppo_episode_seed, validate_dev_range)

# --- frozen Experiment 2 constants (do not change) ---------------------------
WARM_START_PARAMETER_SHA256 = (
    "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b")
CORPUS_FINGERPRINT_SHA256 = (
    "2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e")
CORPUS_STATES = 4440
CORPUS_LOBBIES = 100
CORPUS_SEED_BASE = 10_590_000

PRIMARY_ITERS = (0, 40, 80, 160, 320)
EPISODES_PER_ITER = 16
TRAINING_ITERS = 320
TOTAL_EPISODES = TRAINING_ITERS * EPISODES_PER_ITER  # 5120
SHAPING_HORIZON = 40

DEV_EVAL_BASE = DEV_SEED_START          # 10_550_000
DEV_EVAL_GAMES = 1000
DEV_EVAL_LAST = DEV_EVAL_BASE + DEV_EVAL_GAMES - 1   # 10_550_999
MIXED_GAMES = 500
MIXED_FIELD = "greedy4_random3"

NEW_SEEDS = (1, 2, 3)
ALL_SEEDS = (0, 1, 2, 3)
SEED0_DIR = "results/ppo_budget_v1"
MULTI_DIR = "results/ppo_multiseed_v1"

# Pre-specified within-seed pairs (first − second; positive = first worse).
WITHIN_SEED_PAIRS: Tuple[Tuple[int, int], ...] = (
    (40, 0), (80, 0), (160, 0), (320, 0),
    (80, 40), (160, 40), (320, 40),
    (160, 80), (320, 80),
)

U_SHAPE_LABELS = (
    "U-like / transient improvement",
    "monotonic improvement",
    "monotonic degradation",
    "mostly flat/noisy",
    "other",
)


def episodes(iteration: int) -> int:
    return int(iteration) * EPISODES_PER_ITER


def seed_dir(seed: int) -> str:
    if seed == 0:
        return SEED0_DIR
    return os.path.join(MULTI_DIR, f"seed_{seed}")


def planned_ppo_span(training_seed: int,
                     iters: int = TRAINING_ITERS,
                     episodes_per_iter: int = EPISODES_PER_ITER) -> Tuple[int, int]:
    """Inclusive [first, last] BGEnv seeds for one PPO trajectory."""
    return (ppo_episode_seed(training_seed, 1),
            ppo_episode_seed(training_seed, iters * episodes_per_iter))


def training_seeds_isolated(seeds: Sequence[int] = ALL_SEEDS,
                            iters: int = TRAINING_ITERS,
                            episodes_per_iter: int = EPISODES_PER_ITER) -> List[Dict]:
    """Verify planned PPO episode seeds stay outside DEV and TEST."""
    rows = []
    for s in seeds:
        lo, hi = planned_ppo_span(s, iters, episodes_per_iter)
        overlap = check_training_range(f"ml.train_ppo seed={s}", lo, hi)
        rows.append({
            "training_seed": s, "lo": lo, "hi": hi,
            "overlaps_dev": overlaps_dev_range(lo, hi),
            "overlaps_test": overlaps_eval_range(lo, hi),
            "isolated": (not overlap and not overlaps_dev_range(lo, hi)
                         and not overlaps_eval_range(lo, hi)),
        })
    return rows


def assert_training_seeds_isolated(seeds: Sequence[int] = ALL_SEEDS) -> None:
    rows = training_seeds_isolated(seeds)
    bad = [r for r in rows if not r["isolated"]]
    if bad:
        raise ValueError(f"planned PPO seeds overlap reserved eval intervals: {bad}")


def assert_dev_eval_seeds() -> None:
    """The Experiment 2/3 DEV block is 10,550,000–10,550,999, inside DEV."""
    validate_dev_range(DEV_EVAL_BASE, DEV_EVAL_GAMES)
    assert DEV_EVAL_BASE == 10_550_000
    assert DEV_EVAL_LAST == 10_550_999
    assert DEV_SEED_START <= DEV_EVAL_BASE <= DEV_EVAL_LAST <= DEV_SEED_END
    assert not (EVAL_SEED_START <= DEV_EVAL_BASE <= EVAL_SEED_END)
    assert not (EVAL_SEED_START <= DEV_EVAL_LAST <= EVAL_SEED_END)


def ci_includes_zero(ci: Sequence[float]) -> bool:
    return ci[0] <= 0.0 <= ci[1]


def ci_excludes_zero(ci: Sequence[float]) -> bool:
    return not ci_includes_zero(ci)


def pair_key(it: int, ref: int) -> str:
    return f"iter{it}-iter{ref}"


def load_dev_result(directory: str, iteration: int,
                    field: str = "greedy") -> Dict:
    path = os.path.join(directory, "dev", f"iter{iteration:03d}_vs_{field}.json")
    return load_result(path)


def eval_seed_record(result: Mapping) -> Dict:
    return {
        "base_seed": result["base_seed"],
        "games": result["games"],
        "seed_range": list(result["seed_range"]),
        "evaluation_split": result.get("evaluation_split"),
        "field": result["field"],
    }


def assert_eval_seeds_match_experiment2(result: Mapping,
                                        field: str = "greedy") -> None:
    """Every checkpoint must reuse Experiment 2's DEV seed interval."""
    expected_games = DEV_EVAL_GAMES if field == "greedy" else MIXED_GAMES
    expected_last = DEV_EVAL_BASE + expected_games - 1
    if result.get("evaluation_split") != "dev":
        raise ValueError(f"evaluation_split must be 'dev', got "
                         f"{result.get('evaluation_split')!r}")
    if result["base_seed"] != DEV_EVAL_BASE:
        raise ValueError(f"DEV base seed {result['base_seed']} != {DEV_EVAL_BASE}")
    if result["games"] != expected_games:
        raise ValueError(f"games {result['games']} != {expected_games}")
    if list(result["seed_range"]) != [DEV_EVAL_BASE, expected_last]:
        raise ValueError(f"seed_range {result['seed_range']} != "
                         f"[{DEV_EVAL_BASE}, {expected_last}]")
    if result["field"] != field:
        raise ValueError(f"field {result['field']!r} != {field!r}")


def within_seed_paired(greedy_by_iter: Mapping[int, Mapping],
                       bootstrap_seed: int = 0) -> Dict[str, Dict]:
    """The nine pre-specified paired comparisons for one training seed."""
    out: Dict[str, Dict] = {}
    for it, ref in WITHIN_SEED_PAIRS:
        row = compare_pair(greedy_by_iter[it], greedy_by_iter[ref],
                           seed=bootstrap_seed)
        row = dict(row)
        row["iteration"] = it
        row["reference_iteration"] = ref
        row["ci_excludes_zero"] = ci_excludes_zero(row["ci95"])
        if row["mean_diff"] < 0 and row["ci_excludes_zero"]:
            row["reading"] = "first better (CI excludes 0)"
        elif row["mean_diff"] > 0 and row["ci_excludes_zero"]:
            row["reading"] = "first worse (CI excludes 0)"
        else:
            row["reading"] = "indistinguishable (CI includes 0)"
        out[pair_key(it, ref)] = row
    return out


def load_within_seed_paired(directory: str, bootstrap_seed: int = 0) -> Dict[str, Dict]:
    greedy = {it: load_dev_result(directory, it, "greedy")
              for it in PRIMARY_ITERS}
    for blob in greedy.values():
        assert_eval_seeds_match_experiment2(blob, "greedy")
    return within_seed_paired(greedy, bootstrap_seed=bootstrap_seed)


def _mean(xs: Sequence[float]) -> float:
    return float(st.mean(xs)) if xs else float("nan")


def _std(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return float(st.stdev(xs))  # sample std, n-1; n=4 is small


def summarize_numbers(xs: Sequence[float]) -> Dict:
    vals = [float(x) for x in xs]
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "median": float(st.median(vals)) if vals else float("nan"),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "std": _std(vals),
        "values": vals,
        "note": "n=4 training seeds; descriptive only, not a population CI",
    }


def placements_by_iter(directory: str) -> Dict[int, float]:
    return {it: float(load_dev_result(directory, it)["metrics"]["avg_placement"])
            for it in PRIMARY_ITERS}


def classify_ushape(placements: Mapping[int, float],
                    paired: Optional[Mapping[str, Mapping]] = None) -> Dict:
    """Documented U-shape / trajectory-shape rule.

    Inputs are greedy mean placements (lower is better) at the five primary
    iterations, optionally with the pre-specified paired bootstrap rows.

    Mid-budget is {80, 160} (1,280 and 2,560 episodes). Late is iteration 320
    (5,120 episodes).

    1. U-like / transient improvement
       mid-budget best mean < iter0 mean  AND  iter320 mean > mid-budget best.
    2. monotonic improvement
       the five-point sequence is non-increasing and iter320 < iter0,
       and the curve is not U-like.
    3. monotonic degradation
       the five-point sequence is non-decreasing and iter320 > iter0,
       and the curve is not U-like.
    4. mostly flat/noisy
       every paired CI vs iter0 includes zero (or, if no paired rows, every
       |mean − iter0| < 0.05).
    5. other
       none of the above.

    The rule uses means for the shape label so a single noisy CI cannot
    flip the qualitative class. CI flags are recorded alongside:
    ``significant_mid_gain`` (a mid-budget − iter0 CI is entirely < 0) and
    ``significant_late_regression`` (iter320 − mid-best CI is entirely > 0).
    """
    p = {int(k): float(v) for k, v in placements.items()}
    missing = [it for it in PRIMARY_ITERS if it not in p]
    if missing:
        raise ValueError(f"classify_ushape missing iterations {missing}")

    mid_iters = (80, 160)
    mid_best_it = min(mid_iters, key=lambda it: (p[it], it))
    mid_gain = p[mid_best_it] < p[0]
    late_reg = p[320] > p[mid_best_it]
    seq = [p[it] for it in PRIMARY_ITERS]
    mono_imp = (all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
                and p[320] < p[0])
    mono_deg = (all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
                and p[320] > p[0])

    sig_mid = False
    sig_late = False
    vs0_all_zero = None
    if paired is not None:
        sig_mid = any(
            paired[pair_key(it, 0)]["mean_diff"] < 0
            and paired[pair_key(it, 0)]["ci_excludes_zero"]
            for it in mid_iters
            if pair_key(it, 0) in paired)
        late_key = pair_key(320, mid_best_it)
        if late_key in paired:
            sig_late = (paired[late_key]["mean_diff"] > 0
                        and paired[late_key]["ci_excludes_zero"])
        vs0_keys = [pair_key(it, 0) for it in PRIMARY_ITERS if it]
        vs0_all_zero = all(k in paired and not paired[k]["ci_excludes_zero"]
                           for k in vs0_keys)
    else:
        vs0_all_zero = all(abs(p[it] - p[0]) < 0.05 for it in PRIMARY_ITERS if it)

    if mid_gain and late_reg:
        label = "U-like / transient improvement"
    elif mono_imp:
        label = "monotonic improvement"
    elif mono_deg:
        label = "monotonic degradation"
    elif vs0_all_zero:
        label = "mostly flat/noisy"
    else:
        label = "other"

    return {
        "label": label,
        "mid_best_iteration": mid_best_it,
        "mid_best_episodes": episodes(mid_best_it),
        "mid_gain_mean": mid_gain,
        "late_regression_mean": late_reg,
        "significant_mid_gain": sig_mid,
        "significant_late_regression": sig_late,
        "monotonic_improvement_sequence": mono_imp,
        "monotonic_degradation_sequence": mono_deg,
        "vs_iter0_all_cis_include_zero": vs0_all_zero,
        "placements": {str(it): p[it] for it in PRIMARY_ITERS},
        "rule": ("U-like iff min(p80,p160) < p0 AND p320 > that mid-best; "
                 "else monotonic if the five-point sequence is monotone and "
                 "the endpoint differs from iter0; else flat if all vs-iter0 "
                 "CIs include 0 (or |Δ|<0.05 without CIs); else other."),
    }


def freeze_count(confusion_matrix: Mapping[str, Mapping[str, int]]) -> int:
    """How many diagnostic states the compared policy assigned to freeze."""
    return int(sum(row.get("freeze", 0) for row in confusion_matrix.values()))


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_seed_curve(directory: str) -> Dict:
    return load_json(os.path.join(directory, "learning_curve.json"))


def load_seed_drift(directory: str) -> Dict:
    return load_json(os.path.join(directory, "policy_drift.json"))


def load_seed_categories(directory: str) -> Dict:
    return load_json(os.path.join(directory, "action_category_drift.json"))


def assert_warmstart_hash(parameter_sha256: str) -> None:
    if parameter_sha256 != WARM_START_PARAMETER_SHA256:
        raise ValueError(
            f"warm-start parameter_sha256 {parameter_sha256} != "
            f"frozen Experiment 2 hash {WARM_START_PARAMETER_SHA256}")


def assert_corpus_fingerprint(fingerprint: str) -> None:
    if fingerprint != CORPUS_FINGERPRINT_SHA256:
        raise ValueError(
            f"corpus fingerprint {fingerprint} != "
            f"frozen historical {CORPUS_FINGERPRINT_SHA256}")


def load_seed_bundle(seed: int) -> Dict:
    """Load one training seed's committed artifacts (seed 0 = Experiment 2)."""
    directory = seed_dir(seed)
    curve = load_seed_curve(directory)
    drift = load_seed_drift(directory)
    cats = load_seed_categories(directory)
    paired = load_within_seed_paired(directory)
    greedy = {it: load_dev_result(directory, it, "greedy")
              for it in PRIMARY_ITERS}
    mixed = {it: load_dev_result(directory, it, MIXED_FIELD)
             for it in PRIMARY_ITERS}
    for blob in greedy.values():
        assert_eval_seeds_match_experiment2(blob, "greedy")
    for blob in mixed.values():
        assert_eval_seeds_match_experiment2(blob, MIXED_FIELD)
    assert_corpus_fingerprint(drift["corpus"]["fingerprint_sha256"])
    if drift["corpus"]["states"] != CORPUS_STATES:
        raise ValueError(f"seed {seed} corpus states {drift['corpus']['states']} "
                         f"!= {CORPUS_STATES}")
    iter0 = next(c for c in curve["curve"] if c["iteration"] == 0)
    assert_warmstart_hash(iter0["parameter_sha256"])
    placements = {c["iteration"]: c["greedy_avg"] for c in curve["curve"]}
    shape = classify_ushape(placements, paired)
    return {
        "training_seed": seed,
        "source_dir": directory,
        "source": "experiment_2" if seed == 0 else "experiment_3",
        "curve": curve,
        "drift": drift,
        "categories": cats,
        "paired": paired,
        "greedy": greedy,
        "mixed": mixed,
        "placements": placements,
        "shape": shape,
    }


def cross_seed_table(bundles: Mapping[int, Mapping]) -> Dict:
    """iter × seed greedy averages, then mean/median/min/max/std across seeds."""
    by_iter = {}
    for it in PRIMARY_ITERS:
        vals = {s: float(bundles[s]["placements"][it]) for s in sorted(bundles)}
        by_iter[str(it)] = {
            "iteration": it,
            "cumulative_episodes": episodes(it),
            "per_seed": vals,
            **summarize_numbers([vals[s] for s in sorted(vals)]),
        }
    return {
        "replication_unit": "training seed (n=4); not 4000 independent games",
        "by_iteration": by_iter,
    }


def question_a(bundles: Mapping[int, Mapping]) -> Dict:
    """Does ~1,280-episode improvement reproduce? iter80 − iter0 per seed."""
    rows = []
    for s in sorted(bundles):
        pair = bundles[s]["paired"][pair_key(80, 0)]
        if pair["mean_diff"] < 0:
            direction = "improve"
        elif pair["mean_diff"] > 0:
            direction = "worsen"
        else:
            direction = "tie"
        rows.append({
            "training_seed": s,
            "mean_diff": pair["mean_diff"],
            "ci95": pair["ci95"],
            "ci_excludes_zero": pair["ci_excludes_zero"],
            "direction": direction,
        })
    n_improve = sum(r["direction"] == "improve" for r in rows)
    n_worsen = sum(r["direction"] == "worsen" for r in rows)
    n_sig_improve = sum(r["direction"] == "improve" and r["ci_excludes_zero"]
                        for r in rows)
    n_sig_worsen = sum(r["direction"] == "worsen" and r["ci_excludes_zero"]
                       for r in rows)
    return {
        "comparison": "iter80 - iter0 (positive = iter80 worse)",
        "per_seed": rows,
        "n_improve": n_improve,
        "n_worsen": n_worsen,
        "n_ci_excludes_zero_improve": n_sig_improve,
        "n_ci_excludes_zero_worsen": n_sig_worsen,
        "n_seeds": len(rows),
    }


def question_b(bundles: Mapping[int, Mapping]) -> Dict:
    """Does performance decay after the transient? iter320 − iter80 per seed."""
    rows = []
    for s in sorted(bundles):
        pair = bundles[s]["paired"][pair_key(320, 80)]
        if pair["mean_diff"] > 0:
            direction = "regress"
        elif pair["mean_diff"] < 0:
            direction = "continue_improving"
        else:
            direction = "tie"
        if not pair["ci_excludes_zero"]:
            direction = "indistinguishable"
        rows.append({
            "training_seed": s,
            "mean_diff": pair["mean_diff"],
            "ci95": pair["ci95"],
            "ci_excludes_zero": pair["ci_excludes_zero"],
            "direction": direction,
        })
    return {
        "comparison": "iter320 - iter80 (positive = iter320 worse)",
        "per_seed": rows,
        "n_regress": sum(r["direction"] == "regress" for r in rows),
        "n_continue_improving": sum(r["direction"] == "continue_improving"
                                    for r in rows),
        "n_indistinguishable": sum(r["direction"] == "indistinguishable"
                                   for r in rows),
        "n_seeds": len(rows),
    }


def _curve_row(bundle: Mapping, iteration: int) -> Dict:
    return next(c for c in bundle["curve"]["curve"] if c["iteration"] == iteration)


def drift_replication(bundles: Mapping[int, Mapping]) -> Dict:
    """Iter-320 drift plus whether the seed's best checkpoint is closer to
    the expert / warm start than later checkpoints (descriptive only)."""
    at_320 = []
    best_vs_later = []
    for s in sorted(bundles):
        rows = {c["iteration"]: c for c in bundles[s]["curve"]["curve"]}
        r320 = rows[320]
        at_320.append({
            "training_seed": s,
            "expert_agreement": r320["expert_agreement"],
            "warmstart_agreement": r320["warmstart_agreement"],
            "kl_from_warmstart": r320["kl_from_warmstart"],
            "corpus_entropy": r320["corpus_entropy"],
        })
        # Best = lowest greedy mean among primary checkpoints.
        best_it = min(PRIMARY_ITERS, key=lambda it: rows[it]["greedy_avg"])
        later = [it for it in PRIMARY_ITERS if it > best_it]
        if later:
            later_it = later[-1]  # the last later checkpoint (320 if best < 320)
            best_vs_later.append({
                "training_seed": s,
                "best_iteration": best_it,
                "later_iteration": later_it,
                "best_expert_agreement": rows[best_it]["expert_agreement"],
                "later_expert_agreement": rows[later_it]["expert_agreement"],
                "best_kl": rows[best_it]["kl_from_warmstart"],
                "later_kl": rows[later_it]["kl_from_warmstart"],
                "best_has_higher_expert_agreement":
                    rows[best_it]["expert_agreement"]
                    > rows[later_it]["expert_agreement"],
                "best_has_lower_kl":
                    rows[best_it]["kl_from_warmstart"]
                    < rows[later_it]["kl_from_warmstart"],
            })
    expert = [r["expert_agreement"] for r in at_320]
    warm = [r["warmstart_agreement"] for r in at_320]
    kl = [r["kl_from_warmstart"] for r in at_320]
    # Pearson correlation across seeds: best-minus-later expert vs nothing
    # — just the descriptive counts requested.
    return {
        "iter320": {
            "per_seed": at_320,
            "expert_agreement": summarize_numbers(expert),
            "warmstart_agreement": summarize_numbers(warm),
            "kl_from_warmstart": summarize_numbers(kl),
            "entropy": summarize_numbers([r["corpus_entropy"] for r in at_320]),
        },
        "best_checkpoint_vs_later": best_vs_later,
        "n_best_higher_expert_than_later": sum(
            r["best_has_higher_expert_agreement"] for r in best_vs_later),
        "n_best_lower_kl_than_later": sum(
            r["best_has_lower_kl"] for r in best_vs_later),
        "note": "descriptive only; no causal claim",
    }


def category_replication(bundles: Mapping[int, Mapping]) -> Dict:
    """Action-category drift at iter320 across training seeds."""
    per_seed = []
    for s in sorted(bundles):
        ckpts = {c["checkpoint"]: c for c in bundles[s]["categories"]["checkpoints"]}
        row = ckpts["iter_320.pt"]
        vs_e = row["vs_expert"]
        freeze = freeze_count(vs_e["confusion_matrix"])
        per_seed.append({
            "training_seed": s,
            "expert_agreement": vs_e["overall_agreement"],
            "n_disagreements": vs_e["n_disagreements"],
            "disagreement_share_by_category":
                vs_e["disagreement_share_by_category"],
            "contribution_to_total_drift":
                vs_e["contribution_to_total_drift"],
            "top_transitions": vs_e["top_transitions"],
            "freeze_count": freeze,
            "freeze_rate": freeze / vs_e["n_states"],
        })
    freeze_counts = [r["freeze_count"] for r in per_seed]
    tempo = {}
    for cat in ("roll", "end", "play", "freeze"):
        tempo[cat] = summarize_numbers(
            [r["disagreement_share_by_category"].get(cat) or 0.0
             for r in per_seed])
    return {
        "per_seed": per_seed,
        "freeze": summarize_numbers(freeze_counts),
        "tempo_disagreement_share": tempo,
        "categories": list(CATEGORIES),
        "note": "same action-category mapping as Experiment 2; not redefined",
    }


def rl_block_means(diag: Sequence[Mapping]) -> Dict[str, Dict]:
    def block(rows):
        keys = ("adv_mean", "adv_std", "adv_mean_abs", "adv_frac_positive",
                "adv_frac_negative", "adv_frac_zero", "return_mean",
                "return_std", "value_pred_mean", "value_pred_std",
                "value_explained_variance", "placement_mean", "placement_std",
                "shaping_reward_sum", "terminal_reward_sum", "entropy",
                "approx_kl", "clip_frac", "grad_norm", "pi_loss", "v_loss",
                "rollout_avg_placement")
        out = {}
        for k in keys:
            xs = [r[k] for r in rows if r.get(k) is not None]
            out[k] = _mean(xs) if xs else None
        return out

    return {
        "iters_1_40": block([r for r in diag if r["iter"] <= 40]),
        "iters_41_160": block([r for r in diag if 40 < r["iter"] <= 160]),
        "iters_161_320": block([r for r in diag if r["iter"] > 160]),
    }


def load_rl_signal(directory: str) -> Dict:
    path = os.path.join(directory, "rl_signal.json")
    if os.path.isfile(path):
        blob = load_json(path)
        if "blocks" in blob and "per_iteration" in blob:
            return blob
    diag_path = os.path.join(directory, "train_diag.jsonl")
    diag = [json.loads(l) for l in open(diag_path)]
    return {"per_iteration": diag, "blocks": rl_block_means(diag)}


def rl_comparison(bundles: Mapping[int, Mapping]) -> Dict:
    """Internal PPO metrics vs whether the trajectory improved at iter80."""
    rows = []
    for s in sorted(bundles):
        rl = load_rl_signal(bundles[s]["source_dir"])
        q = bundles[s]["paired"][pair_key(80, 0)]
        rows.append({
            "training_seed": s,
            "iter80_minus_iter0": q["mean_diff"],
            "improved_at_80": q["mean_diff"] < 0,
            "blocks": rl["blocks"],
        })
    improved = [r for r in rows if r["improved_at_80"]]
    not_imp = [r for r in rows if not r["improved_at_80"]]

    def _avg_block(group, block, key):
        xs = [r["blocks"][block][key] for r in group
              if r["blocks"][block].get(key) is not None]
        return _mean(xs) if xs else None

    keys = ("entropy", "approx_kl", "clip_frac", "value_explained_variance",
            "adv_mean_abs", "grad_norm")
    contrast = {}
    for block in ("iters_1_40", "iters_41_160", "iters_161_320"):
        contrast[block] = {
            k: {
                "improved": _avg_block(improved, block, k),
                "not_improved": _avg_block(not_imp, block, k),
            } for k in keys
        }
    return {
        "per_seed": rows,
        "n_improved_at_80": len(improved),
        "n_not_improved_at_80": len(not_imp),
        "block_contrast_improved_vs_not": contrast,
        "note": "descriptive only; diagnostics were not used to alter training",
    }


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return None
    return num / (denx * deny)


def assemble_replication(bundles: Mapping[int, Mapping]) -> Dict:
    table = cross_seed_table(bundles)
    qa = question_a(bundles)
    qb = question_b(bundles)
    shapes = {s: bundles[s]["shape"] for s in sorted(bundles)}
    n_transient = sum(
        shapes[s]["label"] == "U-like / transient improvement" for s in shapes)
    drift = drift_replication(bundles)
    cats = category_replication(bundles)
    rl = rl_comparison(bundles)
    # Descriptive correlation: iter80−iter0 vs iter320 expert agreement.
    x = [bundles[s]["paired"][pair_key(80, 0)]["mean_diff"]
         for s in sorted(bundles)]
    y = [_curve_row(bundles[s], 320)["expert_agreement"]
         for s in sorted(bundles)]
    return {
        "experiment": "Replay Experiment 3 — Multi-Seed PPO Budget Replication",
        "evaluation_split": "dev",
        "replication_unit": "training seed",
        "n_training_seeds": len(bundles),
        "n_training_seeds_note": "n=4 is small; no strong population claims",
        "warm_start_parameter_sha256": WARM_START_PARAMETER_SHA256,
        "corpus_fingerprint_sha256": CORPUS_FINGERPRINT_SHA256,
        "dev_eval_seeds": [DEV_EVAL_BASE, DEV_EVAL_LAST],
        "cross_seed_summary": table,
        "question_a_1280_episode_replication": qa,
        "question_b_late_regression": qb,
        "ushape": {
            "per_seed": {str(s): shapes[s] for s in shapes},
            "n_transient_improvement": n_transient,
            "n_trajectories": len(shapes),
            "statement": (f"{n_transient} / {len(shapes)} trajectories show "
                          "transient improvement followed by regression"),
        },
        "drift_replication": drift,
        "category_replication": cats,
        "rl_signal_comparison": rl,
        "exploratory_corr_iter80_gain_vs_iter320_expert": pearson(x, y),
        "seed0_source": SEED0_DIR,
    }


def outcome_and_recommendation(analysis: Mapping) -> Dict:
    """Pick the single best-supported outcome and one Experiment 4 rec."""
    qa = analysis["question_a_1280_episode_replication"]
    qb = analysis["question_b_late_regression"]
    n_trans = analysis["ushape"]["n_transient_improvement"]
    n = analysis["n_training_seeds"]
    n_new_improve = sum(
        r["training_seed"] != 0 and r["direction"] == "improve"
        for r in qa["per_seed"])
    n_new = sum(r["training_seed"] != 0 for r in qa["per_seed"])
    labels = [analysis["ushape"]["per_seed"][str(s)]["label"]
              for s in range(n)]
    distinct = set(labels)

    if n_trans >= 3:
        outcome = "A"
        outcome_text = ("Outcome A — transient improvement replicates: most "
                        "independent trajectories improve around 1,280–2,560 "
                        "episodes and later regress.")
        rec = ("Recommend Experiment 4 — PPO policy anchoring "
               "(current PPO vs PPO + KL penalty toward BC prior). "
               "Hypothesis: PPO contains useful improvement signal, but "
               "unconstrained optimization eventually drifts too far from "
               "the useful imitation prior.")
    elif n_new_improve == 0:
        outcome = "B"
        outcome_text = ("Outcome B — seed 0 was a lucky excursion: most new "
                        "trajectories do not reproduce the improvement.")
        rec = ("Recommend revisiting whether seed 0's iteration-80 gain was "
               "stochastic rather than algorithmic. Do not build an "
               "intervention around a non-replicated result.")
    elif len(distinct) >= 3 or n_trans <= 1:
        outcome = "C"
        outcome_text = ("Outcome C — PPO trajectories are highly variable: "
                        "different training seeds produce qualitatively "
                        "different curves.")
        rec = ("Recommend Experiment 4 — PPO stability / rollout variance "
               "study (rollout batch size, training variance, opponent "
               "sampling variance). Do NOT jump to KL anchoring if drift "
               "is not consistent.")
    else:
        outcome = "D"
        outcome_text = ("Outcome D — another clear pattern (see analysis).")
        rec = ("Recommend Experiment 4 — PPO stability / rollout variance "
               "study (rollout batch size, training variance, opponent "
               "sampling variance). Do NOT jump to KL anchoring if drift "
               "is not consistent.")

    # If A-like but new seeds never improved, prefer B over A.
    if outcome == "A" and n_new_improve == 0:
        outcome = "B"
        outcome_text = ("Outcome B — seed 0 was a lucky excursion: most new "
                        "trajectories do not reproduce the improvement.")
        rec = ("Recommend revisiting whether seed 0's iteration-80 gain was "
               "stochastic rather than algorithmic. Do not build an "
               "intervention around a non-replicated result.")

    return {
        "outcome": outcome,
        "outcome_text": outcome_text,
        "recommendation": rec,
        "n_transient": n_trans,
        "n_new_seeds_iter80_improve": n_new_improve,
        "n_new_seeds": n_new,
        "n_iter320_regress_vs_iter80": qb["n_regress"],
        "shape_labels": labels,
    }


def write_json(path: str, blob) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
        f.write("\n")
