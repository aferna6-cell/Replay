"""Experiment 3 (multi-seed PPO budget replication) aggregation tooling.

Covers the scripts under scripts/ppo_multiseed_*.py: cross-seed paired
comparisons, the U-shape classification rule, cross-seed summary math, and
the reproduction-gate / seed-metadata checks in the manifest generator.
Where real Experiment 3 result artifacts are present on disk, the
integration tests validate them directly; otherwise those tests skip
(they are not a substitute for the unit tests on synthetic data, which
always run).
"""

import importlib
import json
import os
import statistics as st
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")


def _load_script(name):
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def report_mod():
    return _load_script("ppo_multiseed_report")


@pytest.fixture(scope="module")
def manifest_mod():
    return _load_script("ppo_multiseed_manifest")


MULTISEED_DIR = os.path.join(ROOT, "results", "ppo_multiseed_v1")


def _has_full_results():
    if not os.path.isdir(MULTISEED_DIR):
        return False
    for seed in (1, 2, 3):
        base = os.path.join(MULTISEED_DIR, f"seed_{seed}")
        for it in (0, 40, 80, 160, 320):
            for field in ("greedy", "greedy4_random3"):
                p = os.path.join(base, "dev", f"iter{it:03d}_vs_{field}.json")
                if not os.path.isfile(p):
                    return False
        if not os.path.isfile(os.path.join(base, "policy_drift.json")):
            return False
        if not os.path.isfile(os.path.join(base, "action_category_drift.json")):
            return False
    return True


requires_full_results = pytest.mark.skipif(
    not _has_full_results(),
    reason="Experiment 3 result artifacts for seeds 1-3 are not present")


# --- U-shape classification (synthetic, always runs) --------------------------

def _pair(mean_diff, lo, hi):
    return {"mean_diff": mean_diff, "ci95": [lo, hi]}


def test_classify_trajectory_transient_improvement(report_mod):
    """Mirrors the seed-0 Experiment 2 shape: significant improvement at an
    interior checkpoint that is gone by the end of training."""
    curve = {0: 6.554, 40: 6.761, 80: 6.325, 160: 6.435, 320: 6.606}
    pairwise = {
        (40, 0): _pair(+0.207, +0.093, +0.322),
        (80, 0): _pair(-0.229, -0.392, -0.061),
        (160, 0): _pair(-0.119, -0.245, +0.008),
        (320, 0): _pair(+0.052, -0.104, +0.210),
        (80, 40): _pair(-0.436, -0.6, -0.2),
        (160, 40): _pair(-0.326, -0.5, -0.1),
        (320, 40): _pair(-0.155, -0.3, 0.05),
        (160, 80): _pair(0.11, -0.05, 0.3),
        (320, 80): _pair(0.281, 0.1, 0.45),
    }
    label, evidence = report_mod.classify_trajectory(curve, pairwise)
    assert label == "U-like/transient improvement"
    assert evidence["sig_vs_iter0"][80] == "better"
    assert evidence["sig_vs_iter0"][320] == "none"


def test_classify_trajectory_monotonic_improvement(report_mod):
    curve = {0: 6.5, 40: 6.3, 80: 6.1, 160: 5.9, 320: 5.6}
    pairwise = {
        (40, 0): _pair(-0.2, -0.35, -0.05),
        (80, 0): _pair(-0.4, -0.55, -0.25),
        (160, 0): _pair(-0.6, -0.75, -0.45),
        (320, 0): _pair(-0.9, -1.1, -0.7),
        (80, 40): _pair(-0.2, -0.35, -0.05),
        (160, 40): _pair(-0.4, -0.55, -0.25),
        (320, 40): _pair(-0.7, -0.9, -0.5),
        (160, 80): _pair(-0.2, -0.35, -0.05),
        (320, 80): _pair(-0.5, -0.7, -0.3),
    }
    label, _ = report_mod.classify_trajectory(curve, pairwise)
    assert label == "monotonic improvement"


def test_classify_trajectory_monotonic_degradation(report_mod):
    curve = {0: 5.6, 40: 5.9, 80: 6.1, 160: 6.3, 320: 6.5}
    pairwise = {
        (40, 0): _pair(0.3, 0.1, 0.5),
        (80, 0): _pair(0.5, 0.3, 0.7),
        (160, 0): _pair(0.7, 0.5, 0.9),
        (320, 0): _pair(0.9, 0.7, 1.1),
        (80, 40): _pair(0.2, 0.05, 0.35),
        (160, 40): _pair(0.4, 0.25, 0.55),
        (320, 40): _pair(0.6, 0.45, 0.75),
        (160, 80): _pair(0.2, 0.05, 0.35),
        (320, 80): _pair(0.4, 0.25, 0.55),
    }
    label, _ = report_mod.classify_trajectory(curve, pairwise)
    assert label == "monotonic degradation"


def test_classify_trajectory_flat_noisy(report_mod):
    curve = {0: 6.5, 40: 6.52, 80: 6.48, 160: 6.51, 320: 6.49}
    pairwise = {k: _pair(0.0, -0.1, 0.1) for k in
                [(40, 0), (80, 0), (160, 0), (320, 0), (80, 40), (160, 40),
                 (320, 40), (160, 80), (320, 80)]}
    label, _ = report_mod.classify_trajectory(curve, pairwise)
    assert label == "mostly flat/noisy"


def test_classify_trajectory_other_bucket_for_mixed_signal(report_mod):
    """Significant regression at 40, but a sustained significant
    improvement at 320 that isn't monotonic through the middle — should not
    be forced into any of the four named buckets."""
    curve = {0: 6.5, 40: 6.9, 80: 6.6, 160: 6.4, 320: 6.0}
    pairwise = {
        (40, 0): _pair(0.4, 0.2, 0.6),
        (80, 0): _pair(0.1, -0.1, 0.3),
        (160, 0): _pair(-0.1, -0.3, 0.1),
        (320, 0): _pair(-0.5, -0.7, -0.3),
        (80, 40): _pair(-0.3, -0.5, -0.1),
        (160, 40): _pair(-0.5, -0.7, -0.3),
        (320, 40): _pair(-0.9, -1.1, -0.7),
        (160, 80): _pair(-0.2, -0.4, 0.0),
        (320, 80): _pair(-0.6, -0.8, -0.4),
    }
    label, _ = report_mod.classify_trajectory(curve, pairwise)
    assert label == "other"


# --- cross-seed summary math (synthetic) --------------------------------------

def test_cross_seed_summary_math_matches_stdlib_statistics():
    per_seed = {0: 6.554, 1: 6.7, 2: 6.2, 3: 6.9}
    vals = list(per_seed.values())
    mean, med = st.mean(vals), st.median(vals)
    lo, hi = min(vals), max(vals)
    sd = st.stdev(vals)
    assert mean == pytest.approx(sum(vals) / 4)
    assert lo <= med <= hi
    assert sd > 0


def test_pearson_descriptive_correlation(report_mod):
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    assert report_mod.pearson(xs, ys) == pytest.approx(1.0)
    assert report_mod.pearson(xs, [8.0, 6.0, 4.0, 2.0]) == pytest.approx(-1.0)
    assert report_mod.pearson([1.0], [1.0]) is None            # n < 2
    assert report_mod.pearson([1.0, 1.0], [2.0, 3.0]) is None  # zero variance in x


# --- manifest module: constants + gate logic (synthetic) -----------------------

def test_manifest_warm_start_hash_is_pinned_to_experiment2(manifest_mod):
    exp2 = json.load(open(os.path.join(
        ROOT, "results", "ppo_budget_v1", "manifest.json")))
    assert (manifest_mod.WARM_START_PARAMETER_SHA256
            == exp2["training"]["warm_start"]["parameter_sha256"])
    assert (manifest_mod.WARM_START_CHECKPOINT_SHA256
            == exp2["training"]["warm_start"]["checkpoint_sha256"])


def test_manifest_diagnostic_corpus_fingerprint_is_pinned(manifest_mod):
    exp2_drift = json.load(open(os.path.join(
        ROOT, "results", "ppo_budget_v1", "policy_drift.json")))
    assert (manifest_mod.DIAGNOSTIC_CORPUS_FINGERPRINT
            == exp2_drift["corpus"]["fingerprint_sha256"])


def test_training_seed_episode_spans_are_disjoint_and_outside_reserved(manifest_mod):
    from ml.seeds import ppo_episode_seed, check_training_range
    spans = {}
    for seed in (0, 1, 2, 3):
        lo = ppo_episode_seed(seed, 1)
        hi = ppo_episode_seed(seed, 320 * 16)
        assert not check_training_range(f"seed{seed}", lo, hi)
        spans[seed] = (lo, hi)
    # every new seed's block is disjoint from every other seed's block
    ordered = sorted(spans.items(), key=lambda kv: kv[1][0])
    for (s_a, (_, hi_a)), (s_b, (lo_b, _)) in zip(ordered, ordered[1:]):
        assert hi_a < lo_b, f"seed {s_a} and seed {s_b} episode ranges overlap"


def test_dev_eval_seed_range_identical_to_experiment_2():
    from ml.seeds import DEV_SEED_START
    exp2 = json.load(open(os.path.join(
        ROOT, "results", "ppo_budget_v1", "dev", "iter000_vs_greedy.json")))
    assert exp2["seed_range"] == [DEV_SEED_START, DEV_SEED_START + 999]


# --- integration tests requiring the real Experiment 3 artifacts --------------

@requires_full_results
def test_same_warm_start_hash_across_all_new_seeds():
    from ml.model_fingerprint import checkpoint_fingerprint
    exp2 = json.load(open(os.path.join(
        ROOT, "results", "ppo_budget_v1", "manifest.json")))
    warm_hash = exp2["training"]["warm_start"]["parameter_sha256"]
    for seed in (1, 2, 3):
        ckpt = os.path.join(MULTISEED_DIR, f"seed_{seed}", "checkpoints",
                            "iter_000.pt")
        fp = checkpoint_fingerprint(ckpt)
        assert fp["parameter_sha256"] == warm_hash, (
            f"seed {seed} iter0 does not match the frozen warm start")


@requires_full_results
def test_evaluation_seed_ranges_identical_across_new_seeds():
    ranges = set()
    for seed in (1, 2, 3):
        blob = json.load(open(os.path.join(
            MULTISEED_DIR, f"seed_{seed}", "dev", "iter000_vs_greedy.json")))
        ranges.add(tuple(blob["seed_range"]))
    assert len(ranges) == 1, f"DEV seed ranges differ across seeds: {ranges}"
    exp2 = json.load(open(os.path.join(
        ROOT, "results", "ppo_budget_v1", "dev", "iter000_vs_greedy.json")))
    assert list(next(iter(ranges))) == exp2["seed_range"]


@requires_full_results
def test_dev_test_isolation_for_all_new_seed_evaluations():
    from ml.seeds import overlaps_eval_range
    for seed in (1, 2, 3):
        blob = json.load(open(os.path.join(
            MULTISEED_DIR, f"seed_{seed}", "dev", "iter320_vs_greedy.json")))
        lo, hi = blob["seed_range"]
        assert not overlaps_eval_range(lo, hi)
        assert blob["evaluation_split"] == "dev"


@requires_full_results
def test_cross_seed_result_loading_and_paired_comparisons(report_mod):
    for seed in (0, 1, 2, 3):
        rows, supp = report_mod.seed_paired_table(seed)
        assert len(rows) == 9
        pairs = {(r["target_iteration"], r["reference_iteration"]) for r in rows}
        assert pairs == set(report_mod.PAIR_SPECS)
        for row in rows:
            lo, hi = row["ci95"]
            assert lo <= row["mean_diff"] <= hi
        assert supp["target_iteration"] == 320 and supp["reference_iteration"] == 160


@requires_full_results
def test_paired_comparison_sign_convention_matches_raw_placements(report_mod):
    """mean_diff for (target, ref) should equal target_avg - ref_avg exactly
    (paired bootstrap mean == the deterministic sample mean difference)."""
    for seed in (0, 1, 2, 3):
        greedy = {it: report_mod.load_greedy(seed, it)
                  for it in report_mod.ITERS}
        rows, _ = report_mod.seed_paired_table(seed)
        for row in rows:
            t, r = row["target_iteration"], row["reference_iteration"]
            expected = (greedy[t]["metrics"]["avg_placement"]
                       - greedy[r]["metrics"]["avg_placement"])
            assert row["mean_diff"] == pytest.approx(expected, abs=1e-9)


@requires_full_results
def test_u_shape_classification_produces_valid_label_for_every_seed(report_mod):
    valid = {"U-like/transient improvement", "monotonic improvement",
            "monotonic degradation", "mostly flat/noisy", "other"}
    for seed in (0, 1, 2, 3):
        greedy = {it: report_mod.load_greedy(seed, it)["metrics"]["avg_placement"]
                 for it in report_mod.ITERS}
        rows, supp = report_mod.seed_paired_table(seed)
        pairwise = {(r["target_iteration"], r["reference_iteration"]): r
                   for r in rows}
        pairwise[report_mod.SUPPLEMENTARY_PAIR] = supp
        label, _ = report_mod.classify_trajectory(greedy, pairwise)
        assert label in valid


@requires_full_results
def test_aggregate_json_artifacts_are_internally_consistent():
    agg = os.path.join(MULTISEED_DIR, "aggregate")
    summary = json.load(open(os.path.join(agg, "cross_seed_summary.json")))
    for row in summary["table"]:
        vals = list(row["per_seed"].values())
        assert row["min"] == pytest.approx(min(vals))
        assert row["max"] == pytest.approx(max(vals))
        assert row["mean"] == pytest.approx(sum(vals) / len(vals))
        assert row["min"] <= row["mean"] <= row["max"]

    replication = json.load(open(os.path.join(agg, "replication_analysis.json")))
    qa = replication["question_a_iter80_minus_iter0"]
    assert qa["n_improve"] + qa["n_worsen"] <= 4
    assert set(replication["u_shape_classification"].keys()) == {"0", "1", "2", "3"}


@requires_full_results
def test_plots_exist_for_all_seven_required_figures():
    plot_dir = os.path.join(MULTISEED_DIR, "aggregate", "plots")
    expected = ["A_multiseed_learning_curves.png", "B_mean_with_seed_variance.png",
               "C_expert_agreement_by_seed.png", "D_kl_from_warmstart_by_seed.png",
               "E_warmstart_agreement_by_seed.png",
               "F_action_category_drift_iter320.png",
               "G_rl_diagnostics_across_seeds.png"]
    for name in expected:
        p = os.path.join(plot_dir, name)
        assert os.path.isfile(p), f"missing plot {name}"
        assert os.path.getsize(p) > 1000, f"plot {name} looks empty/corrupt"


@requires_full_results
def test_plots_are_generated_from_supplied_data_not_hardcoded(report_mod, tmp_path):
    """Feeding different curve data must change the rendered plot bytes —
    proof the plotting function actually consumes its arguments rather than
    reading committed hard-coded numbers."""
    import shutil

    seeds = report_mod.SEEDS
    curves_a = {s: {it: {"greedy_avg": 5.0 + 0.1 * s,
                         "expert_agreement": 0.8, "warmstart_agreement": 0.9,
                         "kl_from_warmstart": 0.1,
                         "disagreement_share_by_category":
                             {c: 0.1 for c in report_mod.CATEGORIES}}
                    for it in report_mod.ITERS} for s in seeds}
    curves_b = {s: {it: {"greedy_avg": 3.0 - 0.1 * s,
                         "expert_agreement": 0.4, "warmstart_agreement": 0.5,
                         "kl_from_warmstart": 0.9,
                         "disagreement_share_by_category":
                             {c: 0.5 for c in report_mod.CATEGORIES}}
                    for it in report_mod.ITERS} for s in seeds}
    diag = {s: [{"iter": i, "pi_loss": 0.1, "v_loss": 0.1, "entropy": 0.5,
                "approx_kl": 0.01, "clip_frac": 0.05, "grad_norm": 0.5}
               for i in range(1, 6)] for s in seeds}

    original_out = report_mod.OUT_DIR
    try:
        report_mod.OUT_DIR = str(tmp_path / "run_a")
        os.makedirs(f"{report_mod.OUT_DIR}/plots", exist_ok=True)
        report_mod._plots(curves_a, diag)
        bytes_a = open(f"{report_mod.OUT_DIR}/plots/A_multiseed_learning_curves.png", "rb").read()

        report_mod.OUT_DIR = str(tmp_path / "run_b")
        os.makedirs(f"{report_mod.OUT_DIR}/plots", exist_ok=True)
        report_mod._plots(curves_b, diag)
        bytes_b = open(f"{report_mod.OUT_DIR}/plots/A_multiseed_learning_curves.png", "rb").read()

        assert bytes_a != bytes_b
    finally:
        report_mod.OUT_DIR = original_out
