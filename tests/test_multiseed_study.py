"""Experiment 3 machinery: cross-seed aggregation, the pre-specified paired
comparisons, the documented U-shape classification rule, warm-start and
evaluation-seed controls, and DEV/TEST isolation of the multi-seed runs.

The rule/math tests are self-contained. The artifact tests validate the
committed ``results/ppo_multiseed_v1`` outputs and skip only when those
artifacts are absent (e.g. a fresh clone mid-reproduction)."""

import importlib.util
import json
import os
import statistics as st

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(REPO, "results", "ppo_multiseed_v1")
SEED0_DIR = os.path.join(REPO, "results", "ppo_budget_v1")
NEW_SEEDS = [1, 2, 3]
ITERS = [0, 40, 80, 160, 320]

_HAVE_ARTIFACTS = os.path.isfile(os.path.join(DIR, "manifest.json"))
needs_artifacts = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="multi-seed artifacts not generated yet")


def _report_mod():
    """Import scripts/ppo_multiseed_report.py as a module."""
    path = os.path.join(REPO, "scripts", "ppo_multiseed_report.py")
    spec = importlib.util.spec_from_file_location("ppo_multiseed_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(target, ref, mean, lo, hi):
    return {"iteration": target, "reference_iteration": ref,
            "label": f"iter{target}-iter{ref}", "mean_diff": mean,
            "ci95": [lo, hi], "ci_excludes_zero": not (lo <= 0 <= hi)}


def _rows(overrides):
    """The nine pre-specified comparisons, defaulting to null effects."""
    mod = _report_mod()
    rows = []
    for target, ref in mod.PAIRS:
        o = overrides.get((target, ref))
        rows.append(_row(target, ref, *o) if o
                    else _row(target, ref, 0.0, -0.1, 0.1))
    return rows


# --- the documented U-shape classification rule --------------------------------
def test_prespecified_pairs_are_the_nine_from_the_spec():
    mod = _report_mod()
    assert mod.PAIRS == [(40, 0), (80, 0), (160, 0), (320, 0),
                         (80, 40), (160, 40), (320, 40),
                         (160, 80), (320, 80)]


def test_classify_u_like_transient_improvement():
    mod = _report_mod()
    # iter80 clearly better than iter0, iter320 clearly worse than iter80
    rows = _rows({(80, 0): (-0.3, -0.5, -0.1), (320, 80): (+0.3, +0.1, +0.5)})
    cat, why = mod.classify_curve(rows)
    assert cat == "U-like transient improvement"
    assert "80" in why


def test_classify_monotonic_improvement():
    mod = _report_mod()
    rows = _rows({(320, 0): (-0.4, -0.6, -0.2), (80, 0): (-0.2, -0.4, -0.05),
                  (320, 80): (-0.1, -0.3, 0.05)})
    cat, _ = mod.classify_curve(rows)
    assert cat == "monotonic improvement"


def test_classify_monotonic_degradation():
    mod = _report_mod()
    rows = _rows({(320, 0): (+0.4, +0.2, +0.6), (160, 0): (+0.2, +0.05, +0.4)})
    cat, _ = mod.classify_curve(rows)
    assert cat == "monotonic degradation"


def test_classify_mostly_flat_noisy():
    mod = _report_mod()
    cat, _ = mod.classify_curve(_rows({}))
    assert cat == "mostly flat / noisy"


def test_classify_other_for_mixed_clear_signals():
    mod = _report_mod()
    # A clear mid dip without a clear later regression, plus a clear endpoint
    # regression vs iter0 — fits neither the U rule nor the monotone rules.
    rows = _rows({(80, 0): (-0.3, -0.5, -0.1), (320, 0): (+0.3, +0.1, +0.5),
                  (320, 80): (+0.1, -0.05, +0.3)})
    cat, _ = mod.classify_curve(rows)
    assert cat == "other"


def test_u_rule_requires_regression_from_the_same_mid_checkpoint():
    mod = _report_mod()
    # clear improvement at 80 but no clear later regression from 80 or 40 —
    # must not classify as U-like
    rows = _rows({(80, 0): (-0.3, -0.5, -0.1), (320, 80): (+0.1, -0.05, +0.3)})
    cat, _ = mod.classify_curve(rows)
    assert cat != "U-like transient improvement"
    # iteration-160 improvement alone cannot trigger the U clause because
    # diff(320-160) is not among the nine pre-specified comparisons
    rows = _rows({(160, 0): (-0.3, -0.5, -0.1)})
    cat, _ = mod.classify_curve(rows)
    assert cat == "other"


# --- aggregate summary math -----------------------------------------------------
def test_cross_seed_stats_known_values():
    mod = _report_mod()
    s = mod.cross_seed_stats([6.0, 6.5, 6.1, 6.4])
    assert s["mean"] == pytest.approx(6.25)
    assert s["median"] == pytest.approx(6.25)
    assert s["min"] == 6.0 and s["max"] == 6.5
    assert s["std"] == pytest.approx(st.stdev([6.0, 6.5, 6.1, 6.4]))
    assert s["n_seeds"] == 4


# --- artifact-level controls ----------------------------------------------------
@needs_artifacts
def test_manifest_records_training_seed_metadata():
    m = json.load(open(f"{DIR}/manifest.json"))
    blocks = m["training"]["per_seed"]
    assert [b["training_seed"] for b in blocks] == NEW_SEEDS
    for b in blocks:
        s = b["training_seed"]
        assert f"--seed {s} " in b["command"]
        assert b["iteration0_reproduction_gate"]["passed"] is True
        # frozen recipe markers, verbatim from Experiment 2
        assert "--iters 320" in b["command"]
        assert "--episodes 16" in b["command"]
        assert "--shaping-horizon 40" in b["command"]
        assert "--from-bc ml/policy_bc.pt" in b["command"]


@needs_artifacts
def test_same_warm_start_parameter_hash_across_all_seeds():
    m = json.load(open(f"{DIR}/manifest.json"))
    warm = m["warm_start"]["parameter_sha256"]
    # matches the committed Experiment 2 manifest
    m2 = json.load(open(f"{SEED0_DIR}/manifest.json"))
    assert warm == m2["training"]["warm_start"]["parameter_sha256"]
    # every seed's iteration 0 equals the frozen warm start
    for s in NEW_SEEDS:
        ck = json.load(open(f"{DIR}/seed_{s}/checkpoints.json"))
        assert ck["warm_start_parameter_sha256"] == warm
        iter0 = next(c for c in ck["checkpoints"] if c["iteration"] == 0)
        assert iter0["parameter_sha256"] == warm
        drift = json.load(open(f"{DIR}/seed_{s}/policy_drift.json"))
        ref = drift["reference"]
        assert ref["parameter_sha256"] == warm
    seed0_drift = json.load(open(f"{SEED0_DIR}/policy_drift.json"))
    assert seed0_drift["reference"]["parameter_sha256"] == warm


@needs_artifacts
def test_evaluation_seed_equality_across_all_runs():
    """Every checkpoint of every seed scored the identical DEV games."""
    ref_g = json.load(open(f"{SEED0_DIR}/dev/iter000_vs_greedy.json"))
    ref_m = json.load(
        open(f"{SEED0_DIR}/dev/iter000_vs_greedy4_random3.json"))
    for s in NEW_SEEDS:
        for it in ITERS:
            g = json.load(
                open(f"{DIR}/seed_{s}/dev/iter{it:03d}_vs_greedy.json"))
            assert g["seed_range"] == ref_g["seed_range"]
            assert g["games"] == ref_g["games"] == 1000
            assert g["field"] == "greedy"
            mx = json.load(open(f"{DIR}/seed_{s}/dev/"
                                f"iter{it:03d}_vs_greedy4_random3.json"))
            assert mx["seed_range"] == ref_m["seed_range"]
            assert mx["games"] == ref_m["games"] == 500


@needs_artifacts
def test_dev_test_isolation_of_the_multiseed_runs():
    from ml import seeds
    m = json.load(open(f"{DIR}/manifest.json"))
    # evaluation stayed inside DEV
    lo, hi = m["evaluation"]["primary_field"]["seed_range"]
    assert seeds.DEV_SEED_START <= lo <= hi <= seeds.DEV_SEED_END
    assert not seeds.overlaps_eval_range(lo, hi)
    # training episode seeds touched neither DEV nor TEST
    for b in m["training"]["per_seed"]:
        tlo, thi = b["episode_seed_span"]
        assert not seeds.overlaps_eval_range(tlo, thi)
        assert not seeds.overlaps_dev_range(tlo, thi)
        assert b["episode_seeds_outside_dev"]
        assert b["episode_seeds_outside_test"]
        # and the recorded span is the real scheme's span
        s = b["training_seed"]
        assert tlo == seeds.ppo_episode_seed(s, 1)
        assert thi == seeds.ppo_episode_seed(s, 320 * 16)
    # nothing in the artifact tree mentions a TEST-range base seed
    assert m["evaluation"]["test_usage"].startswith("Benchmark v1 TEST was "
                                                    "not run")


@needs_artifacts
def test_cross_seed_loading_and_summary_math():
    mod = _report_mod()
    summary = json.load(open(f"{DIR}/aggregate/cross_seed_summary.json"))
    table = summary["budget_table"]
    assert [r["iteration"] for r in table] == ITERS
    for row in table:
        by_seed = row["greedy_avg_by_seed"]
        assert sorted(by_seed.keys()) == ["0", "1", "2", "3"]
        vals = [by_seed[k] for k in sorted(by_seed)]
        a = row["across_seeds"]
        assert a["mean"] == pytest.approx(st.mean(vals))
        assert a["median"] == pytest.approx(st.median(vals))
        assert a["min"] == pytest.approx(min(vals))
        assert a["max"] == pytest.approx(max(vals))
        assert a["std"] == pytest.approx(st.stdev(vals))
    # the seed-0 column is the committed Experiment 2 curve, unmodified
    exp2 = json.load(open(f"{SEED0_DIR}/learning_curve.json"))["curve"]
    exp2_by_iter = {c["iteration"]: c["greedy_avg"] for c in exp2}
    for row in table:
        assert row["greedy_avg_by_seed"]["0"] == \
            pytest.approx(exp2_by_iter[row["iteration"]])
    # and per-seed averages match the underlying DEV result files
    for s in NEW_SEEDS:
        for row in table:
            it = row["iteration"]
            g = json.load(
                open(f"{DIR}/seed_{s}/dev/iter{it:03d}_vs_greedy.json"))
            assert row["greedy_avg_by_seed"][str(s)] == \
                pytest.approx(g["metrics"]["avg_placement"])


@needs_artifacts
def test_within_seed_paired_comparisons_recompute():
    """The committed paired numbers must equal a fresh recomputation from
    the committed per-game placements."""
    from ml.analyze_benchmark import compare_pair, load_result
    paired = json.load(open(f"{DIR}/aggregate/paired_results.json"))
    for s in ["0", "1", "2", "3"]:
        rows = paired["per_seed"][s]
        assert [(r["iteration"], r["reference_iteration"]) for r in rows] == \
            [(40, 0), (80, 0), (160, 0), (320, 0), (80, 40), (160, 40),
             (320, 40), (160, 80), (320, 80)]
        d = SEED0_DIR if s == "0" else f"{DIR}/seed_{s}"
        # spot-recompute the two replication-critical comparisons
        for target, ref in [(80, 0), (320, 80)]:
            a = load_result(f"{d}/dev/iter{target:03d}_vs_greedy.json")
            b = load_result(f"{d}/dev/iter{ref:03d}_vs_greedy.json")
            fresh = compare_pair(a, b, seed=0)
            row = next(r for r in rows if r["iteration"] == target
                       and r["reference_iteration"] == ref)
            assert row["mean_diff"] == pytest.approx(fresh["mean_diff"])
            assert row["ci95"] == pytest.approx(fresh["ci95"])


@needs_artifacts
def test_replication_analysis_consistent_with_paired_results():
    mod = _report_mod()
    paired = json.load(open(f"{DIR}/aggregate/paired_results.json"))
    analysis = json.load(open(f"{DIR}/aggregate/replication_analysis.json"))
    qa = analysis["question_a_1280_episode_improvement"]
    qb = analysis["question_b_post_transient_decay"]
    n_improve = n_regress = 0
    for s in ["0", "1", "2", "3"]:
        rows = paired["per_seed"][s]
        r80 = next(r for r in rows if r["label"] == "iter80-iter0")
        r320 = next(r for r in rows if r["label"] == "iter320-iter80")
        assert qa["per_seed"][s]["mean_diff"] == pytest.approx(r80["mean_diff"])
        assert qb["per_seed"][s]["mean_diff"] == \
            pytest.approx(r320["mean_diff"])
        n_improve += r80["mean_diff"] < 0
        n_regress += r320["mean_diff"] > 0
        # the committed classification must equal a fresh application of the
        # documented rule to the committed paired rows
        cat, _ = mod.classify_curve(rows)
        assert analysis["u_shape_classification"][s]["category"] == cat
    assert qa["n_seeds_improving"] == n_improve
    assert qb["n_regressing"] == n_regress
    n_transient = sum(
        1 for s in ["0", "1", "2", "3"]
        if analysis["u_shape_classification"][s]["category"]
        == "U-like transient improvement")
    assert analysis["u_shape_summary"].startswith(f"{n_transient} / 4")


# --- plots are generated from result JSON ---------------------------------------
def test_plots_consume_result_json_not_hardcoded_values(tmp_path, monkeypatch):
    """_plots must render whatever curve data it is handed — synthetic
    inputs in a temp dir must produce all seven plot files."""
    pytest.importorskip("matplotlib")
    mod = _report_mod()
    monkeypatch.setattr(mod, "AGG", str(tmp_path))
    curves, diags, cats = {}, {}, {}
    for s in mod.SEEDS:
        curves[s] = [{"iteration": it, "cumulative_episodes": it * 16,
                      "greedy_avg": 6.0 + 0.1 * s + 0.01 * i,
                      "expert_agreement": 0.8 - 0.05 * i,
                      "warmstart_agreement": 1.0 - 0.1 * i,
                      "kl_from_warmstart": 0.2 * i}
                     for i, it in enumerate(mod.ITERS)]
        diags[s] = [{"iter": k, "adv_mean_abs": 0.1,
                     "value_explained_variance": 0.0, "entropy": 1.0,
                     "approx_kl": 0.01} for k in range(1, 4)]
        cats[s] = {"iter_320.pt": {"vs_expert": {
            "disagreement_share_by_category":
                {c: 0.1 for c in ["buy", "play", "sell", "roll", "level",
                                  "freeze", "end"]}}}}
    mod._plots(curves, diags, cats)
    expected = ["A_dev_learning_curves_by_seed.png",
                "B_dev_mean_across_seeds.png",
                "C_expert_agreement_by_seed.png",
                "D_kl_from_warmstart_by_seed.png",
                "E_warmstart_agreement_by_seed.png",
                "F_category_drift_iter320_by_seed.png",
                "G_rl_diagnostics_by_seed.png"]
    for name in expected:
        assert (tmp_path / "plots" / name).is_file(), name
