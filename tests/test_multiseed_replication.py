"""Experiment 3 machinery: cross-training-seed loading, within-seed paired
comparisons, the pre-specified curve-shape rule, aggregate descriptives, and
the plot inputs.

The synthetic fixtures below are deliberately not the measured values, so a
test that passes proves the *logic*, not the numbers. The few tests that do
read committed artifacts read JSON only (never checkpoints) and assert
protocol invariants: one frozen warm start, identical DEV seeds everywhere,
and no contact with the reserved Benchmark v1 TEST interval.
"""

import importlib.util
import json
import os
import re

import pytest

from ml import seeds
from ml.replication import (COMPARISONS, FLAT_RANGE, PRIMARY_ITERS,
                            SHAPE_CLASSES, build_curve, build_plot_data,
                            category_replication, classify_curve,
                            cross_seed_summary, describe, drift_vs_performance,
                            effect_across_seeds, episodes, exploratory_ci,
                            freeze_stats, iteration_of, load_seed_bundle,
                            paired_table, rl_blocks, significance, spearman)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTISEED = os.path.join(REPO, "results", "ppo_multiseed_v1")
EXP2 = os.path.join(REPO, "results", "ppo_budget_v1")
WARMSTART_SHA = ("094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea"
                 "11246d473b")
NEW_SEEDS = [1, 2, 3]


def _report_module():
    path = os.path.join(REPO, "scripts", "ppo_multiseed_report.py")
    spec = importlib.util.spec_from_file_location("ppo_multiseed_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- synthetic fixtures -------------------------------------------------------
def _dev_result(agent, placements, field="greedy"):
    """A minimal DEV result JSON of the shape ml.analyze_benchmark accepts."""
    n = len(placements)
    return {
        "benchmark_version": "Replay DEV Evaluation (Benchmark v1 machinery)",
        "evaluation_split": "dev", "agent": agent, "field": field,
        "games": n, "base_seed": seeds.DEV_SEED_START,
        "seed_range": [seeds.DEV_SEED_START, seeds.DEV_SEED_START + n - 1],
        "environment": {"env": "hsbg_coach.bg_env.BGEnv"},
        "placements": list(placements),
        "avg_placement_ci95": {"low": 0.0, "high": 0.0},
        "metrics": {"games": n, "avg_placement": sum(placements) / n,
                    "median_placement": 5.0, "std_placement": 1.0,
                    "top4_rate": 0.25, "win_rate": 0.05,
                    "placement_counts": {str(p): placements.count(p)
                                         for p in range(1, 9)}},
    }


def _paired_rows(sig_by_pair, means=None):
    """Fake paired rows carrying only what classify_curve reads."""
    rows = []
    for target, ref in COMPARISONS:
        s = sig_by_pair.get((target, ref), "none")
        ci = {"better": [-0.30, -0.10], "worse": [0.10, 0.30],
              "none": [-0.10, 0.10]}[s]
        rows.append({"iteration": target, "reference_iteration": ref,
                     "label": f"iter{target} - iter{ref}", "ci95": ci,
                     "significance": s,
                     "mean_diff": (means or {}).get((target, ref),
                                                    sum(ci) / 2)})
    return rows


def _curve(training_seed, greedy, expert=None, kl=None):
    rows = []
    for i, it in enumerate(PRIMARY_ITERS):
        rows.append({
            "training_seed": training_seed, "iteration": it,
            "cumulative_episodes": episodes(it), "greedy_avg": greedy[i],
            "greedy_ci95": {"low": greedy[i] - 0.1, "high": greedy[i] + 0.1},
            "greedy_top4": 0.2, "greedy_win": 0.03,
            "mixed_avg": greedy[i] - 2.0,
            "expert_agreement": (expert or [0.8] * 5)[i],
            "warmstart_agreement": 1.0 - 0.1 * i,
            "kl_from_warmstart": (kl or [0.0, 0.2, 0.3, 0.5, 1.1])[i],
            "corpus_entropy": 0.5 + 0.01 * i,
            "expert_disagreement_by_category": {
                "buy": 0.7, "play": 0.1 * (i + 1), "sell": 0.2,
                "roll": 0.05 * i, "level": 0.1, "freeze": None,
                "end": 0.02 * i},
            "drift_contribution_by_category": {c: 0.1 for c in
                                               ("buy", "play", "sell", "roll",
                                                "level", "freeze", "end")},
        })
    return rows


# --- protocol invariants over the committed artifacts -------------------------
def _metadata(seed):
    path = os.path.join(MULTISEED, f"seed_{seed}", "checkpoint_metadata.json")
    if not os.path.isfile(path):
        pytest.skip(f"seed {seed} artifacts not present")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_every_seed_starts_from_the_same_frozen_warm_start(seed):
    """The training-seed control: BC is fixed so only PPO randomness varies.
    A different iteration-0 hash would mean we measured BC randomness too."""
    meta = _metadata(seed)
    iter0 = next(c for c in meta["checkpoints"] if c["iteration"] == 0)
    assert iter0["parameter_sha256"] == WARMSTART_SHA
    assert meta["iter0_matches_frozen_warm_start"] is True
    assert meta["training_seed"] == seed


def test_warm_start_hash_is_identical_across_all_training_seeds():
    hashes = {s: _metadata(s)["iter0_parameter_sha256"] for s in NEW_SEEDS}
    assert len(set(hashes.values())) == 1, hashes
    assert set(hashes.values()) == {WARMSTART_SHA}


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_training_seed_metadata_is_recorded_per_seed(seed):
    """Each artifact must say which PPO training seed produced it — otherwise
    cross-seed aggregation could silently mix trajectories."""
    for name in ("policy_drift.json", "action_category_drift.json"):
        path = os.path.join(MULTISEED, f"seed_{seed}", name)
        if not os.path.isfile(path):
            pytest.skip(f"seed {seed} artifacts not present")
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["training_seed"] == seed
    trained = {c["iteration"] for c in _metadata(seed)["checkpoints"]}
    assert set(PRIMARY_ITERS) <= trained


def test_evaluation_seeds_are_identical_across_every_run():
    """Pairing is only valid because every checkpoint of every training seed
    met the same DEV lobbies — including Experiment 2's seed-0 runs."""
    expected = {"greedy": (1000, [seeds.DEV_SEED_START,
                                  seeds.DEV_SEED_START + 999]),
                "greedy4_random3": (500, [seeds.DEV_SEED_START,
                                          seeds.DEV_SEED_START + 499])}
    checked = 0
    for root in [EXP2] + [os.path.join(MULTISEED, f"seed_{s}")
                          for s in NEW_SEEDS]:
        for it in PRIMARY_ITERS:
            for field, (games, rng) in expected.items():
                path = os.path.join(root, "dev",
                                    f"iter{it:03d}_vs_{field}.json")
                if not os.path.isfile(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    blob = json.load(f)
                assert blob["evaluation_split"] == "dev"
                assert blob["base_seed"] == seeds.DEV_SEED_START
                assert blob["games"] == games
                assert blob["seed_range"] == rng
                assert len(blob["placements"]) == games
                checked += 1
    if checked == 0:
        pytest.skip("no DEV artifacts present")


def test_dev_and_test_isolation_for_experiment_3():
    """No evaluated lobby and no PPO episode of seeds 1-3 may touch TEST, and
    no training episode may touch DEV either."""
    seeds.validate_dev_range(seeds.DEV_SEED_START, 1000)
    with pytest.raises(ValueError):                 # DEV cannot reach TEST
        seeds.validate_eval_range(seeds.DEV_SEED_START, 1000)
    for s in NEW_SEEDS:
        lo = seeds.ppo_episode_seed(s, 1)
        hi = seeds.ppo_episode_seed(s, 320 * 16)
        assert not seeds.overlaps_eval_range(lo, hi)
        assert not seeds.overlaps_dev_range(lo, hi)
        assert not seeds.check_training_range("ml.train_ppo", lo, hi)


def test_drift_corpus_is_the_frozen_experiment_1_corpus():
    for s in NEW_SEEDS:
        path = os.path.join(MULTISEED, f"seed_{s}", "policy_drift.json")
        if not os.path.isfile(path):
            pytest.skip("drift artifacts not present")
        with open(path, encoding="utf-8") as f:
            corpus = json.load(f)["corpus"]
        assert corpus["fingerprint_sha256"].startswith("2ec217b353bd")
        assert corpus["states"] == 4440
        assert corpus["lobbies"] == 100
        assert corpus["seed_base"] == seeds.DEV_SEED_START + 40_000


# --- loading ------------------------------------------------------------------
def test_iteration_of_parses_checkpoint_filenames():
    assert iteration_of("iter_080.pt") == 80
    assert iteration_of("/x/y/iter_000.pt") == 0
    assert iteration_of("iter_320.pt") == 320


def test_load_seed_bundle_and_build_curve_round_trip(tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    greedy = {0: [8, 8, 6, 6], 40: [8, 8, 8, 6], 80: [4, 4, 6, 6],
              160: [5, 5, 6, 6], 320: [7, 7, 6, 6]}
    for it, pl in greedy.items():
        (dev / f"iter{it:03d}_vs_greedy.json").write_text(
            json.dumps(_dev_result(f"s9_iter{it:03d}", pl)))
        (dev / f"iter{it:03d}_vs_greedy4_random3.json").write_text(
            json.dumps(_dev_result(f"s9_iter{it:03d}", [p - 2 for p in pl],
                                   field="greedy4_random3")))
    drift = {"checkpoints": [
        {"checkpoint": f"iter_{it:03d}.pt", "parameter_sha256": "p",
         "checkpoint_sha256": "c", "expert_agreement": 0.8 - 0.05 * i,
         "warmstart_agreement": 1.0 - 0.1 * i, "kl_from_warmstart_mean": 0.2 * i,
         "entropy_mean": 0.5, "value_mean": 0.1, "value_std": 0.2}
        for i, it in enumerate(PRIMARY_ITERS)]}
    cats = {"checkpoints": [
        {"checkpoint": f"iter_{it:03d}.pt", "vs_expert": {
            "n_states": 10, "n_disagreements": 2, "overall_agreement": 0.8,
            "confusion_matrix": {"roll": {"roll": 8, "freeze": 2}},
            "reference_category_counts": {"roll": 10, "freeze": 0},
            "disagreement_share_by_category": {"roll": 0.2},
            "contribution_to_total_drift": {"roll": 1.0}}}
        for it in PRIMARY_ITERS]}
    (tmp_path / "policy_drift.json").write_text(json.dumps(drift))
    (tmp_path / "action_category_drift.json").write_text(json.dumps(cats))
    (tmp_path / "diag.jsonl").write_text("".join(
        json.dumps({"iter": i, "entropy": 0.5, "approx_kl": 0.01,
                    "rollout_avg_placement": 5.0}) + "\n"
        for i in range(1, 321)))

    bundle = load_seed_bundle(9, str(dev), str(tmp_path / "policy_drift.json"),
                              str(tmp_path / "action_category_drift.json"),
                              str(tmp_path / "diag.jsonl"))
    assert bundle["training_seed"] == 9
    assert sorted(bundle["greedy"]) == PRIMARY_ITERS
    curve = build_curve(bundle)
    assert [r["iteration"] for r in curve] == PRIMARY_ITERS
    assert [r["cumulative_episodes"] for r in curve] == [0, 640, 1280, 2560, 5120]
    assert curve[2]["greedy_avg"] == pytest.approx(5.0)     # [4,4,6,6]
    assert curve[2]["mixed_avg"] == pytest.approx(3.0)
    assert curve[0]["warmstart_agreement"] == 1.0
    blocks = rl_blocks(bundle["diag"])
    assert blocks["iters_1_40"]["n"] == 40
    assert blocks["iters_41_160"]["n"] == 120
    assert blocks["iters_161_320"]["n"] == 160
    assert blocks["iters_161_320"]["entropy"] == pytest.approx(0.5)


# --- within-seed paired comparisons -------------------------------------------
def test_paired_table_computes_the_nine_pre_specified_comparisons():
    by_iter = {0: _dev_result("i0", [8, 8, 8, 8]),
               40: _dev_result("i40", [8, 8, 8, 8]),
               80: _dev_result("i80", [6, 6, 6, 6]),
               160: _dev_result("i160", [7, 7, 7, 7]),
               320: _dev_result("i320", [8, 8, 8, 8])}
    rows = paired_table(by_iter)
    assert len(rows) == 9
    assert [(r["iteration"], r["reference_iteration"]) for r in rows] == \
        list(COMPARISONS)
    got = {r["label"]: r for r in rows}
    assert got["iter80 - iter0"]["mean_diff"] == pytest.approx(-2.0)
    assert got["iter80 - iter0"]["significance"] == "better"
    assert got["iter320 - iter80"]["mean_diff"] == pytest.approx(2.0)
    assert got["iter320 - iter80"]["significance"] == "worse"
    assert got["iter40 - iter0"]["mean_diff"] == pytest.approx(0.0)
    assert got["iter40 - iter0"]["significance"] == "none"
    # deterministic: the same inputs give the same bootstrap CI
    assert paired_table(by_iter)[0]["ci95"] == rows[0]["ci95"]


def test_paired_table_refuses_mismatched_evaluation_seeds():
    a = _dev_result("i0", [8, 8, 8, 8])
    b = _dev_result("i80", [6, 6, 6, 6])
    b["base_seed"] = seeds.DEV_SEED_START + 1
    with pytest.raises(ValueError, match="not paired-comparable"):
        paired_table({0: a, 80: b, 40: a, 160: a, 320: a})


def test_significance_reads_the_ci_not_the_point_estimate():
    assert significance([-0.4, -0.1]) == "better"
    assert significance([0.1, 0.4]) == "worse"
    assert significance([-0.4, 0.1]) == "none"
    assert significance([0.0, 0.4]) == "none"       # touching zero is not sig


# --- the pre-specified curve-shape rule ---------------------------------------
def test_shape_rule_labels_a_u_shaped_trajectory():
    avg = {0: 6.55, 40: 6.76, 80: 6.32, 160: 6.43, 320: 6.61}
    rows = _paired_rows({(80, 0): "better", (40, 0): "worse",
                         (80, 40): "better", (160, 40): "better",
                         (320, 80): "worse"})
    got = classify_curve(avg, rows)
    assert got["shape_class"] == SHAPE_CLASSES[0]
    assert got["transient_peak_iterations"] == [80]
    assert got["descriptive"]["best_budget_iteration"] == 80
    assert got["descriptive"]["point_estimate_transient"] is True


def test_shape_rule_needs_both_the_gain_and_the_regression():
    avg = {0: 6.55, 40: 6.76, 80: 6.32, 160: 6.43, 320: 6.35}
    # significant gain at 80, but iter320 is NOT significantly worse than it
    got = classify_curve(avg, _paired_rows({(80, 0): "better"}))
    assert got["shape_class"] != SHAPE_CLASSES[0]
    # and a regression without a significant gain is not a U either
    got2 = classify_curve(avg, _paired_rows({(320, 80): "worse"}))
    assert got2["shape_class"] != SHAPE_CLASSES[0]


def test_shape_rule_labels_monotonic_improvement_and_degradation():
    up = {0: 6.6, 40: 6.4, 80: 6.2, 160: 6.0, 320: 5.8}
    got = classify_curve(up, _paired_rows({(320, 0): "better"}))
    assert got["shape_class"] == SHAPE_CLASSES[1]
    down = {0: 5.8, 40: 6.0, 80: 6.2, 160: 6.4, 320: 6.6}
    got = classify_curve(down, _paired_rows({(320, 0): "worse"}))
    assert got["shape_class"] == SHAPE_CLASSES[2]
    # a significant endpoint effect with a non-monotone path is NOT monotone
    wiggly = {0: 6.6, 40: 6.9, 80: 6.2, 160: 6.4, 320: 5.8}
    got = classify_curve(wiggly, _paired_rows({(320, 0): "better"}))
    assert got["shape_class"] == SHAPE_CLASSES[4]


def test_shape_rule_labels_flat_and_other():
    flat = {0: 6.50, 40: 6.52, 80: 6.48, 160: 6.55, 320: 6.51}
    got = classify_curve(flat, _paired_rows({}))
    assert got["shape_class"] == SHAPE_CLASSES[3]
    assert got["descriptive"]["range"] <= FLAT_RANGE
    # same significance picture but a wide span is not "flat"
    wide = {0: 6.10, 40: 6.60, 80: 6.20, 160: 6.55, 320: 6.30}
    assert classify_curve(wide, _paired_rows({})
                          )["shape_class"] == SHAPE_CLASSES[4]
    # significant comparisons that match no shape fall through to "other"
    got = classify_curve(flat, _paired_rows({(160, 40): "worse"}))
    assert got["shape_class"] == SHAPE_CLASSES[4]
    assert got["significant_comparisons"] == ["iter160 - iter40 worse"]


def test_shape_rule_never_invents_a_class():
    for avg, sig in [({0: 1, 40: 2, 80: 3, 160: 4, 320: 5}, {}),
                     ({0: 5, 40: 4, 80: 3, 160: 2, 320: 1}, {(320, 0): "worse"}),
                     ({0: 6, 40: 6, 80: 6, 160: 6, 320: 6}, {(80, 0): "worse"})]:
        assert classify_curve(avg, _paired_rows(sig))["shape_class"] \
            in SHAPE_CLASSES


# --- aggregate math -----------------------------------------------------------
def test_describe_and_cross_seed_summary_math():
    d = describe([1.0, 2.0, 3.0, 6.0])
    assert d["n"] == 4 and d["mean"] == pytest.approx(3.0)
    assert d["median"] == pytest.approx(2.5)
    assert d["min"] == 1.0 and d["max"] == 6.0
    assert d["sd"] == pytest.approx(2.1602468994692865)   # sample sd, n-1

    avg = {0: {0: 6.0, 40: 7.0, 80: 5.0, 160: 5.5, 320: 6.5},
           1: {0: 6.0, 40: 6.0, 80: 6.0, 160: 6.0, 320: 6.0}}
    s = cross_seed_summary(avg)
    assert s["training_seeds"] == [0, 1]
    row40 = next(r for r in s["per_budget"] if r["iteration"] == 40)
    assert row40["cumulative_episodes"] == 640
    assert row40["by_seed"] == {"0": 7.0, "1": 6.0}
    assert row40["mean"] == pytest.approx(6.5)
    assert row40["min"] == 6.0 and row40["max"] == 7.0
    # individual seeds are always kept alongside the summary statistics
    assert all("by_seed" in r for r in s["per_budget"])


def test_exploratory_ci_is_labeled_and_widens_with_spread():
    tight = exploratory_ci([0.10, 0.11, 0.09, 0.10])
    wide = exploratory_ci([-0.40, 0.50, 0.10, -0.20])
    assert tight["label"] == "exploratory" and "n=4" in tight["caveat"]
    assert (wide["ci95"][1] - wide["ci95"][0]) > \
        (tight["ci95"][1] - tight["ci95"][0])
    assert exploratory_ci([0.1])["ci95"] is None


def test_effect_across_seeds_counts_directions_and_significance():
    paired = {0: _paired_rows({(80, 0): "better"},
                              means={(80, 0): -0.229}),
              1: _paired_rows({}, means={(80, 0): -0.05}),
              2: _paired_rows({(80, 0): "worse"}, means={(80, 0): 0.30}),
              3: _paired_rows({}, means={(80, 0): 0.02})}
    e = effect_across_seeds(paired, 80, 0)
    assert e["comparison"] == "iter80 - iter0"
    assert [r["training_seed"] for r in e["per_seed"]] == [0, 1, 2, 3]
    assert e["n_point_estimate_better"] == 2      # -0.229, -0.05
    assert e["n_point_estimate_worse"] == 2
    assert e["n_ci_excludes_zero_better"] == 1
    assert e["n_ci_excludes_zero_worse"] == 1
    assert e["n_ci_includes_zero"] == 2
    assert e["across_seed_effect"]["label"] == "exploratory"


# --- drift / category helpers -------------------------------------------------
def test_freeze_stats_counts_a_new_action_the_expert_never_takes():
    row = {"vs_expert": {
        "n_states": 100, "n_disagreements": 20, "overall_agreement": 0.8,
        "confusion_matrix": {"end": {"end": 50, "freeze": 12},
                             "roll": {"roll": 30, "freeze": 3},
                             "freeze": {}},
        "reference_category_counts": {"end": 62, "roll": 33, "freeze": 0},
        "disagreement_share_by_category": {"end": 0.2},
        "contribution_to_total_drift": {"end": 1.0}}}
    f = freeze_stats(row)
    assert f["freeze_selections"] == 15
    assert f["freeze_rate"] == pytest.approx(0.15)
    assert f["freeze_appears"] is True
    assert f["expert_freeze_states"] == 0
    row["vs_expert"]["confusion_matrix"] = {"end": {"end": 100}}
    assert freeze_stats(row)["freeze_appears"] is False
    assert category_replication(row)["overall_agreement"] == 0.8


def test_spearman_and_drift_vs_performance_are_descriptive():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) is None
    curve = _curve(1, [6.6, 6.7, 6.3, 6.4, 6.9],
                   expert=[0.85, 0.78, 0.82, 0.74, 0.42])
    got = drift_vs_performance(curve)
    assert got["best_iteration"] == 80
    assert got["best_expert_agreement"] == pytest.approx(0.82)
    assert got["later_mean_expert_agreement"] == pytest.approx(0.58)
    assert got["best_has_higher_expert_agreement_than_later"] is True
    assert got["best_has_lower_kl_than_later"] is True
    assert "no causal claim" in got["note"]


# --- plots read result data, never typed-in numbers ---------------------------
def test_build_plot_data_mirrors_its_inputs():
    curves = {0: _curve(0, [6.554, 6.761, 6.325, 6.435, 6.606]),
              1: _curve(1, [6.1, 6.2, 6.3, 6.4, 6.5])}
    cats = {s: {"disagreement_share_by_category":
                {"roll": 0.1 * s, "end": 0.2, "freeze": None},
                "freeze_selections": 7 * s} for s in (0, 1)}
    blocks = {s: {n: {"entropy": 0.5 + s} for n in
                  ("iters_1_40", "iters_41_160", "iters_161_320")}
              for s in (0, 1)}
    d = build_plot_data(curves, cats, blocks)
    assert d["training_seeds"] == [0, 1]
    assert d["episodes"] == [0, 640, 1280, 2560, 5120]
    assert d["greedy_avg"][0] == [6.554, 6.761, 6.325, 6.435, 6.606]
    assert d["greedy_mean_across_seeds"][0] == pytest.approx((6.554 + 6.1) / 2)
    assert d["category_disagreement_iter320"][1][d["categories"].index("roll")] \
        == pytest.approx(0.1)
    # a category the confusion never saw becomes 0.0, not a crash
    assert d["category_disagreement_iter320"][0][
        d["categories"].index("freeze")] == 0.0
    assert d["freeze_selections_iter320"] == {0: 0, 1: 7}
    # changing the inputs changes the plotted series
    curves[1] = _curve(1, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert build_plot_data(curves, cats, blocks)["greedy_avg"][1] == \
        [1.0, 2.0, 3.0, 4.0, 5.0]


def test_plot_renderer_draws_only_supplied_series(tmp_path):
    mod = _report_module()
    curves = {0: _curve(0, [6.5, 6.7, 6.3, 6.4, 6.6]),
              1: _curve(1, [6.0, 6.1, 6.2, 6.3, 6.4])}
    cats = {s: {"disagreement_share_by_category": {"roll": 0.3},
                "freeze_selections": s} for s in (0, 1)}
    blocks = {s: {n: {"rollout_avg_placement": 5.5, "entropy": 0.5,
                      "approx_kl": 0.01, "clip_frac": 0.06,
                      "value_explained_variance": 0.6, "adv_mean_abs": 0.2}
                  for n in ("iters_1_40", "iters_41_160", "iters_161_320")}
              for s in (0, 1)}
    mod._plots(build_plot_data(curves, cats, blocks), out=str(tmp_path))
    made = sorted(p for p in os.listdir(tmp_path) if p.endswith(".png"))
    assert [p[0] for p in made] == list("ABCDEFG")
    assert all(os.path.getsize(tmp_path / p) > 0 for p in made)


def test_plot_code_contains_no_measured_values():
    """Every number a figure draws must come from the result JSON. Guard
    against a value being typed into the plotting code by checking that no
    numeric literal in the report script equals a measured DEV placement."""
    path = os.path.join(REPO, "scripts", "ppo_multiseed_report.py")
    with open(path, encoding="utf-8") as f:
        literals = {float(m) for m in
                    re.findall(r"(?<![\w.])\d+\.\d+", f.read())}
    measured = set()
    for root in [EXP2] + [os.path.join(MULTISEED, f"seed_{s}")
                          for s in NEW_SEEDS]:
        for it in PRIMARY_ITERS:
            p = os.path.join(root, "dev", f"iter{it:03d}_vs_greedy.json")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)["metrics"]
                measured |= {round(m["avg_placement"], 3),
                             round(m["std_placement"], 3),
                             m["top4_rate"], m["win_rate"]}
    assert not (literals & measured), sorted(literals & measured)
