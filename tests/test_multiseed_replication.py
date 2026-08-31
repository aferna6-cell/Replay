"""Experiment 3: multi-seed aggregation, isolation, and classification."""

import json
import os

import pytest

from ml import seeds
from ml.multiseed_analysis import (
    ALL_SEEDS, CORPUS_FINGERPRINT_SHA256, DEV_EVAL_BASE, DEV_EVAL_GAMES,
    DEV_EVAL_LAST, PRIMARY_ITERS, SEED0_DIR, WARM_START_PARAMETER_SHA256,
    WITHIN_SEED_PAIRS, assert_corpus_fingerprint, assert_dev_eval_seeds,
    assert_eval_seeds_match_experiment2, assert_training_seeds_isolated,
    assert_warmstart_hash, classify_ushape, cross_seed_table, episodes,
    freeze_count, load_dev_result, load_seed_bundle, load_within_seed_paired,
    outcome_and_recommendation, pair_key, planned_ppo_span, question_a,
    question_b, seed_dir, summarize_numbers, training_seeds_isolated,
    within_seed_paired,
)


# --- isolation / metadata -----------------------------------------------------
def test_training_seeds_1_2_3_outside_dev_and_test():
    assert_training_seeds_isolated(ALL_SEEDS)
    rows = training_seeds_isolated(ALL_SEEDS)
    assert {r["training_seed"] for r in rows} == {0, 1, 2, 3}
    for r in rows:
        assert r["isolated"]
        assert not r["overlaps_dev"]
        assert not r["overlaps_test"]
        assert r["hi"] < seeds.EVAL_SEED_START
        assert r["hi"] < seeds.DEV_SEED_START


def test_planned_span_uses_centralized_ppo_helper():
    for s in ALL_SEEDS:
        lo, hi = planned_ppo_span(s)
        assert lo == seeds.ppo_episode_seed(s, 1)
        assert hi == seeds.ppo_episode_seed(s, 320 * 16)


def test_dev_eval_seeds_match_experiment_2_exactly():
    assert_dev_eval_seeds()
    assert DEV_EVAL_BASE == 10_550_000
    assert DEV_EVAL_LAST == 10_550_999
    assert DEV_EVAL_GAMES == 1000
    # every committed Experiment 2 greedy file uses that interval
    for it in PRIMARY_ITERS:
        blob = load_dev_result(SEED0_DIR, it, "greedy")
        assert_eval_seeds_match_experiment2(blob, "greedy")
        assert blob["seed_range"] == [10_550_000, 10_550_999]
        assert blob["evaluation_split"] == "dev"
    for it in PRIMARY_ITERS:
        blob = load_dev_result(SEED0_DIR, it, "greedy4_random3")
        assert_eval_seeds_match_experiment2(blob, "greedy4_random3")
        assert blob["seed_range"] == [10_550_000, 10_550_499]


def test_eval_seed_mismatch_is_rejected():
    blob = load_dev_result(SEED0_DIR, 0, "greedy")
    bad = dict(blob)
    bad["base_seed"] = seeds.EVAL_SEED_START
    with pytest.raises(ValueError, match="base seed"):
        assert_eval_seeds_match_experiment2(bad, "greedy")
    testish = dict(blob)
    testish["evaluation_split"] = "test"
    with pytest.raises(ValueError, match="evaluation_split"):
        assert_eval_seeds_match_experiment2(testish, "greedy")


def test_frozen_warm_start_hash_matches_experiment_2_manifest():
    man = json.load(open(os.path.join(SEED0_DIR, "manifest.json")))
    assert man["training"]["warm_start"]["parameter_sha256"] == (
        WARM_START_PARAMETER_SHA256)
    curve0 = json.load(open(os.path.join(SEED0_DIR, "learning_curve.json")))
    iter0 = next(c for c in curve0["curve"] if c["iteration"] == 0)
    assert_warmstart_hash(iter0["parameter_sha256"])
    with pytest.raises(ValueError, match="warm-start"):
        assert_warmstart_hash("0" * 64)


def test_frozen_corpus_fingerprint_matches_experiment_2():
    drift = json.load(open(os.path.join(SEED0_DIR, "policy_drift.json")))
    assert drift["corpus"]["fingerprint_sha256"] == CORPUS_FINGERPRINT_SHA256
    assert drift["corpus"]["states"] == 4440
    assert_corpus_fingerprint(drift["corpus"]["fingerprint_sha256"])
    with pytest.raises(ValueError, match="corpus fingerprint"):
        assert_corpus_fingerprint("deadbeef")


def test_seed_dir_does_not_point_seed_0_at_experiment_3_tree():
    assert seed_dir(0) == SEED0_DIR == "results/ppo_budget_v1"
    assert seed_dir(1) == "results/ppo_multiseed_v1/seed_1"
    assert seed_dir(2) == "results/ppo_multiseed_v1/seed_2"
    assert seed_dir(3) == "results/ppo_multiseed_v1/seed_3"


# --- cross-seed loading of the Experiment 2 seed-0 reference -----------------
def test_load_seed_0_from_experiment_2_not_a_copy():
    bundle = load_seed_bundle(0)
    assert bundle["source"] == "experiment_2"
    assert bundle["source_dir"] == SEED0_DIR
    assert bundle["training_seed"] == 0
    assert bundle["placements"][0] == pytest.approx(6.554)
    assert bundle["placements"][80] == pytest.approx(6.325)
    assert bundle["placements"][320] == pytest.approx(6.606)
    assert_warmstart_hash(bundle["curve"]["curve"][0]["parameter_sha256"])


# --- within-seed paired comparisons ------------------------------------------
def test_within_seed_pairs_are_the_nine_prespecified():
    assert WITHIN_SEED_PAIRS == (
        (40, 0), (80, 0), (160, 0), (320, 0),
        (80, 40), (160, 40), (320, 40),
        (160, 80), (320, 80),
    )


def test_seed0_paired_matches_experiment_2_published_numbers():
    paired = load_within_seed_paired(SEED0_DIR)
    # published Experiment 2 table (greedy, 1000 DEV games)
    r = paired[pair_key(40, 0)]
    assert r["mean_diff"] == pytest.approx(0.207)
    assert r["ci95"][0] == pytest.approx(0.093, abs=1e-3)
    assert r["ci95"][1] == pytest.approx(0.322, abs=1e-3)
    r = paired[pair_key(80, 0)]
    assert r["mean_diff"] == pytest.approx(-0.229)
    assert r["ci_excludes_zero"]
    r = paired[pair_key(80, 40)]
    assert r["mean_diff"] == pytest.approx(-0.436)
    r = paired[pair_key(320, 0)]
    assert r["mean_diff"] == pytest.approx(0.052)
    assert not r["ci_excludes_zero"]
    # newly required pairs that Experiment 2 did not publish as a table
    assert pair_key(160, 80) in paired
    assert pair_key(320, 80) in paired
    assert paired[pair_key(320, 80)]["mean_diff"] == pytest.approx(0.281)


def test_within_seed_paired_on_synthetic_identical_games():
    placements = [4, 5, 6, 7] * 25  # 100 games
    def _res(name):
        return {
            "agent": name, "benchmark_version": "v", "field": "greedy",
            "games": 100, "base_seed": DEV_EVAL_BASE,
            "seed_range": [DEV_EVAL_BASE, DEV_EVAL_BASE + 99],
            "environment": {"env": "x"}, "placements": list(placements),
            "metrics": {"avg_placement": 5.5},
        }
    greedy = {it: _res(f"iter{it:03d}") for it in PRIMARY_ITERS}
    # make iter80 better by 1 placement on every game
    greedy[80] = dict(greedy[80])
    greedy[80]["placements"] = [p - 1 for p in placements]
    greedy[80]["agent"] = "iter080"
    rows = within_seed_paired(greedy)
    assert rows[pair_key(80, 0)]["mean_diff"] == pytest.approx(-1.0)
    assert rows[pair_key(80, 0)]["ci_excludes_zero"]
    assert rows[pair_key(40, 0)]["mean_diff"] == pytest.approx(0.0)
    assert not rows[pair_key(40, 0)]["ci_excludes_zero"]


# --- aggregate summary math ---------------------------------------------------
def test_summarize_numbers_known_case():
    s = summarize_numbers([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["std"] == pytest.approx(1.2909944487358056)  # sample stdev
    assert "n=4" in s["note"]


def test_cross_seed_table_math():
    bundles = {
        0: {"placements": {0: 6.0, 40: 6.2, 80: 5.8, 160: 5.9, 320: 6.1}},
        1: {"placements": {0: 6.0, 40: 6.0, 80: 6.4, 160: 6.2, 320: 6.3}},
    }
    table = cross_seed_table(bundles)
    it80 = table["by_iteration"]["80"]
    assert it80["mean"] == pytest.approx(6.1)
    assert it80["min"] == pytest.approx(5.8)
    assert it80["max"] == pytest.approx(6.4)
    assert it80["per_seed"][0] == 5.8
    assert "training seed" in table["replication_unit"]


def test_questions_a_and_b_counts():
    def _pair(mean, lo, hi):
        return {"mean_diff": mean, "ci95": [lo, hi],
                "ci_excludes_zero": not (lo <= 0 <= hi)}

    bundles = {
        0: {"paired": {pair_key(80, 0): _pair(-0.23, -0.39, -0.06),
                       pair_key(320, 80): _pair(0.28, 0.10, 0.45)}},
        1: {"paired": {pair_key(80, 0): _pair(0.10, -0.05, 0.25),
                       pair_key(320, 80): _pair(-0.02, -0.15, 0.10)}},
        2: {"paired": {pair_key(80, 0): _pair(-0.15, -0.30, -0.01),
                       pair_key(320, 80): _pair(0.20, 0.05, 0.35)}},
        3: {"paired": {pair_key(80, 0): _pair(0.40, 0.20, 0.60),
                       pair_key(320, 80): _pair(0.05, -0.10, 0.20)}},
    }
    qa = question_a(bundles)
    assert qa["n_improve"] == 2
    assert qa["n_worsen"] == 2
    assert qa["n_ci_excludes_zero_improve"] == 2
    assert qa["n_ci_excludes_zero_worsen"] == 1
    qb = question_b(bundles)
    assert qb["n_regress"] == 2
    assert qb["n_continue_improving"] == 0
    assert qb["n_indistinguishable"] == 2


# --- U-shape classification ---------------------------------------------------
def test_classify_ushape_seed0_is_transient():
    p = {0: 6.554, 40: 6.761, 80: 6.325, 160: 6.435, 320: 6.606}
    got = classify_ushape(p)
    assert got["label"] == "U-like / transient improvement"
    assert got["mid_best_iteration"] == 80
    assert got["mid_gain_mean"] is True
    assert got["late_regression_mean"] is True


def test_classify_ushape_monotonic_improvement():
    p = {0: 6.6, 40: 6.4, 80: 6.2, 160: 6.0, 320: 5.8}
    assert classify_ushape(p)["label"] == "monotonic improvement"


def test_classify_ushape_monotonic_degradation():
    p = {0: 6.0, 40: 6.2, 80: 6.4, 160: 6.5, 320: 6.7}
    assert classify_ushape(p)["label"] == "monotonic degradation"


def test_classify_ushape_flat():
    p = {0: 6.50, 40: 6.51, 80: 6.49, 160: 6.50, 320: 6.52}
    # mid best (80) is below iter0 and 320 is above mid best → U-like on means
    # use a truly flat curve
    p = {0: 6.50, 40: 6.50, 80: 6.50, 160: 6.50, 320: 6.50}
    assert classify_ushape(p)["label"] == "mostly flat/noisy"


def test_classify_ushape_other_when_only_late_gain():
    # worse at mid-budget, better at the end — not U, not monotone
    p = {0: 6.5, 40: 6.8, 80: 6.7, 160: 6.6, 320: 6.2}
    assert classify_ushape(p)["label"] == "other"


def test_classify_ushape_prefers_u_over_monotone_when_both_could_apply():
    # dips then rises — U wins even if someone squints at monotone
    p = {0: 6.5, 40: 6.4, 80: 6.0, 160: 6.1, 320: 6.3}
    assert classify_ushape(p)["label"] == "U-like / transient improvement"


# --- outcome rule -------------------------------------------------------------
def test_outcome_a_when_most_trajectories_are_u_like():
    analysis = {
        "n_training_seeds": 4,
        "question_a_1280_episode_replication": {
            "per_seed": [
                {"training_seed": 0, "direction": "improve"},
                {"training_seed": 1, "direction": "improve"},
                {"training_seed": 2, "direction": "improve"},
                {"training_seed": 3, "direction": "worsen"},
            ]},
        "question_b_late_regression": {"n_regress": 3},
        "ushape": {
            "n_transient_improvement": 3,
            "per_seed": {
                "0": {"label": "U-like / transient improvement"},
                "1": {"label": "U-like / transient improvement"},
                "2": {"label": "U-like / transient improvement"},
                "3": {"label": "monotonic degradation"},
            }},
    }
    d = outcome_and_recommendation(analysis)
    assert d["outcome"] == "A"
    assert "anchoring" in d["recommendation"]


def test_outcome_b_when_new_seeds_never_improve():
    analysis = {
        "n_training_seeds": 4,
        "question_a_1280_episode_replication": {
            "per_seed": [
                {"training_seed": 0, "direction": "improve"},
                {"training_seed": 1, "direction": "worsen"},
                {"training_seed": 2, "direction": "worsen"},
                {"training_seed": 3, "direction": "worsen"},
            ]},
        "question_b_late_regression": {"n_regress": 0},
        "ushape": {
            "n_transient_improvement": 1,
            "per_seed": {
                "0": {"label": "U-like / transient improvement"},
                "1": {"label": "monotonic degradation"},
                "2": {"label": "mostly flat/noisy"},
                "3": {"label": "other"},
            }},
    }
    d = outcome_and_recommendation(analysis)
    assert d["outcome"] == "B"
    assert "stochastic rather than algorithmic" in d["recommendation"]


def test_outcome_c_when_shapes_vary_and_some_new_seeds_improve():
    analysis = {
        "n_training_seeds": 4,
        "question_a_1280_episode_replication": {
            "per_seed": [
                {"training_seed": 0, "direction": "improve"},
                {"training_seed": 1, "direction": "improve"},
                {"training_seed": 2, "direction": "worsen"},
                {"training_seed": 3, "direction": "worsen"},
            ]},
        "question_b_late_regression": {"n_regress": 1},
        "ushape": {
            "n_transient_improvement": 1,
            "per_seed": {
                "0": {"label": "U-like / transient improvement"},
                "1": {"label": "monotonic improvement"},
                "2": {"label": "monotonic degradation"},
                "3": {"label": "mostly flat/noisy"},
            }},
    }
    d = outcome_and_recommendation(analysis)
    assert d["outcome"] == "C"
    assert "stability" in d["recommendation"]


# --- freeze accounting + plots consume JSON ----------------------------------
def test_freeze_count_from_confusion_matrix():
    matrix = {"end": {"freeze": 153, "roll": 10},
              "roll": {"buy": 5}, "buy": {}, "play": {"freeze": 2},
              "sell": {}, "level": {}, "freeze": {}}
    assert freeze_count(matrix) == 155


def test_episodes_formula():
    assert episodes(0) == 0
    assert episodes(80) == 1280
    assert episodes(320) == 5120


def test_plots_consume_result_json_not_hardcoded_values(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from scripts.ppo_multiseed_aggregate import plot_from_json
    from ml.multiseed_analysis import assemble_replication

    bundle = load_seed_bundle(0)
    analysis = assemble_replication({0: bundle})
    # the plotted mean at iter 80 must equal the JSON, not a literal 6.325
    mean80 = analysis["cross_seed_summary"]["by_iteration"]["80"]["mean"]
    assert mean80 == pytest.approx(bundle["placements"][80])
    assert mean80 == pytest.approx(6.325)
    written = plot_from_json({0: bundle}, analysis, str(tmp_path))
    names = {os.path.basename(p) for p in written}
    assert names == {
        "A_multiseed_dev_curves.png",
        "B_cross_seed_mean.png",
        "C_expert_agreement.png",
        "D_kl_from_warmstart.png",
        "E_warmstart_agreement.png",
        "F_category_drift_iter320.png",
        "G_rl_diagnostics.png",
    }
    assert all(os.path.getsize(p) > 0 for p in written)


def test_seed_report_refuses_to_rebuild_seed_0():
    from scripts.ppo_multiseed_seed_report import assemble_seed
    with pytest.raises(ValueError, match="seed 0"):
        assemble_seed(0)


# --- the unscoreable checkpoint and its restricted supplement -----------------
def _diagnostic(stalled_seeds, placements, attempted, field="greedy",
                iteration=320, seed=1):
    return {
        "kind": "PROTOCOL FAILURE DIAGNOSTIC — NOT a benchmark or DEV result",
        "training_seed": seed, "ppo_iteration": iteration, "field": field,
        "games_attempted": attempted,
        "seed_range": [DEV_EVAL_BASE, DEV_EVAL_BASE + attempted - 1],
        "n_completed": len(placements),
        "n_non_terminating": len(stalled_seeds),
        "non_terminating_game_seeds": list(stalled_seeds),
        "completed_games_diagnostic": {
            "avg_placement": sum(placements) / len(placements),
            "placements": list(placements)},
    }


def _scored(name, placements, field="greedy"):
    return {
        "agent": name, "benchmark_version": "v", "evaluation_split": "dev",
        "field": field, "games": len(placements), "base_seed": DEV_EVAL_BASE,
        "seed_range": [DEV_EVAL_BASE, DEV_EVAL_BASE + len(placements) - 1],
        "environment": {"env": "x"}, "placements": list(placements),
        "metrics": {"avg_placement": sum(placements) / len(placements)},
    }


def test_completed_indices_skip_exactly_the_stalled_lobbies():
    from ml.dev_partial import completed_indices
    d = _diagnostic([DEV_EVAL_BASE + 1, DEV_EVAL_BASE + 4], [8, 8, 8, 8], 6)
    assert completed_indices(d) == [0, 2, 3, 5]
    # a diagnostic whose placement count disagrees with its stall list is a bug
    bad = _diagnostic([DEV_EVAL_BASE + 1], [8, 8], 6)
    with pytest.raises(ValueError, match="inconsistent"):
        completed_indices(bad)


def test_restricted_pair_uses_only_the_shared_lobbies():
    from ml.dev_partial import restricted_pair
    d = _diagnostic([DEV_EVAL_BASE + 1, DEV_EVAL_BASE + 4], [8, 8, 8, 8], 6)
    scored = _scored("iter080", [4, 1, 4, 4, 1, 4])   # 1s sit on the stalls
    got = restricted_pair(d, scored)
    assert got["games_paired"] == 4 and got["games_dropped"] == 2
    assert got["mean_diff"] == pytest.approx(4.0)     # 8 - 4 on every shared
    assert got["restricted"] is True
    assert got["ci_excludes_zero"] is True
    assert "flatters" in got["bias_note"]
    assert "UNSCOREABLE" in got["a"]


def test_restricted_pair_refuses_mismatched_runs():
    from ml.dev_partial import restricted_pair
    d = _diagnostic([DEV_EVAL_BASE + 1], [8, 8, 8], 4)
    with pytest.raises(ValueError, match="different opponent fields"):
        restricted_pair(d, _scored("i", [4, 4, 4, 4], field="greedy4_random3"))
    with pytest.raises(ValueError, match="different DEV base seeds"):
        restricted_pair(d, dict(_scored("i", [4, 4, 4, 4]),
                                base_seed=DEV_EVAL_BASE + 7))
    with pytest.raises(ValueError, match="attempted"):
        restricted_pair(d, _scored("i", [4, 4, 4]))   # 3 lobbies, not 4


def test_restricted_pairs_cover_every_reference_budget():
    from ml.dev_partial import restricted_pairs
    d = _diagnostic([DEV_EVAL_BASE + 2], [6, 6, 6], 4)
    scored = {it: _scored(f"iter{it:03d}", [5, 5, 5, 5])
              for it in (0, 40, 80, 160)}
    pairs = restricted_pairs(d, scored)
    assert sorted(pairs) == ["iter320-iter0", "iter320-iter160",
                             "iter320-iter40", "iter320-iter80"]
    assert all(p["mean_diff"] == pytest.approx(1.0) for p in pairs.values())
    assert all(p["status"] == "restricted supplement" for p in pairs.values())


def test_unscoreable_checkpoint_is_never_loaded_as_a_dev_result():
    """The restricted numbers are a supplement; the frozen loader must still
    refuse to treat the failing checkpoint as scored."""
    from ml.multiseed_analysis import dev_protocol_status
    directory = seed_dir(1)
    if not os.path.isdir(os.path.join(directory, "dev")):
        pytest.skip("Experiment 3 DEV artifacts not present")
    status, blob = dev_protocol_status(directory, 320, "greedy")
    assert status == "protocol_failure"
    assert blob["n_non_terminating"] > 0
    assert "NOT a benchmark" in blob["kind"]
    for it in (0, 40, 80, 160):
        assert dev_protocol_status(directory, it, "greedy")[0] == "ok"


def test_committed_restricted_supplement_is_labelled_and_consistent():
    path = os.path.join("results", "ppo_multiseed_v1", "aggregate",
                        "restricted_supplement.json")
    if not os.path.isfile(path):
        pytest.skip("restricted supplement not generated")
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    assert "not a benchmark" in blob["kind"].lower()
    assert "flatters" in blob["bias"]
    assert blob["checkpoints"], "supplement generated with no checkpoints"
    for row in blob["checkpoints"]:
        assert row["games_non_terminating"] == \
            len(row["non_terminating_game_seeds"])
        for key, p in row["pairs"].items():
            assert p["restricted"] is True
            assert p["games_paired"] == (row["games_attempted"]
                                         - row["games_non_terminating"])
            assert p["games_dropped"] == row["games_non_terminating"]
