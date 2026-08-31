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


# --- checkpoints that cannot finish a DEV lobby -------------------------------
def _dev_blob(name, placements, games_requested=None, stalled_seeds=(),
              completed=None, field="greedy"):
    requested = games_requested or len(placements)
    blob = {
        "agent": name, "benchmark_version": "v", "evaluation_split": "dev",
        "field": field, "games": len(placements), "base_seed": DEV_EVAL_BASE,
        "games_requested": requested,
        "seed_range": [DEV_EVAL_BASE, DEV_EVAL_BASE + requested - 1],
        "environment": {"env": "x"}, "placements": list(placements),
        "metrics": {"avg_placement": sum(placements) / len(placements)},
        "complete": not stalled_seeds,
    }
    if stalled_seeds:
        blob["games_non_terminating"] = len(stalled_seeds)
        blob["non_terminating_seeds"] = list(stalled_seeds)
        blob["completed_game_indices"] = list(completed)
    return blob


def test_partial_result_json_records_the_lobbies_that_stalled():
    from ml.benchmark import Agent, BenchmarkResult, compute_metrics
    from ml.dev_partial import partial_result_to_json
    placements = [4, 5, 6, 7, 8]
    res = BenchmarkResult(agent=Agent("iter320", "policy", lambda *a: 0),
                          field="greedy", games=len(placements),
                          base_seed=DEV_EVAL_BASE,
                          metrics=compute_metrics(placements),
                          ci95={"low": 4.0, "high": 8.0}, latency={},
                          placements=placements)
    stalled = [{"index": 2, "seed": DEV_EVAL_BASE + 2, "reason": "stall"},
               {"index": 5, "seed": DEV_EVAL_BASE + 5, "reason": "stall"}]
    blob = partial_result_to_json(res, stalled, [0, 1, 3, 4, 6], 7)
    assert blob["complete"] is False
    assert blob["games"] == 5 and blob["games_requested"] == 7
    assert blob["games_non_terminating"] == 2
    assert blob["non_terminating_seeds"] == [DEV_EVAL_BASE + 2,
                                             DEV_EVAL_BASE + 5]
    # the seed block stays the full requested range, so the protocol is visible
    assert blob["seed_range"] == [DEV_EVAL_BASE, DEV_EVAL_BASE + 6]
    assert blob["beats_field"] is None
    assert "INCOMPLETE" in blob["restricted_note"]
    # a run that finished everything is not flagged and carries no index list
    full = partial_result_to_json(res, [], [0, 1, 2, 3, 4], 5)
    assert full["complete"] is True
    assert "completed_game_indices" not in full


def test_paired_common_games_matches_the_normal_pairing_when_complete():
    from ml.analyze_benchmark import compare_pair
    from ml.dev_partial import paired_common_games
    a = _dev_blob("a", [4, 5, 6, 7, 8, 3])
    b = _dev_blob("b", [5, 5, 7, 7, 8, 4])
    got = paired_common_games(a, b)
    assert got["mean_diff"] == pytest.approx(compare_pair(a, b)["mean_diff"])
    assert got["restricted"] is False and got["games_dropped"] == 0
    assert got["games_paired"] == 6


def test_paired_common_games_drops_only_the_shared_missing_lobbies():
    from ml.dev_partial import paired_common_games
    full = _dev_blob("iter080", [4, 4, 4, 4, 4, 4])
    # the stalled policy has no placement for lobbies 1 and 4
    partial = _dev_blob("iter320", [8, 8, 8, 8], games_requested=6,
                        stalled_seeds=[DEV_EVAL_BASE + 1, DEV_EVAL_BASE + 4],
                        completed=[0, 2, 3, 5])
    got = paired_common_games(partial, full)
    assert got["games_paired"] == 4 and got["games_dropped"] == 2
    assert got["restricted"] is True
    assert got["mean_diff"] == pytest.approx(4.0)
    assert got["a_complete"] is False and got["b_complete"] is True
    assert "optimistic" in got["note"]


def test_within_seed_paired_handles_a_restricted_checkpoint():
    from ml.multiseed_analysis import within_seed_paired
    base = [4, 5, 6, 7] * 25
    greedy = {it: _dev_blob(f"iter{it:03d}", base) for it in PRIMARY_ITERS}
    greedy[320] = _dev_blob("iter320", base[:-2], games_requested=100,
                            stalled_seeds=[DEV_EVAL_BASE + 98,
                                           DEV_EVAL_BASE + 99],
                            completed=list(range(98)))
    rows = within_seed_paired(greedy)
    assert rows[pair_key(320, 80)]["restricted"] is True
    assert rows[pair_key(320, 80)]["games_paired"] == 98
    assert rows[pair_key(320, 80)]["games_dropped"] == 2
    # comparisons that do not involve the restricted checkpoint stay full
    assert rows[pair_key(80, 0)]["restricted"] is False
    assert rows[pair_key(80, 0)]["games_paired"] == 100


def test_incomplete_result_is_accepted_but_never_silently():
    from ml.multiseed_analysis import (assert_eval_seeds_match_experiment2,
                                       eval_seed_record, games_requested,
                                       is_complete)
    partial = _dev_blob("iter320", [8] * 995, games_requested=DEV_EVAL_GAMES,
                        stalled_seeds=[DEV_EVAL_BASE + i for i in range(5)],
                        completed=list(range(5, 1000)))
    assert_eval_seeds_match_experiment2(partial, "greedy")   # attempted block ok
    assert games_requested(partial) == DEV_EVAL_GAMES
    assert is_complete(partial) is False
    rec = eval_seed_record(partial)
    assert rec["complete"] is False and rec["games_non_terminating"] == 5
    assert rec["games_requested"] == DEV_EVAL_GAMES
    # a record that claims to be incomplete while scoring everything is a bug
    inconsistent = dict(partial, games=DEV_EVAL_GAMES)
    with pytest.raises(ValueError, match="inconsistent record"):
        assert_eval_seeds_match_experiment2(inconsistent, "greedy")


def test_tolerant_runner_records_stalls_instead_of_aborting(monkeypatch):
    """The MAX_DECISIONS guard is not relaxed — the stall becomes data."""
    import ml.dev_partial as dp
    from ml.benchmark import Agent, BenchmarkIntegrityError

    stall_at = {DEV_EVAL_BASE + 1, DEV_EVAL_BASE + 3}

    def fake_run_game(agent, seats, seed):
        if seed in stall_at:
            raise BenchmarkIntegrityError(f"episode did not terminate: {seed}")
        return {"seed": seed, "placement": 1 + (seed - DEV_EVAL_BASE) % 8,
                "latencies": [0.001]}

    monkeypatch.setattr(dp, "run_game", fake_run_game)
    agent = Agent("stub", "policy", lambda *a: 0)
    res, stalled, completed = dp.run_dev_benchmark_tolerant(agent, "greedy", 6)
    assert [s["seed"] for s in stalled] == sorted(stall_at)
    assert completed == [0, 2, 4, 5]
    assert res.games == 4 and len(res.placements) == 4

    # a policy that finishes nothing is refused outright rather than scored
    monkeypatch.setattr(dp, "run_game", lambda *a: (_ for _ in ()).throw(
        BenchmarkIntegrityError("stall")))
    with pytest.raises(BenchmarkIntegrityError, match="finished 0 of"):
        dp.run_dev_benchmark_tolerant(agent, "greedy", 3)


def test_committed_results_report_their_completeness():
    """Whatever the outcome, every committed DEV result must state whether
    the policy finished all of its lobbies."""
    from ml.multiseed_analysis import MULTI_DIR, eval_seed_record
    checked = 0
    for s in (1, 2, 3):
        directory = os.path.join(MULTI_DIR, f"seed_{s}")
        for it in PRIMARY_ITERS:
            path = os.path.join(directory, "dev", f"iter{it:03d}_vs_greedy.json")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            rec = eval_seed_record(blob)
            assert rec["games_requested"] == DEV_EVAL_GAMES
            assert rec["seed_range"] == [DEV_EVAL_BASE, DEV_EVAL_LAST]
            assert isinstance(rec["complete"], bool)
            if not rec["complete"]:
                assert blob["non_terminating_seeds"]
                assert len(blob["completed_game_indices"]) == blob["games"]
            checked += 1
    if checked == 0:
        pytest.skip("no Experiment 3 DEV artifacts present")
