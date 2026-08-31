"""Experiment 3 protocol-failure machinery.

Seed 1's iteration-320 policy loops forever in some DEV lobbies, so the
benchmark integrity guard refuses to score it (a silent 8th place would
corrupt the numbers). These tests pin how the multi-seed analysis represents
such an unscoreable checkpoint: explicit statuses everywhere, no fabricated
placements, and the documented ordering extension (unscoreable == strictly
worse than any scoreable checkpoint) applied only where stated."""

import json
import os

import pytest

np = pytest.importorskip("numpy")

from ml.multiseed_analysis import (WITHIN_SEED_PAIRS,  # noqa: E402
                                   classify_ushape, cross_seed_table,
                                   dev_protocol_status, pair_key,
                                   question_b, within_seed_paired,
                                   write_json)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED1_DEV = os.path.join(REPO, "results", "ppo_multiseed_v1", "seed_1", "dev")


def _scored_blob(avg, n=4):
    # minimal single-result DEV blob compare_pair can pair
    placements = [max(1, min(8, int(avg))) for _ in range(n)]
    return {"agent": "x", "benchmark_version": "dev", "field": "greedy",
            "games": n, "base_seed": 10_550_000,
            "seed_range": [10_550_000, 10_550_000 + n - 1],
            "environment": {}, "metrics": {"avg_placement": avg},
            "placements": placements}


# --- dev_protocol_status --------------------------------------------------------
def test_dev_protocol_status_prefers_scored_result(tmp_path):
    d = str(tmp_path)
    write_json(os.path.join(d, "dev", "iter080_vs_greedy.json"),
               _scored_blob(6.5))
    status, blob = dev_protocol_status(d, 80, "greedy")
    assert status == "ok" and blob["games"] == 4


def test_dev_protocol_status_reads_failure_diagnostic(tmp_path):
    d = str(tmp_path)
    write_json(os.path.join(d, "dev",
                            "iter320_vs_greedy.protocol_failure.json"),
               {"kind": "PROTOCOL FAILURE DIAGNOSTIC — NOT a benchmark or "
                        "DEV result",
                "n_non_terminating": 5, "games_attempted": 1000})
    status, blob = dev_protocol_status(d, 320, "greedy")
    assert status == "protocol_failure"
    assert blob["n_non_terminating"] == 5


def test_dev_protocol_status_raises_when_nothing_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        dev_protocol_status(str(tmp_path), 160, "greedy")


# --- paired comparisons with an unscoreable side --------------------------------
def test_within_seed_paired_marks_unscoreable_pairs():
    greedy = {it: _scored_blob(6.5) for it in (0, 40, 80, 160)}
    reason = {320: "iteration 320 failed the frozen DEV protocol"}
    rows = within_seed_paired(greedy, unscoreable=reason)
    assert set(rows) == {pair_key(it, ref) for it, ref in WITHIN_SEED_PAIRS}
    for (it, ref) in WITHIN_SEED_PAIRS:
        row = rows[pair_key(it, ref)]
        if 320 in (it, ref):
            assert row["status"] == "unscoreable"
            assert "mean_diff" not in row          # nothing fabricated
            assert row["unscoreable_iterations"] == [320]
        else:
            assert row["status"] == "ok"
            assert "mean_diff" in row and "ci95" in row


# --- the documented ordering extension in classify_ushape ------------------------
def _placements(p0, p40, p80, p160, p320=None):
    p = {0: p0, 40: p40, 80: p80, 160: p160}
    if p320 is not None:
        p[320] = p320
    return p


def test_classify_unscoreable_with_mid_gain_is_u_like_by_extension():
    shape = classify_ushape(_placements(6.5, 6.6, 6.3, 6.4),
                            unscoreable=[320])
    assert shape["label"].startswith("other (iter320 unscoreable")
    assert shape["extension_label"] == "U-like / transient improvement"
    assert shape["late_regression_mean"] is True
    assert shape["unscoreable_iterations"] == [320]


def test_classify_unscoreable_without_mid_gain_is_degradation_by_extension():
    shape = classify_ushape(_placements(6.5, 6.8, 7.1, 6.9),
                            unscoreable=[320])
    assert shape["label"].startswith("other (iter320 unscoreable")
    assert shape["extension_label"] == "monotonic degradation"


def test_classify_rejects_unscoreable_iterations_other_than_320():
    with pytest.raises(ValueError, match="only iteration 320"):
        classify_ushape(_placements(6.5, 6.6, 6.3, 6.4, 6.7),
                        unscoreable=[80])


def test_classify_full_curve_reports_no_unscoreable():
    shape = classify_ushape(_placements(6.5, 6.6, 6.3, 6.4, 6.7))
    assert shape["unscoreable_iterations"] == []
    assert shape["extension_label"] == shape["label"]


# --- cross-seed table and question B with a missing number ----------------------
def _bundle(placements, unscoreable=None, paired=None):
    return {"placements": placements, "unscoreable": unscoreable or {},
            "paired": paired or {}}


def test_cross_seed_table_excludes_unscoreable_seed_from_stats():
    bundles = {
        0: _bundle({0: 6.5, 40: 6.7, 80: 6.3, 160: 6.4, 320: 6.6}),
        1: _bundle({0: 6.5, 40: 6.8, 80: 7.1, 160: 6.9},
                   unscoreable={320: "protocol failure"}),
    }
    table = cross_seed_table(bundles)
    row = table["by_iteration"]["320"]
    assert row["per_seed"] == {0: 6.6}
    assert row["unscoreable_seeds"] == {1: "protocol failure"}
    assert row["n"] == 1 and row["mean"] == pytest.approx(6.6)
    full = table["by_iteration"]["80"]
    assert full["per_seed"] == {0: 6.3, 1: 7.1}
    assert full["mean"] == pytest.approx(6.7)


def test_question_b_counts_protocol_failure_as_its_own_category():
    ok_pair = {"status": "ok", "mean_diff": 0.3, "ci95": [0.1, 0.5],
               "ci_excludes_zero": True}
    bad_pair = {"status": "unscoreable", "reason": "non-terminating",
                "iteration": 320, "reference_iteration": 80}
    bundles = {
        0: _bundle({}, paired={pair_key(320, 80): ok_pair}),
        1: _bundle({}, paired={pair_key(320, 80): bad_pair}),
    }
    qb = question_b(bundles)
    assert qb["n_regress"] == 1
    assert qb["n_regress_protocol_failure"] == 1
    assert qb["n_regress_including_protocol_failures"] == 2
    fail_row = next(r for r in qb["per_seed"] if r["training_seed"] == 1)
    assert fail_row["direction"] == "regress_protocol_failure"
    assert fail_row["mean_diff"] is None


# --- the committed seed-1 artifacts ----------------------------------------------
_HAVE_SEED1 = os.path.isfile(
    os.path.join(SEED1_DEV, "iter320_vs_greedy.protocol_failure.json"))


@pytest.mark.skipif(not _HAVE_SEED1, reason="seed 1 artifacts not present")
def test_seed1_iter320_failure_artifact_is_labeled_and_unscored():
    # no scored DEV result may exist alongside the failure diagnostic
    assert not os.path.isfile(
        os.path.join(SEED1_DEV, "iter320_vs_greedy.json"))
    blob = json.load(open(
        os.path.join(SEED1_DEV, "iter320_vs_greedy.protocol_failure.json")))
    assert blob["kind"].startswith("PROTOCOL FAILURE DIAGNOSTIC")
    assert blob["n_non_terminating"] >= 1
    assert (blob["n_completed"] + blob["n_non_terminating"]
            == blob["games_attempted"] == 1000)
    assert blob["seed_range"] == [10_550_000, 10_550_999]
    # every non-terminating lobby is one of the protocol's own game seeds
    for s in blob["non_terminating_game_seeds"]:
        assert 10_550_000 <= s <= 10_550_999
    # the loop probes identify a dominant repeated action
    assert blob["loop_probes"], "loop probes missing"
    for probe in blob["loop_probes"]:
        assert probe["tail_share"] >= 0.5