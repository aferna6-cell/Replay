"""Focused tests for Experiment 3 multi-seed aggregation / isolation."""

import json
import os

import pytest

from ml import seeds
from ml.model_fingerprint import checkpoint_fingerprint
from ml.multiseed_analysis import (
    EXP2_WARM_START_PARAMETER_SHA256,
    PAIRED_CONTRASTS,
    PRIMARY_ITERS,
    ci_excludes_zero,
    classify_u_shape,
    episodes,
    replication_questions,
    summarize_across_seeds,
    within_seed_paired,
)


def test_exp2_warm_start_hash_constant():
    assert EXP2_WARM_START_PARAMETER_SHA256 == (
        "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b")


def test_planned_ppo_seeds_outside_dev_and_test():
    for s in (0, 1, 2, 3):
        lo = seeds.ppo_episode_seed(s, 1)
        hi = seeds.ppo_episode_seed(s, 320 * 16)
        assert not seeds.overlaps_eval_range(lo, hi)
        assert not seeds.overlaps_dev_range(lo, hi)
        assert seeds.check_training_range(f"t{s}", lo, hi) is False


def test_dev_eval_seed_range_matches_exp2():
    seeds.validate_dev_range(seeds.DEV_SEED_START, 1000)
    assert seeds.DEV_SEED_START == 10_550_000
    assert seeds.eval_game_seed(seeds.DEV_SEED_START, 999) == 10_550_999


def test_test_interval_untouched_by_exp3_helpers():
    with pytest.raises(ValueError):
        seeds.validate_dev_range(seeds.EVAL_SEED_START, 10)
    assert seeds.EVAL_SEED_START == 10_250_000


def test_episodes_helper():
    assert [episodes(i) for i in PRIMARY_ITERS] == [0, 640, 1280, 2560, 5120]


def _fake_result(agent, placements, field="greedy"):
    return {
        "agent": agent,
        "benchmark_version": "Replay DEV Evaluation (Benchmark v1 machinery)",
        "field": field,
        "games": len(placements),
        "base_seed": seeds.DEV_SEED_START,
        "seed_range": [seeds.DEV_SEED_START,
                       seeds.DEV_SEED_START + len(placements) - 1],
        "environment": {"env": "BGEnv"},
        "placements": list(placements),
        "metrics": {
            "avg_placement": sum(placements) / len(placements),
            "median_placement": sorted(placements)[len(placements) // 2],
            "std_placement": 0.0,
            "top4_rate": 0.0,
            "win_rate": 0.0,
            "placement_counts": {},
            "games": len(placements),
        },
        "avg_placement_ci95": {"low": 0, "high": 8},
    }


def test_within_seed_paired_contrasts_and_sign():
    g0 = _fake_result("iter000", [6, 7, 8, 5] * 10)
    g80 = _fake_result("iter080", [5, 6, 7, 4] * 10)
    g40 = _fake_result("iter040", [6, 7, 8, 5] * 10)
    g160 = _fake_result("iter160", [6, 7, 8, 5] * 10)
    g320 = _fake_result("iter320", [7, 8, 9, 6] * 10)
    greedy = {0: g0, 40: g40, 80: g80, 160: g160, 320: g320}
    rows = within_seed_paired(greedy)
    assert [(r["iteration"], r["reference_iteration"]) for r in rows] == \
        PAIRED_CONTRASTS
    mid = next(r for r in rows if r["label"] == "iter80-iter0")
    assert mid["mean_diff"] == pytest.approx(-1.0)
    assert ci_excludes_zero(mid["ci95"])
    late = next(r for r in rows if r["label"] == "iter320-iter0")
    assert late["mean_diff"] == pytest.approx(1.0)


def test_u_shape_classification_exp2_like():
    def row(a, b, mean, lo, hi):
        return {"iteration": a, "reference_iteration": b, "mean_diff": mean,
                "ci95": [lo, hi], "label": f"iter{a}-iter{b}"}

    rows = [
        row(40, 0, 0.2, 0.1, 0.3),
        row(80, 0, -0.23, -0.39, -0.06),
        row(160, 0, -0.12, -0.25, 0.01),
        row(320, 0, 0.05, -0.1, 0.2),
        row(80, 40, -0.4, -0.6, -0.2),
        row(160, 40, -0.3, -0.45, -0.2),
        row(320, 40, -0.15, -0.3, 0.0),
        row(160, 80, 0.1, -0.05, 0.25),
        row(320, 80, 0.28, 0.1, 0.45),
    ]
    u = classify_u_shape(rows)
    assert u["class"] == "u_like_transient_improvement"


def test_u_shape_flat_and_monotonic():
    def row(a, b, mean, lo, hi):
        return {"iteration": a, "reference_iteration": b, "mean_diff": mean,
                "ci95": [lo, hi], "label": f"iter{a}-iter{b}"}

    flat = [row(a, b, 0.01, -0.1, 0.1) for a, b in PAIRED_CONTRASTS]
    assert classify_u_shape(flat)["class"] == "mostly_flat_noisy"

    mono = [
        row(40, 0, -0.1, -0.2, -0.05),
        row(80, 0, -0.2, -0.3, -0.1),
        row(160, 0, -0.3, -0.4, -0.2),
        row(320, 0, -0.4, -0.5, -0.3),
        row(80, 40, -0.1, -0.2, -0.05),
        row(160, 40, -0.2, -0.3, -0.1),
        row(320, 40, -0.3, -0.4, -0.2),
        row(160, 80, -0.1, -0.2, -0.05),
        row(320, 80, -0.2, -0.3, -0.1),
    ]
    assert classify_u_shape(mono)["class"] == "monotonic_improvement"


def test_cross_seed_summary_math_keeps_individuals():
    curves = {
        0: [{"iteration": it, "greedy_avg": 6.0 + it / 1000,
             "expert_agreement": 0.8, "warmstart_agreement": 0.9,
             "kl_from_warmstart": 0.1, "mixed_avg": 4.0}
            for it in PRIMARY_ITERS],
        1: [{"iteration": it, "greedy_avg": 7.0 + it / 1000,
             "expert_agreement": 0.7, "warmstart_agreement": 0.8,
             "kl_from_warmstart": 0.2, "mixed_avg": 4.5}
            for it in PRIMARY_ITERS],
    }
    summary = summarize_across_seeds(curves)
    s0 = summary[0]
    assert s0["n_seeds"] == 2
    assert s0["greedy_avg_mean"] == pytest.approx(6.5)
    assert s0["greedy_avg_min"] == 6.0
    assert s0["greedy_avg_max"] == 7.0
    assert {p["training_seed"] for p in s0["per_seed"]} == {0, 1}


def test_replication_questions_counts():
    def rows(mid_diff, late_diff):
        out = []
        for a, b in PAIRED_CONTRASTS:
            if (a, b) == (80, 0):
                m = mid_diff
            elif (a, b) == (320, 80):
                m = late_diff
            else:
                m = 0.0
            if abs(m) >= 0.1:
                lo, hi = (m - 0.02, m + 0.02)
            else:
                lo, hi = (-0.1, 0.1)
            out.append({"iteration": a, "reference_iteration": b,
                        "mean_diff": m, "ci95": [lo, hi],
                        "label": f"iter{a}-iter{b}"})
        return out

    per = {
        0: rows(-0.2, 0.3),
        1: rows(-0.15, 0.25),
        2: rows(0.1, -0.1),
        3: rows(-0.05, 0.02),
    }
    for r in per[3]:
        if r["label"] in ("iter80-iter0", "iter320-iter80"):
            r["ci95"] = [-0.1, 0.1]
    q = replication_questions(per)
    assert q["question_A_iter80_minus_iter0"]["n_improve"] == 3
    assert q["question_A_iter80_minus_iter0"]["n_worsen"] == 1
    assert 0 in q["question_B_iter320_minus_iter80"]["seeds_regress"]
    assert 1 in q["question_B_iter320_minus_iter80"]["seeds_regress"]


def test_warm_start_file_matches_exp2_when_present():
    path = "ml/policy_bc.pt"
    if not os.path.isfile(path):
        pytest.skip("warm-start checkpoint not present in this environment")
    fp = checkpoint_fingerprint(path)
    assert fp["parameter_sha256"] == EXP2_WARM_START_PARAMETER_SHA256


def test_aggregate_plots_read_from_json(tmp_path, monkeypatch):
    """Plots must be driven by JSON curve values, not hard-coded series."""
    root = tmp_path / "results" / "ppo_multiseed_v1"
    for s in (0, 1, 2, 3):
        d = root / f"seed_{s}"
        (d / "dev").mkdir(parents=True)
        curve = []
        for it in PRIMARY_ITERS:
            placements = [5 + (it // 40) % 3] * 20
            for field, games in (("greedy", 20), ("greedy4_random3", 20)):
                blob = _fake_result(f"s{s}_iter{it}", placements[:games], field)
                blob["avg_placement_ci95"] = {"low": 5.0, "high": 7.0}
                (d / "dev" / f"iter{it:03d}_vs_{field}.json").write_text(
                    json.dumps(blob))
            curve.append({
                "training_seed": s, "iteration": it,
                "cumulative_episodes": episodes(it),
                "greedy_avg": 6.0 + 0.1 * s + it / 1000,
                "greedy_ci95": {"low": 5, "high": 7},
                "greedy_median": 6, "greedy_std": 1,
                "greedy_top4": 0.1, "greedy_win": 0.05,
                "greedy_placement_counts": {},
                "mixed_avg": 4.0, "mixed_ci95": {"low": 3, "high": 5},
                "mixed_top4": 0.5, "mixed_win": 0.1,
                "expert_agreement": 0.8 - it / 1000,
                "warmstart_agreement": 1.0 - it / 800,
                "kl_from_warmstart": it / 500,
                "corpus_entropy": 1.0, "value_mean": 0.0, "value_std": 0.1,
                "parameter_sha256": "x", "checkpoint_sha256": "y",
                "expert_disagreement_by_category": {
                    c: 0.1 for c in
                    ["buy", "play", "sell", "roll", "level", "freeze", "end"]},
                "drift_contribution_by_category": {},
            })
        (d / "learning_curve.json").write_text(json.dumps({
            "training_seed": s, "curve": curve}))
        drift_ck = []
        cat_ck = []
        for it in PRIMARY_ITERS:
            drift_ck.append({
                "checkpoint": f"iter_{it:03d}.pt",
                "expert_agreement": 0.8, "warmstart_agreement": 0.9,
                "kl_from_warmstart_mean": 0.1, "entropy_mean": 1.0,
                "value_mean": 0.0, "value_std": 0.1,
                "parameter_sha256": "x", "checkpoint_sha256": "y",
            })
            cat_ck.append({
                "checkpoint": f"iter_{it:03d}.pt",
                "vs_expert": {
                    "overall_agreement": 0.8,
                    "disagreement_share_by_category": {
                        c: 0.1 for c in
                        ["buy", "play", "sell", "roll", "level", "freeze", "end"]
                    },
                    "contribution_to_total_drift": {},
                    "confusion_matrix": {
                        c: {"freeze": 0} for c in
                        ["buy", "play", "sell", "roll", "level", "freeze", "end"]},
                    "top_transitions": [],
                },
            })
        (d / "policy_drift.json").write_text(json.dumps({
            "corpus": {"fingerprint_sha256": "2ec217b353bd" + "0" * 52},
            "checkpoints": drift_ck}))
        (d / "action_category_drift.json").write_text(json.dumps({
            "checkpoints": cat_ck}))
        blocks = {b: {k: 0.1 for k in (
            "adv_mean_abs", "value_explained_variance", "entropy", "clip_frac",
            "adv_mean", "adv_std", "adv_frac_positive", "adv_frac_negative",
            "return_mean", "return_std", "value_pred_mean", "value_pred_std",
            "placement_std", "shaping_reward_sum", "terminal_reward_sum",
            "approx_kl", "grad_norm", "pi_loss", "v_loss")}
            for b in ("iters_1_40", "iters_41_160", "iters_161_320")}
        (d / "rl_signal.json").write_text(json.dumps({"blocks": blocks}))

    import importlib
    import scripts.ppo_multiseed_aggregate as agg
    importlib.reload(agg)
    monkeypatch.setattr(agg, "ROOT", str(root))
    monkeypatch.setattr(agg, "AGG", str(root / "aggregate"))
    assert agg.main() == 0
    plots = root / "aggregate" / "plots"
    for name in ("A_multiseed_dev_greedy.png", "B_mean_with_individuals.png",
                 "C_expert_agreement.png", "D_kl_from_warmstart.png",
                 "E_warmstart_agreement.png",
                 "F_category_disagreement_iter320.png",
                 "G_rl_signal_blocks.png"):
        assert (plots / name).is_file(), name
    summary = json.loads(
        (root / "aggregate" / "cross_seed_summary.json").read_text())
    assert summary["per_seed_curves"]["0"][0]["greedy_avg"] == pytest.approx(6.0)
