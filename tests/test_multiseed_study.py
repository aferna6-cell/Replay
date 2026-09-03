"""Tests for Experiment 3: Multi-Seed PPO Budget Replication machinery,
cross-seed aggregation, paired comparisons, seed isolation, and U-shape classification.
"""

import json
import math
import os
import statistics as st

import pytest

np = pytest.importorskip("numpy")

from ml import seeds
from ml.analyze_benchmark import compare_pair, load_result
from ml.model_fingerprint import checkpoint_fingerprint, checkpoint_parameter_sha256
from ml.seeds import (DEV_SEED_START, DEV_SEED_END, EVAL_SEED_START, EVAL_SEED_END,
                      overlaps_dev_range, overlaps_eval_range, ppo_episode_seed)
from scripts.ppo_multiseed_report import classify_ushape

BASE_DIR = "results/ppo_multiseed_v1"
AGG_DIR = os.path.join(BASE_DIR, "aggregate")
PLOTS_DIR = os.path.join(AGG_DIR, "plots")
EXP2_DIR = "results/ppo_budget_v1"


def test_warm_start_parameter_hash_identical_across_seeds():
    """Verify that all seed iter_000 checkpoints match the warm-start parameter hash."""
    warm_fp = checkpoint_fingerprint("ml/policy_bc.pt")
    warm_sha = warm_fp["parameter_sha256"]
    assert warm_sha is not None and len(warm_sha) == 64

    for s in [1, 2, 3]:
        ckpt_path = f"{BASE_DIR}/seed_{s}/checkpoints/iter_000.pt"
        if os.path.exists(ckpt_path):
            fp = checkpoint_fingerprint(ckpt_path)
            assert fp["parameter_sha256"] == warm_sha, (
                f"Seed {s} iter_000 parameter hash {fp['parameter_sha256']} "
                f"does not match warm start {warm_sha}"
            )


def test_training_seed_metadata_and_ranges():
    """Verify training seed ranges and episode calculations for 320 iters x 16 episodes."""
    for s in [0, 1, 2, 3]:
        first_ep_seed = ppo_episode_seed(s, 1)
        last_ep_seed = ppo_episode_seed(s, 320 * 16)
        expected_first = s * 1_000_003 + 1
        expected_last = s * 1_000_003 + 5120
        assert first_ep_seed == expected_first
        assert last_ep_seed == expected_last
        assert last_ep_seed - first_ep_seed == 5119


def test_dev_test_seed_isolation():
    """Verify that all training episode seeds for seeds 0, 1, 2, 3 are strictly disjoint
    from both DEV and TEST evaluation intervals."""
    for s in [0, 1, 2, 3]:
        first_ep = ppo_episode_seed(s, 1)
        last_ep = ppo_episode_seed(s, 5120)
        assert not overlaps_dev_range(first_ep, last_ep), (
            f"Seed {s} training range [{first_ep}, {last_ep}] touches DEV interval!"
        )
        assert not overlaps_eval_range(first_ep, last_ep), (
            f"Seed {s} training range [{first_ep}, {last_ep}] touches TEST interval!"
        )


def test_dev_evaluation_seed_equality_across_runs():
    """Verify that all DEV evaluations use the exact identical 1000 greedy seeds (10550000..10550999)
    and 500 mixed seeds (10550000..10550499) across all seeds and iterations."""
    for s in [1, 2, 3]:
        for it in [0, 40, 80, 160, 320]:
            g_path = f"{BASE_DIR}/seed_{s}/dev/iter{it:03d}_vs_greedy.json"
            if os.path.exists(g_path):
                g_data = json.load(open(g_path))
                assert g_data["games"] == 1000
                assert g_data["seed_range"] == [10550000, 10550999]
                assert len(g_data["placements"]) == 1000
                assert g_data["evaluation_split"] == "dev"

            m_path = f"{BASE_DIR}/seed_{s}/dev/iter{it:03d}_vs_greedy4_random3.json"
            if os.path.exists(m_path):
                m_data = json.load(open(m_path))
                assert m_data["games"] == 500
                assert m_data["seed_range"] == [10550000, 10550499]
                assert len(m_data["placements"]) == 500
                assert m_data["evaluation_split"] == "dev"


def test_cross_seed_result_loading():
    """Verify that cross_seed_summary.json loads and contains all 4 seeds and 5 iterations."""
    summary_path = f"{AGG_DIR}/cross_seed_summary.json"
    assert os.path.exists(summary_path), f"Missing {summary_path}"
    data = json.load(open(summary_path))

    assert data["seeds"] == [0, 1, 2, 3]
    assert data["iterations"] == [0, 40, 80, 160, 320]
    for s in [0, 1, 2, 3]:
        seed_key = f"seed_{s}"
        assert seed_key in data["table_by_seed"]
        for it in [0, 40, 80, 160, 320]:
            assert f"iter_{it:03d}" in data["table_by_seed"][seed_key]
            val = data["table_by_seed"][seed_key][f"iter_{it:03d}"]
            assert 1.0 <= val <= 8.0


def test_aggregate_summary_math():
    """Verify that by_budget calculations match statistics module exactly."""
    summary_path = f"{AGG_DIR}/cross_seed_summary.json"
    data = json.load(open(summary_path))

    for it in [0, 40, 80, 160, 320]:
        b = data["by_budget"][str(it)]
        vals = [b["seed_placements"][f"seed_{s}"] for s in [0, 1, 2, 3]]
        assert b["mean"] == pytest.approx(st.mean(vals), rel=1e-5)
        assert b["median"] == pytest.approx(st.median(vals), rel=1e-5)
        assert b["min"] == min(vals)
        assert b["max"] == max(vals)
        assert b["std"] == pytest.approx(st.stdev(vals), rel=1e-5)


def test_within_seed_paired_comparisons_math():
    """Verify within-seed paired results artifact math and conventions."""
    paired_path = f"{AGG_DIR}/paired_results.json"
    assert os.path.exists(paired_path), f"Missing {paired_path}"
    data = json.load(open(paired_path))

    for s in [0, 1, 2, 3]:
        seed_pairs = data["by_seed"][f"seed_{s}"]
        for pair_key, p_data in seed_pairs.items():
            diff = p_data["mean_diff"]
            ci = p_data["ci95"]
            assert ci[0] <= ci[1], f"CI not ordered for {pair_key} in seed {s}"
            assert p_data["n_games"] == 1000
            # Check verdict logic
            if ci[0] <= 0.0 <= ci[1]:
                assert "no clear difference" in p_data["verdict"]


def test_ushape_classification_logic():
    """Unit test the descriptive U-shape classification function with synthetic inputs."""
    # 1. Synthetic U-shape: improves at iter 80, regresses at iter 320
    synth_u = {
        "iter040_vs_iter000": {"ci95": [-0.1, 0.1], "mean_diff": 0.0},
        "iter080_vs_iter000": {"ci95": [-0.35, -0.05], "mean_diff": -0.20},
        "iter160_vs_iter000": {"ci95": [-0.25, 0.01], "mean_diff": -0.12},
        "iter320_vs_iter000": {"ci95": [-0.08, 0.15], "mean_diff": 0.04},
        "iter320_vs_iter080": {"ci95": [0.10, 0.40], "mean_diff": 0.25},
    }
    c_u = classify_ushape([], synth_u)
    assert c_u["classification"] == "U-like / transient improvement"

    # 2. Synthetic Flat/Noisy: all CIs vs iter0 include zero
    synth_flat = {
        "iter040_vs_iter000": {"ci95": [-0.05, 0.05], "mean_diff": 0.0},
        "iter080_vs_iter000": {"ci95": [-0.08, 0.08], "mean_diff": 0.0},
        "iter160_vs_iter000": {"ci95": [-0.06, 0.06], "mean_diff": 0.0},
        "iter320_vs_iter000": {"ci95": [-0.07, 0.07], "mean_diff": 0.0},
        "iter320_vs_iter080": {"ci95": [-0.05, 0.05], "mean_diff": 0.0},
    }
    c_flat = classify_ushape([], synth_flat)
    assert c_flat["classification"] == "mostly flat/noisy"

    # 3. Synthetic Transient Degradation (Other): worsens at iter 80
    synth_degrade = {
        "iter040_vs_iter000": {"ci95": [-0.02, 0.10], "mean_diff": 0.04},
        "iter080_vs_iter000": {"ci95": [0.30, 0.60], "mean_diff": 0.45},
        "iter160_vs_iter000": {"ci95": [0.25, 0.55], "mean_diff": 0.40},
        "iter320_vs_iter000": {"ci95": [-0.08, 0.08], "mean_diff": 0.0},
        "iter320_vs_iter080": {"ci95": [-0.55, -0.25], "mean_diff": -0.40},
    }
    c_deg = classify_ushape([], synth_degrade)
    assert c_deg["classification"] == "other"


def test_replication_analysis_questions():
    """Verify replication questions A and B match computed statistics."""
    rep_path = f"{AGG_DIR}/replication_analysis.json"
    assert os.path.exists(rep_path), f"Missing {rep_path}"
    data = json.load(open(rep_path))

    # Question A checks
    qa = data["question_a_1280_episodes"]["summary"]
    assert qa["seeds_improving_point"] == 2  # Seed 0 and Seed 1 improved by point estimate
    assert qa["seeds_worsening_point"] == 2  # Seeds 2 and 3 worsened by point estimate
    assert qa["seeds_significant_improvement"] == 1  # Only Seed 0 improved with CI < 0
    assert qa["seeds_significant_worsening"] == 1    # Seed 3 degraded with CI > 0
    assert qa["seeds_ci_excluding_zero"] == 2

    # Question B checks
    qb = data["question_b_decay_after_improvement"]["summary"]
    assert qb["seeds_regressing"] == 1  # Only Seed 0 regressed
    assert qb["seeds_improving"] == 1   # Seed 3 improved after its iter80 dip
    assert qb["seeds_indistinguishable"] == 2


def test_plots_exist_and_non_empty():
    """Verify all 7 required plots (A-G) are created and valid non-empty files."""
    plots = [
        "A_multiseed_dev_learning_curves.png",
        "B_cross_seed_mean_curve.png",
        "C_expert_agreement.png",
        "D_kl_from_warmstart.png",
        "E_warmstart_agreement.png",
        "F_action_category_drift.png",
        "G_ppo_diagnostics.png",
    ]
    for p in plots:
        path = os.path.join(PLOTS_DIR, p)
        assert os.path.exists(path), f"Missing plot: {path}"
        assert os.path.getsize(path) > 1000, f"Plot {path} is suspiciously small ({os.path.getsize(path)} bytes)"


def test_manifest_schema_and_integrity():
    """Verify manifest.json contents, software versions, and frozen recipe elements."""
    m_path = f"{BASE_DIR}/manifest.json"
    assert os.path.exists(m_path), f"Missing {m_path}"
    m = json.load(open(m_path))

    assert m["experiment"] == "Replay Experiment 3 — Multi-Seed PPO Budget Replication"
    assert m["single_variable"] == "PPO training seed (seeds 0, 1, 2, 3)"
    assert m["seeds"] == [0, 1, 2, 3]
    assert len(m["unchanged_recipe"]) >= 15
    assert m["warm_start"]["identical_across_all_seeds"] is True
    assert m["evaluation"]["primary_field"]["games"] == 1000
    assert m["evaluation"]["intermediate_diagnostic_field"]["games"] == 500
