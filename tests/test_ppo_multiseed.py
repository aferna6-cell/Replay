"""Experiment 3 protocol, safety, and artifact invariants."""

import json
from pathlib import Path

import pytest

from ml import seeds
from ml.ppo_multiseed import (ALL_SEEDS, CORPUS_FINGERPRINT, CORPUS_STATES,
                              DEV_BASE_SEED, EPISODES_PER_ITERATION,
                              GREEDY_GAMES, ITERATIONS, MIXED_GAMES,
                              SHAPING_HORIZON, TRAINING_SEEDS,
                              WARMSTART_PARAMETER_SHA256, eval_command,
                              train_command, training_seed_span,
                              validate_protocol)


def test_protocol_is_exactly_frozen_experiment2_recipe():
    assert TRAINING_SEEDS == (1, 2, 3)
    assert ALL_SEEDS == (0, 1, 2, 3)
    assert ITERATIONS == (0, 40, 80, 160, 320)
    assert EPISODES_PER_ITERATION == 16
    assert SHAPING_HORIZON == 40
    for seed in TRAINING_SEEDS:
        command = train_command(seed)
        joined = " ".join(command)
        assert "--iters 320" in joined
        assert "--episodes 16" in joined
        assert f"--seed {seed}" in joined
        assert "--shaping 1.0 --shaping-horizon 40" in joined
        assert "--save-iters 0,40,80,160,320" in joined
        assert WARMSTART_PARAMETER_SHA256 in command


def test_training_spans_do_not_touch_dev_or_test():
    validate_protocol()
    for seed in TRAINING_SEEDS:
        lo, hi = training_seed_span(seed)
        assert (lo, hi) == (
            seeds.ppo_episode_seed(seed, 1),
            seeds.ppo_episode_seed(seed, 5120),
        )
        assert not seeds.overlaps_dev_range(lo, hi)
        assert not seeds.overlaps_eval_range(lo, hi)


def test_dev_commands_use_only_prespecified_ranges_and_fields():
    assert DEV_BASE_SEED == 10_550_000
    assert GREEDY_GAMES == 1000 and MIXED_GAMES == 500
    for seed in TRAINING_SEEDS:
        for iteration in ITERATIONS:
            greedy = eval_command(seed, iteration, "greedy")
            mixed = eval_command(seed, iteration, "greedy4_random3")
            assert greedy[greedy.index("--games") + 1] == "1000"
            assert mixed[mixed.index("--games") + 1] == "500"
            assert greedy[greedy.index("--seed") + 1] == "10550000"
            assert mixed[mixed.index("--seed") + 1] == "10550000"
            assert "10250000" not in greedy + mixed
    with pytest.raises(ValueError):
        eval_command(1, 0, "random")


def test_seed_zero_is_never_a_new_training_or_eval_command():
    with pytest.raises(ValueError):
        train_command(0)
    with pytest.raises(ValueError):
        eval_command(0, 80, "greedy")


def test_experiment2_warmstart_and_corpus_are_pinned():
    manifest = json.loads(Path("results/ppo_budget_v1/manifest.json").read_text())
    drift = json.loads(Path("results/ppo_budget_v1/policy_drift.json").read_text())
    assert (manifest["training"]["warm_start"]["parameter_sha256"] ==
            WARMSTART_PARAMETER_SHA256)
    assert drift["corpus"]["states"] == CORPUS_STATES == 4440
    assert drift["corpus"]["fingerprint_sha256"] == CORPUS_FINGERPRINT


def test_required_warmstart_hash_guard_fails_closed(tmp_path, monkeypatch):
    from ml import model_fingerprint
    from ml.train_ppo import main

    fake = tmp_path / "warm.pt"
    fake.write_bytes(b"not used because hash is mocked")
    monkeypatch.setattr(model_fingerprint, "checkpoint_parameter_sha256",
                        lambda _: "wrong")
    with pytest.raises(SystemExit):
        main([
            "--iters", "0", "--episodes", "1", "--from-bc", str(fake),
            "--require-from-bc-parameter-sha256", WARMSTART_PARAMETER_SHA256,
        ])


def test_cross_seed_bootstrap_is_deterministic_and_descriptive():
    from scripts.ppo_multiseed_report import _seed_boot

    a = _seed_boot([-0.2, 0.0, 0.1, 0.3], seed=7, resamples=1000)
    b = _seed_boot([-0.2, 0.0, 0.1, 0.3], seed=7, resamples=1000)
    assert a == b
    assert a["n_training_seeds"] == 4
    assert a["mean"] == pytest.approx(0.05)
    assert a["min"] == -0.2 and a["max"] == 0.3


def test_action_category_mapping_includes_freeze():
    from hsbg_coach.bg_env import A_FREEZE
    from ml.action_categories import CATEGORIES, action_category
    assert "freeze" in CATEGORIES
    assert action_category(A_FREEZE) == "freeze"


def test_integrity_evaluator_never_imputes_unfinished_game(monkeypatch):
    from types import SimpleNamespace
    from ml import dev_integrity
    from ml.benchmark import BenchmarkIntegrityError

    agent = SimpleNamespace(name="policy", checkpoint="policy.pt")

    def fake_run(_agent, _seats, seed):
        if seed == DEV_BASE_SEED + 1:
            raise BenchmarkIntegrityError("did not terminate")
        return {"placement": 4, "latencies": [0.001]}

    monkeypatch.setattr(dev_integrity, "run_game", fake_run)
    result = dev_integrity.run_integrity_dev(
        agent, "greedy", 3, DEV_BASE_SEED)
    assert result["placements_nullable"] == [4, None, 4]
    assert result["games_completed"] == 2
    assert result["games_unfinished"] == 1
    assert result["complete_case_metrics"]["avg_placement"] == 4
    assert result["mean_placement_sensitivity_bounds"] == [3.0, 16 / 3]
    assert result["failures"][0]["seed"] == DEV_BASE_SEED + 1
