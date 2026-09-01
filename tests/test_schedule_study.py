"""Tests for Experiment 6 schedule study (post-run gates)."""

import json
import os

import pytest

from ml.experiment_contract import load_contract
from ml.kl_schedule import EXPERIMENT_6_KL_SCHEDULE

SCHEDULE_DIR = "results/ppo_schedule_v1"
DOSE_DIR = "results/ppo_dose_v1"


def test_schedule_contract_matches_dose_warm_start():
    if not os.path.isfile(os.path.join(SCHEDULE_DIR, "contract.json")):
        pytest.skip("Experiment 6 contract not created yet")
    sched = load_contract(os.path.join(SCHEDULE_DIR, "contract.json"))
    dose = load_contract(os.path.join(DOSE_DIR, "contract.json"))
    assert (sched["expected_warm_start_parameter_sha256"]
            == dose["expected_warm_start_parameter_sha256"])


def test_schedule_contract_has_frozen_schedule():
    if not os.path.isfile(os.path.join(SCHEDULE_DIR, "contract.json")):
        pytest.skip("Experiment 6 contract not created yet")
    contract = load_contract(os.path.join(SCHEDULE_DIR, "contract.json"))
    assert contract["arms"]["treatment"]["kl_schedule"] == EXPERIMENT_6_KL_SCHEDULE


def test_success_criteria_frozen_thresholds():
    if not os.path.isfile(os.path.join(SCHEDULE_DIR, "contract.json")):
        pytest.skip("Experiment 6 contract not created yet")
    sc = load_contract(os.path.join(SCHEDULE_DIR, "contract.json"))["success_criteria"]
    assert sc["max_mean_delta_vs_bc"] == -0.02
    assert sc["max_scheduled_cross_seed_mean"] == pytest.approx(6.530)
    assert sc["seed_beat_threshold"] == -0.01
    assert sc["max_worst_seed_delta_vs_bc"] == 0.05


@pytest.mark.skipif(not os.path.isfile(
    os.path.join(SCHEDULE_DIR, "control_code_equivalence.json")),
                    reason="control equivalence gate not run yet")
def test_control_code_equivalence_passed():
    data = json.load(open(os.path.join(SCHEDULE_DIR,
                                        "control_code_equivalence.json")))
    assert data["control_code_equivalence_passed"] is True


@pytest.mark.skipif(not os.path.isdir(SCHEDULE_DIR),
                    reason="Experiment 6 not run yet")
def test_schedule_gate_results():
    gate = json.load(open(os.path.join(SCHEDULE_DIR, "gate_results.json")))
    assert gate["all_passed"] is True
