"""Tests for Experiment 5 KL anchoring dose-response machinery."""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from ml.experiment_contract import build_contract, load_contract
from ml.model_fingerprint import checkpoint_fingerprint
from ml.policy_net import PolicyNet, save_policy
from ml.tokens import token_dim
from hsbg_coach.synergy import load_embeddings

DOSE_DIR = "results/ppo_dose_v1"
MATCHED_DIR = "results/ppo_matched_ab_v1"


def test_dose_contract_kl_values(tmp_path):
    emb = load_embeddings()
    net = PolicyNet(token_dim(emb))
    path = str(tmp_path / "warm.pt")
    save_policy(net, path, {"kind": "bc"})
    contract = build_contract(path, kl_coef_values=[0.0, 0.01, 0.03, 0.1])
    assert contract["kl_coef_values"] == [0.0, 0.01, 0.03, 0.1]


def test_dose_warm_start_matches_4b():
    if not os.path.isfile(os.path.join(DOSE_DIR, "contract.json")):
        pytest.skip("Experiment 5 contract not created yet")
    dose = load_contract(os.path.join(DOSE_DIR, "contract.json"))
    matched = load_contract(os.path.join(MATCHED_DIR, "contract.json"))
    assert (dose["expected_warm_start_parameter_sha256"]
            == matched["expected_warm_start_parameter_sha256"])


@pytest.mark.skipif(not os.path.isdir(DOSE_DIR),
                    reason="Experiment 5 not run yet")
def test_dose_gate_results_all_passed():
    gate = json.load(open(os.path.join(DOSE_DIR, "gate_results.json")))
    assert gate["all_passed"] is True


@pytest.mark.skipif(not os.path.isdir(DOSE_DIR),
                    reason="Experiment 5 not run yet")
def test_new_arms_share_warm_start_hash():
    contract = load_contract(os.path.join(DOSE_DIR, "contract.json"))
    expected = contract["expected_warm_start_parameter_sha256"]
    for kl in ("beta001", "beta003"):
        for seed in range(4):
            ckpt = os.path.join(DOSE_DIR, kl, f"seed_{seed}",
                                "checkpoints", "iter_000.pt")
            if not os.path.isfile(ckpt):
                pytest.skip(f"missing {ckpt}")
            fp = checkpoint_fingerprint(ckpt)
            assert fp["parameter_sha256"] == expected


@pytest.mark.skipif(not os.path.isdir(os.path.join(DOSE_DIR, "aggregate")),
                    reason="Experiment 5 analysis not run yet")
def test_dose_analysis_structure():
    analysis = json.load(open(
        os.path.join(DOSE_DIR, "aggregate", "dose_analysis.json")))
    assert analysis["experiment"].startswith("Replay Experiment 5")
    assert "bc_warm_start_baseline" in analysis
    assert "outcome_classification" in analysis
    assert len(analysis["arms"]) == 4


@pytest.mark.skipif(not os.path.isdir(os.path.join(DOSE_DIR, "aggregate")),
                    reason="Experiment 5 analysis not run yet")
def test_dose_has_four_beta_arms():
    analysis = json.load(open(
        os.path.join(DOSE_DIR, "aggregate", "dose_analysis.json")))
    coefs = sorted(a["kl_coef"] for a in analysis["arms"])
    assert coefs == [0.0, 0.01, 0.03, 0.1]
