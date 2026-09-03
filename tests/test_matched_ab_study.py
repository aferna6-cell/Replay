"""Tests for ml.experiment_contract and Experiment 4b matched A/B machinery."""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from ml.experiment_contract import (ContractViolation, build_contract,
                                    enforce_runtime_match, load_contract,
                                    save_contract, verify_identical_placements,
                                    verify_matched_iter0_pair, verify_warm_start,
                                    runtime_fingerprint, ppo_recipe)
from ml.model_fingerprint import (checkpoint_fingerprint,
                                    checkpoint_parameter_sha256,
                                    parameter_sha256)
from ml.policy_net import PolicyNet, save_policy
from ml.tokens import token_dim
from hsbg_coach.synergy import load_embeddings


def test_runtime_fingerprint_has_required_keys():
    rt = runtime_fingerprint()
    for key in ("python_version", "torch_version", "numpy_version",
                "torch_device", "torch_cuda_available"):
        assert key in rt


def test_build_contract_from_checkpoint(tmp_path):
    emb = load_embeddings()
    net = PolicyNet(token_dim(emb))
    path = str(tmp_path / "warm.pt")
    save_policy(net, path, {"kind": "bc"})
    contract = build_contract(path)
    assert len(contract["expected_warm_start_parameter_sha256"]) == 64
    assert contract["training_seeds"] == [0, 1, 2, 3]
    assert contract["kl_coef_values"] == [0.0, 0.1]


def test_verify_warm_start_pass_and_fail(tmp_path):
    emb = load_embeddings()
    net = PolicyNet(token_dim(emb))
    path = str(tmp_path / "warm.pt")
    save_policy(net, path, {"kind": "bc"})
    sha = checkpoint_parameter_sha256(path)
    verify_warm_start(path, sha)
    with pytest.raises(ContractViolation):
        verify_warm_start(path, "0" * 64)


def test_verify_matched_iter0_pair(tmp_path):
    emb = load_embeddings()
    net = PolicyNet(token_dim(emb))
    sha = parameter_sha256(net.state_dict())
    a = str(tmp_path / "a.pt")
    b = str(tmp_path / "b.pt")
    save_policy(net, a, {"kind": "ppo", "iter": 0})
    save_policy(net, b, {"kind": "ppo", "iter": 0})
    verify_matched_iter0_pair(a, b, sha)


def test_verify_identical_placements(tmp_path):
    payload = {"placements": [3, 5, 2, 7], "games": 4}
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(payload))
    pb.write_text(json.dumps(payload))
    verify_identical_placements(str(pa), str(pb))
    payload2 = {"placements": [3, 5, 2, 8], "games": 4}
    pc = tmp_path / "c.json"
    pc.write_text(json.dumps(payload2))
    with pytest.raises(ContractViolation):
        verify_identical_placements(str(pa), str(pc))


def test_contract_roundtrip(tmp_path):
    emb = load_embeddings()
    net = PolicyNet(token_dim(emb))
    path = str(tmp_path / "warm.pt")
    save_policy(net, path, {"kind": "bc"})
    contract = build_contract(path)
    cpath = str(tmp_path / "contract.json")
    save_contract(cpath, contract)
    loaded = load_contract(cpath)
    enforce_runtime_match(loaded)


def test_ppo_recipe_frozen_fields():
    r = ppo_recipe()
    assert r["iterations"] == 320
    assert r["shaping_horizon"] == 40
    assert r["lr"] == 3e-4


@pytest.mark.skipif(not os.path.isdir("results/ppo_matched_ab_v1"),
                    reason="Experiment 4b not run yet")
def test_gate_results_all_passed():
    gate = json.load(open("results/ppo_matched_ab_v1/gate_results.json"))
    assert gate["all_passed"] is True
    expected = gate["expected_warm_start_parameter_sha256"]
    assert len(expected) == 64


@pytest.mark.skipif(not os.path.isdir("results/ppo_matched_ab_v1"),
                    reason="Experiment 4b not run yet")
def test_all_runs_share_warm_start_hash():
    contract = load_contract("results/ppo_matched_ab_v1/contract.json")
    expected = contract["expected_warm_start_parameter_sha256"]
    for kl in ("beta0", "beta01"):
        for seed in range(4):
            ckpt = (f"results/ppo_matched_ab_v1/{kl}/seed_{seed}/"
                    f"checkpoints/iter_000.pt")
            fp = checkpoint_fingerprint(ckpt)
            assert fp["parameter_sha256"] == expected


@pytest.mark.skipif(not os.path.isdir("results/ppo_matched_ab_v1/aggregate"),
                    reason="Experiment 4b analysis not run yet")
def test_matched_ab_analysis_structure():
    analysis = json.load(open(
        "results/ppo_matched_ab_v1/aggregate/matched_ab_analysis.json"))
    assert analysis["experiment"].startswith("Replay Experiment 4b")
    assert "paired_by_seed" in analysis
    assert "outcome_classification" in analysis
    assert len(analysis["paired_by_seed"]) == 4
