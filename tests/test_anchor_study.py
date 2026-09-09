"""Tests for Experiment 4: PPO Policy Anchoring machinery and result artifacts."""

import json
import os

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ml.analyze_benchmark import compare_pair, load_result
from ml.model_fingerprint import checkpoint_fingerprint
from ml.policy_drift import masked_kl
from ml.policy_net import PolicyNet
from ml.seeds import overlaps_dev_range, overlaps_eval_range, ppo_episode_seed
from ml.tokens import token_dim
from ml.train_ppo import _update
from hsbg_coach.synergy import load_embeddings

DIR = "results/ppo_anchor_v1"
BASELINE_DIR = "results/ppo_budget_v1"
ITERS = [0, 40, 80, 160, 320]


def test_kl_penalty_changes_gradients():
    """Anchor KL must affect optimization when kl_coef > 0."""
    emb = load_embeddings()
    net = PolicyNet(token_dim(emb))
    anchor = PolicyNet(token_dim(emb))
    anchor.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    n = 32
    toks = torch.randn(n, 7, token_dim(emb))
    mask = torch.ones(n, 7)
    zones = torch.zeros(n, 7, dtype=torch.long)
    ctx = torch.randn(n, 8)
    legal = torch.ones(n, 28)
    acts = torch.zeros(n, dtype=torch.long)
    old_logp = torch.zeros(n)
    adv = torch.randn(n)
    ret = torch.randn(n)
    batch = (toks, mask, zones, ctx, legal, acts, old_logp, adv, ret)

    w0 = {k: v.clone() for k, v in net.state_dict().items()}
    _update(net, opt, batch, anchor_net=None, kl_coef=0.0)
    w_no_kl = {k: v.clone() for k, v in net.state_dict().items()}

    net.load_state_dict(w0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    _update(net, opt, batch, anchor_net=anchor, kl_coef=0.1)
    w_kl = {k: v.clone() for k, v in net.state_dict().items()}

    key = next(k for k in w_no_kl if "pi" in k and "weight" in k)
    assert not torch.allclose(w_no_kl[key], w_kl[key])


def test_masked_kl_is_nonnegative():
    legal = torch.tensor([[1.0] * 28])
    p = torch.randn(1, 28)
    q = torch.randn(1, 28)
    assert float(masked_kl(p, q, legal).item()) >= 0.0


def test_training_seed_isolation():
    first = ppo_episode_seed(0, 1)
    last = ppo_episode_seed(0, 320 * 16)
    assert not overlaps_dev_range(first, last)
    assert not overlaps_eval_range(first, last)


@pytest.mark.skipif(not os.path.isdir(DIR), reason="Experiment 4 not run yet")
def test_anchored_dev_seed_ranges():
    for it in ITERS:
        path = f"{DIR}/dev/iter{it:03d}_vs_greedy.json"
        data = json.load(open(path))
        assert data["games"] == 1000
        assert data["seed_range"] == [10550000, 10550999]
        assert data["evaluation_split"] == "dev"


@pytest.mark.skipif(not os.path.isdir(DIR), reason="Experiment 4 not run yet")
def test_anchored_learning_curve_vs_baseline():
    curve = json.load(open(f"{DIR}/learning_curve.json"))["curve"]
    assert len(curve) == len(ITERS)
    for row in curve:
        assert "baseline_greedy_avg" in row
        assert "kl_from_warmstart" in row
        assert "baseline_kl_from_warmstart" in row


@pytest.mark.skipif(not os.path.isdir(DIR), reason="Experiment 4 not run yet")
def test_baseline_comparison_paired_structure():
    cmp = json.load(open(f"{DIR}/baseline_comparison.json"))
    assert len(cmp["comparisons"]) == len(ITERS)
    for row in cmp["comparisons"]:
        assert "mean_diff" in row and "ci95" in row


@pytest.mark.skipif(not os.path.isdir(DIR), reason="Experiment 4 not run yet")
def test_manifest_kl_coef():
    m = json.load(open(f"{DIR}/manifest.json"))
    assert m["experiment"] == "Replay Experiment 4 — PPO Policy Anchoring"
    assert m["training"]["anchor"]["kl_coef"] == 0.1


@pytest.mark.skipif(not os.path.isdir(BASELINE_DIR), reason="no baseline")
def test_can_pair_anchored_vs_baseline():
    if not os.path.isfile(f"{DIR}/dev/iter080_vs_greedy.json"):
        pytest.skip("anchored iter80 not evaluated")
    a = load_result(f"{DIR}/dev/iter080_vs_greedy.json")
    b = load_result(f"{BASELINE_DIR}/dev/iter080_vs_greedy.json")
    row = compare_pair(a, b, seed=0)
    assert len(row["ci95"]) == 2


@pytest.mark.skipif(not os.path.isdir(DIR), reason="Experiment 4 not run yet")
def test_warm_start_matches_bc():
    from tests.ml_testutil import skip_unless_files

    warm_path = "ml/policy_bc.pt"
    ckpt_path = f"{DIR}/checkpoints/iter_000.pt"
    skip_unless_files(warm_path, ckpt_path)
    warm = checkpoint_fingerprint(warm_path)
    ckpt = checkpoint_fingerprint(ckpt_path)
    assert ckpt["parameter_sha256"] == warm["parameter_sha256"]


def test_warm_start_matches_bc_recorded_manifest():
    """Clean-checkout evidence: committed Experiment 4 hashes, no *.pt bytes."""
    m_path = f"{DIR}/manifest.json"
    if not os.path.isfile(m_path):
        pytest.skip("Experiment 4 manifest not present")
    m = json.load(open(m_path))
    warm = m["training"]["warm_start"]["parameter_sha256"]
    assert warm and len(warm) == 64
    iter0 = next(ck["parameter_sha256"] for ck in m["checkpoints"]
                 if ck["iteration"] == 0)
    assert iter0 == warm
