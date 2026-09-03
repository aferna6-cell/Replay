"""DEV split, PPO diagnostic instrumentation, and drift-metric tests."""

import json
import os

import pytest

np = pytest.importorskip("numpy")

from ml import seeds
from ml.benchmark import make_agent
from ml.dev_benchmark import (DEV_VERSION, dev_result_to_json, main as dev_main,
                              run_dev_benchmark, save_dev_json)


# --- DEV seed interval --------------------------------------------------------
def test_dev_interval_finite_disjoint_from_test():
    assert seeds.DEV_SEED_START < seeds.DEV_SEED_END
    assert (seeds.DEV_SEED_END < seeds.EVAL_SEED_START
            or seeds.DEV_SEED_START > seeds.EVAL_SEED_END)
    seeds.validate_dev_range(seeds.DEV_SEED_START, 200)      # must not raise
    seeds.validate_dev_range(seeds.DEV_SEED_START,
                             seeds.DEV_SEED_END - seeds.DEV_SEED_START + 1)


def test_training_schemes_cannot_reach_dev_interval():
    # Same audit style as the TEST interval, through the real helpers.
    for base in range(0, 2001):
        assert not seeds.overlaps_dev_range(
            seeds.ppo_episode_seed(base, 1),
            seeds.ppo_episode_seed(base, 500_000))
        assert not seeds.overlaps_dev_range(
            seeds.midgame_lobby_seed(base, 0),
            seeds.midgame_lobby_seed(base, 40_000 - 1))
    assert not seeds.overlaps_dev_range(0, 150_000)          # additive schemes
    # Honest documented collision bounds:
    assert seeds.overlaps_dev_range(seeds.ppo_episode_seed(10, 1),
                                    seeds.ppo_episode_seed(10, 549_970))
    assert seeds.overlaps_dev_range(seeds.midgame_lobby_seed(105, 0),
                                    seeds.midgame_lobby_seed(105, 49_685))


def test_validate_dev_range_rejects_test_and_training_seeds():
    with pytest.raises(ValueError, match="Replay DEV"):
        seeds.validate_dev_range(seeds.EVAL_SEED_START, 10)  # TEST seeds
    with pytest.raises(ValueError, match="Replay DEV"):
        seeds.validate_dev_range(0, 10)                      # training range
    with pytest.raises(ValueError, match="--games"):
        seeds.validate_dev_range(seeds.DEV_SEED_START, 0)
    with pytest.raises(ValueError, match="reserved"):
        run_dev_benchmark(make_agent("random"), "greedy", games=5,
                          base_seed=seeds.EVAL_SEED_START)


def test_check_training_range_covers_dev_interval(capsys):
    assert seeds.check_training_range(
        "t", seeds.DEV_SEED_START, seeds.DEV_SEED_START + 1) is True
    assert "DEV" in capsys.readouterr().err


def test_dev_cli_rejects_test_seeds(capsys):
    with pytest.raises(SystemExit):
        dev_main(["--agent", "random", "--games", "5",
                  "--seed", str(seeds.EVAL_SEED_START)])
    capsys.readouterr()


# --- DEV result labeling ------------------------------------------------------
def test_dev_results_are_explicitly_labeled_and_unpairable_with_test(tmp_path):
    from ml.analyze_benchmark import load_result, verify_paired
    from ml.benchmark import run_benchmark, save_json
    dev = run_dev_benchmark(make_agent("random"), "greedy", games=3)
    blob = dev_result_to_json(dev)
    assert blob["evaluation_split"] == "dev"
    assert blob["benchmark_version"] == DEV_VERSION != "Replay Benchmark v1"
    assert "NOT Benchmark v1" in blob["seed_policy"]
    assert blob["seed_range"] == [seeds.DEV_SEED_START, seeds.DEV_SEED_START + 2]

    dev_p = str(tmp_path / "dev.json")
    save_dev_json(dev, dev_p)
    test_res = run_benchmark(make_agent("random"), "greedy", games=3)
    test_p = str(tmp_path / "test.json")
    save_json(test_res, test_p)
    assert json.load(open(test_p))["evaluation_split"] == "test"
    with pytest.raises(ValueError, match="not paired-comparable"):
        verify_paired(load_result(dev_p), load_result(test_p))
    # dev-vs-dev with equal config pairs fine
    verify_paired(load_result(dev_p), load_result(dev_p))


# --- drift metric math --------------------------------------------------------
def _torch():
    return pytest.importorskip("torch")


def test_masked_kl_identity_is_zero():
    torch = _torch()
    from ml.policy_drift import masked_kl
    logits = torch.tensor([[1.0, 2.0, float("-inf"), 0.5]])
    legal = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    kl = masked_kl(logits, logits, legal)
    assert torch.allclose(kl, torch.zeros(1), atol=1e-6)


def test_masked_kl_known_two_action_value():
    torch = _torch()
    import math
    from ml.policy_drift import masked_kl
    # p = [0.5, 0.5], q = [e/(e+1), 1/(e+1)] over 2 legal actions
    lp = torch.tensor([[0.0, 0.0, float("-inf")]])
    lq = torch.tensor([[1.0, 0.0, float("-inf")]])
    legal = torch.tensor([[1.0, 1.0, 0.0]])
    e = math.e
    expected = 0.5 * math.log(0.5 / (e / (e + 1))) + 0.5 * math.log(0.5 / (1 / (e + 1)))
    assert masked_kl(lp, lq, legal).item() == pytest.approx(expected, abs=1e-6)


def test_masked_kl_ignores_illegal_actions():
    torch = _torch()
    from ml.policy_drift import masked_kl
    lp = torch.tensor([[0.3, 1.1, float("-inf")]])
    lq1 = torch.tensor([[0.9, -0.2, float("-inf")]])
    lq2 = lq1.clone()                                # illegal logit irrelevant
    kl1 = masked_kl(lp, lq1, torch.tensor([[1.0, 1.0, 0.0]]))
    kl2 = masked_kl(lp, lq2, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(kl1, kl2)
    assert torch.isfinite(kl1).all()


def test_drift_metrics_agreements():
    torch = _torch()
    from ml.policy_drift import drift_metrics
    # 3 states, 3 actions, all legal. argmax_k = [0,1,2]; ref = [0,1,1]; expert = [0,2,2]
    logits_k = torch.eye(3) * 5.0
    logits_ref = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 5.0, 1.0]])
    legal = torch.ones(3, 3)
    expert = torch.tensor([0, 2, 2])
    tensors = (None, None, None, None, legal, expert)
    m = drift_metrics(logits_k, torch.tensor([1.0, 2.0, 3.0]), logits_ref, tensors)
    assert m["expert_agreement"] == pytest.approx(2 / 3)
    assert m["warmstart_agreement"] == pytest.approx(2 / 3)
    assert m["value_mean"] == pytest.approx(2.0)
    assert m["kl_from_warmstart_mean"] >= 0.0


def test_corpus_is_deterministic_and_dev_ranged():
    from ml.policy_drift import CORPUS_SEED_BASE, CORPUS_LOBBIES, build_corpus
    assert seeds.DEV_SEED_START <= CORPUS_SEED_BASE
    assert CORPUS_SEED_BASE + CORPUS_LOBBIES - 1 <= seeds.DEV_SEED_END
    # corpus seeds must not collide with default dev-eval games at DEV start
    assert CORPUS_SEED_BASE > seeds.DEV_SEED_START + 10_000
    s1, f1 = build_corpus(lobbies=2)
    s2, f2 = build_corpus(lobbies=2)
    assert f1 == f2 and len(s1) == len(s2) > 0


# --- PPO instrumentation ------------------------------------------------------
def _tiny_bc_checkpoint(tmp_path):
    torch = _torch()
    from hsbg_coach.synergy import load_embeddings
    from ml.policy_net import PolicyNet, save_policy
    from ml.tokens import token_dim
    torch.manual_seed(0)
    net = PolicyNet(token_dim(load_embeddings()))
    path = str(tmp_path / "warm.pt")
    save_policy(net, path, {"kind": "bc"})
    return path


def _state_dicts_equal(a, b):
    torch = _torch()
    return (a.keys() == b.keys()
            and all(torch.equal(a[k], b[k]) for k in a))


def test_ppo_checkpoint_schedule_and_training_unchanged(tmp_path):
    """One tiny PPO run with snapshots+diagnostics vs one without: the saved
    schedule must exist, iteration 0 must be the exact warm-start weights,
    and the final trained weights must be identical in both runs."""
    torch = _torch()
    from ml.policy_net import load_policy
    from ml.train_ppo import main as ppo_main
    warm = _tiny_bc_checkpoint(tmp_path)
    out_plain = str(tmp_path / "plain.pt")
    out_diag = str(tmp_path / "diag.pt")
    ckpt_dir = str(tmp_path / "ckpts")
    diag_log = str(tmp_path / "diag.jsonl")
    base = ["--iters", "2", "--episodes", "2", "--seed", "0",
            "--shaping", "1.0", "--eval-episodes", "1", "--from-bc", warm]
    ppo_main(base + ["--out", out_plain])
    ppo_main(base + ["--out", out_diag, "--save-iters", "0,1,2",
                     "--save-dir", ckpt_dir, "--diag-log", diag_log])

    # checkpoint schedule
    files = sorted(os.listdir(ckpt_dir))
    assert files == ["iter_000.pt", "iter_001.pt", "iter_002.pt"]
    # iteration 0 == exact warm-start weights, before any PPO update
    assert _state_dicts_equal(load_policy(ckpt_dir + "/iter_000.pt").state_dict(),
                              load_policy(warm).state_dict())
    # diagnostics + snapshotting did not perturb training
    assert _state_dicts_equal(load_policy(out_plain).state_dict(),
                              load_policy(out_diag).state_dict())
    # final snapshot equals the final saved model
    assert _state_dicts_equal(load_policy(ckpt_dir + "/iter_002.pt").state_dict(),
                              load_policy(out_diag).state_dict())
    # diag log carries the required per-iteration fields
    rows = [json.loads(l) for l in open(diag_log)]
    assert [r["iter"] for r in rows] == [1, 2]
    for r in rows:
        for key in ("pi_loss", "v_loss", "entropy", "approx_kl", "clip_frac",
                    "grad_norm", "rollout_avg_placement", "shaping",
                    "league_size", "steps"):
            assert key in r
        assert 0.0 <= r["clip_frac"] <= 1.0
        assert r["grad_norm"] >= 0.0


def test_ppo_save_iters_requires_save_dir(tmp_path):
    from ml.train_ppo import main as ppo_main
    with pytest.raises(SystemExit):
        ppo_main(["--iters", "1", "--save-iters", "0"])


def test_diag_stats_math():
    """approx KL and clip fraction on known inputs."""
    torch = _torch()
    from ml.train_ppo import diag_stats
    # identical policies: ratio == 1 everywhere -> kl 0, nothing clipped
    lp = torch.tensor([-1.0, -2.0, -0.5])
    kl, cf = diag_stats(lp, lp, clip=0.2)
    assert kl == pytest.approx(0.0) and cf == pytest.approx(0.0)
    # new policy uniformly more likely by +0.1 nats: kl = -0.1, ratio e^0.1
    kl, cf = diag_stats(lp, lp + 0.1, clip=0.2)
    assert kl == pytest.approx(-0.1, abs=1e-6)
    assert cf == pytest.approx(0.0)                  # e^0.1-1 = 0.105 < 0.2
    # one sample far outside the trust region -> exactly 1/3 clipped
    kl, cf = diag_stats(lp, lp + torch.tensor([0.0, 0.0, 1.0]), clip=0.2)
    assert cf == pytest.approx(1 / 3)
    assert kl == pytest.approx(-1 / 3, abs=1e-6)


def test_paired_dev_comparison():
    """Two DEV runs on identical config pair game-by-game."""
    from ml.analyze_benchmark import compare_pair, paired_diff
    from ml.dev_benchmark import dev_result_to_json
    rnd = dev_result_to_json(run_dev_benchmark(
        make_agent("random", name="Rando"), "greedy", games=5))
    gre = dev_result_to_json(run_dev_benchmark(
        make_agent("greedy", name="Greed"), "greedy", games=5))
    row = compare_pair(rnd, gre, seed=0)
    assert row["n"] == 5
    # random must not place better than greedy: diff (random - greedy) > 0
    assert row["mean_diff"] > 0
    assert row["ci95"][0] <= row["mean_diff"] <= row["ci95"][1]
    # pairing is game-by-game on the same seeds, not a difference of means
    manual = paired_diff(rnd["placements"], gre["placements"], seed=0)
    assert manual["mean_diff"] == pytest.approx(row["mean_diff"])
