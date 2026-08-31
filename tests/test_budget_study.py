"""Experiment 2 machinery: parameter fingerprints, frozen shaping schedule,
mixed DEV diagnostic fields, action categories, RL-signal metrics."""

import json
import os

import pytest

np = pytest.importorskip("numpy")

from ml import seeds
from ml.action_categories import (CATEGORIES, action_category, confusion,
                                  top_transitions)
from ml.model_fingerprint import (checkpoint_fingerprint,
                                  checkpoint_parameter_sha256, file_sha256,
                                  parameter_sha256)


def _torch():
    return pytest.importorskip("torch")


# --- parameter fingerprint ----------------------------------------------------
def _net(seed=0):
    torch = _torch()
    from hsbg_coach.synergy import load_embeddings
    from ml.policy_net import PolicyNet
    from ml.tokens import token_dim
    torch.manual_seed(seed)
    return PolicyNet(token_dim(load_embeddings()))


def test_parameter_hash_is_filename_independent(tmp_path):
    """The exact bug Experiment 1 found: identical weights under different
    filenames must share a parameter hash even though their bytes differ."""
    from ml.policy_net import save_policy
    net = _net()
    a, b = str(tmp_path / "policy_ppo.pt"), str(tmp_path / "ppo_repro.pt")
    save_policy(net, a, {"kind": "ppo", "iters": 40})
    save_policy(net, b, {"kind": "ppo", "iters": 40})
    assert checkpoint_parameter_sha256(a) == checkpoint_parameter_sha256(b)
    assert file_sha256(a) != file_sha256(b)        # raw artifact hash differs
    fp = checkpoint_fingerprint(a)
    assert fp["parameter_sha256"] == checkpoint_parameter_sha256(a)
    assert fp["checkpoint_sha256"] == file_sha256(a)


def test_parameter_hash_changes_when_a_weight_changes():
    torch = _torch()
    net = _net()
    before = parameter_sha256(net.state_dict())
    assert parameter_sha256(net.state_dict()) == before      # deterministic
    with torch.no_grad():
        net.pi[1].bias[0] += 1e-4
    assert parameter_sha256(net.state_dict()) != before


def test_parameter_hash_covers_key_dtype_and_shape():
    torch = _torch()
    base = {"a": torch.zeros(2, 3), "b": torch.ones(4)}
    h = parameter_sha256(base)
    assert parameter_sha256({"b": torch.ones(4), "a": torch.zeros(2, 3)}) == h
    assert parameter_sha256({"a": torch.zeros(3, 2), "b": torch.ones(4)}) != h
    assert parameter_sha256({"z": torch.zeros(2, 3), "b": torch.ones(4)}) != h
    assert parameter_sha256(
        {"a": torch.zeros(2, 3, dtype=torch.float64), "b": torch.ones(4)}) != h
    # non-contiguous / device-transposed views must not change the hash
    t = torch.zeros(3, 2).t()
    assert parameter_sha256({"a": t, "b": torch.ones(4)}) == h


# --- frozen shaping schedule --------------------------------------------------
def _shaping_series(iters, horizon, initial=1.0):
    """Mirror of the anneal in ml.train_ppo.main (see the test below that
    pins it against the real trainer)."""
    return [initial * max(0.0, 1.0 - it / max(1, horizon * 0.7))
            for it in range(iters)]


def test_shaping_horizon_preserves_original_40_iteration_schedule():
    original = _shaping_series(40, 40)
    extended = _shaping_series(320, 40)
    assert extended[:40] == original            # first 40 iters unchanged
    assert all(s == 0.0 for s in extended[40:])  # and pinned at 0 afterwards
    # Without the mechanism, a 320-iteration run would stretch the schedule
    # and change the reward for the first 40 iterations too.
    naive = _shaping_series(320, 320)
    assert naive[:40] != original
    assert naive[39] > 0.8                       # still nearly full shaping


def test_shaping_series_matches_trainer_formula(tmp_path, monkeypatch):
    """Pin the helper above against the real trainer's logged shaping."""
    from ml.policy_net import save_policy
    from ml.train_ppo import main as ppo_main
    warm = str(tmp_path / "warm.pt")
    save_policy(_net(), warm, {"kind": "bc"})
    log = str(tmp_path / "d.jsonl")
    ppo_main(["--iters", "3", "--episodes", "1", "--seed", "0",
              "--shaping", "1.0", "--shaping-horizon", "40",
              "--eval-episodes", "1", "--from-bc", warm,
              "--out", str(tmp_path / "o.pt"), "--diag-log", log])
    logged = [json.loads(l)["shaping"] for l in open(log)]
    assert logged == pytest.approx(_shaping_series(3, 40))


# --- action categories --------------------------------------------------------
def test_action_category_mapping_matches_env_action_space():
    from hsbg_coach.bg_env import (A_END, A_FREEZE, A_LEVEL, A_ROLL,
                                   N_ACTIONS)
    assert action_category(0) == action_category(6) == "buy"
    assert action_category(7) == action_category(16) == "play"
    assert action_category(17) == action_category(23) == "sell"
    assert action_category(A_ROLL) == "roll"
    assert action_category(A_LEVEL) == "level"
    assert action_category(A_FREEZE) == "freeze"
    assert action_category(A_END) == "end"
    # every action in the space maps to a declared category, none left over
    got = {action_category(a) for a in range(N_ACTIONS)}
    assert got == set(CATEGORIES)
    with pytest.raises(ValueError):
        action_category(N_ACTIONS)
    with pytest.raises(ValueError):
        action_category(-1)


def test_confusion_accounting_known_case():
    #        expert:  buy  buy   roll  end   level
    ref = [0, 1, 24, 27, 25]
    #      compared:  buy  roll  roll  end   buy      -> 2 disagreements
    cmp_ = [0, 24, 24, 27, 3]
    c = confusion(ref, cmp_)
    assert c["n_states"] == 5 and c["n_disagreements"] == 2
    assert c["overall_agreement"] == pytest.approx(0.6)
    assert c["reference_category_counts"]["buy"] == 2
    # of the 2 expert-buy states, 1 changed
    assert c["disagreement_share_by_category"]["buy"] == pytest.approx(0.5)
    assert c["disagreement_share_by_category"]["level"] == pytest.approx(1.0)
    assert c["disagreement_share_by_category"]["roll"] == pytest.approx(0.0)
    assert c["disagreement_share_by_category"]["sell"] is None   # never seen
    # contributions to total drift sum to 1 over categories that had any
    contrib = [v for v in c["contribution_to_total_drift"].values() if v]
    assert sum(contrib) == pytest.approx(1.0)
    assert c["confusion_matrix"]["buy"]["roll"] == 1
    assert c["confusion_matrix"]["buy"]["buy"] == 1
    # rows of the matrix sum to the reference counts
    for cat, row in c["confusion_matrix"].items():
        assert sum(row.values()) == c["reference_category_counts"][cat]
    top = top_transitions(c)
    assert {"from": "buy", "to": "roll", "count": 1} in top
    assert all(r["from"] != r["to"] for r in top)


def test_confusion_identical_lists_have_no_drift():
    ref = [0, 7, 17, 24, 25, 26, 27]
    c = confusion(ref, ref)
    assert c["n_disagreements"] == 0
    assert c["overall_agreement"] == pytest.approx(1.0)
    assert all(v in (0.0, None)
               for v in c["disagreement_share_by_category"].values())


def test_confusion_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        confusion([0, 1], [0])


# --- RL signal ----------------------------------------------------------------
def test_rl_signal_known_values():
    from ml.train_ppo import rl_signal
    adv = [1.0, -1.0, 0.0, 2.0]
    ret = [1.0, 2.0, 3.0, 4.0]
    val = [1.0, 2.0, 3.0, 4.0]            # perfect predictions
    s = rl_signal(adv, ret, val, [1, 2, 3, 4], [0.1, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0])
    assert s["adv_mean"] == pytest.approx(0.5)
    assert s["adv_mean_abs"] == pytest.approx(1.0)
    assert s["adv_frac_positive"] == pytest.approx(0.5)
    assert s["adv_frac_zero"] == pytest.approx(0.25)
    assert s["adv_frac_negative"] == pytest.approx(0.25)
    assert s["value_explained_variance"] == pytest.approx(1.0)
    assert s["placement_mean"] == pytest.approx(2.5)
    assert s["placement_distinct"] == 4
    assert s["shaping_reward_sum"] == pytest.approx(0.1)
    assert s["terminal_reward_sum"] == pytest.approx(1.0)


def test_rl_signal_explained_variance_degenerates_safely():
    from ml.train_ppo import rl_signal
    # value head predicting the mean only -> EV 0; constant returns -> None
    s = rl_signal([0.0, 0.0], [1.0, 3.0], [2.0, 2.0], [1, 2], [0.0], [0.0])
    assert s["value_explained_variance"] == pytest.approx(0.0)
    flat = rl_signal([0.0], [2.0], [9.0], [1], [0.0], [0.0])
    assert flat["value_explained_variance"] is None


# --- mixed DEV diagnostic field -----------------------------------------------
def test_dev_diagnostic_field_composition_and_determinism():
    from hsbg_coach.bg_env import greedy_policy, random_policy
    from ml.benchmark import FIELD_SIZE, make_agent
    from ml.dev_benchmark import (DEV_DIAGNOSTIC_FIELDS, dev_field_seats,
                                  dev_result_to_json, field_composition,
                                  run_dev_benchmark)
    seats = dev_field_seats("greedy4_random3")
    assert len(seats) == FIELD_SIZE
    assert seats == [greedy_policy] * 4 + [random_policy] * 3   # fixed seats
    assert field_composition("greedy4_random3") == (
        "seat1=greedy, seat2=greedy, seat3=greedy, seat4=greedy, "
        "seat5=random, seat6=random, seat7=random")
    # deterministic across runs on the same seeds
    a = run_dev_benchmark(make_agent("greedy"), "greedy4_random3", games=4)
    b = run_dev_benchmark(make_agent("greedy"), "greedy4_random3", games=4)
    assert a.placements == b.placements
    # and genuinely intermediate: easier than all-greedy on the same seeds
    hard = run_dev_benchmark(make_agent("greedy"), "greedy", games=4)
    assert np.mean(a.placements) <= np.mean(hard.placements)
    # labeled as a diagnostic, with the 4.5 threshold explicitly withheld
    blob = dev_result_to_json(a)
    assert blob["evaluation_split"] == "dev"
    assert blob["field_kind"] == "dev diagnostic (mixed opponents)"
    assert blob["beats_field"] is None and blob["beat_field_threshold"] is None
    assert "not a beat-the-field threshold" in blob["threshold_note"].lower()
    assert "greedy4_random3" in DEV_DIAGNOSTIC_FIELDS


def test_test_benchmark_rejects_dev_diagnostic_fields():
    """The mixed diagnostic must never leak into the TEST benchmark."""
    from ml.benchmark import field_seats, run_benchmark, make_agent
    with pytest.raises(ValueError, match="unknown field"):
        field_seats("greedy4_random3")
    with pytest.raises(ValueError, match="unknown field"):
        run_benchmark(make_agent("random"), "greedy4_random3", games=1)


def test_dev_and_test_seed_intervals_do_not_overlap():
    assert not seeds.overlaps_eval_range(seeds.DEV_SEED_START,
                                         seeds.DEV_SEED_END)
    assert not seeds.overlaps_dev_range(seeds.EVAL_SEED_START,
                                        seeds.EVAL_SEED_END)
    # the 1000-game Experiment 2 DEV block stays inside DEV
    seeds.validate_dev_range(seeds.DEV_SEED_START, 1000)
