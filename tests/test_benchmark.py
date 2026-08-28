"""Replay Benchmark v1 tests — determinism, seed policy, integrity, metrics,
serialization.

Deliberately small game counts: unit tests prove the machinery, not agent
strength (that's what the real benchmark runs are for).
"""

import hashlib
import json
import os

import pytest

np = pytest.importorskip("numpy")

from ml import seeds
from ml.benchmark import (
    Agent, BEAT_FIELD_THRESHOLD, BENCHMARK_VERSION, BenchmarkIntegrityError,
    EVAL_SEED_BASE, bootstrap_ci, compare_files, compute_metrics,
    latency_stats, main, make_agent, result_to_json, run_benchmark, run_game,
    save_json, suite_to_json,
)
from hsbg_coach.bg_env import N_ACTIONS, greedy_policy


# --- seed policy --------------------------------------------------------------
# These exercise the REAL seed-generation helpers the trainers call
# (ml/seeds.py), not copies of their arithmetic.
def _span_overlaps(lo, hi):
    return seeds.overlaps_eval_range(lo, hi)


def test_eval_interval_is_finite_and_default_fits_1000_games():
    assert seeds.EVAL_SEED_START == EVAL_SEED_BASE
    assert seeds.EVAL_SEED_END > seeds.EVAL_SEED_START
    seeds.validate_eval_range(EVAL_SEED_BASE, 1000)          # must not raise
    capacity = seeds.EVAL_SEED_END - seeds.EVAL_SEED_START + 1
    seeds.validate_eval_range(seeds.EVAL_SEED_START, capacity)


def test_bc_and_dagger_default_seeds_stay_out_of_eval_interval():
    # ml/bc.py defaults: base 0, 150 demo lobbies, 2 DAgger rounds x 80.
    used = [seeds.bc_lobby_seed(0, i) for i in range(150)]
    for rnd in (1, 2):
        base = seeds.dagger_round_base(0, rnd)
        used += [seeds.bc_lobby_seed(base, i) for i in range(80)]
    assert not any(seeds.EVAL_SEED_START <= s <= seeds.EVAL_SEED_END
                   for s in used)
    # Even a heavy config (base 0, 9 rounds x 5000 lobbies) stays below.
    assert seeds.bc_lobby_seed(seeds.dagger_round_base(0, 9), 4999) \
        < seeds.EVAL_SEED_START


def test_ppo_episode_seeds_cannot_reach_eval_interval_for_any_base():
    # PPO blocks are [seed*1000003+1, seed*1000003+episodes]. For EVERY base
    # seed, a single run needs >=249_970 episodes (defaults: 640) to touch
    # the reserved interval — check the actual helper across a wide base range.
    for base in range(0, 2001):
        lo = seeds.ppo_episode_seed(base, 1)
        hi = seeds.ppo_episode_seed(base, 200_000)      # far beyond realistic
        assert not _span_overlaps(lo, hi), f"ppo base {base} overlaps"


def test_ppo_documented_collision_bound_is_honest():
    # We do NOT claim mathematical separation: base seed 10 with >=249_970
    # episodes DOES reach the interval. The tests above rely on runs being
    # smaller; this pins the documented bound.
    assert not _span_overlaps(seeds.ppo_episode_seed(10, 1),
                              seeds.ppo_episode_seed(10, 249_969))
    assert _span_overlaps(seeds.ppo_episode_seed(10, 1),
                          seeds.ppo_episode_seed(10, 249_970))


def test_midgame_lobby_seeds_cannot_reach_eval_interval_for_any_base():
    # midgame blocks are [seed*100003, seed*100003+lobbies-1]; for EVERY base
    # a run needs >49_694 lobbies (defaults: 300, calibrate: 60).
    for base in range(0, 2001):
        lo = seeds.midgame_lobby_seed(base, 0)
        hi = seeds.midgame_lobby_seed(base, 40_000 - 1)
        assert not _span_overlaps(lo, hi), f"midgame base {base} overlaps"
    # Documented honest bound: base 102 with >49_694 lobbies collides.
    assert not _span_overlaps(seeds.midgame_lobby_seed(102, 0),
                              seeds.midgame_lobby_seed(102, 49_693))
    assert _span_overlaps(seeds.midgame_lobby_seed(102, 0),
                          seeds.midgame_lobby_seed(102, 49_694))


def test_legacy_eval_seeds_stay_out_of_eval_interval():
    # ml/rl_common.evaluate_policy default base 9000.
    assert not _span_overlaps(seeds.legacy_eval_seed(9000, 0),
                              seeds.legacy_eval_seed(9000, 9999))


def test_check_training_range_warns_only_on_overlap(capsys):
    assert seeds.check_training_range("t", 0, 100) is False
    assert "WARNING" not in capsys.readouterr().err
    assert seeds.check_training_range(
        "t", seeds.EVAL_SEED_START, seeds.EVAL_SEED_START + 1) is True
    assert "WARNING" in capsys.readouterr().err


def test_validate_eval_range_rejects_bad_requests():
    with pytest.raises(ValueError, match="--games"):
        seeds.validate_eval_range(EVAL_SEED_BASE, 0)
    with pytest.raises(ValueError, match="--games"):
        seeds.validate_eval_range(EVAL_SEED_BASE, -10)
    with pytest.raises(ValueError, match="reserved"):
        seeds.validate_eval_range(0, 10)                    # below interval
    with pytest.raises(ValueError, match="reserved"):
        seeds.validate_eval_range(seeds.EVAL_SEED_END, 2)   # runs past end
    with pytest.raises(ValueError, match="reserved"):
        run_benchmark(make_agent("random"), "greedy", games=5, base_seed=0)


def test_cli_rejects_invalid_games_and_out_of_range_seed(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--agent", "random", "--games", "0"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        main(["--agent", "random", "--games", "-10"])
    with pytest.raises(SystemExit):
        main(["--agent", "random", "--games", "5", "--seed", "9000"])
    capsys.readouterr()


# --- determinism --------------------------------------------------------------
def test_same_seed_same_results():
    agent = make_agent("random")
    a = run_benchmark(agent, "greedy", games=3, base_seed=EVAL_SEED_BASE)
    b = run_benchmark(agent, "greedy", games=3, base_seed=EVAL_SEED_BASE)
    assert a.placements == b.placements
    assert a.metrics == b.metrics
    assert a.ci95 == b.ci95                      # bootstrap is seeded too


def test_different_seeds_differ():
    agent = make_agent("random")
    a = run_benchmark(agent, "greedy", games=5, base_seed=EVAL_SEED_BASE)
    b = run_benchmark(agent, "greedy", games=5, base_seed=EVAL_SEED_BASE + 5000)
    # Different seed ranges must produce different trajectories; placements
    # could still coincide by chance, so compare the finer-grained signal of
    # per-game decision counts alongside placements.
    assert (a.placements != b.placements
            or a.latency["decisions"] != b.latency["decisions"])


# --- metrics ------------------------------------------------------------------
def test_compute_metrics_known_values():
    m = compute_metrics([1, 2, 3, 4, 5, 6, 7, 8])
    assert m["games"] == 8
    assert m["avg_placement"] == pytest.approx(4.5)
    assert m["median_placement"] == pytest.approx(4.5)
    assert m["top4_rate"] == pytest.approx(0.5)
    assert m["win_rate"] == pytest.approx(1 / 8)
    assert all(m["placement_counts"][str(p)] == 1 for p in range(1, 9))


def test_placement_distribution_sums_to_games():
    placements = [1, 1, 4, 4, 4, 8, 2]
    m = compute_metrics(placements)
    assert sum(m["placement_counts"].values()) == m["games"] == len(placements)
    assert m["top4_rate"] == pytest.approx(6 / 7)
    assert m["win_rate"] == pytest.approx(2 / 7)


def test_compute_metrics_rejects_empty():
    with pytest.raises(ValueError):
        compute_metrics([])


def test_bootstrap_ci_deterministic_and_bounded():
    data = [1, 3, 3, 5, 8, 8, 2, 4, 6, 7]
    a = bootstrap_ci(data, seed=42)
    b = bootstrap_ci(data, seed=42)
    assert a == b
    assert min(data) <= a["low"] <= a["high"] <= max(data)
    c = bootstrap_ci(data, seed=43)
    assert c != a                                # seed actually drives it


def test_bootstrap_ci_constant_data_collapses():
    ci = bootstrap_ci([4, 4, 4, 4], seed=0)
    assert ci["low"] == ci["high"] == 4.0


def test_latency_stats():
    s = latency_stats([0.001, 0.002, 0.003, 0.004])
    assert s["decisions"] == 4
    assert s["mean_ms"] == pytest.approx(2.5)
    assert s["p50_ms"] >= s["mean_ms"] - 2.5     # sanity: same order of magnitude
    assert latency_stats([]) == {"decisions": 0}


# --- game / agent integrity ---------------------------------------------------
def test_run_game_agents_act_legally_and_finish():
    for kind in ("random", "greedy"):
        g = run_game(make_agent(kind), greedy_policy, seed=EVAL_SEED_BASE)
        assert 1 <= g["placement"] <= 8
        assert len(g["latencies"]) >= 1


def _agent_returning(value):
    return Agent("Bad", "random", lambda obs, mask, rng: value)


def test_negative_action_is_rejected_not_negatively_indexed():
    # mask[-1] is END TURN and legal — Python negative indexing must never
    # silently launder -1 into a legal action.
    with pytest.raises(BenchmarkIntegrityError, match="out-of-range"):
        run_game(_agent_returning(-1), greedy_policy, seed=EVAL_SEED_BASE)


def test_action_beyond_mask_length_is_rejected():
    with pytest.raises(BenchmarkIntegrityError, match="out-of-range"):
        run_game(_agent_returning(N_ACTIONS), greedy_policy,
                 seed=EVAL_SEED_BASE)


def test_in_range_but_masked_illegal_action_is_rejected():
    # Action 7 (play hand slot 0) is illegal at reset: the hand is empty.
    with pytest.raises(BenchmarkIntegrityError, match="illegal"):
        run_game(_agent_returning(7), greedy_policy, seed=EVAL_SEED_BASE)


def test_non_integer_action_is_rejected():
    with pytest.raises(BenchmarkIntegrityError, match="non-integer"):
        run_game(_agent_returning(2.5), greedy_policy, seed=EVAL_SEED_BASE)


def test_unfinished_episode_raises_instead_of_scoring_8th(monkeypatch):
    class NeverDoneEnv:
        def __init__(self, seed=None, opponent_policies=None):
            pass

        def reset(self, seed=None):
            return {}

        def legal_mask(self, seat):
            return [True] * N_ACTIONS

        def step(self, action):
            return {}, 0.0, False, {}            # never terminates

    monkeypatch.setattr("ml.benchmark.BGEnv", NeverDoneEnv)
    agent = Agent("Staller", "random", lambda obs, mask, rng: 0)
    with pytest.raises(BenchmarkIntegrityError) as e:
        run_game(agent, greedy_policy, seed=EVAL_SEED_BASE)
    msg = str(e.value)
    assert "did not terminate" in msg
    assert "Staller" in msg                      # tested agent
    assert str(EVAL_SEED_BASE) in msg            # game seed
    assert "400 decisions" in msg                # decision count (MAX_DECISIONS)


def test_make_agent_errors():
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("alphazero")
    with pytest.raises(ValueError, match="requires --checkpoint"):
        make_agent("policy")
    with pytest.raises(ValueError, match="not found"):
        make_agent("policy", checkpoint="ml/no_such_checkpoint.pt")


def test_make_agent_rejects_malformed_checkpoint(tmp_path):
    pytest.importorskip("torch")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(ValueError, match="could not load"):
        make_agent("policy", checkpoint=str(bad))


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown field"):
        run_benchmark(make_agent("random"), "psychic", games=1)


# --- policy-checkpoint agents (torch) -----------------------------------------
def _tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from hsbg_coach.synergy import load_embeddings
    from ml.policy_net import PolicyNet, save_policy
    from ml.tokens import token_dim
    torch.manual_seed(0)
    net = PolicyNet(token_dim(load_embeddings()))
    path = str(tmp_path / "policy_tiny.pt")
    save_policy(net, path, {"kind": "test"})
    return path


def test_policy_agent_deterministic_fingerprinted_and_reads_only(tmp_path):
    path = _tiny_checkpoint(tmp_path)
    before = open(path, "rb").read()
    agent = make_agent("policy", checkpoint=path, name="Tiny")
    a = run_benchmark(agent, "greedy", games=2, base_seed=EVAL_SEED_BASE)
    b = run_benchmark(agent, "greedy", games=2, base_seed=EVAL_SEED_BASE)
    assert a.placements == b.placements          # argmax decisions, seeded env
    assert open(path, "rb").read() == before     # benchmark never writes weights
    assert agent.checkpoint == "policy_tiny.pt"  # basename only, no abs path
    # sha256 fingerprints the exact evaluated bytes.
    assert agent.checkpoint_sha256 == hashlib.sha256(before).hexdigest()
    blob = result_to_json(a)
    assert blob["checkpoint_sha256"] == agent.checkpoint_sha256


# --- serialization ------------------------------------------------------------
def test_json_output_schema_and_roundtrip(tmp_path):
    res = run_benchmark(make_agent("random"), "greedy", games=3)
    out = tmp_path / "nested" / "dir" / "res.json"   # parents auto-created
    save_json(res, str(out))
    blob = json.loads(out.read_text())
    assert blob["benchmark_version"] == BENCHMARK_VERSION
    assert blob["agent"] == "Random" and blob["field"] == "greedy"
    assert blob["games"] == 3
    assert blob["seed_range"] == [EVAL_SEED_BASE, EVAL_SEED_BASE + 2]
    assert sum(blob["metrics"]["placement_counts"].values()) == 3
    assert blob["beat_field_threshold"] == BEAT_FIELD_THRESHOLD
    assert "timestamp" in blob and "avg_placement_ci95" in blob
    assert blob["checkpoint_sha256"] is None     # scripted agent: no model
    assert "git_commit" in blob                  # sha, sha-dirty, or null
    assert os.path.sep not in (blob["checkpoint"] or "")   # no machine paths


def test_suite_json_wrapper_schema(tmp_path):
    r1 = run_benchmark(make_agent("random"), "greedy", games=2)
    r2 = run_benchmark(make_agent("greedy"), "greedy", games=2)
    blob = suite_to_json([r1, r2])
    assert blob["benchmark_version"] == BENCHMARK_VERSION
    assert [r["agent"] for r in blob["results"]] == ["Random", "Greedy"]
    assert all(r["benchmark_version"] == BENCHMARK_VERSION
               for r in blob["results"])


def test_compare_consumes_single_and_suite_files(tmp_path):
    r1 = run_benchmark(make_agent("random", name="Rando"), "greedy", games=2)
    r2 = run_benchmark(make_agent("greedy", name="Greed"), "greedy", games=2)
    single = str(tmp_path / "single.json")
    suite = str(tmp_path / "suite.json")
    save_json(r1, single)
    (tmp_path / "suite.json").write_text(json.dumps(suite_to_json([r2])))
    table = compare_files([single, suite])
    assert "Rando" in table and "Greed" in table
    order = sorted([r1, r2], key=lambda r: r.metrics["avg_placement"])
    assert table.find(order[0].agent.name) < table.find(order[1].agent.name)
    assert "WARNING" not in table                # same version/field/games/seed
    assert "Identities:" in table                # model identity is explicit

    r3 = run_benchmark(make_agent("random", name="OtherSeed"), "greedy",
                       games=2, base_seed=EVAL_SEED_BASE + 77)
    p3 = str(tmp_path / "c.json")
    save_json(r3, p3)
    assert "WARNING" in compare_files([single, p3])  # mismatched seeds flagged


def test_compare_shows_checkpoint_fingerprint_without_erroring(tmp_path):
    # Two different models sharing a filename must be distinguishable — and
    # differing hashes are the point of a comparison, never a warning.
    path = _tiny_checkpoint(tmp_path)
    a = run_benchmark(make_agent("policy", path, "ModelA"), "greedy", games=2)
    pa = str(tmp_path / "a.json")
    save_json(a, pa)
    out = compare_files([pa])
    assert "WARNING" not in out
    assert a.agent.checkpoint_sha256[:12] in out
