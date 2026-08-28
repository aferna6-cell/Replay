"""Replay Benchmark v1 tests — determinism, metrics, serialization, safety.

Deliberately small game counts: unit tests prove the machinery, not agent
strength (that's what the real benchmark runs are for).
"""

import json
import os

import pytest

np = pytest.importorskip("numpy")

from ml.benchmark import (
    Agent, BEAT_FIELD_THRESHOLD, BENCHMARK_VERSION, EVAL_SEED_BASE,
    bootstrap_ci, compare_files, compute_metrics, latency_stats, make_agent,
    result_to_json, run_benchmark, run_game, save_json,
)
from hsbg_coach.bg_env import greedy_policy


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


def test_eval_seed_base_is_separated_from_training():
    assert EVAL_SEED_BASE >= 100_000


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


# --- game / agent safety ------------------------------------------------------
def test_run_game_agents_act_legally_and_finish():
    for kind in ("random", "greedy"):
        g = run_game(make_agent(kind), greedy_policy, seed=EVAL_SEED_BASE)
        assert 1 <= g["placement"] <= 8
        assert len(g["latencies"]) >= 1


def test_run_game_flags_illegal_agent():
    cheat = Agent("Cheat", "random", lambda obs, mask, rng: mask.index(False))
    with pytest.raises(RuntimeError, match="illegal"):
        run_game(cheat, greedy_policy, seed=EVAL_SEED_BASE)


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


def test_policy_agent_deterministic_and_reads_only(tmp_path):
    path = _tiny_checkpoint(tmp_path)
    before = open(path, "rb").read()
    agent = make_agent("policy", checkpoint=path, name="Tiny")
    a = run_benchmark(agent, "greedy", games=2, base_seed=EVAL_SEED_BASE)
    b = run_benchmark(agent, "greedy", games=2, base_seed=EVAL_SEED_BASE)
    assert a.placements == b.placements          # argmax decisions, seeded env
    assert open(path, "rb").read() == before     # benchmark never writes weights
    assert agent.checkpoint == "policy_tiny.pt"  # basename only, no abs path


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
    assert os.path.sep not in (blob["checkpoint"] or "")   # no machine paths


def test_compare_files_sorts_and_warns(tmp_path):
    r1 = run_benchmark(make_agent("random", name="Rando"), "greedy", games=2)
    r2 = run_benchmark(make_agent("greedy", name="Greed"), "greedy", games=2)
    p1, p2 = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    save_json(r1, p1)
    save_json(r2, p2)
    table = compare_files([p1, p2])
    assert "Agent" in table and "Rando" in table and "Greed" in table
    order = sorted([r1, r2], key=lambda r: r.metrics["avg_placement"])
    assert table.find(order[0].agent.name) < table.find(order[1].agent.name)
    assert "WARNING" not in table                # same version/field/games/seed

    r3 = run_benchmark(make_agent("random", name="OtherSeed"), "greedy",
                       games=2, base_seed=EVAL_SEED_BASE + 77)
    p3 = str(tmp_path / "c.json")
    save_json(r3, p3)
    assert "WARNING" in compare_files([p1, p3])  # mismatched seeds flagged
