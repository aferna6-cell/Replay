"""Paired-analysis layer tests (ml/analyze_benchmark.py)."""

import json

import pytest

np = pytest.importorskip("numpy")

from ml.analyze_benchmark import (
    compare_pair, load_result, main, paired_diff, verify_paired,
)
from ml.benchmark import EVAL_SEED_BASE, make_agent, run_benchmark, save_json


# --- paired difference math ---------------------------------------------------
def test_paired_diff_known_values():
    d = paired_diff([1, 1, 1, 1], [3, 3, 3, 3], seed=0)
    assert d["n"] == 4
    assert d["mean_diff"] == pytest.approx(-2.0)
    assert d["ci95"] == [-2.0, -2.0]             # constant diffs collapse


def test_paired_diff_zero_centered():
    d = paired_diff([1, 2, 3], [3, 2, 1], seed=0)   # diffs [-2, 0, 2]
    assert d["mean_diff"] == pytest.approx(0.0)
    assert d["ci95"][0] <= 0.0 <= d["ci95"][1]


def test_paired_diff_deterministic_and_seeded():
    pa, pb = [1, 4, 2, 8, 5, 7], [3, 3, 3, 3, 3, 3]
    assert paired_diff(pa, pb, seed=7) == paired_diff(pa, pb, seed=7)
    assert paired_diff(pa, pb, seed=7) != paired_diff(pa, pb, seed=8)


def test_paired_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        paired_diff([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="equal-length"):
        paired_diff([], [])


# --- pairing verification -----------------------------------------------------
def _fake(agent="A", games=3, base=EVAL_SEED_BASE, field="greedy",
          placements=None):
    return {"benchmark_version": "Replay Benchmark v1", "agent": agent,
            "field": field, "games": games, "base_seed": base,
            "seed_range": [base, base + games - 1],
            "environment": {"env": "bg", "n_players": 8},
            "metrics": {}, "placements": placements or [1] * games}


def test_verify_paired_accepts_matching_runs():
    verify_paired(_fake("A"), _fake("B"))        # must not raise


@pytest.mark.parametrize("kw", [
    {"games": 4, "placements": [1] * 4},         # different game counts
    {"base": EVAL_SEED_BASE + 1},                # different seeds
    {"field": "random"},                         # different opponent field
])
def test_verify_paired_rejects_mismatches(kw):
    with pytest.raises(ValueError, match="not paired-comparable"):
        verify_paired(_fake("A"), _fake("B", **kw))


def test_verify_paired_rejects_environment_mismatch():
    b = _fake("B")
    b["environment"] = {"env": "bg", "n_players": 4}
    with pytest.raises(ValueError, match="environment"):
        verify_paired(_fake("A"), b)


def test_compare_pair_verdicts():
    a, b = _fake("A", placements=[1, 1, 1]), _fake("B", placements=[8, 8, 8])
    assert compare_pair(a, b)["verdict"] == "A places better"   # lower wins
    assert compare_pair(b, a)["verdict"] == "A places better"
    tie = compare_pair(_fake("A", placements=[1, 8, 4]),
                       _fake("B", placements=[8, 1, 4]))
    assert "no clear difference" in tie["verdict"]


# --- result loading -----------------------------------------------------------
def test_load_result_roundtrip_and_seed_alignment(tmp_path):
    res = run_benchmark(make_agent("random"), "greedy", games=3)
    path = str(tmp_path / "r.json")
    save_json(res, path)
    blob = load_result(path)
    assert blob["placements"] == res.placements
    assert blob["seed_range"] == [EVAL_SEED_BASE, EVAL_SEED_BASE + 2]


def test_load_result_rejects_suite_wrapper_and_bad_shapes(tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"benchmark_version": "x", "results": []}))
    with pytest.raises(ValueError, match="single-result"):
        load_result(str(suite))
    trunc = _fake("A")
    trunc["placements"] = trunc["placements"][:-1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(trunc))
    with pytest.raises(ValueError, match="placements length"):
        load_result(str(bad))
    old = _fake("A")
    del old["placements"]
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="no per-game placements"):
        load_result(str(legacy))


def test_cli_end_to_end(tmp_path, capsys):
    r1 = run_benchmark(make_agent("random", name="Rando"), "greedy", games=3)
    r2 = run_benchmark(make_agent("greedy", name="Greed"), "greedy", games=3)
    p1, p2 = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    save_json(r1, p1)
    save_json(r2, p2)
    out_json = str(tmp_path / "pairs.json")
    assert main([p1, p2, "--json-out", out_json]) == 0
    printed = capsys.readouterr().out
    assert "Rando" in printed and "Greed" in printed
    pairs = json.load(open(out_json))["pairs"]
    assert len(pairs) == 1 and pairs[0]["n"] == 3
    # deterministic: identical runs of the analysis agree exactly
    assert main([p1, p2]) == 0
    assert pairs[0] == compare_pair(load_result(p1), load_result(p2))
