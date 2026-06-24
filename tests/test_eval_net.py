"""Board-evaluation net tests. Feature/dataset plumbing is stdlib+numpy and
always runs; the training + model round-trip need torch and skip without it."""

import json
import os

import pytest

np = pytest.importorskip("numpy")

from ml.board_features import (
    board_vector, feature_dim, minion_from_population, minion_from_snapshot, _minion,
)
from ml.board_dataset import (
    build_hero_vocab, to_arrays, group_split, trajectory_examples, UNKNOWN_HERO,
)
from hsbg_coach import cards


EMB = {"A": [1.0, 0.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0, 0.0],
       "C": [0.0, 0.0, 1.0, 0.0], "D": [0.0, 0.0, 0.0, 1.0]}


def mk(name, atk=3, health=3, tribe="Beast"):
    return _minion(name, atk, health, 2, False, False, False, False, [tribe])


# --- features ---------------------------------------------------------------
def test_board_vector_length_matches_feature_dim():
    v = board_vector([mk("A"), mk("B")], EMB)
    assert v.shape[0] == feature_dim(EMB)


def test_population_and_snapshot_normalizers():
    byname = cards.by_name(cards.load_kb())
    pop = minion_from_population(
        {"cardID": "ZZZ", "tags": {"ATK": "9", "HEALTH": "7", "PREMIUM": "1"}},
        {"ZZZ": "Test Minion"}, byname)
    assert pop["name"] == "Test Minion" and pop["atk"] == 9 and pop["golden"]
    snap = minion_from_snapshot({"name": "Test Minion", "attack": 4, "health": 5}, byname)
    assert snap["atk"] == 4 and snap["health"] == 5


# --- dataset plumbing -------------------------------------------------------
def test_group_split_holds_out_whole_groups():
    ex = [{"minions": [mk("A")], "hero": f"h{i%4}", "label": 3.0,
           "group": f"h{i%4}"} for i in range(40)]
    tr, va = group_split(ex, val_frac=0.25, seed=1)
    assert tr and va
    assert set(e["group"] for e in tr).isdisjoint(e["group"] for e in va)


def test_to_arrays_standardizes_and_shapes():
    ex = [{"minions": [mk("A"), mk("B")], "hero": "h0", "label": 2.0, "group": "g"},
          {"minions": [mk("C"), mk("D")], "hero": "h1", "label": 5.0, "group": "g"}]
    stoi = build_hero_vocab(ex)
    X, hero, y, stats = to_arrays(ex, EMB, stoi)
    assert X.shape[0] == 2 and hero.shape[0] == 2 and y.tolist() == [2.0, 5.0]
    assert UNKNOWN_HERO in stoi


def test_trajectory_ingestion_continual(tmp_path):
    """A recorded game with a placement becomes a labeled example."""
    name = next(c.name for c in cards.load_kb().values())
    rec = {"placement": 1, "game_id": "g1",
           "state": {"board": [{"name": name, "attack": 5, "health": 5},
                               {"name": name, "attack": 4, "health": 4}]}}
    p = tmp_path / "20260624.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    ex = trajectory_examples(str(tmp_path))
    assert len(ex) == 1 and ex[0]["label"] == 1.0 and len(ex[0]["minions"]) == 2


# --- model (needs torch) ----------------------------------------------------
def test_eval_net_learns_board_strength():
    pytest.importorskip("torch")
    from ml.eval_net import train, evaluate
    # Label = 8 - (#strong cards): boards with A/B finish better than C/D.
    rng = np.random.default_rng(0)
    ex = []
    for i in range(400):
        strong = i % 2 == 0
        cs = ["A", "B"] if strong else ["C", "D"]
        ex.append({"minions": [mk(cs[0]), mk(cs[1])], "hero": "h0",
                   "label": 2.0 if strong else 6.0, "group": f"g{i}"})
    stoi = build_hero_vocab(ex)
    tr, va = group_split(ex, seed=2)
    Xtr, htr, ytr, st = to_arrays(tr, EMB, stoi)
    Xva, hva, yva, _ = to_arrays(va, EMB, stoi, stats=st)
    model, hist = train(Xtr, htr, ytr, n_heroes=len(stoi), epochs=30,
                        val=(Xva, hva, yva), verbose=False)
    assert hist["val_mae"] < 1.0           # learns the 2-vs-6 split cleanly


def test_eval_model_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from ml.eval_net import train, EvalModel
    ex = [{"minions": [mk("A"), mk("B")], "hero": "h0", "label": 3.0, "group": "g"}
          for _ in range(20)]
    stoi = build_hero_vocab(ex)
    X, hero, y, stats = to_arrays(ex, EMB, stoi)
    model, _ = train(X, hero, y, n_heroes=len(stoi), epochs=5, verbose=False)
    em = EvalModel(model, stoi, stats, EMB)
    out = str(tmp_path / "m.pt")
    em.save(out)
    loaded = EvalModel.load(out, EMB)
    a = em.predict([mk("A"), mk("B")], "h0")["placement"]
    b = loaded.predict([mk("A"), mk("B")], "h0")["placement"]
    assert abs(a - b) < 1e-4
