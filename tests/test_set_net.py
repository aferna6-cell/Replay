"""Set-transformer eval net tests. Token plumbing is numpy-only; the training
and round-trip tests need torch and skip without it."""

import os

import pytest

np = pytest.importorskip("numpy")

from ml.board_features import _minion
from ml.tokens import (
    board_tokens, minion_token, token_dim, examples_to_arrays, KB_KEYWORDS,
)

EMB = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]}


def mk(name, atk=3, health=3, tier=2, tribe="Beast", **kw):
    return _minion(name, atk, health, tier, kw.get("golden", False),
                   kw.get("divine", False), kw.get("reborn", False),
                   kw.get("taunt", False), [tribe])


def ex(names, label, turn=7):
    return {"minions": [mk(n) for n in names], "hero": "UNKNOWN",
            "label": label, "state": {"turn": turn, "tavern_tier": 3,
                                      "gold": 5, "hero_health": 25},
            "group": f"g{label}", "turn": turn}


# --- tokens -------------------------------------------------------------------
def test_token_dim_and_shape():
    t = minion_token(mk("A"), EMB, byname={})
    assert t.shape[0] == token_dim(EMB)


def test_board_tokens_mask_and_padding():
    toks, mask = board_tokens([mk("A"), mk("B")], EMB, byname={})
    assert toks.shape == (7, token_dim(EMB)) and mask.tolist()[:3] == [1, 1, 0]
    assert not toks[2].any()


def test_tokens_distinguish_individual_cards():
    """The failure mode of mean-pooling: two different boards must produce
    different token sets even when their mean embedding is similar."""
    a, _ = board_tokens([mk("A"), mk("B")], EMB, byname={})
    b, _ = board_tokens([mk("B"), mk("A")], EMB, byname={})
    assert (a[0] != b[0]).any()          # per-slot identity is preserved


def test_examples_to_arrays_shapes():
    exs = [ex(["A", "B"], 3.0), ex(["C"], 6.0)]
    toks, mask, ctx, hero, y, stats = examples_to_arrays(
        exs, EMB, {"UNKNOWN": 0}, byname={})
    assert toks.shape[0] == 2 and ctx.shape == (2, 8)
    assert y.tolist() == [3.0, 6.0]
    toks2, *_ = examples_to_arrays(exs, EMB, {"UNKNOWN": 0}, ctx_stats=stats,
                                   byname={})
    assert np.allclose(toks, toks2)


# --- soft targets + net (torch) -------------------------------------------------
def test_soft_targets_interpolate():
    torch = pytest.importorskip("torch")
    from ml.set_net import soft_targets
    t = soft_targets(np.array([3.25, 1.0, 8.0]))
    assert np.allclose(t.sum(axis=1), 1.0)
    assert np.isclose(t[0, 2], 0.75) and np.isclose(t[0, 3], 0.25)
    assert t[1, 0] == 1.0 and t[2, 7] == 1.0


def _tiny_dataset(n=160, seed=0):
    """Boards of A's are strong (place ~2), boards of C's weak (place ~7)."""
    rng = np.random.default_rng(seed)
    exs = []
    for i in range(n):
        good = rng.random() < 0.5
        names = ["A", "B"] if good else ["C"]
        exs.append(ex(names, 2.0 if good else 7.0, turn=int(rng.integers(4, 12))))
        exs[-1]["group"] = f"g{i % 8}"
    return exs


def test_training_learns_separation_and_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.set_net import SetEvalModel, train
    from ml.tokens import examples_to_arrays

    exs = _tiny_dataset()
    stoi = {"UNKNOWN": 0}
    toks, mask, ctx, hero, y, stats = examples_to_arrays(exs, EMB, stoi, byname={})
    model, hist = train((toks, mask, ctx, hero, y), n_heroes=1, epochs=25,
                        val=(toks, mask, ctx, hero, y), verbose=False, seed=0)
    assert hist["val_mae"] < 1.5                     # separates strong from weak

    wrapper = SetEvalModel(model, stoi, stats, EMB)
    wrapper._byname = {}
    good = wrapper.predict([mk("A"), mk("B")], state={"turn": 7})
    bad = wrapper.predict([mk("C")], state={"turn": 7})
    assert good["placement"] < bad["placement"]
    assert len(good["dist"]) == 8
    assert abs(sum(good["dist"]) - 1.0) < 1e-4

    path = str(tmp_path / "set_net.pt")
    wrapper.save(path)
    loaded = SetEvalModel.load(path, EMB)
    loaded._byname = {}
    again = loaded.predict([mk("A"), mk("B")], state={"turn": 7})
    assert abs(again["placement"] - good["placement"]) < 1e-5


def test_scorer_prefers_set_net(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from ml.set_net import SetEvalNet, SetEvalModel
    from ml import eval_scorer
    from ml.tokens import token_dim

    net = SetEvalNet(token_dim(EMB), 1)
    wrapper = SetEvalModel(net, {"UNKNOWN": 0},
                           (np.zeros(8), np.ones(8)), EMB)
    path = str(tmp_path / "set_net.pt")
    wrapper.save(path)
    monkeypatch.setattr(eval_scorer, "load_embeddings", lambda: EMB)
    scorer = eval_scorer.load_default_scorer(
        model_path=str(tmp_path / "missing.pt"), set_model_path=path)
    assert scorer is not None and scorer.name == "set_net"
    eq = scorer.equity([{"name": "A", "attack": 3, "health": 3}], "UNKNOWN")
    assert 0.0 <= eq <= 1.0


# --- mid-game dataset ------------------------------------------------------------
def test_midgame_examples_have_labels_and_context():
    from ml.midgame_dataset import generate_examples
    exs = generate_examples(lobbies=2, seed=1)
    assert len(exs) > 20
    turns = {e["turn"] for e in exs}
    assert min(turns) <= 3 and max(turns) >= 8       # covers early AND late
    for e in exs[:10]:
        assert 1 <= e["label"] <= 8
        assert e["state"]["turn"] == e["turn"]
        assert e["minions"] and all("name" in m for m in e["minions"])
