"""card2vec tests. Training tests skip without torch; the *use* of committed
embeddings (cosine/learned_fit) is stdlib and always runs."""

import os
import pytest

from hsbg_coach.synergy import load_embeddings, _cosine, learned_fit, score_card
from hsbg_coach.cards import CardKnowledge


def K(name, tribes=None, text=""):
    return CardKnowledge(card_id=name, name=name, tier=1, attack=2, health=2,
                         tribes=tribes or [], keywords=[], text=text)


# --- stdlib inference path (no torch needed) --------------------------------
def test_cosine_basics():
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def test_learned_fit_prefers_similar_cards():
    emb = {"A": [1.0, 0.0], "B": [0.9, 0.1], "C": [0.0, 1.0]}
    near = learned_fit("A", ["B"], emb)
    far = learned_fit("A", ["C"], emb)
    assert near > far


def test_score_card_adds_learned_term():
    emb = {"Cand": [1.0, 0.0], "OnBoard": [0.95, 0.05]}
    v = score_card(K("Cand"), [K("OnBoard")], embeddings=emb)
    assert any("learned synergy" in r for r in v.reasons)


def test_committed_embeddings_are_coherent():
    emb = load_embeddings()
    if not emb:
        pytest.skip("no committed card2vec.json")
    # Murloc Tidecaller-style: its nearest neighbor should be strongly similar.
    name = next(iter(emb))
    sims = sorted((_cosine(emb[name], v) for n, v in emb.items() if n != name),
                  reverse=True)
    assert sims and sims[0] > 0.3            # top neighbor is clearly related


# --- training path (needs torch) --------------------------------------------
def test_card2vec_learns_cooccurrence():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from ml.card2vec import train_embeddings, similarity
    # Two clusters that always co-occur: {A,B,C} and {X,Y,Z}.
    boards = ([["A", "B", "C"]] * 60) + ([["X", "Y", "Z"]] * 60)
    vecs = train_embeddings(boards, dim=16, epochs=15, neg_k=3, min_count=2,
                            verbose=False)
    assert set(vecs) >= {"A", "B", "C", "X", "Y", "Z"}
    # Within-cluster similarity should beat cross-cluster.
    assert similarity(vecs, "A", "B") > similarity(vecs, "A", "X")
