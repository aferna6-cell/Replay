"""Train card2vec on the winning boards and save embeddings.

  python -m ml.train_card2vec --period past-seven --dim 48 --epochs 5

Writes data/cards/card2vec.json: {"cards": {name: [floats], ...}, "_meta": {...}}.
By default fetches the comp data live; pass --comp-source / --cards-source to use
local files.
"""

import argparse
import json
import os

from .board_corpus import extract_boards
from .card2vec import train_embeddings, most_similar

_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cards", "card2vec.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train card2vec on winning boards")
    p.add_argument("--period", default="past-seven")
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--comp-source", help="local comp JSON (else fetch live)")
    p.add_argument("--cards-source", help="local HearthstoneJSON cards.json")
    p.add_argument("--out", default=_OUT)
    a = p.parse_args(argv)

    print("Extracting winning boards…")
    kw = {}
    if a.cards_source:
        kw["cards_source"] = a.cards_source
    boards = extract_boards(comp_source=a.comp_source, period=a.period, **kw)
    print(f"{len(boards)} winning boards.")
    vectors = train_embeddings(boards, dim=a.dim, epochs=a.epochs,
                               min_count=a.min_count)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rounded = {name: [round(float(x), 4) for x in vec] for name, vec in vectors.items()}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"_meta": {"dim": a.dim, "period": a.period, "vocab": len(vectors)},
                   "cards": rounded}, fh)
    print(f"Saved {len(vectors)} card vectors -> {a.out}")
    # Sanity peek.
    import numpy as np
    vecs = {k: np.asarray(v) for k, v in vectors.items()}
    for probe in list(vecs)[:1]:
        sims = most_similar(vecs, probe, 5)
        print(f"  most similar to {probe}: {[s[0] for s in sims]}")


if __name__ == "__main__":
    main()
