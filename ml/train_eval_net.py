"""Train the board-evaluation net on the meta — and on your own games.

  # population only (the meta prior)
  python -m ml.train_eval_net --comp-source comp.json --cards-source cards.json

  # fold in your recorded games (continual learning)
  python -m ml.train_eval_net --comp-source comp.json --trajectories data/

Reports val MAE (placement units) and Pearson r (does a higher predicted finish
track the real one), split by comp/game so val groups are unseen. Saves to
ml/eval_net.pt.
"""

import argparse
import os

from hsbg_coach.synergy import load_embeddings
from .board_dataset import (
    build_examples, trajectory_examples, build_hero_vocab, to_arrays, group_split,
)
from .eval_net import train, EvalModel

_OUT = os.path.join(os.path.dirname(__file__), "eval_net.pt")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the board-evaluation net")
    p.add_argument("--comp-source", help="local comp JSON (else fetch live)")
    p.add_argument("--cards-source", help="local HearthstoneJSON cards.json")
    p.add_argument("--period", default="past-seven")
    p.add_argument("--trajectories", help="dir of recorded *.jsonl games to fold in")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default=_OUT)
    a = p.parse_args(argv)

    emb = load_embeddings()
    if not emb:
        print("No card2vec.json — run `python -m ml.train_card2vec` first.")
        return 1

    print("Building dataset…")
    kw = {"cards_source": a.cards_source} if a.cards_source else {}
    examples = build_examples(comp_source=a.comp_source, period=a.period, **kw)
    print(f"  population boards: {len(examples)}")
    if a.trajectories:
        traj = trajectory_examples(a.trajectories)
        print(f"  your recorded boards: {len(traj)}")
        examples += traj
    if not examples:
        print("No examples.")
        return 1

    hero_stoi = build_hero_vocab(examples)
    train_ex, val_ex = group_split(examples)
    Xtr, htr, ytr, stats = to_arrays(train_ex, emb, hero_stoi)
    Xva, hva, yva, _ = to_arrays(val_ex, emb, hero_stoi, stats=stats)
    print(f"  train {len(train_ex)}  val {len(val_ex)}  "
          f"features {Xtr.shape[1]}  heroes {len(hero_stoi)}")

    model, hist = train(Xtr, htr, ytr, n_heroes=len(hero_stoi),
                        epochs=a.epochs, val=(Xva, hva, yva))
    print(f"\nval MAE {hist['val_mae']:.3f} placements | "
          f"val Pearson r {hist['val_r']:.3f}")

    EvalModel(model, hero_stoi, stats, emb).save(a.out)
    print(f"Saved -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
