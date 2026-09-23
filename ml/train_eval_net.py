"""Train the board-evaluation net.

HSReplay-snapshot mode (no Firestone, no VODs):

  python -m ml.train_eval_net ^
    --hsreplay-snapshot data/hsreplay/36.6.1 ^
    --trajectories data/train_perfect_thin150 ^
    --epochs 40

Firestone population mode (legacy):

  python -m ml.train_eval_net --comp-source comp.json --trajectories data/
"""
import argparse
import os

from hsbg_coach.synergy import load_embeddings
from .board_dataset import (
    build_examples,
    trajectory_examples,
    build_hero_vocab,
    to_arrays,
    group_split,
)
from .hsreplay_snapshot_dataset import build_hsreplay_snapshot_examples
from .eval_net import train, EvalModel

_OUT = os.path.join(os.path.dirname(__file__), "eval_net.pt")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the board-evaluation net")
    p.add_argument("--comp-source", help="local Firestone comp JSON (else fetch live)")
    p.add_argument("--cards-source", help="local HearthstoneJSON cards.json")
    p.add_argument("--period", default="past-seven")
    p.add_argument(
        "--hsreplay-snapshot",
        help="local HSReplay snapshot dir (data/hsreplay/36.6.1). "
             "When set, skips Firestone entirely.",
    )
    p.add_argument(
        "--trajectories",
        help="dir of recorded *.jsonl boards to fold in "
             "(use data/train_perfect_thin150 for perfect boards only)",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default=_OUT)
    p.add_argument(
        "--with-context",
        action="store_true",
        help="fold whole game state into features, not just the board",
    )
    a = p.parse_args(argv)

    emb = load_embeddings()
    if not emb:
        print("No card2vec.json - run `python -m ml.train_card2vec` first.")
        return 1

    print("Building dataset...")
    if a.hsreplay_snapshot:
        examples = build_hsreplay_snapshot_examples(a.hsreplay_snapshot)
        print(f"  population boards (hsreplay snapshot): {len(examples)}")
    else:
        kw = {"cards_source": a.cards_source} if a.cards_source else {}
        examples = build_examples(
            comp_source=a.comp_source, period=a.period, **kw
        )
        print(f"  population boards (firestone): {len(examples)}")

    if a.trajectories:
        traj = trajectory_examples(a.trajectories)
        print(f"  trajectory boards: {len(traj)}")
        examples += traj

    if not examples:
        print("No examples.")
        return 1

    hero_stoi = build_hero_vocab(examples)
    train_ex, val_ex = group_split(examples)
    Xtr, htr, ytr, stats = to_arrays(
        train_ex, emb, hero_stoi, with_context=a.with_context
    )
    Xva, hva, yva, _ = to_arrays(
        val_ex, emb, hero_stoi, stats=stats, with_context=a.with_context
    )
    print(
        f"  train {len(train_ex)}  val {len(val_ex)}  "
        f"features {Xtr.shape[1]}  heroes {len(hero_stoi)}"
    )

    model, hist = train(
        Xtr, htr, ytr, n_heroes=len(hero_stoi), epochs=a.epochs, val=(Xva, hva, yva)
    )
    print(
        f"\nval MAE {hist['val_mae']:.3f} placements | "
        f"val Pearson r {hist['val_r']:.3f}"
    )

    EvalModel(model, hero_stoi, stats, emb, with_context=a.with_context).save(a.out)
    print(f"Saved -> {a.out}  (with_context={a.with_context})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
