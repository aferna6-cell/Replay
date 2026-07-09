"""Train the set-transformer board brain on mid-game states.

  # env self-play mid-game data (the default, no network needed)
  python -m ml.train_set_net --midgame-lobbies 300 --epochs 30

  # fold in population final boards and/or your own recorded games
  python -m ml.train_set_net --comp-source comp.json --trajectories data/

Reports val MAE / Pearson r overall AND per turn bucket — the calibration view
that exposed the old net's mid-game blind spot. Saves ml/set_net.pt, which
`board_value.get_scorer()` prefers over the MLP automatically.
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from hsbg_coach.synergy import load_embeddings
from .board_dataset import (
    build_examples, trajectory_examples, build_hero_vocab, group_split,
)
from .midgame_dataset import generate_examples
from .set_net import SetEvalModel, train, evaluate
from .tokens import examples_to_arrays

_OUT = os.path.join(os.path.dirname(__file__), "set_net.pt")


def per_turn_mae(model, val_ex, val_data) -> dict:
    import torch
    toks, mask, ctx, hero, y = [torch.from_numpy(a) for a in val_data]
    model.eval()
    with torch.no_grad():
        pred = model.expected_placement(model(toks, mask, ctx, hero)).numpy()
    buckets = defaultdict(list)
    for e, p, actual in zip(val_ex, pred, val_data[4]):
        t = e.get("turn") or (e.get("state") or {}).get("turn") or 0
        key = "early(1-5)" if t <= 5 else "mid(6-9)" if t <= 9 else "late(10+)"
        buckets[key].append((abs(p - actual), p, actual))
    return {k: (float(np.mean([b[0] for b in v])),
                float(np.mean([b[1] for b in v])),
                float(np.mean([b[2] for b in v])), len(v))
            for k, v in sorted(buckets.items())}


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the set-transformer eval net")
    p.add_argument("--midgame-lobbies", type=int, default=300,
                   help="env self-play lobbies to generate (0 = skip)")
    p.add_argument("--comp-source", help="population comp JSON (else skip)")
    p.add_argument("--cards-source", help="local HearthstoneJSON cards.json")
    p.add_argument("--period", default="past-seven")
    p.add_argument("--trajectories", help="dir of recorded *.jsonl games")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=_OUT)
    a = p.parse_args(argv)

    emb = load_embeddings()
    if not emb:
        print("No card2vec.json — run `python -m ml.train_card2vec` first.")
        return 1

    examples = []
    if a.midgame_lobbies:
        print(f"Generating {a.midgame_lobbies} env self-play lobbies…")
        mid = generate_examples(a.midgame_lobbies, seed=a.seed)
        print(f"  mid-game states: {len(mid)}")
        examples += mid
    if a.comp_source:
        kw = {"cards_source": a.cards_source} if a.cards_source else {}
        pop = build_examples(comp_source=a.comp_source, period=a.period, **kw)
        print(f"  population final boards: {len(pop)}")
        examples += pop
    if a.trajectories:
        traj = trajectory_examples(a.trajectories)
        print(f"  your recorded boards: {len(traj)}")
        examples += traj
    if not examples:
        print("No examples.")
        return 1

    hero_stoi = build_hero_vocab(examples)
    train_ex, val_ex = group_split(examples, seed=a.seed)
    ttoks, tmask, tctx, thero, ty, ctx_stats = examples_to_arrays(
        train_ex, emb, hero_stoi)
    vtoks, vmask, vctx, vhero, vy, _ = examples_to_arrays(
        val_ex, emb, hero_stoi, ctx_stats=ctx_stats)
    print(f"  train {len(train_ex)}  val {len(val_ex)}  "
          f"token dim {ttoks.shape[-1]}  heroes {len(hero_stoi)}")

    model, hist = train((ttoks, tmask, tctx, thero, ty),
                        n_heroes=len(hero_stoi), epochs=a.epochs, seed=a.seed,
                        val=(vtoks, vmask, vctx, vhero, vy))
    print(f"\nval MAE {hist['val_mae']:.3f} placements | "
          f"val Pearson r {hist['val_r']:.3f}")
    print("\nCalibration by game stage (MAE | mean pred | mean actual | n):")
    for k, (mae, mp, ma, n) in per_turn_mae(
            model, val_ex, (vtoks, vmask, vctx, vhero, vy)).items():
        print(f"  {k:11s}  {mae:.3f} | {mp:.2f} | {ma:.2f} | {n}")

    SetEvalModel(model, hero_stoi, ctx_stats, emb).save(a.out)
    print(f"Saved -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
