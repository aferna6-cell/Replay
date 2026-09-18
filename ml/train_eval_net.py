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
    mix_population_and_personal,
)
from hsbg_coach import config as coach_config
from hsbg_coach.learn import write_learn_meta
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
    p.add_argument("--with-context", action="store_true",
                   help="fold the whole game state (tier/gold/hp/turn/opponents/"
                        "trinkets/anomaly) into the features, not just the board")
    p.add_argument("--dry-run", action="store_true",
                   help="validate trajectories + print adaptive mix; no train/network")
    a = p.parse_args(argv)

    if a.dry_run:
        from hsbg_coach.learn import validate_trajectories, model_sidecar
        data = a.trajectories or coach_config.DATA_DIR
        st = validate_trajectories(data)
        print(f"dry-run trajectories: games={st['n_games']} rows={st['n_rows']} "
              f"placement={st['n_with_placement']}")
        print(f"dry-run mix: pop={st['population_weight']} "
              f"personal={st['personal_weight']}")
        side = model_sidecar(a.out)
        print(f"dry-run model exists={side['exists']} sha={side.get('file_sha256_12')}")
        return 0 if st["ok"] else 1

    emb = load_embeddings()
    if not emb:
        print("No card2vec.json — run `python -m ml.train_card2vec` first.")
        return 1

    print("Building dataset…")
    kw = {"cards_source": a.cards_source} if a.cards_source else {}
    pop = build_examples(comp_source=a.comp_source, period=a.period, **kw)
    print(f"  population boards: {len(pop)}")
    traj = trajectory_examples(a.trajectories) if a.trajectories else []
    n_games = coach_config.count_recorded_games(a.trajectories) if a.trajectories else 0
    print(f"  your recorded boards: {len(traj)}  (games={n_games})")
    examples, pw, pop_w, reps = mix_population_and_personal(pop, traj, n_games)
    if traj:
        print(f"  adaptive mix: personal_weight={pw:.3f} population_weight={pop_w:.3f} "
              f"personal_upsample×{reps} → train_pool={len(examples)}")
    if not examples:
        print("No examples.")
        return 1

    hero_stoi = build_hero_vocab(examples)
    train_ex, val_ex = group_split(examples)
    Xtr, htr, ytr, stats = to_arrays(train_ex, emb, hero_stoi,
                                     with_context=a.with_context)
    Xva, hva, yva, _ = to_arrays(val_ex, emb, hero_stoi, stats=stats,
                                 with_context=a.with_context)
    print(f"  train {len(train_ex)}  val {len(val_ex)}  "
          f"features {Xtr.shape[1]}  heroes {len(hero_stoi)}")

    model, hist = train(Xtr, htr, ytr, n_heroes=len(hero_stoi),
                        epochs=a.epochs, val=(Xva, hva, yva))
    print(f"\nval MAE {hist['val_mae']:.3f} placements | "
          f"val Pearson r {hist['val_r']:.3f}")

    EvalModel(model, hero_stoi, stats, emb,
              with_context=a.with_context).save(a.out)
    write_learn_meta(
        a.out,
        n_personal_games=n_games,
        n_population=len(pop),
        n_personal_examples=len(traj),
        personal_w=pw if traj else 0.0,
        population_w=pop_w if traj else 1.0,
        with_context=a.with_context,
        epochs=a.epochs,
    )
    print(f"Saved -> {a.out}  (with_context={a.with_context})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
