"""Calibration diagnostic — is the board brain honest on MID-GAME states?

The old eval net trained on final boards and was queried on mid-game boards;
this script quantifies that distribution shift and verifies the fix. It builds
a fresh env-generated validation set (never seen by either model) and reports,
per game stage, each model's MAE and mean-predicted vs mean-actual placement.

  python -m ml.calibrate --lobbies 60

A model is *calibrated* when mean-pred ≈ mean-actual in every bucket; the old
net's mid-game rows are where it drifts.
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from hsbg_coach.synergy import load_embeddings

_DIR = os.path.dirname(__file__)


def _bucket(turn: int) -> str:
    return "early(1-5)" if turn <= 5 else "mid(6-9)" if turn <= 9 else "late(10+)"


def _report(name: str, preds, examples) -> None:
    rows = defaultdict(list)
    for p, e in zip(preds, examples):
        rows[_bucket(e["turn"])].append((abs(p - e["label"]), p, e["label"]))
    print(f"\n{name}  (MAE | mean pred | mean actual | n)")
    for k in ("early(1-5)", "mid(6-9)", "late(10+)"):
        v = rows.get(k)
        if not v:
            continue
        print(f"  {k:11s}  {np.mean([x[0] for x in v]):.3f} | "
              f"{np.mean([x[1] for x in v]):.2f} | "
              f"{np.mean([x[2] for x in v]):.2f} | {len(v)}")
    allv = [x for v in rows.values() for x in v]
    print(f"  {'overall':11s}  {np.mean([x[0] for x in allv]):.3f} | "
          f"{np.mean([x[1] for x in allv]):.2f} | "
          f"{np.mean([x[2] for x in allv]):.2f} | {len(allv)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Calibration check on mid-game states")
    p.add_argument("--lobbies", type=int, default=60)
    p.add_argument("--seed", type=int, default=777)   # disjoint from training
    a = p.parse_args(argv)

    emb = load_embeddings()
    from .midgame_dataset import generate_examples
    examples = generate_examples(a.lobbies, seed=a.seed)
    print(f"validation states: {len(examples)} from {a.lobbies} fresh lobbies")

    ran = False
    set_path = os.path.join(_DIR, "set_net.pt")
    if os.path.isfile(set_path):
        from .set_net import SetEvalModel
        m = SetEvalModel.load(set_path, emb)
        preds = [m.predict(e["minions"], e["hero"], state=e["state"])["placement"]
                 for e in examples]
        _report("set_net (attention, mid-game trained)", preds, examples)
        ran = True

    old_path = os.path.join(_DIR, "eval_net.pt")
    if os.path.isfile(old_path):
        from .eval_net import EvalModel
        m = EvalModel.load(old_path, emb)
        preds = [m.predict(e["minions"], e["hero"], state=e["state"])["placement"]
                 for e in examples]
        _report("eval_net (MLP, final-board trained)", preds, examples)
        ran = True

    if not ran:
        print("No trained model found (ml/set_net.pt or ml/eval_net.pt).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
